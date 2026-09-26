/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.vector.shuffle.flight

import java.nio.charset.StandardCharsets
import java.nio.file.Files

import org.apache.arrow.flight._
import org.apache.arrow.flight.auth2.BearerCredentialWriter
import org.apache.arrow.flight.grpc.CredentialCallOption
import org.apache.arrow.memory.RootAllocator
import org.apache.spark.sql.SparkSession

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/**
 * #288 slice 3: two executor JVMs (`local-cluster[2,1,1024]`), so a reducer's fetch of the other
 * executor's map output is a remote `DoGet` against that executor's Flight server; the rows equal
 * Spark's and the shuffle read metrics show remote blocks. With `spark.authenticate` on, a client
 * without the secret is refused.
 */
class FlightShuffleClusterSuite extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  private def session(authenticate: Boolean): SparkSession = {
    val tempDir = Files.createTempDirectory("vecruntime-flight")
    // The workers launch executor JVMs: under spark.testing the launcher needs no SPARK_HOME jars and
    // the executors take the test JVM's classpath and its Vector API flags.
    System.setProperty("spark.testing", "true")
    System.setProperty("spark.test.home", tempDir.toString)
    // The launcher, in testing mode, derives the Scala version from a build directory under spark.test.home.
    Files.createDirectories(tempDir.resolve("launcher/target/scala-2.13"))
    Files.createDirectories(
      tempDir.resolve("assembly/target/scala-2.13/jars")
    ) // may be empty: the classpath is extraClassPath
    val jvmArgs = Seq(
      "--add-modules=jdk.incubator.vector",
      "--enable-native-access=ALL-UNNAMED",
      "--sun-misc-unsafe-memory-access=allow"
    ).mkString(" ")
    val b = SparkSession.builder()
      .master("local-cluster[2,1,1024]")
      .appName("FlightShuffleClusterSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.sql.warehouse.dir", tempDir.resolve("wh").toString)
      .config("spark.executor.extraClassPath", System.getProperty("java.class.path"))
      .config("spark.executor.extraJavaOptions", jvmArgs)
      .config("spark.plugins", "io.vecruntime.spark.VectorPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.sql.vector.shuffle.VectorShuffleManager")
      .config("spark.vector.shuffle.enabled", "true")
      .config("spark.vector.shuffle.backend", "flight")
      .config("spark.vector.exec.strictFloatingPoint", "true")
      // Small record batches: a block holds several, each with its own string dictionary (#338).
      .config("spark.vector.shuffle.batchBytes", "16k")
    if (authenticate)
      b.config("spark.authenticate", "true").config("spark.authenticate.secret", "flight-shuffle-test-secret")
    b.getOrCreate()
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
    SparkSession.clearActiveSession()
    SparkSession.clearDefaultSession()
  }

  test("a grouped aggregate across two executors fetches the other executor's map outputs over Flight") {
    spark = session(authenticate = true)
    val df = spark.range(0, 40000, 1, 8).selectExpr("id", "cast(id % 53 as int) as k", "cast(id % 300 as string) as s")
    df.createOrReplaceTempView("t")
    val sql = "select k, count(*) c, count(distinct s) ds, max(s) ms from t group by k"
    val ours = spark.sql(sql).collect().toSeq.sortBy(_.toString)
    spark.sessionState.conf.setConfString("spark.vector.enabled", "false")
    val expected =
      try spark.sql(sql).collect().toSeq.sortBy(_.toString)
      finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
    assert(ours === expected)
    assert(ours.size === 53)
    // Remote fetches happened: the metrics of the last query's shuffle read stages.
    val listener = new org.apache.spark.scheduler.SparkListener {
      var remote = 0L
      override def onStageCompleted(e: org.apache.spark.scheduler.SparkListenerStageCompleted): Unit =
        remote += e.stageInfo.taskMetrics.shuffleReadMetrics.remoteBlocksFetched
    }
    spark.sparkContext.addSparkListener(listener)
    spark.sql(sql).collect()
    spark.sparkContext.listenerBus.waitUntilEmpty()
    assert(listener.remote > 0, "no remote block was fetched: both executors served only themselves")
  }

  test("with spark.authenticate on, a DoGet without the secret is refused and one with it is served") {
    // The same Producer and SecretAuthenticator the executors' servers are built with, on a server of
    // this test's own: looking the executors' servers up in the JVM-wide registry served the call
    // unauthenticated whenever another suite's server (no secret) was still registered under the same
    // executor id -- the flaky failure of every fourth gate.
    val allocator = new RootAllocator()
    val block = Array.emptyByteArray
    val producer = new FlightShuffle.Producer(
      (_, _, _) => new org.apache.spark.network.buffer.NioManagedBuffer(java.nio.ByteBuffer.wrap(block)),
      allocator
    )
    val server = FlightServer.builder(allocator, Location.forGrpcInsecure("127.0.0.1", 0), producer)
      .headerAuthenticator(new FlightShuffle.SecretAuthenticator("flight-shuffle-test-secret")).build()
    server.start()
    val client = FlightClient.builder(allocator, Location.forGrpcInsecure("127.0.0.1", server.getPort)).build()
    try {
      val ticket = FlightShuffle.ticket(0, 0L, 0)
      val e = intercept[FlightRuntimeException] {
        val s = client.getStream(ticket)
        try s.next()
        finally s.close()
      }
      assert(e.status().code() === FlightStatusCode.UNAUTHENTICATED, e.toString)
      // With the secret the call is authenticated and the (empty) block is served.
      val s =
        client.getStream(ticket, new CredentialCallOption(new BearerCredentialWriter("flight-shuffle-test-secret")))
      try s.next()
      finally s.close()
      val e3 = intercept[FlightRuntimeException] {
        val s = client.getStream(ticket, new CredentialCallOption(new BearerCredentialWriter("wrong")))
        try s.next()
        finally s.close()
      }
      assert(e3.status().code() === FlightStatusCode.UNAUTHENTICATED, e3.toString)
    } finally {
      client.close()
      server.close()
      allocator.close()
    }
  }

  test(
    "#338: a remote block of several record batches, each with its own string dictionary, decodes every batch right"
  ) {
    // A block is one record batch per `batchBytes` of held rows; with a small cap a reduce partition's
    // block holds many, and the writer gives each its own replacement dictionary (the slice's distinct
    // strings). Flight writes a stream's dictionaries once, at its start: before #338 the client decoded
    // batches 2..n against batch 1's dictionary -- out-of-bounds indices, or the wrong string silently.
    // Strings unique per row make every batch's dictionary different (batchBytes is set on the session).
    {
      val df = spark.range(0, 200000, 1, 8).selectExpr(
        "id",
        "cast(id % 7 as int) as k",
        "concat('s-', cast(id as string)) as s"
      )
      df.createOrReplaceTempView("u")
      // The shuffle carries s; the aggregate reads it back on the other executor.
      val sql = "select k, count(distinct s) ds, max(s) ms, min(s) mn from u group by k order by k"
      val ours = spark.sql(sql).collect().toSeq
      spark.sessionState.conf.setConfString("spark.vector.enabled", "false")
      val expected =
        try spark.sql(sql).collect().toSeq
        finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
      assert(ours === expected)
      assert(ours.map(_.getLong(1)).sum === 200000L)
    }
  }
}
