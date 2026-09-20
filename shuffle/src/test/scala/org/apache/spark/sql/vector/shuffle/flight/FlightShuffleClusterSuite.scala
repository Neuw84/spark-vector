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
    val tempDir = Files.createTempDirectory("spark-vector-flight")
    // The workers launch executor JVMs: under spark.testing the launcher needs no SPARK_HOME jars and
    // the executors take the test JVM's classpath and its Vector API flags.
    System.setProperty("spark.testing", "true")
    System.setProperty("spark.test.home", tempDir.toString)
    // The launcher, in testing mode, derives the Scala version from a build directory under spark.test.home.
    Files.createDirectories(tempDir.resolve("launcher/target/scala-2.13"))
    Files.createDirectories(tempDir.resolve("assembly/target/scala-2.13/jars")) // may be empty: the classpath is extraClassPath
    val jvmArgs = Seq("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
      "--sun-misc-unsafe-memory-access=allow").mkString(" ")
    val b = SparkSession.builder()
      .master("local-cluster[2,1,1024]")
      .appName("FlightShuffleClusterSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.sql.warehouse.dir", tempDir.resolve("wh").toString)
      .config("spark.executor.extraClassPath", System.getProperty("java.class.path"))
      .config("spark.executor.extraJavaOptions", jvmArgs)
      .config("spark.plugins", "io.sparkvector.spark.VectorPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.sql.vector.shuffle.VectorShuffleManager")
      .config("spark.vector.shuffle.enabled", "true")
      .config("spark.vector.shuffle.backend", "flight")
      .config("spark.vector.exec.strictFloatingPoint", "true")
      // Small record batches: a block holds several, each with its own string dictionary (#338).
      .config("spark.vector.shuffle.batchBytes", "16k")
    if (authenticate) b.config("spark.authenticate", "true").config("spark.authenticate.secret", "flight-shuffle-test-secret")
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
    val expected = try spark.sql(sql).collect().toSeq.sortBy(_.toString) finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
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
    // The executors' servers are reachable by their registered locations; the driver plugin holds them.
    // One of THIS session's executors: the registry is a JVM-wide map and another suite's (unauthenticated)
    // server may still be registered when the suites share a JVM -- `anyLocation` then served the call.
    val execIds = spark.sparkContext.getExecutorIds()
    assert(execIds.nonEmpty, "no executors")
    val loc = execIds.iterator.flatMap(id => Option(org.apache.spark.sql.vector.shuffle.flight.FlightRegistry.driverReceive(
      org.apache.spark.sql.vector.shuffle.flight.LookupFlight(id)).asInstanceOf[FlightLocation])).nextOption()
      .getOrElse(fail(s"no Flight server registered for executors $execIds"))
    val allocator = new RootAllocator()
    val client = FlightClient.builder(allocator, Location.forGrpcInsecure(loc.host, loc.port)).build()
    try {
      val ticket = FlightShuffle.ticket(0, 0L, 0)
      val e = intercept[FlightRuntimeException] {
        val s = client.getStream(ticket)
        try s.next() finally s.close()
      }
      assert(e.status().code() === FlightStatusCode.UNAUTHENTICATED, e.toString)
      // With the secret the call is authenticated; the block may not exist, which is a different error.
      val e2 = intercept[FlightRuntimeException] {
        val s = client.getStream(ticket, new CredentialCallOption(new BearerCredentialWriter("flight-shuffle-test-secret")))
        try s.next() finally s.close()
      }
      assert(e2.status().code() !== FlightStatusCode.UNAUTHENTICATED, e2.toString)
      val e3 = intercept[FlightRuntimeException] {
        val s = client.getStream(ticket, new CredentialCallOption(new BearerCredentialWriter("wrong")))
        try s.next() finally s.close()
      }
      assert(e3.status().code() === FlightStatusCode.UNAUTHENTICATED, e3.toString)
    } finally {
      client.close()
      allocator.close()
    }
  }

  test("#338: a remote block of several record batches, each with its own string dictionary, decodes every batch right") {
    // A block is one record batch per `batchBytes` of held rows; with a small cap a reduce partition's
    // block holds many, and the writer gives each its own replacement dictionary (the slice's distinct
    // strings). Flight writes a stream's dictionaries once, at its start: before #338 the client decoded
    // batches 2..n against batch 1's dictionary -- out-of-bounds indices, or the wrong string silently.
    // Strings unique per row make every batch's dictionary different (batchBytes is set on the session).
    {
      val df = spark.range(0, 200000, 1, 8).selectExpr("id", "cast(id % 7 as int) as k", "concat('s-', cast(id as string)) as s")
      df.createOrReplaceTempView("u")
      // The shuffle carries s; the aggregate reads it back on the other executor.
      val sql = "select k, count(distinct s) ds, max(s) ms, min(s) mn from u group by k order by k"
      val ours = spark.sql(sql).collect().toSeq
      spark.sessionState.conf.setConfString("spark.vector.enabled", "false")
      val expected = try spark.sql(sql).collect().toSeq finally spark.sessionState.conf.setConfString("spark.vector.enabled", "true")
      assert(ours === expected)
      assert(ours.map(_.getLong(1)).sum === 200000L)
    }
  }
}
