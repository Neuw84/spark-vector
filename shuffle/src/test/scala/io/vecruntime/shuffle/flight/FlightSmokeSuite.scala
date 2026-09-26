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
package io.vecruntime.shuffle.flight

import java.nio.charset.StandardCharsets

import org.apache.arrow.flight._
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.{IntVector, VectorSchemaRoot}
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.scalatest.funsuite.AnyFunSuite

/**
 * #288 slice 3, the first check: Flight's gRPC transport is grpc-netty 1.71 compiled against Netty
 * 4.1, running here on the Netty 4.2 Spark bundles. A server that starts, serves a `DoGet` and
 * stops is the evidence that the combination works on this classpath.
 */
class FlightSmokeSuite extends AnyFunSuite {

  test("a FlightServer starts on an ephemeral port and serves a DoGet over Spark's Netty") {
    val allocator = new RootAllocator()
    val schema = new Schema(java.util.List.of(new Field("v", FieldType.nullable(new ArrowType.Int(32, true)), null)))
    val producer = new NoOpFlightProducer {
      override def getStream(
          context: FlightProducer.CallContext,
          ticket: Ticket,
          listener: FlightProducer.ServerStreamListener
      ): Unit = {
        val root = VectorSchemaRoot.create(schema, allocator)
        try {
          val v = root.getVector("v").asInstanceOf[IntVector]
          v.allocateNew(3); v.set(0, 1); v.set(1, 2); v.set(2, 3); v.setValueCount(3)
          root.setRowCount(3)
          listener.start(root)
          listener.putNext()
          listener.completed()
        } finally root.close()
      }
    }
    val server = FlightServer.builder(allocator, Location.forGrpcInsecure("127.0.0.1", 0), producer).build()
    server.start()
    try {
      assert(server.getPort > 0)
      val client = FlightClient.builder(allocator, Location.forGrpcInsecure("127.0.0.1", server.getPort)).build()
      try {
        val stream = client.getStream(new Ticket("t".getBytes(StandardCharsets.UTF_8)))
        try {
          assert(stream.next())
          val root = stream.getRoot
          assert(root.getRowCount === 3)
          assert(root.getVector("v").asInstanceOf[IntVector].get(2) === 3)
          assert(!stream.next())
        } finally stream.close()
      } finally client.close()
    } finally {
      server.close()
      allocator.close()
    }
  }
}
