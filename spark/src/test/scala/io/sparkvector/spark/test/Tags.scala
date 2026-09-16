package io.sparkvector.spark.test

import org.scalatest.Tag

/** Suites that require the Comet jar on the classpath (enabled with the `comet` Maven profile). */
object CometTest extends Tag("io.sparkvector.spark.test.CometTest")

/** Suites that require the Iceberg Spark runtime on the classpath (`iceberg` Maven profile). */
object IcebergTest extends Tag("io.sparkvector.spark.test.IcebergTest")

/** Placeholder tag used to exclude nothing when a profile is active. */
object NoTest extends Tag("io.sparkvector.spark.test.NoTest")
