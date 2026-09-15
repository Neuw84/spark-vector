package io.sparkvector.spark.test

import org.scalatest.Tag

/** Suites that require the Comet jar on the classpath (enabled with the `comet` Maven profile). */
object CometTest extends Tag("io.sparkvector.spark.test.CometTest")

/** Placeholder tag used to exclude nothing when the `comet` profile is active. */
object NoTest extends Tag("io.sparkvector.spark.test.NoTest")
