/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package io.sparkvector.spark.test

import org.scalatest.Tag

/** Suites that require the Comet jar on the classpath (enabled with the `comet` Maven profile). */
object CometTest extends Tag("io.sparkvector.spark.test.CometTest")

/** Suites that require the Iceberg Spark runtime on the classpath (`iceberg` Maven profile). */
object IcebergTest extends Tag("io.sparkvector.spark.test.IcebergTest")

/** Placeholder tag used to exclude nothing when a profile is active. */
object NoTest extends Tag("io.sparkvector.spark.test.NoTest")
