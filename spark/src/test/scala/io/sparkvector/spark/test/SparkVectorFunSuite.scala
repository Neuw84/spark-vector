package io.sparkvector.spark.test

import java.nio.file.{Files, Path}
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/**
 * Base class for suites that need a local SparkSession. The session is created once per suite,
 * without the plugin unless [[extraSparkConf]] enables it.
 */
abstract class SparkVectorFunSuite extends AnyFunSuite with BeforeAndAfterAll {

  protected var spark: SparkSession = _
  private var tempDir: Path = _

  /** Configuration applied on top of the defaults; override to enable the plugin. */
  protected def extraSparkConf: Map[String, String] = Map.empty

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    tempDir = Files.createTempDirectory("spark-vector-test")
    val builder = SparkSession
      .builder()
      .master("local[2]")
      .appName(getClass.getSimpleName)
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.sql.warehouse.dir", tempDir.resolve("warehouse").toString)
      .config("spark.driver.host", "localhost")
    extraSparkConf.foreach { case (k, v) => builder.config(k, v) }
    spark = builder.getOrCreate()
  }

  override protected def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.stop()
        spark = null
      }
      SparkSession.clearActiveSession()
      SparkSession.clearDefaultSession()
    } finally {
      deleteRecursively(tempDir)
      super.afterAll()
    }
  }

  /** A fresh sub-directory under the suite's temp dir. */
  protected def newTempPath(name: String): String = {
    val p = tempDir.resolve(name)
    Files.createDirectories(p.getParent)
    p.toString
  }

  private def deleteRecursively(p: Path): Unit = {
    if (p != null && Files.exists(p)) {
      Files
        .walk(p)
        .sorted(java.util.Comparator.reverseOrder[Path]())
        .forEach(f => Files.deleteIfExists(f))
    }
  }
}
