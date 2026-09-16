package org.apache.spark.sql.vector.ui

import scala.util.control.NonFatal

import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging

/**
 * Attaches [[VectorAccelerationTab]] to a running application's UI.
 *
 * `SparkContext.ui`, `SparkUI.attachTab` and `SparkUITab` are all `private[spark]`, so the call has
 * to be made from inside the `org.apache.spark` package; `io.sparkvector.spark.VectorPlugin` goes
 * through here.
 */
object VectorUi extends Logging {

  /**
   * Registers the listener and the tab. Does nothing when the UI is disabled (either by Spark or
   * by `spark.vector.ui.enabled=false`), which is also the case in most tests.
   *
   * @return true when the tab was attached
   */
  def attach(sc: SparkContext, enabled: Boolean, retainedExecutions: Int): Boolean = {
    if (!enabled) {
      logDebug("spark-vector: UI tab disabled")
      false
    } else {
      sc.ui match {
        case None =>
          logDebug("spark-vector: no Spark UI to attach the acceleration tab to")
          false
        case Some(ui) =>
          try {
            val store = new VectorAccelerationStore(retainedExecutions)
            // The shared queue: this listener is cheap and must not delay the app status queue.
            sc.listenerBus.addToSharedQueue(new VectorAccelerationListener(store))
            new VectorAccelerationTab(store, ui)
            logInfo("spark-vector: attached the Vector Acceleration tab to the Spark UI")
            true
          } catch {
            // A UI failure must never stop the application from starting.
            case NonFatal(e) =>
              logWarning("spark-vector: could not attach the Vector Acceleration tab", e)
              false
          }
      }
    }
  }
}
