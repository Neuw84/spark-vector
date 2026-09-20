import java.io.NotSerializableException;

import org.apache.spark.SparkConf;
import org.apache.spark.serializer.JavaSerializer;
import org.apache.spark.serializer.SerializerInstance;
import scala.reflect.ClassTag$;

/**
 * Serialises one object that is not Serializable with Spark's JavaSerializer.
 *
 * Expected (JDK 17/21, or Spark 4.2.0+ on any JDK): a NotSerializableException whose message carries the
 * "Serialization stack" that SerializationDebugger adds.
 *
 * Observed (Spark 4.0.x / 4.1.x on JDK 24+): java.lang.ExceptionInInitializerError, caused by
 * ClassNotFoundException: sun.security.action.GetBooleanAction, thrown from the static initialiser of
 * SerializationDebugger (core/src/main/scala/org/apache/spark/serializer/SerializationDebugger.scala,
 * `enableDebugging`). An ExceptionInInitializerError is an Error, so inside an executor it reaches
 * SparkUncaughtExceptionHandler and the executor exits with code 50 -- the task's real failure is never
 * reported, and the retried task ends the next executor the same way.
 */
public final class Repro {
  /** Not Serializable on purpose. */
  static final class Unserialisable {
    final Object payload = new Object();
  }

  public static void main(String[] args) {
    System.out.println("java " + Runtime.version() + ", spark " + org.apache.spark.package$.MODULE$.SPARK_VERSION());
    SerializerInstance ser = new JavaSerializer(new SparkConf(false)).newInstance();
    try {
      ser.serialize(new Unserialisable(), ClassTag$.MODULE$.apply(Unserialisable.class));
      System.out.println("UNEXPECTED: serialised a non-serializable object");
      System.exit(2);
    } catch (Throwable t) {
      Throwable root = t;
      while (root.getCause() != null) root = root.getCause();
      System.out.println("thrown: " + t.getClass().getName());
      System.out.println("root:   " + root);
      if (t instanceof NotSerializableException) {
        System.out.println("OK: the expected NotSerializableException (with the debugger's serialization stack)");
        System.exit(0);
      }
      if (t instanceof ExceptionInInitializerError) {
        System.out.println("BUG: SerializationDebugger's initialiser failed -- an Error escapes the serializer (SPARK-55679)");
        System.exit(1);
      }
      System.exit(3);
    }
  }
}
