# Upstream report: backport SPARK-55679 to branch-4.1 (and branch-4.0)

Spark tracks bugs in JIRA (https://issues.apache.org/jira/projects/SPARK); a backport is usually a
comment on the original ticket plus a PR against `branch-4.1` titled `[SPARK-55679][CORE][4.1] ...`.
The text below is written for a JIRA comment on SPARK-55679 (or a new ticket linked to it if the
committers prefer), and the PR description follows it.

---

**Title:** On JDK 24+, `SerializationDebugger` fails to initialise and turns every non-serializable task failure into an executor death (exit code 50) -- please backport SPARK-55679 to branch-4.1

**Affects:** 4.0.x, 4.1.x (verified 4.1.3). Fixed in 4.2.0 by SPARK-55679 (3e4df1c1c1).

**Environment:** Spark 4.1.3, JDK 25.0.4 (also JDK 24: `sun.security.action.*` removed with the Security
Manager, JDK-8338411), Kubernetes executors.

**Description**

`SerializationDebugger.enableDebugging` (core/src/main/scala/org/apache/spark/serializer/SerializationDebugger.scala:74 in 4.1.3) is evaluated in the object's static initialiser by loading `sun.security.action.GetBooleanAction` reflectively. On JDK 24+ that class no longer exists, `SparkClassUtils.classForName` throws `ClassNotFoundException` out of the initialiser, and the first call of `SerializationDebugger.improveException` in a JVM ends with `java.lang.ExceptionInInitializerError`.

`JavaSerializationStream.writeObject` calls `improveException` whenever `ObjectOutputStream` throws `NotSerializableException`. In an executor that path is `Executor.TaskRunner.run` serialising the `ExceptionFailure` of a task whose exception is not `Serializable` (in our case Arrow's `org.apache.arrow.memory.OutOfMemoryException`; any exception with a non-serializable field does it). `ExceptionInInitializerError` is an `Error`: it escapes `NonFatal`, reaches `SparkUncaughtExceptionHandler`, and the executor exits with code 50. The driver never sees the task's real failure -- only `ExecutorLostFailure` -- and the retried task ends the next executor the same way. On an 8-executor job we lost ~40 executors in a minute to one query.

**Executor log (4.1.3, JDK 25)**

```
ERROR SparkUncaughtExceptionHandler: Uncaught exception in thread Thread[#1529,Executor task launch worker for task 4.0 in stage 1463.0 (TID 35561),5,main]
java.lang.ExceptionInInitializerError
	at org.apache.spark.serializer.JavaSerializationStream.writeObject(JavaSerializer.scala:50)
	at org.apache.spark.serializer.JavaSerializerInstance.serialize(JavaSerializer.scala:122)
	at org.apache.spark.executor.Executor$TaskRunner.liftedTree1$1(Executor.scala:1080)
	at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:1077)
Caused by: java.lang.ClassNotFoundException: sun.security.action.GetBooleanAction
	at org.apache.spark.executor.ExecutorClassLoader.findClass(ExecutorClassLoader.scala:121)
	at java.base/java.lang.ClassLoader.loadClass(Unknown Source)
	at java.base/java.lang.Class.forName(Unknown Source)
	at org.apache.spark.util.SparkClassUtils.classForName(SparkClassUtils.scala:42)
	at org.apache.spark.util.SparkClassUtils$.classForName(SparkClassUtils.scala:169)
	at org.apache.spark.serializer.SerializationDebugger$.<clinit>(SerializationDebugger.scala:74)
```

**Minimal reproduction (no cluster)**

```java
import org.apache.spark.SparkConf;
import org.apache.spark.serializer.JavaSerializer;
import scala.reflect.ClassTag$;

public final class Repro {
  static final class Unserialisable { final Object payload = new Object(); } // not Serializable

  public static void main(String[] args) {
    new JavaSerializer(new SparkConf(false)).newInstance()
        .serialize(new Unserialisable(), ClassTag$.MODULE$.apply(Unserialisable.class));
  }
}
```

With `spark-core_2.13:4.1.3` on JDK 25: `java.lang.ExceptionInInitializerError` caused by
`ClassNotFoundException: sun.security.action.GetBooleanAction`. With `spark-core_2.13:4.2.0`, same JDK:
`java.io.NotSerializableException: Repro$Unserialisable` with the serialization stack, as designed. With
4.1.3 on JDK 17/21: the `NotSerializableException`, as designed.

**Expected behaviour**

A non-serializable task failure is reported as `ExceptionFailure` (with the debugger's serialization
stack, or without it if the debugger cannot run); the executor survives.

**Proposed fix**

Backport 3e4df1c1c1 (SPARK-55679) to `branch-4.1` and `branch-4.0`: read
`java.lang.Boolean.getBoolean("sun.io.serialization.extendedDebugInfo")` when
`Runtime.version().feature() >= 24`. Additionally worth considering for all branches: initialise
`enableDebugging` defensively (`try ... catch (NonFatal) => false`) so that no future JDK-internal change
can turn a diagnostic aid into a fatal error, and have `improveException` treat a debugger failure as
"return the original exception".

**Workaround for users on 4.1.x + JDK 24/25:** `spark.serializer.extraDebugInfo=false`. `JavaSerializationStream.writeObject` then rethrows the plain `NotSerializableException` instead of calling the debugger, and `Executor.TaskRunner` already catches that case and reports the failure in its serializable form (without the exception object). Verified in our reading of `Executor.scala`; the fix above remains the right answer.
---

## PR description (branch-4.1)

**Title:** `[SPARK-55679][CORE][4.1] Fix detecting sun.io.serialization.extendedDebugInfo on Java 24+`

### What changes were proposed in this pull request?
Backport of #(master PR for SPARK-55679) to branch-4.1: `SerializationDebugger.enableDebugging` reads the plain system property on Java 24+, where `sun.security.action.GetBooleanAction` no longer exists.

### Why are the changes needed?
On Java 24/25 the object's static initialiser throws `ExceptionInInitializerError` the first time a `NotSerializableException` is improved. In an executor this is fatal (exit code 50) and replaces the task's real failure; see the reproduction in the JIRA comment.

### Does this PR introduce any user-facing change?
No.

### How was this patch tested?
`core/testOnly *SerializationDebuggerSuite` on JDK 25 (before: `ClassNotFoundException: sun.security.action.GetBooleanAction`; after: passes), plus the standalone reproduction above.
