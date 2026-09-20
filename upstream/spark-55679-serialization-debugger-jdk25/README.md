# Spark 4.0/4.1 on JDK 24+: a non-serializable task failure ends the executor (SPARK-55679)

## What we saw

TPC-DS SF100 on EKS (#247/#248), eight executors, Spark 4.1.3 on JDK 25. A map task of q67 failed with
an Arrow `OutOfMemoryException` (#340). Instead of a failed task, **the executor exited with code 50**
(`SparkUncaughtExceptionHandler`), the retried task ended the next executor the same way, and within a
minute forty executors had come and gone. The executor log:

```
ERROR Executor: Exception in task 4.0 in stage 1463.0 (TID 35561)
org.apache.arrow.memory.OutOfMemoryException: Failure allocating buffer.
	...
ERROR SparkUncaughtExceptionHandler: Uncaught exception in thread Thread[#1529,Executor task launch worker for task 4.0 in stage 1463.0 (TID 35561),5,main]
java.lang.ExceptionInInitializerError
	at org.apache.spark.serializer.JavaSerializationStream.writeObject(JavaSerializer.scala:50)
	at org.apache.spark.serializer.JavaSerializerInstance.serialize(JavaSerializer.scala:122)
	at org.apache.spark.executor.Executor$TaskRunner.liftedTree1$1(Executor.scala:1080)
	at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:1077)
Caused by: java.lang.ClassNotFoundException: sun.security.action.GetBooleanAction
	at org.apache.spark.executor.ExecutorClassLoader.findClass(ExecutorClassLoader.scala:121)
	...
	at org.apache.spark.util.SparkClassUtils$.classForName(SparkClassUtils.scala:169)
	at org.apache.spark.serializer.SerializationDebugger$.<clinit>(SerializationDebugger.scala:74)
```

## The mechanism

1. The task's exception is not `Serializable` (Arrow's `OutOfMemoryException` carries an `Optional`).
   `Executor.TaskRunner` serialises the `ExceptionFailure` with `JavaSerializer`; `ObjectOutputStream`
   throws `NotSerializableException`.
2. `JavaSerializationStream.writeObject` catches it and calls `SerializationDebugger.improveException`
   to add the serialization path to the message.
3. `SerializationDebugger`'s static initialiser evaluates `enableDebugging` by loading
   `sun.security.action.GetBooleanAction` reflectively (Spark 4.1.3, `SerializationDebugger.scala:74`).
   That JDK-internal class was removed in JDK 24 (Security Manager removal), so `classForName` throws
   `ClassNotFoundException` **out of the initialiser**: `ExceptionInInitializerError`.
4. `ExceptionInInitializerError` is an `Error`, not caught by `NonFatal`; it escapes the task runner,
   `SparkUncaughtExceptionHandler` ends the JVM with exit code 50. The original failure is never
   reported to the driver, which sees only `ExecutorLostFailure`.

Any non-serializable exception in any task does this on JDK 24+ with Spark 4.0.x/4.1.x. The class is
loaded once per JVM, so it is the *first* such failure per executor that ends it.

## Upstream state

Fixed on `master` by **SPARK-55679** (`3e4df1c1c1`, 2026-02-25, "Fix detecting
`sun.io.serialization.extendedDebugInfo` on Java 25"): `enableDebugging` reads
`java.lang.Boolean.getBoolean("sun.io.serialization.extendedDebugInfo")` when
`Runtime.version().feature() >= 24`. Released in **4.2.0**. **Not in `branch-4.1` or `branch-4.0`**
(checked against `v4.1.3`, `branch-4.1`, `branch-4.0`). JDK 25 is not a documented runtime for 4.1, but
the fix is two lines and the failure mode -- a fatal error replacing every non-serializable task failure --
argues for a backport. `SPARK-ISSUE.md` is the text for the backport request.

## The reproducer

`Repro.java` serialises one non-serializable object with Spark's `JavaSerializer` -- no cluster, no
SparkContext. `pom.xml` pulls `spark-core_2.13` at `${spark.version}` (default 4.1.3).

```
# JDK 24 or 25 on PATH
mvn -q compile exec:java                        # Spark 4.1.3: exit 1, ExceptionInInitializerError
mvn -q -Dspark.version=4.2.0 compile exec:java  # Spark 4.2.0: exit 0, NotSerializableException as designed
```

Observed on JDK 25.0.4:

```
java 25.0.4.1+8-LTS, spark 4.1.3
thrown: java.lang.ExceptionInInitializerError
root:   java.lang.ClassNotFoundException: sun.security.action.GetBooleanAction
BUG: SerializationDebugger's initialiser failed -- an Error escapes the serializer (SPARK-55679)

java 25.0.4.1+8-LTS, spark 4.2.0
thrown: java.io.NotSerializableException
root:   java.io.NotSerializableException: Repro$Unserialisable
OK: the expected NotSerializableException (with the debugger's serialization stack)
```

## What we do on our side (#340)

Independently of the backport: the columnar shuffle turns Arrow's `OutOfMemoryException` into a
serializable `SparkException` at the task boundary, so an Arrow out-of-memory is a failed task, not a
dead executor -- on any JDK.
