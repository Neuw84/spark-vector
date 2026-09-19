package io.sparkvector.spark.comet;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Optional;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.SparkPlan;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The seam that lets a Comet native block sit above one of our operator chains (#280), resolved
 * reflectively like {@link CometBatchBridge}: no compile-time dependency on Comet.
 *
 * <p>Comet's rule converts an operator natively only when its children are Comet's -- native
 * operators, or a <em>sink</em>: {@code CometSinkPlaceHolder(nativeScanOp, original, child)}, whose
 * {@code child.executeColumnar()} the native block pulls through the same JNI iterator as its
 * shuffle reader, zero-copy as long as the batches carry {@code CometVector}s. Our export node
 * emits exactly those (the C Data hand-over the shuffle already uses), so the leaf is that
 * placeholder over {@code VectorToCometExec(chain)}; its Scan proto comes from Comet's own sink
 * serde ({@code CometExchangeSink.convert}, the base {@code CometSink} path for a non-shuffle
 * operator), which applies Comet's type checks. Comet's {@code sparkToColumnar} transition was
 * measured and rejected as the leaf: it wraps leaf nodes only and copies a Spark columnar batch
 * value by value.
 *
 * <p>With the leaf in place, Comet's {@code CometExecRule} applied to the parent subtree builds the
 * native operator above it exactly as it would above one of its own sinks.
 */
public final class CometMixedBridge {
  private static final Logger LOG = LoggerFactory.getLogger(CometMixedBridge.class);

  private final Object exchangeSink;
  private final Method sinkConvert;
  private final Method operatorNewBuilder;
  private final Constructor<?> placeholderCtor;
  private final Constructor<?> execRuleCtor;
  private final Method execRuleApply;
  private final Class<?> nativeExecClass;
  private final Constructor<?> unionCtor;
  private final Object emptySeq;

  private CometMixedBridge(ClassLoader loader) throws ReflectiveOperationException {
    Class<?> sinkModule = Class.forName("org.apache.comet.serde.operator.CometExchangeSink$", true, loader);
    exchangeSink = sinkModule.getField("MODULE$").get(null);
    Class<?> operatorClass = Class.forName("org.apache.comet.serde.OperatorOuterClass$Operator", false, loader);
    Class<?> builderClass = Class.forName("org.apache.comet.serde.OperatorOuterClass$Operator$Builder", false, loader);
    operatorNewBuilder = operatorClass.getMethod("newBuilder");
    sinkConvert = sinkModule.getMethod("convert", SparkPlan.class, builderClass, scala.collection.immutable.Seq.class);
    Class<?> placeholder = Class.forName("org.apache.spark.sql.comet.CometSinkPlaceHolder", false, loader);
    placeholderCtor = placeholder.getConstructor(operatorClass, SparkPlan.class, SparkPlan.class);
    Class<?> rule = Class.forName("org.apache.comet.rules.CometExecRule", false, loader);
    execRuleCtor = rule.getConstructor(SparkSession.class);
    execRuleApply = rule.getMethod("apply", SparkPlan.class);
    nativeExecClass = Class.forName("org.apache.spark.sql.comet.CometNativeExec", false, loader);
    Class<?> union = Class.forName("org.apache.spark.sql.comet.CometUnionExec", false, loader);
    unionCtor = union.getConstructor(SparkPlan.class, scala.collection.immutable.Seq.class, scala.collection.immutable.Seq.class);
    emptySeq = scala.collection.immutable.Nil$.MODULE$;
  }

  /** A bridge if Comet's classes are loadable with the shapes this seam needs, else {@code null}. */
  public static CometMixedBridge tryCreate() {
    ClassLoader loader = Thread.currentThread().getContextClassLoader();
    if (loader == null) {
      loader = CometMixedBridge.class.getClassLoader();
    }
    try {
      return new CometMixedBridge(loader);
    } catch (ClassNotFoundException e) {
      return null;
    } catch (ReflectiveOperationException e) {
      LOG.warn("spark-vector: Comet found but its planner API differs from the expected one; mixed chains stay off", e);
      return null;
    }
  }

  /** Whether {@code plan} is one of Comet's native operators (a native block or a sink placeholder). */
  public boolean isNative(SparkPlan plan) {
    return nativeExecClass.isInstance(plan);
  }

  /**
   * The leaf Comet's native block reads from: Comet's sink placeholder over a one-child {@code
   * CometUnionExec} over {@code export} (our export node over the chain). The union is the
   * pass-through Comet's input walk recognises ({@code foreachUntilCometInput} lists its own JVM
   * operators, never a foreign node), and its batches -- ours, as {@code CometVector}s -- reach native
   * through Comet's {@code ColumnarBatchArrowReader}, which hands the Arrow buffers over without a
   * copy. Empty when Comet's serde refuses a column type of the chain's output.
   */
  public Optional<SparkPlan> leaf(SparkPlan chain, SparkPlan export) {
    try {
      Object builder = operatorNewBuilder.invoke(null);
      Object converted = sinkConvert.invoke(exchangeSink, export, builder, emptySeq);
      scala.Option<?> option = (scala.Option<?>) converted;
      if (option.isEmpty()) {
        return Optional.empty();
      }
      scala.collection.immutable.Seq<SparkPlan> one =
          scala.collection.immutable.List$.MODULE$.<SparkPlan>empty().$colon$colon(export);
      SparkPlan passThrough = (SparkPlan) unionCtor.newInstance(chain, chain.output(), one);
      return Optional.of((SparkPlan) placeholderCtor.newInstance(option.get(), chain, passThrough));
    } catch (InvocationTargetException e) {
      LOG.debug("spark-vector: Comet refused the mixed leaf", e.getCause());
      return Optional.empty();
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Comet mixed leaf construction failed", e);
    }
  }

  /**
   * Comet's own planner rule applied to {@code subtree}, whose children are leaves from {@link #leaf}.
   * The result is Comet's native operator over the leaves when Comet took the operator, else the
   * subtree as it was (Comet's fallback reasons are then on the node, readable through its explain).
   */
  public SparkPlan convertAbove(SparkSession session, SparkPlan subtree) {
    try {
      Object rule = execRuleCtor.newInstance(session);
      return (SparkPlan) execRuleApply.invoke(rule, subtree);
    } catch (InvocationTargetException e) {
      LOG.debug("spark-vector: Comet's rule failed over the mixed subtree", e.getCause());
      return subtree;
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Comet mixed conversion failed", e);
    }
  }
}
