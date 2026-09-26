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
 * The seam that lets a Comet native block sit above one of our operator chains
 * (#280), resolved reflectively like {@link CometBatchBridge}: no compile-time
 * dependency on Comet.
 *
 * <p>Comet's rule converts an operator natively only when its children are
 * Comet's -- native operators, or a <em>sink</em>: {@code
 * CometSinkPlaceHolder(nativeScanOp, original, child)}, whose {@code
 * child.executeColumnar()} the native block pulls through the same JNI iterator
 * as its shuffle reader, zero-copy as long as the batches carry {@code
 * CometVector}s. Our export node emits exactly those (the C Data hand-over the
 * shuffle already uses), so the leaf is that placeholder over {@code
 * VectorToCometExec(chain)}; its Scan proto comes from Comet's own sink serde
 * ({@code CometExchangeSink.convert}, the base {@code CometSink} path for a
 * non-shuffle operator), which applies Comet's type checks. Comet's {@code
 * sparkToColumnar} transition was measured and rejected as the leaf: it wraps
 * leaf nodes only and copies a Spark columnar batch value by value.
 *
 * <p>With the leaf in place, Comet's {@code CometExecRule} applied to the
 * parent subtree builds the native operator above it exactly as it would above
 * one of its own sinks.
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
    private final Object serde;
    private final Method allAggsMix;
    private final Object explainInfo;
    private final Method fallbackReasons;
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
        Class<?> serdeModule = Class.forName("org.apache.comet.serde.QueryPlanSerde$", true, loader);
        serde = serdeModule.getField("MODULE$").get(null);
        allAggsMix = serdeModule.getMethod("allAggsSupportMixedExecution", scala.collection.immutable.Seq.class);
        Class<?> explain = Class.forName("org.apache.comet.ExtendedExplainInfo", false, loader);
        explainInfo = explain.getDeclaredConstructor().newInstance();
        fallbackReasons = explain.getMethod("getFallbackReasons", SparkPlan.class);
        emptySeq = scala.collection.immutable.Nil$.MODULE$;
    }

    /**
     * A bridge if Comet's classes are loadable with the shapes this seam needs,
     * else {@code null}.
     */
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

    /**
     * Comet's own answer to whether these aggregate functions' intermediate
     * buffers are laid out the same way by Spark and by Comet -- when they are,
     * a partial on one engine and a final on the other is sound (sum, min, max,
     * the bit aggregates and a non-decimal avg are; count and the decimal sums
     * are not; the ones that are not are Comet's concern too, and its rule
     * refuses a final over a foreign partial for them).
     */
    public boolean aggregatesMix(scala.collection.immutable.Seq<?> aggregateExpressions) {
        try {
            return (Boolean) allAggsMix.invoke(serde, aggregateExpressions);
        } catch (ReflectiveOperationException e) {
            return false;
        }
    }

    /**
     * Comet's fallback reasons on {@code plan}, one string each, after its rule
     * declined an operator.
     */
    @SuppressWarnings("unchecked")
    public scala.collection.immutable.Seq<String> declineReasons(SparkPlan plan) {
        try {
            return (scala.collection.immutable.Seq<String>) fallbackReasons.invoke(explainInfo, plan);
        } catch (ReflectiveOperationException e) {
            return (scala.collection.immutable.Seq<String>) emptySeq;
        }
    }

    /**
     * Whether {@code plan} is one of Comet's native operators (a native block
     * or a sink placeholder).
     */
    public boolean isNative(SparkPlan plan) {
        return nativeExecClass.isInstance(plan);
    }

    /**
     * Whether {@code plan} is any of Comet's operators: native ones, and its
     * JVM sinks (union, coalesce, the limits, take-ordered), which are {@code
     * CometExec} but not {@code CometNativeExec} and whose batches a native
     * block above reads through Comet's Arrow reader.
     */
    public boolean isComet(SparkPlan plan) {
        String name = plan.getClass().getName();
        return name.startsWith("org.apache.spark.sql.comet.") || name.startsWith("org.apache.comet.");
    }

    /**
     * The leaf Comet's native block reads from: Comet's sink placeholder over a
     * one-child {@code CometUnionExec} over {@code export} (our export node
     * over the chain). The union is the pass-through Comet's input walk
     * recognises ({@code foreachUntilCometInput} lists its own JVM operators,
     * never a foreign node), and its batches -- ours, as {@code CometVector}s
     * -- reach native through Comet's {@code ColumnarBatchArrowReader}, which
     * hands the Arrow buffers over without a copy. Empty when Comet's serde
     * refuses a column type of the chain's output.
     */
    public Optional<SparkPlan> leaf(SparkPlan chain, SparkPlan export) {
        try {
            Object builder = operatorNewBuilder.invoke(null);
            Object converted = sinkConvert.invoke(exchangeSink, export, builder, emptySeq);
            scala.Option<?> option = (scala.Option<?>) converted;
            if (option.isEmpty()) {
                return Optional.empty();
            }
            scala.collection.immutable.Seq<SparkPlan> one = scala.collection.immutable.List$.MODULE$.<SparkPlan>empty().$colon$colon(export);
            // The union's originalPlan must take exactly one child: Comet 1.0's CometUnionExec reads its
            // partitioning through originalPlan.withNewChildren(children), and a chain rooted in a join has two
            // (TPC-H q2 under the allowlist, #281). The export node has one and forwards the chain's.
            SparkPlan passThrough = (SparkPlan) unionCtor.newInstance(export, chain.output(), one);
            return Optional.of((SparkPlan) placeholderCtor.newInstance(option.get(), chain, passThrough));
        } catch (InvocationTargetException e) {
            LOG.debug("spark-vector: Comet refused the mixed leaf", e.getCause());
            return Optional.empty();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Comet mixed leaf construction failed", e);
        }
    }

    /**
     * Comet's own planner rule applied to {@code subtree}, whose children are
     * leaves from {@link #leaf}. The result is Comet's native operator over the
     * leaves when Comet took the operator, else the subtree as it was (Comet's
     * fallback reasons are then on the node, readable through its explain).
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
