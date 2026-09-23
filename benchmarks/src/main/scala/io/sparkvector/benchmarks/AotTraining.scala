package io.sparkvector.benchmarks

import java.nio.file.{Files, Path}

import org.apache.spark.sql.SparkSession

/**
 * The training run behind the executor's AOT cache (#416).
 *
 * The JDK's AOT cache (JEP 483/515) holds the classes a JVM loaded and linked, and the method
 * profiles it gathered, in a training run; a later JVM started with `-XX:AOTCache` skips the
 * class loading, linking and interpreter warm-up for them. At 1000 partitions a query's tasks
 * are short enough that our executors spent their first stages in the interpreter (q18: 2.7x
 * behind Spark cold, parity warm), and a cache trained on other queries took 27% off q18's
 * executor time. This program is that training run, done once at image build: Spark in local
 * mode on our engine and our shuffle, over synthetic tables that carry every lane type, through
 * every operator we have -- scan adapter, project and filter with the expression kernels, hash
 * aggregate (partial and final, over our exchange) on every key kind, broadcast, shuffled and
 * sort-merge joins in every join type, window, sort, top-k, expand and rollup, union, limit,
 * generate, sample -- each statement several times so the profiles are hot. Also the shuffle
 * writer and reader over dictionary strings, decimals and nulls.
 *
 * It is built to run as `train.sh` runs it: on the executor's class path (`/opt/spark/jars`, wildcard)
 * with the executor's JVM options, since the cache is only used by a JVM whose class path and
 * module options match the training run's. What it computes is irrelevant; what it exercises
 * is the point, so the queries are chosen for the code paths they take, not for their answers.
 *
 * {{{
 * java <executor JVM options> -XX:AOTCacheOutput=/opt/spark/aot/executor.aot -cp '/opt/spark/jars/<wildcard>' \
 *   io.sparkvector.benchmarks.AotTraining [--rounds N] [--rows N]
 * java -cp '/opt/spark/jars/<wildcard>' io.sparkvector.benchmarks.AotTraining --module-options   # Spark's launcher defaults, one per line
 * }}}
 */
object AotTraining {

  def main(args: Array[String]): Unit = {
    if (args.contains("--module-options")) {
      // What Spark's launcher prepends to every JVM it starts (driver and executors alike): the
      // training JVM must carry the same module options for the cache to be accepted.
      org.apache.spark.launcher.JavaModuleOptions.defaultModuleOptions().split(
        "\\s+"
      ).filter(_.nonEmpty).foreach(println)
      return
    }
    val rounds = arg(args, "--rounds").map(_.toInt).getOrElse(3)
    val rows = arg(args, "--rows").map(_.toLong).getOrElse(400000L)
    val dir = Files.createTempDirectory("spark-vector-aot")
    val start = System.nanoTime()
    val spark = session(dir)
    try {
      tables(spark, dir, rows)
      val statements = queries
      println(s"[aot-training] ${statements.size} statements x $rounds rounds over $rows rows")
      for (round <- 1 to rounds; (name, sql) <- statements) {
        val t = System.nanoTime()
        val n = spark.sql(sql).collect().length
        if (round == 1) println(f"[aot-training] $name%-28s rows=$n%-8d ${(System.nanoTime() - t) / 1e9}%.1fs")
      }
    } finally {
      spark.stop()
      delete(dir)
    }
    println(f"[aot-training] done in ${(System.nanoTime() - start) / 1e9}%.0fs")
  }

  private def arg(args: Array[String], name: String): Option[String] = {
    val i = args.indexOf(name)
    if (i >= 0 && i + 1 < args.length) Some(args(i + 1)) else None
  }

  /** Local mode on our engine with the `vector-shuffle` configuration the cluster runs use (submit-cluster.sh). */
  private def session(dir: Path): SparkSession = SparkSession.builder()
    .master(s"local[${math.max(2, Runtime.getRuntime.availableProcessors() min 4)}]")
    .appName("spark-vector AOT training")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.host", "localhost")
    .config("spark.sql.warehouse.dir", dir.resolve("wh").toString)
    .config("spark.local.dir", dir.resolve("local").toString)
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.autoBroadcastJoinThreshold", "1m")
    .config("spark.plugins", "io.sparkvector.spark.VectorPlugin")
    .config("spark.shuffle.manager", "org.apache.spark.sql.vector.shuffle.VectorShuffleManager")
    .config("spark.vector.shuffle.enabled", "true")
    .config("spark.vector.exec.strictFloatingPoint", "false")
    .config("spark.vector.exec.sortMergeJoin.enabled", "true")
    .config("spark.sql.parquet.enableVectorizedReader", "true")
    .config("spark.sql.columnVector.offheap.enabled", "true")
    .getOrCreate()

  /**
   * A fact table and three dimensions as parquet (key moduli coprime with the flag's, so joins on one
   * and filters on the other intersect), so the scan adapter and the parquet reader are in
   * the run. Every lane type: boolean, byte, short, int, long, double, decimal(12,2) and (28,6),
   * string (low cardinality -> dictionary encoded, and unique -> plain), date, timestamp; nulls in
   * every nullable column.
   */
  private def tables(spark: SparkSession, dir: Path, rows: Long): Unit = {
    val fact = spark.range(0, rows).selectExpr(
      "id",
      "cast(id % 997 as int) as cust_sk",
      "cast(id % 241 as int) as item_sk",
      "cast(1 + id % 1823 as int) as day_sk",
      "cast(id % 5 as tinyint) as flag8",
      "cast(id % 300 as smallint) as flag16",
      "cast(1 + id % 40 as int) as quantity",
      "cast((id * 7919) % 100000 as decimal(12,2)) / 100 as price",
      "cast((id * 104729) % 1000000000 as decimal(28,6)) / 1000 as ext_amount",
      "cast(id % 97 as double) / 7 as ratio",
      "case when id % 13 = 0 then null else cast(id % 3 = 0 as boolean) end as returned",
      "case when id % 11 = 0 then null else concat('promo_', cast(id % 20 as string)) end as promo",
      "concat('ticket-', cast(id as string)) as ticket",
      "date_add(date '2020-01-01', cast(id % 1826 as int)) as sold_date",
      "timestamp_seconds(1577836800 + (id % 100000) * 37) as sold_ts"
    )
    write(spark, fact, dir, "fact")
    val customer = spark.range(0, 1000).selectExpr(
      "cast(id as int) as c_sk",
      "concat('customer#', cast(id as string)) as c_name",
      "case when id % 17 = 0 then null else concat('state_', cast(id % 50 as string)) end as c_state",
      "cast(id % 7 as int) as c_segment",
      "cast(18 + id % 70 as int) as c_age",
      "cast((id * 31) % 10000 as decimal(12,2)) as c_credit"
    )
    write(spark, customer, dir, "customer")
    val item = spark.range(0, 250).selectExpr(
      "cast(id as int) as i_sk",
      "concat('item ', cast(id as string)) as i_name",
      "concat('brand_', cast(id % 25 as string)) as i_brand",
      "concat('class_', cast(id % 10 as string)) as i_class",
      "concat('cat_', cast(id % 5 as string)) as i_category",
      "cast((id * 13) % 50000 as decimal(12,2)) / 100 as i_price"
    )
    write(spark, item, dir, "item")
    val dates = spark.range(1, 1827).selectExpr(
      "cast(id as int) as d_sk",
      "date_add(date '2020-01-01', cast(id - 1 as int)) as d_date",
      "cast(year(date_add(date '2020-01-01', cast(id - 1 as int))) as int) as d_year",
      "cast(month(date_add(date '2020-01-01', cast(id - 1 as int))) as int) as d_moy",
      "cast(dayofweek(date_add(date '2020-01-01', cast(id - 1 as int))) as int) as d_dow"
    )
    write(spark, dates, dir, "dates")
  }

  private def write(spark: SparkSession, df: org.apache.spark.sql.DataFrame, dir: Path, name: String): Unit = {
    val path = dir.resolve(name).toString
    df.coalesce(4).write.mode("overwrite").parquet(path)
    spark.read.parquet(path).createOrReplaceTempView(name)
  }

  /** The statements, each named for the code path it is there for. */
  private def queries: Seq[(String, String)] = Seq(
    "scan-filter-project" ->
      """select ticket, quantity * price as amount, upper(promo) as p, substr(ticket, 8, 4) as t4, length(promo) as l,
        |  ratio * 2.5 + 1 as r, case when returned then 'R' when returned is null then 'N' else 'K' end as k,
        |  year(sold_date) as y, month(sold_date) as m, cast(sold_ts as date) as d, ext_amount - price as diff,
        |  coalesce(promo, 'none') as pn, flag8 + flag16 as f, quantity between 5 and 20 as mid
        |from fact where price > 100 and quantity < 30 and promo like 'promo_1%' and sold_date >= date '2021-01-01'
        |  and returned is not null and item_sk in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)""".stripMargin,
    "agg-int-keys" ->
      """select cust_sk, quantity, count(*) c, sum(item_sk) q, sum(price) p, avg(ratio) r, min(sold_date) d0, max(sold_ts) t1,
        |  sum(ext_amount) e, count(distinct day_sk) dd, max(promo) mp, min(ticket) mt
        |from fact group by cust_sk, quantity""".stripMargin,
    "agg-string-keys" ->
      """select promo, i_brand, i_class, count(*) c, sum(price * quantity) rev, avg(price) ap, sum(case when returned then 1 else 0 end) ret
        |from fact join item on item_sk = i_sk group by promo, i_brand, i_class having count(*) > 10 order by rev desc limit 100""".stripMargin,
    "agg-decimal-date-keys" ->
      """select price, sold_date, flag8, sum(ext_amount) s, count(*) c, avg(ext_amount) a, max(quantity) mq
        |from fact where day_sk < 400 group by price, sold_date, flag8""".stripMargin,
    "agg-global" ->
      "select count(*), sum(price), sum(ext_amount), avg(ratio), min(sold_ts), max(sold_date), count(distinct promo), sum(quantity) from fact",
    "agg-rollup" ->
      """select i_category, i_class, i_brand, sum(price * quantity) rev, count(*) c, grouping(i_category) g
        |from fact join item on item_sk = i_sk group by rollup(i_category, i_class, i_brand) order by 1, 2, 3 limit 300""".stripMargin,
    "agg-cube-expand" ->
      "select flag8, d_dow, count(*) c, sum(quantity) q from fact join dates on day_sk = d_sk group by cube(flag8, d_dow)",
    "broadcast-joins" ->
      """select c_state, i_category, d_year, count(*) c, sum(price) p
        |from fact join customer on cust_sk = c_sk join item on item_sk = i_sk join dates on day_sk = d_sk
        |where c_age > 30 and i_price > 10 group by c_state, i_category, d_year""".stripMargin,
    "broadcast-join-residual" ->
      """select count(*), sum(price) from fact join item on item_sk = i_sk and price > i_price and quantity > 2
        |join customer on cust_sk = c_sk and c_credit > price""".stripMargin,
    "shuffled-hash-join" ->
      """select /*+ SHUFFLE_HASH(b) */ a.cust_sk, count(*) c, sum(a.price - b.price) d
        |from fact a join fact b on a.ticket = b.ticket and a.cust_sk = b.cust_sk where a.item_sk < 100 group by a.cust_sk""".stripMargin,
    "sort-merge-join" ->
      """select /*+ MERGE(b) */ a.item_sk, count(*) c, sum(b.quantity) q
        |from fact a join fact b on a.cust_sk = b.cust_sk and a.ratio = b.ratio where a.flag8 = 1 and b.flag8 = 2 group by a.item_sk""".stripMargin,
    "sort-merge-join-computed-key" ->
      """select /*+ MERGE(b) */ count(*), sum(a.price) from fact a join fact b on a.cust_sk = b.cust_sk + 1 and a.day_sk = b.day_sk
        |where a.flag8 = 3 and b.flag8 = 4""".stripMargin,
    "outer-semi-anti-joins" ->
      """select
        |  (select count(*) from customer c left join fact f on c.c_sk = f.cust_sk and f.quantity > 38 where f.id is null) as no_big,
        |  (select count(*) from customer c where exists (select 1 from fact f where f.cust_sk = c.c_sk and f.returned)) as has_return,
        |  (select count(*) from item i where i_sk not in (select item_sk from fact where flag8 = 0 and item_sk is not null)) as unsold,
        |  (select count(*) from fact f full outer join customer c on f.cust_sk = c.c_sk and c.c_age < 20) as full_rows,
        |  (select count(*) from fact f right join item i on f.item_sk = i.i_sk where f.price > 900) as right_rows""".stripMargin,
    "window" ->
      """select cust_sk, sold_date, price,
        |  rank() over (partition by cust_sk order by price desc) rk,
        |  sum(price) over (partition by cust_sk order by sold_date rows between 3 preceding and current row) run,
        |  avg(quantity) over (partition by item_sk) aq,
        |  lag(price, 1) over (partition by cust_sk order by sold_date, id) prev,
        |  row_number() over (order by ext_amount desc) rn
        |from fact where day_sk < 200""".stripMargin,
    "window-self-join" ->
      """with v as (select item_sk, d_year, d_moy, sum(price) s,
        |  avg(sum(price)) over (partition by item_sk, d_year) avg_m,
        |  rank() over (partition by item_sk order by d_year, d_moy) rn
        |  from fact join dates on day_sk = d_sk group by item_sk, d_year, d_moy)
        |select v1.item_sk, v1.d_year, v1.s, v0.s prev, v2.s next from v v1
        |join v v0 on v1.item_sk = v0.item_sk and v1.rn = v0.rn + 1
        |join v v2 on v1.item_sk = v2.item_sk and v1.rn = v2.rn - 1 where v1.avg_m > 0 order by v1.item_sk, v1.d_year, v1.d_moy limit 500""".stripMargin,
    "sort-topk" ->
      "select * from fact order by price desc, ticket limit 100",
    "sort-global" ->
      "select cust_sk, item_sk, promo, price from fact where flag8 = 2 order by promo nulls first, price, cust_sk",
    "union-distinct-intersect" ->
      """select count(*) from (
        |  select cust_sk, item_sk from fact where flag8 = 0 union select cust_sk, item_sk from fact where flag8 = 1
        |  union all select cust_sk, item_sk from fact where flag8 = 2 and returned
        |  intersect select cust_sk, item_sk from fact where quantity > 10)""".stripMargin,
    "subqueries-scalar-in" ->
      """select cust_sk, sum(price) s from fact
        |where price > (select avg(price) * 1.2 from fact) and item_sk in (select i_sk from item where i_category = 'cat_1')
        |  and day_sk in (select d_sk from dates where d_moy in (11, 12)) group by cust_sk order by s desc limit 50""".stripMargin,
    "generate-explode" ->
      "select cust_sk, x, count(*) from fact lateral view explode(array(quantity, flag16, day_sk)) t as x where flag8 = 1 group by cust_sk, x",
    "sample-limit" ->
      "select count(*), sum(price) from (select * from fact tablesample (10 percent) limit 20000)",
    "repartition-roundrobin" ->
      "select count(*), sum(ext_amount), max(ticket) from (select /*+ REPARTITION(6) */ * from fact where flag8 < 3)",
    "string-functions" ->
      """select promo, count(*) from (select
        |  concat_ws('|', promo, i_brand, c_name) cw, trim(lower(c_name)) t, replace(ticket, 'ticket-', '') r, instr(c_name, '#') i,
        |  split(i_name, ' ')[0] sp, rpad(promo, 12, '.') rp, ticket like '%99%' l, promo rlike 'promo_1[0-9]' rl, promo
        |  from fact join item on item_sk = i_sk join customer on cust_sk = c_sk where c_state is not null) group by promo""".stripMargin,
    "date-decimal-arithmetic" ->
      """select d_year, d_moy, sum(price * quantity) rev, sum(ext_amount * 1.05) taxed, avg(ext_amount / nullif(quantity, 0)) unit,
        |  sum(cast(price as decimal(28,6)) + ext_amount) wide, datediff(max(sold_date), min(sold_date)) span,
        |  count(case when sold_ts > timestamp '2021-06-01 00:00:00' then 1 end) late
        |from fact join dates on day_sk = d_sk group by d_year, d_moy""".stripMargin
  )

  private def delete(dir: Path): Unit = {
    import scala.jdk.CollectionConverters._
    if (Files.exists(dir)) Files.walk(dir).iterator().asScala.toSeq.reverse.foreach(p => Files.deleteIfExists(p))
  }
}
