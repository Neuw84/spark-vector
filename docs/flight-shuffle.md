---
layout: default
title: Flight shuffle
---

# The columnar shuffle over Arrow Flight

How a shuffle stage's map outputs get from one executor to another without leaving Arrow memory:
the file the map side writes, the Flight service that streams it, the reader that turns it back into
column batches, and the decisions behind each. Written after the 1000-partition work (#411, #416);
the issue numbers point at the measurements that motivated a choice.

- Code: `shuffle/` -- `VectorShuffleExchangeExec` (in `spark/`), `VectorShuffleManager`,
  `PartitionedIpcWriter`, `PartitionedIpcFile`, `VectorShuffleBackend`, `flight/FlightShuffle`.
- Tests: `PartitionedIpcSuite`, `FlightBlockStreamSuite`, `FlightShuffleClusterSuite`, `VectorShuffleSuite`.
- Benchmark: `FlightShuffleBenchmark` (`benchmarks/scripts/run-flight-bench.sh`).

## 1. Why a shuffle of our own

Spark's shuffle is row-based: a columnar operator feeding `ShuffleExchangeExec` converts every batch
to `UnsafeRow`s, the sort-based writer serialises them per partition, and the reducer deserialises
rows the next columnar operator converts back into columns. For a plan whose operators are all ours
that is two conversions and a serialiser round trip per stage, and it was the largest single cost of
the TPC-DS campaign on plain Spark's shuffle (#288). Comet has the same problem and the same answer:
a `ShuffleManager` that handles only its own dependency type and leaves every other shuffle to
Spark's sort shuffle untouched.

Ours moves Arrow record batches. The map side partitions a `ColumnarBatch` with the partitioning
kernels and appends each partition's rows to that partition's builders; the reduce side decodes IPC
messages straight into Arrow vectors the operators read in place. Nothing is a row in between.

## 2. The pieces

```
 map task (executor A)                                   reduce task (executor B)
 ───────────────────────                                 ─────────────────────────
 VectorShuffleExchangeExec                               VectorShuffleReader.read()
   │ partition ids (PartitionKernels)                      │ mapOutputTracker: blocks by executor
   ▼                                                       ├─ local map outputs: file segments
 VectorShuffleWriter                                       │    (dictionary section + range)
   └─ PartitionedIpcWriter ──► data file + index file      └─ remote executors: one DoGet each
        [p0 stream][p1 stream]…[dict section][trailer]           │
                                                                 ▼
 FlightShuffle.Service (one per executor)  ◄──── ticket ── FlightBlockStream (per executor)
   Producer.getStream:                                       ChunkChannel ─► StreamReader
     for each mapId: blockData(shuffle, map, [start,end))         decode units ─► ColumnarBatch
       dictionary section, then the range, in 4 MB chunks
```

The location of every executor's server travels through Spark's plugin: the executor plugin starts
the server on the block manager's host and an ephemeral port and registers `executorId -> host:port`
with the driver plugin; a reducer looks a location up once per executor and caches it
(`FlightRegistry`). Map outputs live in the block resolver's ordinary data/index files, so
`MapStatus`, the map output tracker, speculation and `unregisterShuffle` (by tracked map task id,
#358) are Spark's own.

## 3. The map file

`PartitionedIpcWriter` produces one data file per map task. Its layout, in order:

```
[partition 0 stream][partition 1 stream] … [partition n-1 stream]
[dictionary section]
[trailer: offset:long, length:long, magic "SVDICT11":long]      (24 bytes, always)
[index footer: n, offsets[n], lengths[n], rows[n], bodyLen:int, magic:long]   (tests/tools only)
```

Under Spark the partition offsets and lengths go to the resolver's index file
(`writeMetadataFileAndCommit`) and the footer is not written; the trailer sits at the file's end, so
anyone holding the data file finds the dictionary section without the index. `PartitionReader` (tests)
reads the footer, then the trailer just before it.

### 3.1 A partition's stream

A partition's stream is a sequence of *units*, each a 12-byte header of ours followed by one Arrow
IPC message:

```
[magic "SVB1":int][kind:int][value:int] <IPC message>
   kind 1 (dictionary):   value = column ordinal; the message is a DictionaryBatch, current for that
                          column until the next dictionary unit for it
   kind 2 (record batch): value = bitmap length; then ceil(columns/8) bytes, bit c set = column c is
                          int32 ids over its current dictionary, clear = plain UTF8; then a RecordBatch
```

There is no Schema message (#411: the reader knows the shuffle's schema from the dependency, and a
schema message per block was measurable at 1000 partitions) and no end-of-stream marker (#347: the
streams of several map outputs are read back to back as one, so a boundary needs no marker). The
unit header is what lets a record batch say its own shape and lets a dictionary outlive the record
batch that follows it -- Arrow's own IPC stream would tie both to the schema message.

### 3.2 Strings: one dictionary per column per map file (#416)

A string column travels as int32 ids over a dictionary whenever the dictionary pays. Before #416
every record batch carried its own dictionary, remapped dense from the task's staging dictionary
(#345: a 200-row block should not carry a 3000-entry dictionary). At 1000 partitions a block is a
few hundred rows and the remap, the dictionary batch and the reader's per-block decode were most of
a string column's cost. Now:

- every string column starts in *ids mode* over one task-wide `StringDictionary`, whatever the input's
  encoding -- a dictionary-encoded input batch (an aggregate's output, #377) maps by its entries, a
  plain one is hashed row by row into the same dictionary;
- the record batches carry the staging ids directly; the dictionary is written once, whole, in the
  dictionary section (a dictionary unit per string column with values, one compressed frame);
- a column whose dictionary stops paying is *frozen*: past `DictionaryCapBytes` (32 MB of distinct
  values) or, once `FreezeSampleRows` (4096) non-null rows are seen, more distinct values than
  `dictionaryMaxRatio` (default 0.5) of its rows. Freezing flushes every partition's pending rows as
  ids (they stay valid: the dictionary is written whole at `finish`), then the column travels plain
  UTF8 for the rest of the task. Rare -- the high-cardinality column of a fact table -- and cheaper than
  what the plain path does anyway.

Since a block's bytes alone no longer decode, every transport prepends the map file's dictionary
section to each range it serves: the Flight producer once per map output in a `DoGet`, the local
reader through `PartitionedIpcFile.blockChannel`. Spark's block-transfer backend delivers a block's
bytes alone, so under `spark.vector.shuffle.backend=block` the writer keeps per-block dictionaries
(`fileDictionary = false`), the pre-#416 format.

### 3.3 Staging and flushing

Below `StagingPartitions` (256) partitions each partition has its own builders and a batch's rows are
gathered straight into them. Above it the task's rows are staged once with their partition ids and
grouped at the flush into one reusable set of batch builders (#416: at 1000 partitions the
per-partition path allocated 18,000 vectors for 1.2 MB of output). Either way a partition's pending
rows become a record batch when they reach `batchRows` (8192) or `batchBytes` (1 MB), or when the
task's held bytes reach `bufferBytes` (64 MB).

A partition's IPC bytes are staged raw on the heap and compressed in frames of at most `flushBytes`
(1 MB) through the writer's single zstd context (#411: a native context per partition was 5% of an
executor). Frames go to a per-partition overflow file as they fill; at `finish` the overflow files and
each partition's last frame are concatenated into the data file. Frames of one partition, and of the
consecutive partitions of a range, decompress as one stream.

Arrow's per-buffer body compression is off (`NoCompressionCodec`): the frame compression is where
the bytes shrink, and it sees whole messages. `zstd` through zstd-jni is the default; Arrow's `lz4` is
pure Java and an order of magnitude slower; `none` for tests and benchmarks.

## 4. The reduce side

`VectorShuffleReader.read()` asks the map output tracker for the task's blocks by executor and builds
one iterator of `ColumnarBatch`es:

- **Remote executors**: one `FlightBlockStream` per executor, all opened up front (#347) so every
  server produces at once and gRPC's flow control bounds what each stream buffers. The ticket names
  the shuffle, the task's reduce range `[start, end)` and the executor's map ids
  (`[shuffleId:int][start:int][end:int][count:int][mapIds:long…]`, big-endian); the server answers
  with the blocks back to back.
- **Local map outputs**: one file segment per map for the task's whole partition range (#411:
  `ShuffleBlockBatchId` -- one index lookup and one open per map where it was one per partition times
  the coalescing factor), read with positional reads straight into Arrow memory.

Both feed `PartitionedIpcFile.StreamReader`, the one decoder:

- it reads units; a dictionary unit loads a `VarCharVector` that stays current for its column until
  replaced; a record batch unit reads the shape bitmap, picks the `VectorSchemaRoot` and loader for
  that shape (cached per shape) and loads the batch;
- dictionaries are shared by reference count between the reader and the batches handed out over
  them: a replacement (the next map output's dictionary) does not disturb a batch the operator still
  holds, and a large per-map dictionary is never copied per batch;
- **coalescing** (#411): a batch under `CoalesceRows` (1024) is appended to a pending plain batch
  instead of reaching the operators on its own -- kernel set-up, a hash table's probe round and an
  output batch per input batch were most of a reduce task's time over ten-row blocks. A dictionary-
  encoded column of such a block is decoded into the pending plain column (#438). Because a `DoGet`
  concatenates a range's map outputs into one stream, coalescing crosses map outputs on the Flight
  path; the local path reads each map's segment separately.
- the consumer does not close its input: a batch is closed when the next is produced, or at `close`.

## 5. The Flight service

One `FlightServer` per executor over gRPC on Spark's Netty (`flight-core` lives in `shuffle/` so the
plugin jar never carries gRPC). `Producer.getStream` parses the ticket and, per map id, asks the
block resolver for the range's `ManagedBuffer` (a file segment), prepends the dictionary section and
copies the bytes into a one-column `VarBinary` root, one row per chunk of `ChunkBytes` (4 MB). Not as
Flight record batches (#338: Flight's own framing sends dictionaries once per stream, ours replace
them within one) and not one message per map output (#416: 2,000 map outputs as 2,000 gRPC messages
of a few KB each cost a Flight decode and a flow-control round trip apiece -- a third of a reduce
task's time at 1 TB / 1000 partitions); the chunk fills across map outputs and is sent when full or
when the range ends. The server's executor pool is `spark.vector.shuffle.flight.threads`
(default `max(4, cores)`).

Client side, `FlightBlockStream` wraps the stream in a `ReadableByteChannel` (`ChunkChannel`) the
`StreamReader` pulls from as it needs bytes, so decode overlaps transfer. The decoder is created on
the first `hasNext`, not in the constructor (#416): opening seven servers' streams back to back and
waiting for each first message inside the constructor serialised the waits. Clients are pooled per
JVM per remote location.

**Security.** With `spark.authenticate` on, every call carries Spark's shuffle secret as a bearer
token and an unauthenticated `DoGet` is refused; the server refuses to start when auth is on but no
secret can be read. TLS is the open gap of #288: the server needs PEM material and Spark configures
JKS, so under `spark.ssl.rpc.enabled` the server refuses to start rather than run in the clear --
use `spark.vector.shuffle.backend=block` there. Without `spark.authenticate` the endpoint is as open
as Spark's own block transfer in that configuration.

**Failures.** A remote fetch that fails for any reason other than the reducer's own Arrow memory
(`OutOfMemoryException`) is turned into Spark's `FetchFailedException` carrying the block's map index
(#364), so the scheduler recomputes the lost map outputs instead of failing the job; memory failures
propagate as the reducer's error, wrapped into a serialisable exception (Arrow's is not, and on JDK
24+ an unserialisable task failure ended the executor -- SPARK-55679).

## 6. The backend seam

`VectorShuffleBackend` is the interface between the reader and the transport: `remoteBlocks(address,
blocks, …)` returns batch iterators for one executor's blocks. Three implementations:

| `spark.vector.shuffle.backend` | what it is |
|---|---|
| `flight` (default) | the Flight data plane above |
| `block` | Spark's block transfer against the same files, all of an executor's blocks in one request; per-block dictionaries |
| a class name | a backend with a no-argument constructor -- where a push-based shuffle service (Celeborn-style) would plug in; future work |

## 7. Tuning knobs and what they move

| knob | default | moves |
|---|---|---|
| `spark.vector.shuffle.batchRows` / `batchBytes` | 8192 / 1 MB | record batch size on the map side: larger batches, fewer messages, more map-side memory |
| `spark.vector.shuffle.bufferBytes` | 64 MB | task-wide cap on held rows before a forced flush |
| `spark.vector.shuffle.flushBytes` | 1 MB | compression frame size (one zstd call per frame) |
| `spark.vector.shuffle.compression` | `zstd` | `zstd`, `lz4`, `none` |
| `spark.vector.shuffle.writer.dictionaryMaxRatio` | 0.5 | distinct/rows above which a string column is frozen plain (0 never encodes, 1 always) |
| `spark.vector.shuffle.writer.memoryLimit` | 1 GB | the writer allocator's hard limit (backstop; flushes happen long before) |
| `spark.vector.shuffle.flight.threads` | `max(4, cores)` | server executor pool: concurrent `DoGet`s an executor serves |
| `spark.vector.shuffle.flight.bindHost` | block manager host | server bind address |
| `Producer(chunkBytes)` | 4 MB | bytes per gRPC message (constructor parameter; benchmark sweep) |
| `-Dsparkvector.shuffle.reader.coalesceRows` | 1024 | rows a reader accumulates before handing a batch to the operators |

## 8. Measuring it

Two rulers, for two different questions.

**In-process, JMH** -- `FlightShuffleBenchmark`: two Flight servers on loopback stand in for two
executors, each serving map files the real writer produced; one operation is one reduce task. Four
operations apportion the cost: `reduceTask` (the real path), `reduceTasksConcurrent` (eight reducers
against the same two servers -- where the server pool, chunk size and flow control contend),
`rawTransport` (the bytes undecoded: the transport alone), `localRead` (the same blocks with no
Flight: the floor). Aux counters give rows, batches and bytes per iteration; rows / batches is the
mean batch the operators receive. Parameters: partitions (block size), range width (AQE coalescing),
compression, server threads, chunk bytes, string columns. It measures the CPU side -- serialisation,
chunking, decode, threads -- not bandwidth; that is what the knobs above change. Run it with
`benchmarks/scripts/run-flight-bench.sh` (Spark is a provided dependency, so the jar alone will not
do); each run writes a JSON file, an A/B is two runs and a diff. Host noise on a shared box is
several percent: compare against a measured band.

**On the cluster** -- the TPC-DS matrix (`benchmarks/k8s/run-matrix.sh`) with event logs, read per
stage: shuffle write time and bytes of the map stage, fetch wait and executor time of the reduce
stage. That is the ruler that found #347 (round trips), #411 (per-partition costs), #416 (per-block
costs) and #377 (the dictionary's other side): each began as a stage whose executor time did not
match its bytes.

## 9. Open items

- TLS for the Flight server (#288): PEM from Spark's JKS, or Spark's own SSL context.
- A push-based shuffle service through the backend seam (Celeborn or our own aggregator); the
  `StreamReader` already decodes several map outputs' streams concatenated, which is what an
  aggregated partition would hold.
- A frozen column's dictionary is still written whole; a column frozen very early could drop it.
- The block-transfer backend keeps the per-block format; if it stays, its writer path could carry the
  dictionary section as a prefix instead.
