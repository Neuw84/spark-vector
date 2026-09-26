/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package org.apache.spark.sql.execution.vector

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.joins.{
  EmptyHashedRelation,
  HashedRelation,
  HashedRelationWithAllNullKeys,
  LongHashedRelation
}

/**
 * `HashedRelation` is `private[execution]`; this bridge lives inside that package so the join
 * operators (in `org.apache.spark.sql.vecruntime`) can read the rows of Spark's broadcast relation.
 */
object HashedRelationAccess {

  /**
   * Every build row of the relation exactly once, from a task-local read-only copy. (`keys()`
   * yields a key once per row stored under it, so `keys().flatMap(get)` would repeat rows of
   * duplicated keys.) The returned rows may be reused by the iterator: consume each before
   * advancing.
   */
  def rows(relation: Any): Iterator[InternalRow] = relation.asInstanceOf[HashedRelation].asReadOnlyCopy() match {
    // The two singletons a null-aware build produces: no rows to read in either (the second must
    // be recognised before this point, see allNullKeys -- its accessors throw).
    case EmptyHashedRelation | HashedRelationWithAllNullKeys => Iterator.empty
    // The long-keyed map lists each key once and has no valuesWithKeyIndex.
    case l: LongHashedRelation => l.keys().flatMap(key => l.get(key))
    case r => r.valuesWithKeyIndex().map(_.getValue)
  }

  /** Whether a null-aware build side held a null key: Spark then broadcasts this singleton instead of the rows. */
  def allNullKeys(relation: Any): Boolean = relation == HashedRelationWithAllNullKeys
}
