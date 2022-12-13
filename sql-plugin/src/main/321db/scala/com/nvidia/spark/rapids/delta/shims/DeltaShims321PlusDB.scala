/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
package com.nvidia.spark.rapids.delta.shims

import com.databricks.sql.transaction.tahoe.DeltaUDF
import com.databricks.sql.transaction.tahoe.constraints.Constraints._
import com.databricks.sql.transaction.tahoe.schema.InvariantViolationException

import org.apache.spark.sql.expressions.UserDefinedFunction

object InvariantViolationExceptionShim {
  def apply(c: Check, m: Map[String, Any]): InvariantViolationException = {
    InvariantViolationException(c, m)
  }

  def apply(c: NotNull): InvariantViolationException = {
    InvariantViolationException(c)
  }
}

object ShimDeltaUDF {
  def stringStringUdf(f: String => String): UserDefinedFunction = {
     DeltaUDF.stringStringUdf(f)
  }
}