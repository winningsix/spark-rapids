/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.types._

/**
 * Unit tests for AST batch project optimization
 */
class AstBatchProjectSuite extends AnyFunSuite {

  test("analyzeExpressions should correctly count AST-compatible expressions") {
    // Create some simple bound expressions
    val ref0 = GpuBoundReference(0, LongType, nullable = true)(
      org.apache.spark.sql.catalyst.expressions.ExprId(0), "a")
    val ref1 = GpuBoundReference(1, LongType, nullable = true)(
      org.apache.spark.sql.catalyst.expressions.ExprId(1), "b")
    
    // GpuLiteral should be AST-compatible
    val lit = GpuLiteral(0L, LongType)
    
    // Simple expressions
    val expressions = Seq(ref0, ref1, lit)
    
    val stats = AstBatchProject.analyzeExpressions(expressions)
    
    // GpuBoundReference and GpuLiteral should be AST-compatible
    assert(stats.totalExpressions == 3)
    assert(stats.astCompatible >= 2) // At least refs and literal should work
  }

  test("canConvertToAst should return true for GpuLiteral") {
    val lit = GpuLiteral(42L, LongType)
    assert(AstBatchProject.canConvertToAst(lit))
  }

  test("canConvertToAst should return true for GpuBoundReference") {
    val ref = GpuBoundReference(0, LongType, nullable = true)(
      org.apache.spark.sql.catalyst.expressions.ExprId(0), "a")
    assert(AstBatchProject.canConvertToAst(ref))
  }

  test("AstProjectStats should calculate ratio correctly") {
    val stats = AstBatchProject.AstProjectStats(100, 75, 25)
    assert(stats.astRatio == 0.75)
    assert(stats.toString.contains("75.0%"))
  }

  test("AstProjectStats should handle zero total") {
    val stats = AstBatchProject.AstProjectStats(0, 0, 0)
    assert(stats.astRatio == 0.0)
  }
}


