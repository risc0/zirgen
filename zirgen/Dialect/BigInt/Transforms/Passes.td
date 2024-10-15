// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BIGINT_TRANSFORM_PASSES
#define BIGINT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LowerInv : Pass<"lower-inv", "mlir::ModuleOp"> {
  let summary = "Remove BigInt::InvOp by lowering it to other ops";
  let constructor = "zirgen::BigInt::createLowerInvPass()";
}

def LowerReduce : Pass<"lower-reduce", "mlir::ModuleOp"> {
  let summary = "Remove BigInt::ReduceOp by lowering it to other ops";
  let constructor = "zirgen::BigInt::createLowerReducePass()";
}

def LowerZll : Pass<"lower-zll", "mlir::ModuleOp"> {
  let summary = "Convert bigint to verification ZLL using CheckedBytes ";
  let constructor = "zirgen::BigInt::createLowerZllPass()";
  let dependentDialects = ["zirgen::Iop::IopDialect"];
}

#endif // BIGINT_TRANSFORM_PASSES
