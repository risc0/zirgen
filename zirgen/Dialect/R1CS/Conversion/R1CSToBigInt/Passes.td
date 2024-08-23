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

#ifndef R1CS_CONVERSION_PASSES
#define R1CS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def R1CSToBigInt : Pass<"r1cs-to-bigint", "mlir::ModuleOp"> {
  let summary = "Render R1CS constraints into BigInt operations";
  let constructor = "zirgen::R1CSToBigInt::createR1CSToBigIntPass()";
}

#endif // R1CS_CONVERSION_PASSES
