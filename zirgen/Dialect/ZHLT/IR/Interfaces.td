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

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

#ifndef ZHLT_INTERFACES
#define ZHLT_INTERFACES

def InferInputSegmentSizesOpInterface : OpInterface<"InferInputSegmentSizesOpInterface"> {
  let description = [{
    Interface to infer the segment sizes of a callable with multiple variadic inputs.

    Typically such an operation would need to have an 'inputSegmentSizes' attribute that tracks
    how many concrete inputs are part of each declared input. It is sometimes more straightforward
    to automatically derive this from the operation itself, rather than storing, updating,
    verifying, and by default printing this information as an attribute on the operation.
  }];
  let methods = [
    InterfaceMethod<"use a dynamic 'getter' to define the input segment sizes of a callable",
      "::llvm::SmallVector<int32_t>", "getInputSegmentSizes", (ins)>,
  ];
}

def InferResultSegmentSizesOpInterface : OpInterface<"InferResultSegmentSizesOpInterface"> {
  let description = [{
    Interface to infer the segment sizes of an operation with multiple variadic results.

    Typically such an operation would need to have a 'resultSegmentSizes' attribute that tracks
    how many concrete operands are part of each declared operand. It is sometimes more
    straightforward to automatically derive this from the operation itself, rather than storing,
    updating, verifying, and by default printing this information as an attribute on the operation.
  }];
  let methods = [
    InterfaceMethod<"use a dynamic 'getter' to define the result segment sizes",
      "::llvm::SmallVector<int32_t>", "getResultSegmentSizes", (ins)>,
  ];
}

#endif /* ZHLT_INTERFACES */
