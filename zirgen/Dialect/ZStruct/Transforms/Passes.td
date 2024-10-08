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

#ifndef ZSTRUCT_TRANSFORM_PASSES
#define ZSTRUCT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def OptimizeLayout : Pass<"optimize-layout", "mlir::ModuleOp"> {
  let summary = "Reorder structure members for constraint compatibility";
  let constructor = "zirgen::ZStruct::createOptimizeLayoutPass()";
}

def Unroll : Pass<"unroll"> {
  let summary = "Unroll zhlt map and reduce";
  let description = [{
    Removes zhlt.map and zhlt.reduce instructions by unrolling the loops.
  }];
  let constructor = "zirgen::ZStruct::createUnrollPass()";
  let dependentDialects = ["zirgen::ZStruct::ZStructDialect"];
}

def ExpandLayout : Pass<"expand-layout", "mlir::ModuleOp"> {
  let summary = "Expand global layout constants so that each layout constant only has data for a single component.";
  let constructor = "zirgen::ZStruct::createExpandLayoutPass()";
}

def InlineLayout : Pass<"inline-layout"> {
  let summary = "Inline offsets from layouts";
  let description = [{
    Converts zstruct.load and zstruct.store to zll.{get, get_global, set, set_global} when possible
  }];
  let constructor = "zirgen::ZStruct::createExpandLayoutPass()";
}

def BuffersToArgs : Pass<"buffers-to-args"> {
  let summary = "Convert any calls to zstruct.get_buffer to function arguments";
  let description = [{
     All new arguments are added to the end of any existing function arguments, and
     are added in the order they appear in the BufferAnalysis.

     These new arguments are propagated from any callers.
  }];
  let constructor = "zirgen::ZStruct::createBuffersToArgsPass()";
}

#endif // ZSTRUCT_TRANSFORM_PASSES
