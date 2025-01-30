// Copyright 2025 RISC Zero, Inc.
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

#ifndef BYTECODE_TRANSFORM_PASSES
#define BYTECODE_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def GenExecutor : Pass<"gen-executor", "mlir::ModuleOp"> {
  let description = [{
    Generate a bytecode executor suitable for executing any functions present.
  }];
  let options = [
    Option<"execSymbol", "exec-symbol",
           "std::string", /*default=*/"\"exec_byte_code\"", "Symbol name of the geneerated bytecode executor">,
    Option<"maxArms", "max-arms", "size_t", /*default=*/"256", "Maximum number of arms to generate">,
    Option<"minArmUse", "min-arm-use", "size_t", /*default=*/"5", "Minimum number of uses of a sequence of operations in order to generate an arm for them">,
    Option<"maxCapture", "max-capture", "size_t", /*default=*/"5", "Maximum number of temporary values to capture in an arm">,
    Option<"maxYield", "max-yield", "size_t", /*default=*/"5", "Maximum number of temporary values output from an arm">,
    
  ];
  let statistics = [
    Statistic<"armCount", "arms", "Number of arms generated">,
    Statistic<"maxArmOps", "max-arm-ops", "maximum number of operations in an arm">,
    Statistic<"maxArmUses", "max-arm-uses", "maximum number of uses of an arm">,
    Statistic<"maxArmReplaced", "max-arm-replaced", "maximum number of operations that could potentially be replaced due to an arm">
  ];
  let dependentDialects = ["mlir::func::FuncDialect", "zirgen::ByteCode::ByteCodeDialect"];
}

def Encode : Pass<"encode", "mlir::ModuleOp"> {
  let description = [{
     For each function present, convert it into a EncodedBlockOp full of bytecode.
  }];
  let options = [
    Option<"execSymbol", "exec-symbol",
           "std::string", /*default=*/"\"exec_byte_code\"", "Symbol name bytecode executor">
  ];
  let dependentDialects = ["mlir::func::FuncDialect", "zirgen::ByteCode::ByteCodeDialect"];
  let statistics = [
    Statistic<"origOps", "orig-ops", "Original operation count before encoding">,
    Statistic<"encodedOps", "encoded-ops", "Operation count after encoding">,
    Statistic<"encodedVals", "encoded-vals", "Total encoded values">,
  ];
}

def CloneSimpleZll : Pass<"clone-simple-zll", "mlir::func::FuncOp"> {
  let description = [{
     Clone all "simple" operations for each user.  This should be used for operations
     which are more expensive to encode and decode bytecode to execute than to re-execute.
  }];
}

def ScheduleZll : Pass<"schedule-zll", "mlir::func::FuncOp"> {
  let description = [{
     Schedule functions containing Zll operations, attempting to minimize active register count.
  }];
  let dependentDialects = [
      "mlir::func::FuncDialect",
      "zirgen::ByteCode::ByteCodeDialect",
      "zirgen::Zll::ZllDialect",
  ];
}

def BufferizeZll : Pass<"bufferize-zll", "zirgen::ByteCode::EncodedBlockOp"> {
  let description = [{
     Generate temporary buffer assignments for an encoded bytecode program
  }];
  let dependentDialects = [
      "zirgen::ByteCode::ByteCodeDialect",
      "zirgen::Zll::ZllDialect",
  ];
}

#ifdef NOPE
def GenNaiveEncoding : Pass<"gen-naive-encoding", "mlir::func::ExecutorOp"> {
  let description = [{
     Generates an encoding scheme for executors which encode everything as uint32s.
  }];
  let dependentDialects = [
      "mlir::func::FuncDialect",
      "zirgen::ByteCode::ByteCodeDialect",
  ];
}
#endif

#endif // BYTECODE_TRANSFORM_PASSES
