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

#ifndef ZHLT_TRANSFORM_PASSES
#define ZHLT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def HoistAllocs : Pass<"hoist-allocs", "mlir::ModuleOp"> {
  let summary = "Hoist alloc-only params to callers, then merge";
  let description = [{
    Removes some unnecessary allocations, allowing programs to conservatively
    invoke NondetReg without penalty. When the only use of some parameter is a
    call to NondetReg, that allocation will be hoisted to the callers of the
    component, and its parameter signature will be changed accordingly. The
    result may be a redundant allocation within the outer component, so nested
    calls to NondetReg will also be merged.
  }];
  let constructor = "zirgen::Zhlt::createHoistAllocsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

def StripTests : Pass<"strip-tests", "mlir::ModuleOp"> {
  let summary = "Strip all tests for smaller generated code.";
  let constructor = "zirgen::Zhlt::createStripTestsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

def GenerateSteps : Pass<"generate-steps", "mlir::ModuleOp"> {
  let summary = "Generate top-level StepOps for all externally callable functions ";
  let description = [{
     The following steps are generated:

     step$top: Constructs the component named "Top" and checks its constraints
     step$test$*: A StepOp to run each test
  }];
  let constructor = "zirgen::Zhlt::createStripTestsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

def StripAliasLayoutOps : Pass<"inline-layout", "mlir::ModuleOp"> {
  let summary = "Erase all AliasLayoutOps outside of CheckLayoutFuncOps";
  let description = [{
    ZStruct::AliasLayoutOps are introduced during the lowering to ZHLT, and used
    to indicate layout constraints to GenerateLayoutPass when constructing the
    concrete layout of the circuit. This pass eliminates all AliasLayoutOps that
    aren't inside of a CheckLayoutFuncOp.
  }];
  let constructor = "zirgen::Zhlt::createStripAliasLayoutOpsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

def LowerStepFuncs : Pass<"lower-step-funcs", "mlir::ModuleOp"> {
  let summary = "Convert all functions to StepFuncs";
  let constructor = "zirgen::Zhlt::createLowerStepFuncsPass()";
  let dependentDialects = ["zirgen::Zhlt::ZhltDialect"];
}

def AnalyzeBuffers : Pass<"analyze-buffers", "mlir::ModuleOp"> {
  let summary = "Analyze the buffers needed for this ZHLT circuit";
  let description = [{
     This analyzes the ZHLT buffers needed for a circuit, and saves the buffer descriptions in a BuffersAttr
     on the module.
  }];
  let constructor = "zirgen::Zhlt::createAnalyzeBuffersPass()";
}

def OptimizeParWitgen : Pass<"optimize-par-witgen", "mlir::ModuleOp"> {
  let summary = "Optimize for parallel witness generation";
  let description = [{
     Applies optimizations for parallel witness generation.
        * Flattens, unrolls, and inlines all zstruct structures to minimize using local memory.
        * Erases unused externs and constraint checks.
        * Assumes any other externs called are idempotent
  }];
  let constructor = "zirgen::Zhlt::createOptimizeParWitgenPass()";
}

def OutlineIfs : Pass<"outline-ifs", "mlir::ModuleOp"> {
  let summary = "Move bodies of `if` statements into their own calls";
  let description = [{
     Outlines the body of of all zll.if operations inside of zhlt.step_funcs to
     separate functions.  Captures values when necessary, but prefers to reconstruct
     them from e.g. `zll.get` operations.
  }];
  let constructor = "zirgen::Zhlt::createOutlineIfsPass()";
}

#endif // ZHLT_TRANSFORM_PASSES
