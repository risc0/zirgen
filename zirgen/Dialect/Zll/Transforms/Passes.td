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

#ifndef ZLL_TRANSFORM_PASSES
#define ZLL_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ComputeTaps : Pass<"compute-taps", "mlir::func::FuncOp"> {
  let summary = "Compute tap ID for each tap and save the tap table";
  let constructor = "zirgen::Zll::createComputeTapsPass()";
}

def MakePolynomial : Pass<"make-polynomial", "mlir::func::FuncOp"> {
  let summary = "Compute a polynomial mix of constraints";
  let constructor = "zirgen::Zll::createMakePolynomialPass()";
}

def SplitStage : Pass<"split-stage", "mlir::func::FuncOp"> {
  let summary = "Split a function into one of its stages";
  let constructor = "zirgen::Zll::createSplitStagePass()";
  let options = [
    Option<"stage", "stage-number", "unsigned", /*default=*/"0",
           "Which stage to keep">,
  ];
}

def DropConstraints : Pass<"drop-constraints", "mlir::func::FuncOp"> {
  let summary = "Drop any constraint checking, e.g. EqualZeroOp";
  let constructor = "zirgen::Zll::createDropConstraintsPass()";
}

def InlineFpExt : Pass<"inline-fpext", "mlir::func::FuncOp"> {
  let summary = "Inline FpExt into primitive Fps";
  let constructor = "zirgen::Zll::createInlineFpExtPass()";
}

def AddReductions : Pass<"add-reductions", "mlir::func::FuncOp"> {
  let summary = "Add reduction ops as required to keep computations below the size of the native field";
  let constructor = "zirgen::Zll::createAddReductionsPass()";
  let options = [
    Option<"maxBits", "max-bits", "unsigned", /*default=*/"255", "Number of bits where outer field is still unambigous">,
  ];
}

def IfToMultiply : Pass<"if-to-multiply"> {
  let summary = "Remove EqualZeroOp constraints from if blocks by multiplying with the condition";
  let constructor = "zirgen::Zll::createIfToMultiplyPass()";
}

def MultiplyToIf : Pass<"multiply-to-if"> {
  let summary = "Remove factors from EqualZeroOp constraints by moving them into `if` blocks.";
  let constructor = "zirgen::Zll::createMultiplyToIfPass()";
}

def BalancedSplit : Pass<"balanced-split"> {
  let summary = "Attempt to split up large blocks";
  let constructor = "zirgen::Zll::createBalancedSplitPass()";
  let options = [
    Option<"maxOps", "max-ops", "unsigned", /*default=*/"5000", "Maximum number of operations in a block before splitting it">
    ];
}

def SortForReproducibility : Pass<"sort-for-reproducibility"> {
  let summary = "Attempt to make operation order consistent between compilers and compilations";
  let description = [{
     HACK: Evaluation order of arguments is unspecified in c++ and
     different compilers do it differently. We want our circuit
     compilations to be reproducible, so sortForReproducibility tries
     to undo the varation in argument order.
  }];
  let constructor = "zirgen::Zll::createSortForReproducibilityPass()";
}

#endif // ZLL_TRANSFORM_PASSES
