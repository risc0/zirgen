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

def ExtractPolyMix : Pass<"extract-poly-mix", "mlir::func::FuncOp"> {
  let summary = "Extract poly_mix from zll.and_eqz and zll.and_cond operations";
  let description = [{
     This removes zll.and_eqz, zll.true, and zll.and_cond operations from the functions present.
     It transforms them into array lookups of anew `poly_mix` argument which provides all the powers
     of poly_mix.

     The powers needed must be generated elsewhere; this just assumes that the powers available
     are the same ones as provided by MixPowerAnalysis.
  }];
  let constructor = "zirgen::Zll::createExtractPolyMixPass()";
}

def AnnotatePolyMix : Pass<"annotate-poly-mix", "mlir::func::FuncOp"> {
  let summary = "Add zirgen.mixPowerIndex to zll.and_eqz and zlll.and_cond";
  let description = [{
     This adds the mix power index from MixPowerAnalysis as a `zirgen.mixPowerIndex` attribute
     on zll.and_eqz and zll.and_cond operations
  }];
  let constructor = "zirgen::Zll::createAnnotatePolyMixPass()";
}

def ReorderConstraints : Pass<"reorder-constraints"> {
  let summary = "Optimize by reordering zll.eqz constraints";
  let description = [{
     ReorderConstraints reorders constraints, and puts groups of constraints
     inside a `if(1)` such that they can be better optimized by CSE.
  }];
  let constructor = "zirgen::Zll::createReorderConstraintsPass()";
}

#endif // ZLL_TRANSFORM_PASSES
