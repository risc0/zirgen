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

def MakeVerifyTaps : Pass<"make-verify-taps", "mlir::ModuleOp"> {
  let summary = "Generate code for verifying taps for FRI queries";
  let constructor = "zirgen::Zll::createMakeVerifyTapsPass()";
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

#endif // ZLL_TRANSFORM_PASSES
