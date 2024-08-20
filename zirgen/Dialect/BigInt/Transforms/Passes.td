#ifndef BIGINT_TRANSFORM_PASSES
#define BIGINT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LowerModularInv : Pass<"lower-modular-inv", "mlir::ModuleOp"> {
  let summary = "Remove BigInt::ModularInvOp by lowering it to other ops";
  let constructor = "zirgen::BigInt::createLowerModularInvPass()";
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
