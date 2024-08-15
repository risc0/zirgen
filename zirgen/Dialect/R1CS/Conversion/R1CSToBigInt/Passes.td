#ifndef R1CS_CONVERSION_PASSES
#define R1CS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def R1CSToBigInt : Pass<"r1cs-to-bigint", "mlir::ModuleOp"> {
  let summary = "Render R1CS constraints into BigInt operations";
  let constructor = "zirgen::R1CSToBigInt::createR1CSToBigIntPass()";
}

#endif // R1CS_CONVERSION_PASSES
