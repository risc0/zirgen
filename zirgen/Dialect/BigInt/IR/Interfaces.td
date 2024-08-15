
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

#ifndef BIGINT_INTERFACES
#define BIGINT_INTERFACES

/*
def EvalOp : OpInterface<"EvalOp"> {
  let cppNamespace = "::zirgen::BigInt";
  let methods = [
    InterfaceMethod<"Evaluate this op",
      "::llvm::APInt", "evaluate", (ins "EvalContext&":$ctx, "::llvm::ArrayRef<::llvm::APInt>":$inputs)
    >,
  ];
}
*/

#endif /* BIGINT_INTERFACES */

