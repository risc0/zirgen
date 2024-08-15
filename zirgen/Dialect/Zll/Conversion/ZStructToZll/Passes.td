#ifndef ZLL_CONVERSION_PASSES
#define ZLL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LowerComposites : Pass<"lower-composites", "mlir::func::FuncOp"> {
  let summary = "Lower all composite types to buffers";
  let constructor = "zirgen::ZStructToZll::createLowerCompositesPass()";
}

#endif // ZLL_CONVERSION_PASSES
