#ifndef ZSTRUCT_TRANSFORM_PASSES
#define ZSTRUCT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def OptimizeLayout : Pass<"optimize-layout", "mlir::ModuleOp"> {
  let summary = "Reorder structure members for constraint compatibility";
  let constructor = "zirgen::ZStruct::createOptimizeLayoutPass()";
}

def Unroll : Pass<"unroll", "mlir::ModuleOp"> {
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

def StripAliasLayoutOps : Pass<"inline-layout", "mlir::ModuleOp"> {
  let summary = "Erase all AliasLayoutOps";
  let description = [{
    ZStruct::AliasLayoutOps are introduced during the lowering to ZHLT, and used
    to indicate layout constraints to GenerateLayoutPass when constructing the
    concrete layout of the circuit. This pass eliminates all AliasLayoutOps from
    the module.
  }];
  let constructor = "zirgen::ZStruct::createExpandLayoutPass()";
}


#endif // ZSTRUCT_TRANSFORM_PASSES
