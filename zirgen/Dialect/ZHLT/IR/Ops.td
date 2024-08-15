// Copyright (c) 2023 RISC Zero, Inc.

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "zirgen/Dialect/ZHLT/IR/Dialect.td"
include "zirgen/Dialect/ZStruct/IR/Types.td"

class ZhltOp<string mnemonic, list<Trait> traits = []>
    : Op<ZhltDialect, mnemonic, traits> {
}

def ReturnOp : ZhltOp<"return", [Terminator, ReturnLike]> {
  let summary = "A terminator which marks the result of a component constructor";
  let arguments = (ins Optional<AnyTypeOf<[ZirType, Constraint]>>:$value);
  let builders = [
    OpBuilder<(ins), "build($_builder, $_state, {});">];
  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def GetGlobalLayoutOp : ZhltOp<"get_global_layout", [Pure]> {
  let summary = "Returns the layout associated with a given component in a global buffer";
  let arguments = (ins
      DefaultValuedAttr<StrAttr, "\"global\"">:$buffer,
      StrAttr:$name);
  let results = (outs LayoutType:$out);
  let builders = [
    OpBuilder<
      (ins "mlir::Type":$out, "llvm::StringRef":$name),
      "build($_builder, $_state, out, \"global\", name);">,
  ];
}

def BackOp: ZhltOp<"back", [MemoryEffects<[MemRead]>]> {
  let summary = "Retrieve value from previous cycle";
  let arguments = (ins FlatSymbolRefAttr:$callee,
      IndexAttr:$distance, Optional<IsLayoutType>:$layout);
  let results = (outs ZirType:$out);
  let builders = [
     OpBuilder<(ins "mlir::Type":$out, "llvm::StringRef":$callee, "size_t":$distance, "mlir::Value":$layout),
       "build($_builder, $_state, out, callee, llvm::APInt(64, distance), layout);">
  ];
  let assemblyFormat = [{
     $callee `(` $distance ( `,` $layout^ `:` type($layout) )?
     `)` `->` type($out) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

def MagicOp: ZhltOp<"magic", [Pure]> {
  let summary = "Construct magic value for error recovery";
  let description = [{
    Constructs a value of any type "out of nothing" in order to recover when
    lowering illegal input. This makes it possible to proceed with the lowering
    and detect additional errors, but this operation is always invalid.
  }];
  let results = (outs ZirType:$out);
  let assemblyFormat = [{
    type($out) attr-dict
  }];
  let hasVerifier = 1;
}

