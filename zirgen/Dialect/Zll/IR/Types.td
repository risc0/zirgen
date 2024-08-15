// Copyright (c) 2023 RISC Zero, Inc.

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/Zll/IR/Dialect.td"
include "zirgen/Dialect/Zll/IR/Enums.td"
include "zirgen/Dialect/Zll/IR/Interfaces.td"
include "zirgen/Dialect/Zll/IR/Attrs.td"

#ifndef ZLL_TYPES
#define ZLL_TYPES

class ZllType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZllDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Val : ZllType<"Val", "val", [DeclareTypeInterfaceMethods<CodegenTypeInterface, ["getTypeName", "emitLiteral"]>]> {
  let summary = "An expression which results in a single field element";
  let parameters = (ins
    "::zirgen::Zll::FieldAttr": $field,
    OptionalParameter<"mlir::UnitAttr">: $extended
  );
  let builders = [
      TypeBuilder<(ins), "return get($_ctxt, zirgen::Zll::getDefaultField($_ctxt), /*extended=*/{});">];
  let assemblyFormat = "`<` custom<Field>($field) `` custom<FieldExt>($extended) `>`";
  let extraClassDeclaration = [{
      // DEPRECATED; prefer to construct based on a FieldAttr, using
      // zirgen::Zll::getField or zirgen::Zll::getDefaultField.
      static ValType get(mlir::MLIRContext* ctx, uint64_t fieldP, unsigned fieldK = 1);

      static ValType getBaseType(mlir::MLIRContext* ctx);
      static ValType getExtensionType(mlir::MLIRContext* ctx);

      // DEPRECATED: prefer to use getField() or getExtensionField() to query the field being used.
      uint64_t getFieldP() const;
      unsigned getFieldK() const;

      // Returns the field arithmetic implementation for this value.
      ExtensionField getExtensionField() const;
  }];
}

def String : ZllType<"String", "string", [DeclareTypeInterfaceMethods<CodegenTypeInterface, ["getTypeName", "emitLiteral"]>]> {
  let summary = "An expression which results in text";
}

def Constraint : ZllType<"Constraint", "constraint", [DeclareTypeInterfaceMethods<CodegenTypeInterface, ["getTypeName"]>]> {
  let summary = "An expression representing a constraint";
  let extraClassDefinition = [{
     zirgen::codegen::CodegenIdent<zirgen::codegen::IdentKind::Type>
     $cppClass::getTypeName(zirgen::codegen::CodegenEmitter& cg) const {
        return cg.getStringAttr("MixState");
     }
  }];
}

def Buffer : ZllType<"Buffer", "buffer", [DeclareTypeInterfaceMethods<CodegenTypeInterface, ["getTypeName"]>]> {
  let summary = "A buffer of 1 or more field element values";
  let parameters = (ins
      DefaultValuedParameter<"Zll::ValType", "Zll::ValType::getBaseType($_ctxt)">: $element,
      "unsigned": $size,
      "BufferKind": $kind
  );
  let assemblyFormat = "`<`$size `,` $kind  (`,` $element^)? `>`";
}

def Digest : ZllType<"Digest", "digest", [CodegenTypeInterface]> {
  let summary = "A cryptographic digest";
  let parameters = (ins
      "DigestKind": $kind
  );
  let assemblyFormat = "`<`$kind`>`";
}

def VariadicPack : ZllType<"Variadic", "variadic", [CodegenNeedsCloneType,
                                                    DeclareTypeInterfaceMethods<CodegenTypeInterface, ["getTypeName"]>]> {
  let summary = "the type of values which pack variadic parameters";
  let parameters = (ins "mlir::Type": $element);
  let assemblyFormat = [{ $element }];
}

def CodegenOpaque : ZllType<"CodegenOpaque", "codegen_opaque"> {
  let parameters = (ins StringRefParameter<"the opaque type">:$value);
  let assemblyFormat = [{ $value }];
}

def PolyMix : ZllType<"PolyMix", "poly_mix", [CodegenTypeInterface]> {
}

#endif
