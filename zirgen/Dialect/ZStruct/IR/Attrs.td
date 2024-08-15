include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "zirgen/Dialect/ZStruct/IR/Dialect.td"

class ZStructAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZStructDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def RefAttr : ZStructAttr<"Ref", "ref", [TypedAttrInterface]> {
  let summary = "A layout for a single register.";
  let parameters = (ins
      "uint32_t":$index,
      AttributeSelfTypeParameter<"type", "RefType">:$type);
  let assemblyFormat = [{ $index }];
}

def StructAttr : ZStructAttr<"Struct", "struct", [TypedAttrInterface]> {
  let summary = "A structure of components";
  let parameters = (ins
      "mlir::DictionaryAttr":$fields,
      AttributeSelfTypeParameter<"StructType or LayoutType">:$type);
  let assemblyFormat = [{ $fields }];
}

def UnionAttr : ZStructAttr<"Union", "union", [TypedAttrInterface]> {
  let summary = "A mux of components";
  let parameters = (ins
      "mlir::Attribute":$activeValue,
      AttributeSelfTypeParameter<"UnionType">:$type);
  let assemblyFormat = [{ $activeValue }];
}

def BoundLayoutAttr : ZStructAttr<"BoundLayout", "bound_layout"> {
  let summary = "A symbolic view of a zll.buffer, ";
  let parameters = (ins
      "mlir::StringAttr":$buffer, // Name of buffer
      "mlir::Attribute":$layout  // Layout structure
  );
  let builders = [
      AttrBuilderWithInferredContext<(ins "mlir::StringAttr":$buffer, "mlir::Attribute":$layout), [{
         return $_get(buffer.getContext(), buffer, layout);
      }]>
  ];
  let assemblyFormat = [{ $buffer `as` $layout }];
}

