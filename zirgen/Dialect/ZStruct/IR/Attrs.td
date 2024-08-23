// Copyright 2024 RISC Zero, Inc.
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

