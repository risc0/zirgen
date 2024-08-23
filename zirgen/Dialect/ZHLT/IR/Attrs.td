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

#ifndef ZIRGEN_IR_ATTRS
#define ZIRGEN_IR_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/ZHLT/IR/Dialect.td"

class ZhltAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZhltDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def ComponentTypeAttr : ZhltAttr<"ComponentType", "component_type"> {
  let summary = "A component name, possibly including type arguments";
  let parameters = (ins
    "mlir::StringAttr": $name,
    ArrayRefParameter<"mlir::Attribute", "Type arguments">: $typeArgs
  );

  let extraClassDeclaration = [{
     // Flattens this component into a mangled name including all type arguments, if any.
     std::string getMangledName() {
        std::string str;
        llvm::raw_string_ostream os(str);
        buildMangledName(os);
        return str;
     }

     void buildMangledName(llvm::raw_ostream& os);
  }];

  let builders = [
    AttrBuilder<(ins "mlir::StringRef":$name, "llvm::ArrayRef<mlir::Attribute>":$typeArgs), [{
      return $_get(context, mlir::StringAttr::get(context, name), typeArgs);
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::StringAttr":$name, "llvm::ArrayRef<mlir::Attribute>":$typeArgs), [{
      return $_get(name.getContext(), name, typeArgs);
    }]>,
  ];
  let assemblyFormat = [{ $name `<` $typeArgs `>` }];
}

#endif
