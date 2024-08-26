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
include "zirgen/Dialect/Zll/IR/Dialect.td"

class ZllAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZllDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Tap : ZllAttr<"Tap", "tap"> {
  let summary = "A Tap, which specifies a register and a back.";
  let parameters = (ins
      "uint32_t" : $regGroupId,
      "uint32_t" : $offset,
      "uint32_t" : $back);
  let assemblyFormat = "`<` $regGroupId `,` $offset `,` $back `>`";
}

def TapReg : ZllAttr<"TapReg", "tapReg"> {
  let summary = "A tap register, which has one or more backs";
  let parameters = (ins
      "uint32_t" : $regGroupId,
      "uint32_t" : $offset,
      ArrayRefParameter<"uint32_t", "backs">: $backs,
      "uint32_t" : $comboId);
  let assemblyFormat = "`<` $regGroupId `,` $offset `,` `[` $backs `]` `,` $comboId `>`";
}

def PolynomialAttr : DenseArrayAttrBase<"PolynomialAttr", "uint64_t", "i64"> {
  let summary = "A compile-time constant element of an extension field";
  let constBuilderCall = "mlir::PolynomialAttr::get($_builder.getContext(), $0)";
}

def Field : ZllAttr<"Field", "field"> {
   let summary = "A description of a finite galois field configuration including a field, extension field, and irriducable polynomial.";
   let parameters = (ins
      StringRefParameter<"Identifying name of this field"> : $name,
      AttrParameter<"uint64_t", "Core prime to this field">  : $prime,
      AttrParameter<"size_t", "Number of elements in extension field"> : $extDegree,
      ArrayRefParameter<"uint64_t", "Irreducible polynomial of the field"> : $polynomial);
  let assemblyFormat = [{ $name `,` $prime `,` $extDegree `,` $polynomial }];
  let extraClassDeclaration = [{
      // Returns the field arithmetic implementation for the base
      // field, operating on individual field elements (u64).
      Field getBaseField() const;

      // Returns the field arithmetic implementation for the base
      // field, operating on arrays of 1 field elements (ArrayRef<u64> with size=1).
      ExtensionField getBaseExtensionField() const;

      // Returns the field arithmetic implementation for the extension
      // field, operating on arrays of extSize field elements (ArrayRef<u64>)
      ExtensionField getExtExtensionField() const;
  }];
  let genVerifyDecl = 1;
}

def FieldExt : ZllAttr<"FieldExt", "fieldExt"> {
  let summary = "Marker to signify a field extension as opposed to the base field.";
  let assemblyFormat = [{ `ext` }];
}

#endif // ZIRGEN_IR_ATTRS
