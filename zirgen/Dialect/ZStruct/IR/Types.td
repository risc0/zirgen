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
include "zirgen/Dialect/Zll/IR/Types.td"
include "zirgen/Dialect/ZStruct/IR/Dialect.td"
include "zirgen/Dialect/ZStruct/IR/Enums.td"
include "zirgen/Dialect/ZStruct/IR/Interfaces.td"

class ZStructType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZStructDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def StructType : ZStructType<"Struct", "struct", [
    DeclareTypeInterfaceMethods<CodegenTypeInterface, ["emitTypeDefinition", "getTypeName", "emitLiteral"]>,
    CodegenNeedsCloneType]> {
  let summary = "struct type, representing a component";
  let parameters = (ins
      StringRefParameter<>: $id,
      ArrayRefParameter<"ZStruct::FieldInfo", "struct fields">: $fields
  );
  let hasCustomAssemblyFormat = 1;
}

def LayoutType : ZStructType<"Layout", "layout", [
    DeclareTypeInterfaceMethods<CodegenTypeInterface, ["emitTypeDefinition", "getTypeName", "emitLiteral"]>,
    CodegenLayoutType
  ]> {
  let summary = "Layout type, representing registers for a component to write to.";
  let parameters = (ins
      StringRefParameter<>: $id,
      ArrayRefParameter<"ZStruct::FieldInfo", "struct fields">: $fields,
      // "mux" is a hint to the layout generator that the common "@super" for all fields should point
      // at the same allocated registers.
      //
      // TODO: consider moving mux types into a separate type other than LayoutType?
      // But in this case we'd want a StructLikeType interface or something that lets us
      // retrieve fields without using a TypeSwitch.
      "LayoutKind": $kind
  );
  let builders = [
    TypeBuilder<(ins
      CArg<"::llvm::StringRef">: $id,
      CArg<"::llvm::ArrayRef<ZStruct::FieldInfo>">: $fields), [{
       return $_get($_ctxt, id, fields, LayoutKind::Normal);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def UnionType : ZStructType<"Union", "union"> {
  let summary = "union type, representing a mux";
  let parameters = (ins
      StringRefParameter<>: $id,
      ArrayRefParameter<"ZStruct::FieldInfo", "union fields">: $fields
  );
  let hasCustomAssemblyFormat = 1;
}

def ArrayType : ZStructType<"Array", "array", [
    ArrayLikeTypeInterface,
    DeclareTypeInterfaceMethods<CodegenTypeInterface, ["emitTypeDefinition", "getTypeName", "emitLiteral"]>,
    CodegenNeedsCloneType]> {
  let summary = "array type, a series of same-typed values";
  let parameters = (ins
      "mlir::Type": $element,
      "unsigned": $size
  );
  let assemblyFormat = "`<`$element `,` $size `>`";
  let extraClassDeclaration = [{
    mlir::Value materialize(mlir::Location loc, llvm::ArrayRef<mlir::Value> elements, mlir::OpBuilder& builder);
  }];
  let genVerifyDecl = 1;
}

def LayoutArrayType : ZStructType<"LayoutArray", "layout_array", [
    ArrayLikeTypeInterface,
    DeclareTypeInterfaceMethods<CodegenTypeInterface, ["emitTypeDefinition", "getTypeName", "emitLiteral"]>,
    CodegenOnlyPassByReferenceType,
    CodegenLayoutType]> {
  let summary = "array type, a series of same-typed layouts";
  let parameters = (ins
      "mlir::Type": $element,
      "unsigned": $size
  );
  let assemblyFormat = "`<`$element `,` $size `>`";
  let genVerifyDecl = 1;
}

def Ref : ZStructType<"Ref", "ref", [
    DeclareTypeInterfaceMethods<CodegenTypeInterface, ["getTypeName"]>,
    CodegenLayoutType
  ]> {
  let summary = "reference to a register";
  let parameters = (ins
     DefaultValuedParameter<"Zll::ValType", "Zll::ValType::get($_ctxt, Zll::kFieldPrimeDefault, 1)">: $element
  );
  let assemblyFormat = "(`<` $element^ `>`)?";
  let builders = [
    TypeBuilderWithInferredContext<(ins "Zll::ValType":$element), [{
      return $_get(element.getContext(), element);
    }]>
  ];
}

// TODO: Disambiguate this and the other type named "Tap" in IR/Attrs.td
def ZTap : ZStructType<"Tap", "tap", [DeclareTypeInterfaceMethods<CodegenTypeInterface, ["emitLiteral"]>]> {
  let summary = "reference to a tap";
  let assemblyFormat = "";
}

def ZirType : AnyTypeOf<[UnionType, StructType, LayoutType, ArrayType, LayoutArrayType,
                         Val, Ref, ZTap, String, VariadicPack],
                         "A type that can participate in zirgen structures">;

def IsLayoutType : AnyTypeOf<[LayoutType, Ref, LayoutArrayType],
                             "A type specifying a layout for organizing registers">;
