// Copyright 2025 RISC Zero, Inc.
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

#ifndef BYTECODE_OPS
#define BYTECODE_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "zirgen/Dialect/ByteCode/IR/Attrs.td"
include "zirgen/Dialect/ByteCode/IR/Dialect.td"
include "zirgen/Dialect/ByteCode/IR/Types.td"
include "zirgen/Dialect/ByteCode/Interfaces/Interfaces.td"

class ByteCodeOp<string mnemonic, list<Trait> traits = []>
    : Op<ByteCodeDialect, mnemonic, traits> {
}

def ExecutorOp : ByteCodeOp<"executor", [
    SingleBlock, HasParent<"mlir::ModuleOp">, Symbol, OpAsmOpInterface, IsolatedFromAbove]> {
  let summary = "ExecutorOp represents an executor that executes bytecode that's specifically compiled for it.";
  let description = [{
     Each region contains zero or more operations which execute the given bytecodes.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<FunctionType>:$function_type,
    StrArrayAttr:$argNames,
    // How many bits each integer is encoded in.  Unspecified integer kinds are encoded in 32 bits.
    OptionalAttr<IntKindInfoArrayAttr>: $intKinds
  );
  
  let regions = (region VariadicRegion<SizedRegion<1>>: $arms);
  let assemblyFormat = "$sym_name $argNames $function_type attr-dict-with-keyword $arms";

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "zbytecode"; }
  }];
}

def YieldOp : ByteCodeOp<"yield", [Terminator, HasParent<"ExecutorOp">]> {
  let summary = "Yield the given values to be saved from this arm of the ExecutorOp.";
  let arguments  = (ins
      Variadic<AnyType>: $vals);
  let assemblyFormat = "operands `:` type(operands) attr-dict";
}

def ExitOp : ByteCodeOp<"exit", [Terminator, ParentOneOf<["ExecutorOp", "EncodedBlockOp"]>]> {
  let summary = "Return the given values from the executor.";
  let arguments = (ins Variadic<AnyType>: $vals);
  let assemblyFormat = "operands `:` type(operands) attr-dict";
  let hasVerifier = 1;
}

def LoadOp : ByteCodeOp<"load", [HasParent<"ExecutorOp">]> {
  let results = (outs
      AnyType: $out);
  let assemblyFormat = "type($out) attr-dict";
}

def DecodeOp : ByteCodeOp<"decode", [HasParent<"ExecutorOp">]> {
  let results = (outs
      Index: $out);
  let assemblyFormat = "attr-dict";
}

def GetArgumentOp : ByteCodeOp<"get_argument", [HasParent<"ExecutorOp">]> {
  let summary = "Get a named argument";
  let arguments = (ins StrAttr:$argName);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$argName `:` type(results) attr-dict";
}

def WrappedOp : ByteCodeOp<"wrapped", [AttrSizedOperandSegments, HasParent<"ExecutorOp">]> {
  let summary = "A genericized operation that decodes one or more int args (from getByteCodeIntArgs) from bytecode.";
  let arguments = (ins
      StrAttr: $wrappedOpName,
      Variadic<Index>: $intArgs,
      Variadic<AnyType>: $vals
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$wrappedOpName `(` $intArgs `:` type($intArgs) `)` `,`( `(` $vals^ `:` type($vals)`)`)? `->` type(results) attr-dict";
}

def EncodedBlockOp : ByteCodeOp<"encoded_block", [
    SingleBlock, NoTerminator, HasParent<"mlir::ModuleOp">, Symbol, OpAsmOpInterface, IsolatedFromAbove]> {
  let summary = "A (potentially) partially encoded byte code block";
  let description = [{
     Each region contains zero or more operations which execute the given bytecodes.
  }];
  let arguments = (ins
     SymbolNameAttr:$sym_name,
     OptionalAttr<StrAttr>:$sym_visibility,
     StrArrayAttr: $argNames,
     OptionalAttr<IntKindInfoArrayAttr>: $intKinds,
     OptionalAttr<TempBufArrayAttr>: $tempBufs
  );
  let regions = (region SizedRegion<1>: $body);
  let assemblyFormat = [{ $sym_name $argNames attr-dict-with-keyword $body }];
  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "zbytecode"; }
  }];
}

def EncodedOp : ByteCodeOp<"encoded", [HasParent<"EncodedBlockOp">]> {
  let summary = "A (potentially partially) encoded byte code operation";
  let description = [{
This represents a partially encoded byte code operation.  For each encoded value, it may either
be an integer or a mlir::Value.  In the case of the latter, it still needs to be resolved to a temporary
buffer offset.
  }];
  let arguments = (ins
      // Internal storage is that negative encoded elements are indexes into `values` and `results`.
      // For instance, if `values` has two values, encoded elements correspond as follows:
      // -1: values[0]
      // -2: values[1]
      // -3: results[0]
      DenseI64ArrayAttr:$encoded,
      Variadic<AnyType>:$values);
  let results = (outs Variadic<AnyType>:$results);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
      size_t size() { return getEncoded().size(); }
      EncodedElement getElement(size_t idx);
  }];
  let builders = [
     // Build from a set of partially encoded elements.  Any OpResults present
     // will be used to determine the type of the result, but the values are ignored.
     OpBuilder<(ins "llvm::ArrayRef<zirgen::ByteCode::BuildEncodedElement>":$encoded)>
  ];
}

#endif // BYTECODE_OPS
