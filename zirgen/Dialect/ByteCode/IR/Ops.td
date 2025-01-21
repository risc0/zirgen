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

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "zirgen/Dialect/ByteCode/Interfaces/Interfaces.td"
include "zirgen/Dialect/ByteCode/IR/Types.td"
include "zirgen/Dialect/ByteCode/IR/Attrs.td"
include "zirgen/Dialect/ByteCode/IR/Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class ByteCodeOp<string mnemonic, list<Trait> traits = []>
    : Op<ByteCodeDialect, mnemonic, traits> {
}

def YieldOp : ByteCodeOp<"yield", [Terminator, HasParent<"ExecuteOp">]> {
  let summary = "Yield the given values to be saved from this arm of the ExecuteOp.";
  let arguments  = (ins
      DispatchKeyAttr:$dispatchKey,
      Variadic<AnyType>: $vals,
      // Integer kinds for return value buffers
      ArrayAttr: $intKinds);
  let assemblyFormat = "operands `:` type(operands) `:` $dispatchKey attr-dict";
}

def ExitOp : ByteCodeOp<"exit", [Terminator, HasParent<"ExecuteOp">]> {
  let summary = "Return the given values from the executor.";
  let arguments = (ins DispatchKeyAttr:$dispatchKey, Variadic<AnyType>: $vals);
  let assemblyFormat = "operands `:` type(operands) `:` $dispatchKey attr-dict";
}

def ExecuteOp : ByteCodeOp<"execute", [SingleBlock]> {
  let summary = "ExecuteOp represents an executor that executes bytecode that's specifically compiled for it.";
  let description = [{
     Each region contains zero or more operations which execute the given bytecodes.
  }];
  let arguments = (ins
    EncodedType: $encoded,
    IntKindInfoArrayAttr: $intKinds
  );
  let regions = (region VariadicRegion<SizedRegion<1>>: $arms);
  let results = (outs Variadic<AnyType>: $outs);
  let extraClassDeclaration = [{
     /// Returns the DispatchKey associated with the given arm index
     DispatchKeyAttr getArmDispatchKey(size_t idx);
  }];

  let assemblyFormat = "$encoded attr-dict-with-keyword $arms `->` type(results)";
}

def DefineEncodedOp : ByteCodeOp<"define", [IsolatedFromAbove, Symbol]> {
  let arguments = (ins SymbolNameAttr: $sym_name, EncodedAttr: $encoded);
  let assemblyFormat = "$sym_name `=` $encoded attr-dict";
}

def GetEncodedOp : ByteCodeOp<"get"> {
  let arguments = (ins FlatSymbolRefAttr: $target);
  let results = (outs EncodedType: $result);
  let assemblyFormat = "$target attr-dict";
}

def LoadTemporaryOp : ByteCodeOp<"load", [HasParent<"ExecuteOp">]> {
  let arguments = (ins
      AnyAttr: $intKind);
  let results = (outs
      AnyType: $out);
  let assemblyFormat = "$intKind `->` type($out) attr-dict";
}

def DecodeOp : ByteCodeOp<"decode", [HasParent<"ExecuteOp">]> {
  let arguments = (ins
      AnyAttr: $intKind);
  let results = (outs
      Index: $out);
  let assemblyFormat = "$intKind attr-dict";
}

def OperationOp : ByteCodeOp<"operation"> {
  let summary = "A genericized operation that decodes one or more attribute arguments from bytecode.";
  let arguments = (ins
      StrAttr: $wrappedOpName,
      Variadic<AnyType>: $operands
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$wrappedOpName operands `:` functional-type(operands, results) attr-dict";
}

def TestOp : ByteCodeOp<"test", [
   DeclareOpInterfaceMethods<ByteCodeOpInterface, ["getByteCodeIntArgs"]>
   ]> {
  let summary = "A dummy operation to use for testing";
  let arguments = (ins
      DenseI64ArrayAttr: $intArgs,
      Variadic<NoneType>: $operands
  );
  let results = (outs
      Variadic<NoneType>: $results
  );

  let assemblyFormat = "$intArgs operands `:` functional-type(operands, results) attr-dict";
}
