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

#ifndef BYTECODE_ATTRS
#define BYTECODE_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/ByteCode/IR/Dialect.td"

class ByteCodeAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ByteCodeDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def TempBufInfoAttr : ByteCodeAttr<"TempBufInfo", "temp_buf"> {
  let parameters = (ins
    // Arbitrary attribute used as a uniquing key for integer kinds.
    "mlir::Attribute": $intKind,
    // Number of bits used to encode this kind of int
    "size_t": $size
  );
  let assemblyFormat = [{ $intKind `size` $size }];
}

def IntKindInfoAttr : ByteCodeAttr<"IntKindInfo", "int_kind_info"> {
  let summary = "Information on a set of encoded integers in a bytecode encoding";
  let parameters = (ins
    // Arbitrary attribute used as a uniquing key for integer kinds.
    "mlir::Attribute": $intKind,
    // Number of bits used to encode this kind of int
    "size_t": $encodedBits
  );
  let assemblyFormat = [{ $intKind `u` $encodedBits }];
}

def IntKindInfoArrayAttr : TypedArrayAttrBase<IntKindInfoAttr, "Array of integer kinds">;

def EncodedAttr : ByteCodeAttr<"Encoded", "encoded"> {
  let summary = "A bytecode encoding for passing to an executor";
  let parameters = (ins
    StringRefParameter<>: $encoded,
    // Sizes of any temporary buffers that need to be allocated to execute this bytecode.
    ArrayRefParameter<"zirgen::ByteCode::TempBufInfoAttr">: $tempBufs
  );
  let assemblyFormat = [{ $encoded `temps` $tempBufs }];
}

def DispatchKeyAttr : ByteCodeAttr<"DispatchKey", "dispatch_key"> {
  let summary = "Unique identifier for a byte code operation on concrete types";
  let parameters = (ins
     StringRefParameter<>:$operationName,
     ArrayRefParameter<"mlir::Type">:$operandTypes,
     ArrayRefParameter<"mlir::Type">:$resultTypes,
     // Int kinds of any additional decoded arguments, not including value operands.
     ArrayRefParameter<"mlir::Attribute">:$intKinds,
     // Further unique by any captured block arguments from containing
     // block in case they happen to be the same type.
     ArrayRefParameter<"size_t">:$blockArgNums
  );
  let assemblyFormat = [{
     $operationName `(` $intKinds `,` $operandTypes `,` $blockArgNums `)`
     `->` $resultTypes
  }];
}

#endif // BYTECODE_ATTRS
