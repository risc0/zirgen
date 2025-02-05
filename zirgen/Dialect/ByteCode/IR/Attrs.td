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

def TempBufAttr : ByteCodeAttr<"TempBuf", "temp_buf"> {
  let parameters = (ins
    // Name of this temporary buffer, and the intKind of values used to index into it.
    "mlir::StringAttr": $bufName,
    // Number of elements of this temporary buffer to be allocated.
    "size_t": $size
  );
  let assemblyFormat = [{ $bufName `size` $size }];
}

def TempBufArrayAttr : TypedArrayAttrBase<TempBufAttr, "Array of temporary buffers">;

def IntKindInfoAttr : ByteCodeAttr<"IntKindInfo", "int_kind_info"> {
  let summary = "Information on a set of encoded integers in a bytecode encoding";
  let parameters = (ins
    // Name of this integer kind
    "mlir::StringAttr": $intKind,
    // Number of bits used to encode
    "size_t": $encodedBits
  );
  let assemblyFormat = [{ $intKind `u` $encodedBits }];
}

def IntKindInfoArrayAttr : TypedArrayAttrBase<IntKindInfoAttr, "Array of integer kinds">;

def EncodedAttr : ByteCodeAttr<"Encoded", "encoded"> {
  let summary = "A encoded bytecode program for passing to an executor";
  let parameters = (ins
    // Actual encoded data
    StringRefParameter<>: $encoded,
    // Sizes of any temporary buffers that need to be allocated to execute this bytecode.
    ArrayRefParameter<"zirgen::ByteCode::TempBufAttr">: $tempBufs
  );
  let assemblyFormat = [{ $encoded `temps` $tempBufs }];
}

#endif // BYTECODE_ATTRS
