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

#ifndef BYTECODE_TYPES
#define BYTECODE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/ByteCode/IR/Dialect.td"

class ByteCodeType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ByteCodeDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def EncodedType : ByteCodeType<"Encoded", "encoded"> {
  let summary = "A bytecode encoding for passing to an executor";
}

#endif // BYTECODE_TYPES
