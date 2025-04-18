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
include "zirgen/Dialect/IOP/IR/Dialect.td"
include "zirgen/Dialect/Zll/IR/Types.td"

class IopType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IopDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IOP : IopType<"IOP", "iop", [CodegenTypeInterface]> {
  let summary = "An IOP read stream for the verifier";
}

def AnyScalar : AnyTypeOf<[Val, Digest]>;
