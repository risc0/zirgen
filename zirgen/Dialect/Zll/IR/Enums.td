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

include "mlir/IR/EnumAttr.td"

def BufferKind : I32EnumAttr<"BufferKind", "Buffer Type", [
     I32EnumAttrCase<"Constant", 0, "constant">,
     I32EnumAttrCase<"Mutable", 1, "mutable">,
     I32EnumAttrCase<"Global", 2, "global">,
     I32EnumAttrCase<"Temporary", 3, "temporary">, // per cycle temporary that doesn't need to be saved
]> {
  let cppNamespace = "::zirgen::Zll";
}

def DigestKind : I32EnumAttr<"DigestKind", "Digest Type", [
     I32EnumAttrCase<"Default", 0, "default">,
     I32EnumAttrCase<"Sha256", 1, "sha256">,
     I32EnumAttrCase<"Poseidon2", 2, "poseidon2">,
]> {
  let cppNamespace = "::zirgen::Zll";
}

