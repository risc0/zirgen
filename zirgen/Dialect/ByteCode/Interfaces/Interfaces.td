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

#ifndef BYTECODE_INTERFACES
#define BYTECODE_INTERFACES

def ByteCodeOpInterface : OpInterface<"ByteCodeOpInterface"> {
  let cppNamespace = "::zirgen::ByteCode";
  let methods = [
    InterfaceMethod<[{
         Return arguments needed at runtime to execute this operation.
         All arguments must be unsigned integers that fit into a size_t,
         and every instance of the operation must encode into the same
         number of integers.
      }],
      "void", "getByteCodeIntArgs",
      (ins "llvm::SmallVectorImpl<size_t>&": $intArgs)>
  ];
}

def BitWidthOpInterface : OpInterface<"BitWidthOpInterface"> {
  let cppNamespace = "::zirgen::ByteCode";
  let methods = [
    InterfaceMethod<[{
        Counts how many encoded elements this operation uses.
    }],
    "size_t", "getNumEncodedElements", (ins), "",
    [{ return 1; }]>,
    InterfaceMethod<[{
        Set the bit widths of the encoded elements for this operation
    }], 
    "void", "setEncodedElementsBits", (ins "llvm::ArrayRef<size_t>":$bits), "",
    [{
       assert(bits.size() == 1 && "Must override setEncodedElementsBits if we have more than one element");
       mlir::Builder builder($_op->getContext());
       $_op.setBitsAttr(builder.getIndexAttr(bits[0]));
    }]>
  ];
}

#endif /* BYTECODE_INTERFACES */
