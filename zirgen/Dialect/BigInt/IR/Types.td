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
include "zirgen/Dialect/Zll/IR/Interfaces.td"
include "zirgen/Dialect/BigInt/IR/Dialect.td"

class BigIntType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BigIntDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def BigInt : BigIntType<"BigInt", "bigint", [
   DeclareTypeInterfaceMethods<CodegenTypeInterface, ["getTypeName", "allowDuplicateTypeNames", "emitTypeDefinition"]>,
   CodegenNeedsCloneType
]> {
  let summary = "A big interger value represented as a polynomial";
  let parameters = (ins
    "size_t": $coeffs,  // Number of polynomial coefficents
    "size_t": $maxPos,  // Maximum positive coefficient value
    "size_t": $maxNeg,  // Maximum negative coefficient value
    "size_t": $minBits  // If minBits == 0, no constraint, otherwise N >= 2^(minBits - 1)
  );
  let assemblyFormat = "`<` $coeffs `,` $maxPos `,` $maxNeg `,` $minBits `>`";
  let extraClassDeclaration = [{
    size_t getMaxBits() {
      size_t extraBits = risc0::log2Ceil(getMaxPos() / (1 << kBitsPerCoeff));
      return kBitsPerCoeff * getCoeffs() + extraBits;
    }
    size_t getCarryOffset() {
      size_t coeffMagnitude = std::max(getMaxPos(), getMaxNeg());
      return (coeffMagnitude + 3*kBitsPerCoeff) / kBitsPerCoeff;
    }
    size_t getCarryBytes() {
      size_t maxCarry = getCarryOffset() * 2;
      if (maxCarry < 256) { return 1; }
      maxCarry /= 256;
      if (maxCarry < 256) { return 2; }
      return 4;
    }
    size_t getNormalWitnessSize() {
      return risc0::ceilDiv(getCoeffs(), kCoeffsPerPoly);
    }
    size_t getCarryWitnessSize() {
      return risc0::ceilDiv(getCoeffs(), kCoeffsPerPoly) * getCarryBytes();
    }
  }];
}
