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

#include <map>
#include "llvm/include/llvm/Support/raw_ostream.h"

#include "zirgen/Dialect/BigInt/Bytecode/bibc.h"

namespace zirgen::BigInt::Bytecode {

void Program::clear() {
  types.clear();
  inputs.clear();
  constants.clear();
  ops.clear();
}

void Program::dumpstats() {
  llvm::errs() << "  types.size() == " << types.size() << "\n";
  llvm::errs() << "  inputs.size() == " << inputs.size() << "\n";
  llvm::errs() << "  constants.size() == " << constants.size() << "\n";
  llvm::errs() << "  ops.size() == " << ops.size() << "\n";
}

bool operator<(const Type& l, const Type& r) {
  if (l.coeffs >= r.coeffs) return false;
  if (l.maxPos >= r.maxPos) return false;
  if (l.maxNeg >= r.maxNeg) return false;
  if (l.minBits >= r.minBits) return false;
  return true;
}

} // namespace zirgen::BigInt::Bytecode
