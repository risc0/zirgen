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

#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/include/llvm/ADT/StringExtras.h"

#include "zirgen/components/plonk.h"

namespace zirgen {

std::optional<std::vector<uint64_t>>
PlonkExternHandler::doExtern(llvm::StringRef name,
                             llvm::StringRef extra,
                             llvm::ArrayRef<const Zll::InterpVal*> args,
                             size_t outCount) {
  if (name == "plonkWrite") {
    assert(outCount == 0);
    plonkRows[extra.str()].emplace_back(asFpArray(args));
    return std::vector<uint64_t>{};
  }
  if (name == "plonkRead") {
    assert(!plonkRows[extra.str()].empty());
    std::vector<uint64_t> top = plonkRows[extra.str()].front();
    assert(top.size() == outCount);
    plonkRows[extra.str()].pop_front();
    return top;
  }

  if (name == "plonkWriteAccum") {
    assert(outCount == 0);
    plonkAccumRows[extra.str()].emplace_back(asFpArray(args));
    return std::vector<uint64_t>{};
  }
  if (name == "plonkReadAccum") {
    assert(!plonkAccumRows[extra.str()].empty());
    std::vector<uint64_t> top = plonkAccumRows[extra.str()].front();
    assert(top.size() == outCount);
    plonkAccumRows[extra.str()].pop_front();
    return top;
  }
  return ExternHandler::doExtern(name, extra, args, outCount);
}

void PlonkExternHandler::sort(llvm::StringRef name) {
  FpMat& mat = plonkRows.at(name.str());
  std::sort(mat.begin(), mat.end());
}

void PlonkExternHandler::calcPrefixProducts(Zll::ExtensionField f) {
  for (auto& kv : plonkAccumRows) {
    FpMat& mat = kv.second;

    auto accum = f.One();

    std::vector<uint64_t> newRow;
    for (auto& row : mat) {
      assert((row.size() % f.degree) == 0);
      newRow.clear();
      for (size_t i = 0; i != row.size(); i += f.degree) {
        auto val = llvm::ArrayRef(row).slice(i, f.degree);
        accum = f.Mul(accum, val);
        newRow.insert(newRow.end(), accum.begin(), accum.end());
      }
      assert(newRow.size() == row.size());
      std::swap(newRow, row);
    }
  }
}

} // namespace zirgen
