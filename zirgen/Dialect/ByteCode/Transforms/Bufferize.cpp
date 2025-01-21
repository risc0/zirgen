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

#include "zirgen/Dialect/ByteCode/Transforms/Bufferize.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace zirgen::ByteCode {

std::pair</*intKind=*/mlir::Attribute, /*index=*/size_t>
NaiveBufferize::getKindAndIndex(mlir::Value value) {
  auto intNameAttr = StringAttr::get(value.getContext(), "naive_buf");
  if (indexes.contains(value))
    return std::make_pair(intNameAttr, indexes[value]);
  size_t newIndex = indexes.size();
  indexes[value] = newIndex;
  return std::make_pair(intNameAttr, newIndex);
};

} // namespace zirgen::ByteCode
