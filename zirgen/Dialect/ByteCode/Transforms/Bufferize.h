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

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"

namespace zirgen::ByteCode {

// An interface for a bufferizer to supply buffer indexes for values.
class BufferizeInterface {
public:
  virtual std::pair</*intKind=*/mlir::Attribute, /*index=*/size_t>
  getKindAndIndex(mlir::Value value) = 0;
  virtual ~BufferizeInterface() = default;

protected:
  BufferizeInterface() = default;
  BufferizeInterface(const BufferizeInterface&) = default;
};

// A bufferize interface that assignes each value to a separate index,
// regardless of type, size, or liveness.
class NaiveBufferize : public BufferizeInterface {
public:
  std::pair</*intKind=*/mlir::Attribute, /*index=*/size_t>
  getKindAndIndex(mlir::Value value) override;

private:
  llvm::DenseMap<mlir::Value, size_t> indexes;
};

} // namespace zirgen::ByteCode
