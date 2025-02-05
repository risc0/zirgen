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

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"

namespace zirgen::ByteCode {

#if 0
/// Encodes the given region as bytecodes suitable for executing with the given executor.
/// Requirements:
///    * The region must contain a single block.
///    * It must be possible to represent all operations in the region using
///      the bytecodes defined in the executor.
///
/// In the future, we may supply a IRMapper or similar to this function
/// to allow it to encode a region using an executor built on a
/// different region.
EncodedAttr encodeByteCode(mlir::Region* region,
                           ExecutorOp executor,
                           BufferizeInterface& bufferize,
                           const EncodeOptions& encodeOpts = EncodeOptions());
#endif

// Build an executor supporting the operations present in the given region.
// Any operations without a ByteCodeOpInterface are ignored.
//
// The supplied `encodedInput` value can be used to supply an encoded byte code generated
// by `encodeByteCode`.
//
// `bufferize` is used to gather the integer types and width of temporary value indexes.
ExecutorOp buildExecutor(mlir::Location loc,
                         mlir::Region* region,
                         mlir::Value encodedInput,
                         BufferizeInterface& bufferize);

} // namespace zirgen::ByteCode
