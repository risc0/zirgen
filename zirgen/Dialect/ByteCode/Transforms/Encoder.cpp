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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Executor.h"

using namespace mlir;

namespace zirgen::ByteCode {

namespace {

struct IntKindInfo {
  size_t maxVal = 0;
  size_t intCount = 0;
};

struct Encoder {
  void addInt(mlir::Attribute intKind, size_t val) {
    auto& info = intKinds[intKind];
    info.maxVal = std::max<size_t>(info.maxVal, val);
    ++info.intCount;
    encoded.emplace_back(intKind, val);
  }

  std::string encode(ExecuteOp execOp, const EncodeOptions& opts);

  llvm::MapVector<Attribute, IntKindInfo> intKinds;
  SmallVector<std::pair</* int kind=*/Attribute, /*encoded val=*/size_t>> encoded;
};

std::string Encoder::encode(ExecuteOp execOp, const EncodeOptions& opts) {
  size_t totBytes = 0;
  llvm::DenseMap<Attribute, size_t> bytesPerElem;
  for (auto execInfo : execOp.getIntKinds().getAsRange<IntKindInfoAttr>()) {
    size_t bits = execInfo.getEncodedBits();
    Attribute intKind = execInfo.getIntKind();
    assert(!(bits % 8));
    size_t bytes = bits / 8;

    if (intKinds.contains(intKind)) {
      totBytes += intKinds[intKind].intCount * bytes;
      bytesPerElem[intKind] = bytes;
    } else {
      bytesPerElem[intKind] = 0;
    }
  }

  std::string result;
  result.reserve(totBytes);
  llvm::raw_string_ostream os(result);

  for (auto& [intKind, val] : encoded) {
    if (opts.outputText) {
      os << getNameForIntKind(intKind) << " " << val << "\n";
    } else {
      size_t byteWidth = bytesPerElem[intKind];
      size_t origVal = val;
      while (byteWidth) {
        os.write(val & 0xFF);
        byteWidth--;
        val >>= 8;
      }
      if (val) {
        llvm::errs() << "Value " << origVal << " doesn't fit in " << bytesPerElem[intKind]
                     << " bytes of kind " << intKind << "\n";
        assert(false && "Value out of range for bytes width supplied");
      }
    }
  }

  if (!opts.outputText) {
    assert(result.size() == totBytes);
  }
  return result;
}

} // namespace

EncodedAttr encodeByteCode(Region* region,
                           ExecuteOp executor,
                           BufferizeInterface& bufferize,
                           const EncodeOptions& encodeOpts) {
  std::string encoded;
  llvm::raw_string_ostream os(encoded);

  size_t numArms = executor.getArms().size();
  DenseMap</*dispatch key=*/DispatchKeyAttr, /*arm number=*/size_t> armByKey;
  DenseSet</*intKind=*/Attribute> isTempBuf;

  for (auto idx : llvm::seq(numArms)) {
    DispatchKeyAttr key = executor.getArmDispatchKey(idx);

    bool didInsert = armByKey.try_emplace(key, idx).second;
    if (!didInsert) {
      executor.emitError() << "Duplicate dispatch key in byte code executor: " << key << "\n";
      return {};
    }
  }

  Encoder encoder;

  for (Operation* op : llvm::make_pointer_range(region->front())) {
    DispatchKeyAttr key = getDispatchKey(op);
    if (!armByKey.contains(key)) {
      op->emitError() << "Missing executor arm for this operation";
      return {};
    }
    size_t armIndex = armByKey[key];
    encoder.addInt(getDispatchKeyIntKind(op->getContext()), armIndex);
    auto& arm = executor.getArms()[armIndex];

    SmallVector<size_t> intArgsStorage;
    if (auto bcOp = llvm::dyn_cast<ByteCodeOpInterface>(op)) {
      bcOp.getByteCodeIntArgs(intArgsStorage);
    }
    ArrayRef<size_t> intArgs = intArgsStorage;

    for (auto operand : op->getOperands()) {
      if (llvm::isa<BlockArgument>(operand))
        // Skip function arguments that were passed in from outside
        continue;
      auto [intKind, index] = bufferize.getKindAndIndex(operand);
      encoder.addInt(intKind, index);
      isTempBuf.insert(intKind);
    }

    for (DecodeOp op : arm.front().getOps<DecodeOp>()) {
      if (intArgs.empty()) {
        op.emitError() << "getByteCodeIntArgs supplies fewer int args than expected";
        return {};
      }
      encoder.addInt(op.getIntKind(), intArgs.front());
      intArgs = intArgs.drop_front(1);
    }
    if (!intArgs.empty()) {
      op->emitError() << "getByteCodeIntArgs supplies more int args than expected";
      return {};
    }

    for (auto result : op->getResults()) {
      auto [intKind, index] = bufferize.getKindAndIndex(result);
      encoder.addInt(intKind, index);
      isTempBuf.insert(intKind);
    }
  }

  SmallVector<TempBufInfoAttr> tempBufs;
  for (auto& [intKind, info] : encoder.intKinds) {
    if (!isTempBuf.contains(intKind))
      continue;
    auto tempAttr = TempBufInfoAttr::get(executor.getContext(), intKind, info.maxVal + 1);
    tempBufs.emplace_back(tempAttr);
  }

  return EncodedAttr::get(executor.getContext(), encoder.encode(executor, encodeOpts), tempBufs);
}

} // namespace zirgen::ByteCode
