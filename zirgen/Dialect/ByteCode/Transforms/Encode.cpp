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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "zirgen/Dialect/ByteCode/Analysis/ArmAnalysis.h"
#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Encode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h"

#define DEBUG_TYPE "gen-encoding"

using namespace mlir;

namespace zirgen::ByteCode {

#define GEN_PASS_DEF_GENENCODING
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

namespace {

template <typename T> std::string printVar(T&& var) {
  std::string out;
  llvm::raw_string_ostream os(out);
  std::visit([&](auto v) { os << "VARIANT " << llvm::getTypeName<decltype(v)>() << ": " << v; },
             var);
  return out;
}

LogicalResult notifyMatchFailure(StringRef msg) {
  LLVM_DEBUG(llvm::dbgs() << msg << "\n");
  return failure();
}

SmallVector<StringAttr> getOpNames(MLIRContext* ctx, Region* region) {
  SmallVector<StringAttr> opNames;
  for (auto& op : region->getOps()) {
    if (auto wrappedOp = llvm::dyn_cast<WrappedOp>(op)) {
      opNames.push_back(wrappedOp.getWrappedOpNameAttr());
    } else if (auto exitOp = llvm::dyn_cast<ExitOp>(op)) {
      opNames.push_back(StringAttr::get(ctx, func::ReturnOp::getOperationName()));
    } else if (!llvm::isa<LoadOp, GetArgumentOp, DecodeOp, YieldOp>(op))
      opNames.push_back(op.getName().getIdentifier());
  }
  return opNames;
}

// TODO: How do we pass the llvm hash system to the STL searcher?
struct StringAttrHasher {
  size_t operator()(StringAttr arg) const { return hash_value(arg); }
};

struct EncodeArmPattern {
  EncodeArmPattern(MLIRContext* ctx, size_t armIdx, Region& armRegion)
      : ctx(ctx)
      , armIdx(armIdx)
      , armRegion(armRegion)
      , opNames(getOpNames(ctx, &armRegion))
      , searcher(opNames.begin(), opNames.end()) {
    assert(!opNames.empty());
  }

  void applyToBlock(Block* block,
                    ArrayRef<Operation*> blockVec,
                    ArrayRef<Operation*>& skipped,
                    ArrayRef<Operation*>& remain) const {
    auto mapped = llvm::map_range(
        blockVec, [&](Operation* op) -> StringAttr { return op->getName().getIdentifier(); });
    auto pos = mapped.begin();
    while (pos != mapped.end()) {
      auto [foundStart, foundEnd] = searcher(pos, mapped.end());
      if (foundStart == foundEnd) {
        skipped = blockVec;
        return;
      }

      if (succeeded(matchAndRewriteAt(block, Block::iterator(*foundStart.getCurrent())))) {
        skipped = blockVec.slice(0, foundStart - mapped.begin());
        remain = blockVec.slice(foundEnd - mapped.begin());
        return;
      }

      pos = foundStart;
      assert(pos != mapped.end());
      ++pos;
    }
  }

  LogicalResult matchAndRewriteAt(Block* block, Block::iterator blockIt) const {

    Operation* origOp = &*blockIt;

    // Mapping from values in arm to values in block
    DenseMap<Value, Value> valMapping;
    DenseMap<Value, size_t> intArgMapping;

    llvm::SetVector<Operation*> toReplace;

    DenseSet<Value> armInputs;
    DenseSet<Value> blockResults;

    auto mapVal = [&](auto& mapping, auto armVal, auto blockVal) -> LogicalResult {
      if (mapping.contains(armVal) && mapping[armVal] != blockVal)
        return failure();
      mapping[armVal] = blockVal;
      return success();
    };

    auto processOperands = [&](auto blockOperands, auto armOperands) -> LogicalResult {
      for (auto [blockVal, armVal] : llvm::zip_equal(blockOperands, armOperands)) {
        if (armInputs.contains(armVal) && blockResults.contains(blockVal))
          return notifyMatchFailure("input generated internally");
        if (!armInputs.contains(armVal) && !blockResults.contains(blockVal)) {
          return notifyMatchFailure("unexpected import");
        }
        if (failed(mapVal(valMapping, armVal, blockVal)))
          return failure();
      }
      return success();
    };

    // First pass, map operations.
    for (Operation* armOp : llvm::make_pointer_range(armRegion.getOps())) {
      if (blockIt == block->end())
        return notifyMatchFailure("end of block");
      if (llvm::isa<LoadOp, GetArgumentOp, DecodeOp>(armOp)) {
        armInputs.insert(armOp->getResults().begin(), armOp->getResults().end());
        continue;
      }
      if (llvm::isa<YieldOp, DecodeOp>(armOp)) {
        continue;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Block op: " << *blockIt << "\n"
                     << "Arm op:   " << *armOp << "\n";
      });

      if (auto wrappedOp = llvm::dyn_cast<WrappedOp>(armOp)) {
        if (wrappedOp.getWrappedOpName() != blockIt->getName().getStringRef())
          return notifyMatchFailure("wrapped op mismatch");

        // Process non-int-args operands
        if (processOperands(blockIt->getOperands(), wrappedOp.getVals()).failed())
          return notifyMatchFailure("wrapped capture mismatch");

        // Process int args
        SmallVector<size_t> opIntArgs;
        llvm::cast<ByteCodeOpInterface>(*blockIt).getByteCodeIntArgs(opIntArgs);
        if (opIntArgs.size() != wrappedOp.getIntArgs().size())
          return notifyMatchFailure("int arg quantity mismatch");
        for (auto [intArg, armVal] : llvm::zip_equal(opIntArgs, wrappedOp.getIntArgs()))
          if (failed(mapVal(intArgMapping, armVal, intArg)))
            return notifyMatchFailure("int arg mismatch");
      } else if (llvm::isa<ExitOp>(armOp)) {
        if (func::ReturnOp::getOperationName() != blockIt->getName().getStringRef())
          return notifyMatchFailure("exit op mismatch");

        if (failed(processOperands(blockIt->getOperands(), armOp->getOperands())))
          return notifyMatchFailure("exit capture mismatch");
      } else {
        if (armOp->getName() != blockIt->getName())
          return notifyMatchFailure("op mismatch");

        if (failed(processOperands(blockIt->getOperands(), armOp->getOperands())))
          return notifyMatchFailure("regular op mismatch");
      }
      for (auto [blockVal, armVal] : llvm::zip_equal(blockIt->getResults(), armOp->getResults())) {
        if (failed(mapVal(valMapping, armVal, blockVal)))
          return notifyMatchFailure("result conflict");
        blockResults.insert(blockVal);
      }
      toReplace.insert(&*blockIt);
      ++blockIt;
    }

    // Next pass, generate the encoding
    SmallVector<BuildEncodedElement> encoded;
    encoded.push_back(armIdx);

    DenseMap</*encoded element index=*/size_t, Value> yieldValIdx;
    DenseSet<Value> yieldVals;

    for (Operation* armOp : llvm::make_pointer_range(armRegion.getOps())) {
      if (auto yieldOp = llvm::dyn_cast<YieldOp>(armOp)) {
        for (Value yieldVal : yieldOp.getVals()) {
          assert(valMapping.contains(yieldVal));
          yieldValIdx[encoded.size()] = valMapping.lookup(yieldVal);
          encoded.push_back(Type(yieldVal.getType()));
          yieldVals.insert(valMapping.lookup(yieldVal));
        }
        continue;
      }
      SmallVector<Value> valsToEncode;
      if (auto loadOp = llvm::dyn_cast<LoadOp>(armOp)) {
        valsToEncode.push_back(loadOp);
      } else if (auto decodeOp = llvm::dyn_cast<DecodeOp>(armOp)) {
        valsToEncode.push_back(decodeOp);
      }

      for (Value val : valsToEncode) {
        if (valMapping.contains(val))
          encoded.push_back(Value(valMapping[val]));
        else if (intArgMapping.contains(val))
          encoded.push_back(size_t(intArgMapping[val]));
        else
          return notifyMatchFailure("can't find val to encode");
      }
    }

    // Check for any intermediate results used in the block
    for (Operation* replaceOp : toReplace)
      for (Value result : replaceOp->getResults())
        if (!yieldVals.contains(result))
          for (Operation* user : result.getUsers())
            if (!toReplace.contains(user))
              return notifyMatchFailure("user outside arm");

    OpBuilder builder(origOp);
    auto encodedOp = builder.create<EncodedOp>(origOp->getLoc(), encoded);
    LLVM_DEBUG({
      for (auto enc : encoded)
        llvm::dbgs() << "Encoded: " << printVar(enc) << "\n";
      llvm::dbgs() << "Op: " << *encodedOp << "\n";
    });

    // Remap any uses of the results
    for (size_t idx : llvm::seq(encodedOp.size())) {
      if (std::holds_alternative<Type>(encoded[idx])) {
        assert(yieldValIdx.contains(idx));
        yieldValIdx[idx].replaceAllUsesWith(std::get<OpResult>(encodedOp.getElement(idx)));
      } else {
        assert(!yieldValIdx.contains(idx));
      }
    }

    // Remove all the operations we replaced
    for (Operation* replaceOp : toReplace)
      replaceOp->dropAllReferences();
    for (Operation* replaceOp : toReplace)
      replaceOp->erase();

    return success();
  }

  MLIRContext* ctx;
  size_t armIdx;
  Region& armRegion;
  std::string debugName;

  SmallVector<StringAttr> opNames;
  std::boyer_moore_searcher<SmallVector<StringAttr>::iterator, StringAttrHasher> searcher;
};

} // namespace

struct EncodePatterns {
  SmallVector<std::pair</*benefit=*/size_t, std::unique_ptr<EncodeArmPattern>>> patterns;
  SmallVector<ArrayRef<Operation*>> blockVecs;

  void addFromExecutor(ExecutorOp executor) {
    for (auto [idx, arm] : llvm::enumerate(executor.getArms())) {
      auto pattern = std::make_unique<EncodeArmPattern>(executor.getContext(), idx, arm);
      auto numOps = pattern->opNames.size();
      patterns.emplace_back(numOps, std::move(pattern));
    }

    // Match biggest patterns first
    llvm::sort(patterns, [&](auto& lhs, auto& rhs) { return lhs.first > rhs.first; });
  }

  void applyAllPatterns(Block* block) {
    SmallVector blockVec = llvm::to_vector(llvm::make_pointer_range(*block));
    blockVecs.push_back(blockVec);
    for (auto& [benefit, pattern] : patterns)
      applyToAllBlocks(block, *pattern);
  }

  void applyToAllBlocks(Block* block, EncodeArmPattern& pattern) {
    SmallVector<ArrayRef<Operation*>> nextPatternBlockVecs;
    while (!blockVecs.empty()) {
      auto blockVec = blockVecs.pop_back_val();
      ArrayRef<Operation*> skipped, remain;
      pattern.applyToBlock(block, blockVec, skipped, remain);
      if (!skipped.empty())
        nextPatternBlockVecs.push_back(skipped);
      if (!remain.empty())
        blockVecs.push_back(remain);
    }
    blockVecs = std::move(nextPatternBlockVecs);
  }
};
struct GenEncodingPass : public impl::GenEncodingBase<GenEncodingPass> {
  using GenEncodingBase::GenEncodingBase;

  // When copying pass, start out with no arm statistics
  GenEncodingPass(const GenEncodingPass& rhs) : armStats() {}

  struct ArmStatistic {
    ArmStatistic(GenEncodingPass* pass,
                 size_t armIdx,
                 const char* nameBase,
                 const char* description)
        : nameStorage("arm" + std::to_string(armIdx) + "-" + nameBase)
        , stat(pass, nameStorage.c_str(), description) {}

    Pass::Statistic& operator*() { return stat; }

    std::string nameStorage;
    Pass::Statistic stat;
  };

  struct ArmStatistics {
    ArmStatistics(GenEncodingPass* pass, size_t armIdx) : pass(pass), armIdx(armIdx) {}

    GenEncodingPass* pass;
    size_t armIdx;

    ArmStatistic uses{pass, armIdx, "uses", "Number of times this arm was used "};
    ArmStatistic armSize{pass, armIdx, "armSize", "Number of operations in this arm"};
    ArmStatistic opsReplaced{pass,
                             armIdx,
                             "opsReplaced",
                             "Number of total operations replaced by encoding for this arm"};
  };

  ArmStatistics& getStatisticsForArm(size_t idx) {
    auto it = armStats.find(idx);
    if (it == armStats.end()) {
      it = armStats.try_emplace(idx, std::make_unique<ArmStatistics>(this, idx)).first;
    }
    return *it->second;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    auto execOp = mod.lookupSymbol<ExecutorOp>(execSymbol);
    if (!execOp) {
      mod.emitError() << "Unable to find ExecutorOp named " << execSymbol << "\n";
      signalPassFailure();
      return;
    }

    EncodePatterns patterns;
    patterns.addFromExecutor(execOp);

    for (func::FuncOp funcOp : llvm::make_early_inc_range(mod.getOps<func::FuncOp>())) {
      assert(funcOp.getBody().hasOneBlock());
      OpBuilder builder(funcOp);

      SmallVector<Attribute> argNames;
      for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
        auto argName = funcOp.getArgAttrOfType<StringAttr>(idx, "zirgen.argName");
        if (!argName) {
          argName = builder.getStringAttr("arg" + std::to_string(idx));
        }
        argNames.push_back(argName);
      }

      auto blockOp = builder.create<EncodedBlockOp>(funcOp.getLoc(),
                                                    SymbolTable::getSymbolName(funcOp),
                                                    /*visibility=*/StringAttr(),
                                                    builder.getArrayAttr(argNames),
                                                    /*intKinds=*/ArrayAttr(),
                                                    /*tempBufs=*/ArrayAttr());
      origOps += llvm::range_size(funcOp.getBody().getOps());
      blockOp.getBody().takeBody(funcOp.getBody());
      funcOp.erase();

      Block* block = &blockOp.getBody().front();

      patterns.applyAllPatterns(block);

      // Erase arguments that we were able to eliminate
      blockOp.getBody().front().eraseArguments(0, blockOp.getBody().getNumArguments());

      for (EncodedOp op : blockOp.getBody().getOps<EncodedOp>()) {
        ++encodedOps;
        encodedVals += op.size();
      }
    }
  }

  DenseMap</* arm idx=*/size_t, std::unique_ptr<ArmStatistics>> armStats;
};

} // namespace zirgen::ByteCode
