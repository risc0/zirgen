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

#include "zirgen/Dialect/ByteCode/Analysis/ArmAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "llvm/Support/Debug.h"

// Analyze operations to find sequences of operations that occur more than once,
// and count how many times they're seen.
//
// Terminology:
//
// * Suffix: This is a slice of operations in a block, starting at an
//   arbitrary position and continuing to the end.
//
// * Run: This is a slice of a Suffix, starting at the beginning.
//
// * Arm: This is a set of runs that have an equivalent ArmInfo and
//   can be encoded as the same byte code.
//
// We find similar sequences by calculating ArmInfo for all suffixes,
// and then iteratively increasing the length of the run slices until
// runs have fewer than kMinArmUseCount runs in common for an arm.
//
// Also, no run in an arm may start with an operation which is contained
// in any other run.

#define DEBUG_TYPE "arm-analysis"

using namespace mlir;

namespace zirgen::ByteCode {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ArmInfo& armInfo) {
  os << "[arm: " << armInfo.getOps().size() << " ops, " << armInfo.getCount() << " uses, "
     << armInfo.numLoadVals << " in, " << armInfo.numYieldVals << " out:\n";
  for (auto op : armInfo.getOps()) {
    os << "  " << *op << "\n";
  }
  os << "offsets=";
  llvm::interleaveComma(armInfo.valueOffsets, os);
  os << "]\n";
  return os;
}

namespace {

constexpr size_t kMinArmUseCount = 2;

template <typename R> auto hash_range(const R& range) {
  return llvm::hash_combine_range(std::begin(range), std::end(range));
}

template <typename OuterC, typename InnerC> void mergeReturnValues(OuterC& outer, InnerC&& inner) {
  auto innerIt = inner.begin();
  auto outerIt = outer.begin();

  while (innerIt != inner.end() && outerIt != outer.end()) {
    if (*outerIt == *innerIt) {
      ++innerIt;
      ++outerIt;
    } else
      ++outerIt;
  }
  llvm::append_range(outer, llvm::make_range(innerIt, inner.end()));
}

StringAttr getBlockArgName(BlockArgument arg) {
  auto funcOp = llvm::cast<func::FuncOp>(arg.getOwner()->getParentOp());
  auto argName = funcOp.getArgAttrOfType<StringAttr>(arg.getArgNumber(), "zirgen.argName");
  if (!argName) {
    argName = StringAttr::get(arg.getContext(), "arg" + std::to_string(arg.getArgNumber()));
  }
  return argName;
}

// Uniquing map info for ArmInfo structure to compare if two different
// sets of operations can be assigned to the same arm.
struct UniqueArmMapInfo {
  static inline ArmInfo getEmptyKey() { return ArmInfo{.valueOffsets = {~0ul}}; }
  static inline ArmInfo getTombstoneKey() { return ArmInfo{.valueOffsets = {~0ul - 1}}; }
  static unsigned getHashValue(const ArmInfo& a) {
    return llvm::hash_combine(
        // Operation names
        hash_range(llvm::map_range(a.getOps(), [](Operation* op) { return op->getName(); })),
        // Types of all values consumed and produced
        ValueRange(a.values).getTypes(),
        // Names of captured function arguments
        hash_range(a.funcArgNames),
        // How the operations inside this set are connected
        hash_range(a.valueOffsets));
  }

  static bool isEqual(const ArmInfo& lhs, const ArmInfo& rhs) {
    if (lhs.valueOffsets != rhs.valueOffsets || lhs.opIntArgs != rhs.opIntArgs ||
        lhs.funcArgNames != rhs.funcArgNames || lhs.numLoadVals != rhs.numLoadVals ||
        lhs.numYieldVals != rhs.numYieldVals || lhs.valueOffsets != rhs.valueOffsets)
      return false;

    if (ValueRange(lhs.values).getTypes() != ValueRange(rhs.values).getTypes())
      return false;

    if (!llvm::equal(lhs.getOps(), rhs.getOps(), [](Operation* lhsOp, Operation* rhsOp) {
          return lhsOp->getName() == rhsOp->getName() &&
                 lhsOp->getNumOperands() == rhsOp->getNumOperands() &&
                 lhsOp->getNumResults() == rhsOp->getNumResults();
        }))
      return false;

    return true;
  }
};

// CandidateTrace incrementally constructs a Candidate from a Suffix while calculating the
// associated ArmInfo.
struct CandidateTrace {
  CandidateTrace(ArrayRef<Operation*> ops) : ops(ops) {}

  // Lengthens the Candidate by one, adding the newly included operation to the given poison set.
  LogicalResult advance(DenseSet<Operation*>& poison) {
    if (pos >= ops.size())
      return failure();

    Operation* op = ops[pos++];
    if (locs.empty())
      locs.insert(op->getLoc());
    key.allOps = {ops.slice(0, pos)};
    if (pos > 1)
      poison.insert(op);

    if (auto byteCodeOp = llvm::dyn_cast<ByteCodeOpInterface>(op)) {
      SmallVector<size_t> intArgs;
      byteCodeOp.getByteCodeIntArgs(intArgs);
      key.opIntArgs.push_back(intArgs.size());
    } else {
      key.opIntArgs.push_back(0);
    }

    // Update any temp vals or block args
    for (Value val : op->getOperands()) {
      markValueUsed(op, val);

      if (auto blockArg = dyn_cast<BlockArgument>(val)) {
        if (blockArgIdx.contains(val))
          continue;
        blockArgIdx[val] = blockArgIdx.size();
        key.values.insert(key.values.begin() + key.funcArgNames.size(), val);
        key.funcArgNames.push_back(getBlockArgName(blockArg));
        assert(blockArgIdx.size() == key.funcArgNames.size());
      } else {
        if (resultIdx.contains(val) || armOperandIdx.contains(val))
          continue;
        armOperandIdx[val] = key.numLoadVals;
        key.values.insert(key.values.begin() + key.funcArgNames.size() + key.numLoadVals, val);
        ++key.numLoadVals;
        assert(key.numLoadVals == armOperandIdx.size());
      }
    }

    // Save any results from this operation
    for (Value val : op->getResults()) {
      resultIdx[val] = resultIdx.size();
      key.values.push_back(val);

      if (!val.use_empty())
        usedValuesRemaining[val].insert(val.getUsers().begin(), val.getUsers().end());
    }

    // Generate our offsets for operations
    key.valueOffsets.clear();
    for (Operation* prevOp : key.getOps()) {
      for (Value val : prevOp->getOperands()) {
        if (isa<BlockArgument>(val)) {
          assert(blockArgIdx.contains(val));
          key.valueOffsets.push_back(blockArgIdx[val]);
        } else if (armOperandIdx.contains(val)) {
          key.valueOffsets.push_back(armOperandIdx[val] + blockArgIdx.size());
        } else {
          assert(resultIdx.contains(val));
          key.valueOffsets.push_back(resultIdx[val] + blockArgIdx.size() + key.numLoadVals);
        }
      }
    }

    // Add offsets for yields
    key.numYieldVals = usedValuesRemaining.size();
    for (Value val : llvm::make_first_range(usedValuesRemaining)) {
      assert(resultIdx.contains(val));
      key.valueOffsets.push_back(resultIdx[val] + blockArgIdx.size() + key.numLoadVals);
    }

    return success();
  }

  const ArmInfo& getKey() const { return key; }
  size_t getNumOps() const { return pos; }

  // Returns true if the first operation in this Candidate is poisoned.
  bool isPoisoned(DenseSet<Operation*>& poison) const {
    if (ops.empty())
      return false;
    bool res = poison.contains(ops[0]);
    if (res) {
      LLVM_DEBUG(llvm::dbgs() << ops[0] << " is poisoned\n");
    }
    return res;
  }

  void markValueUsed(Operation* user, Value val) {
    auto it = usedValuesRemaining.find(val);
    if (it == usedValuesRemaining.end())
      return;

    it->second.erase(user);
    if (it->second.empty()) {
      usedValuesRemaining.erase(it);
    }
  }

  ArrayRef<Operation*> ops;
  size_t pos = 0;
  ArmInfo key;

  SetVector<Location> locs;

  DenseMap<Value, size_t> resultIdx;
  DenseMap<Value, size_t> armOperandIdx;
  DenseMap<Value, size_t> blockArgIdx;

  llvm::MapVector<Value, DenseSet<Operation*>> usedValuesRemaining;
};

struct QueueEntry {
  QueueEntry(SmallVector<std::unique_ptr<CandidateTrace>> runs) : runs(std::move(runs)) {}
  QueueEntry() = default;
  QueueEntry(QueueEntry&&) = default;
  QueueEntry(const QueueEntry&) = delete;
  QueueEntry& operator=(const QueueEntry&) = delete;
  QueueEntry& operator=(QueueEntry&&) = default;

  bool operator<(const QueueEntry& rhs) const {
    // Returns true if this is lower priority than rhs
    return runs.size() < rhs.runs.size();
  }
  ArmInfo getArmInfo() const {
    assert(!runs.empty());
    ArmInfo armInfo = runs.front()->key;
    for (auto& run : llvm::drop_begin(runs)) {
      llvm::append_range(armInfo.allOps, run->key.allOps);
    }
    return armInfo;
  }
  SmallVector<std::unique_ptr<CandidateTrace>> runs;
};

} // namespace

struct ArmAnalysisImpl {
  ArmAnalysisImpl(MLIRContext* ctx, std::vector<std::vector<Operation*>>& blockOpStorage)
      : ctx(ctx), blockOpStorage(blockOpStorage) {}
  void addBlock(Block* block);
  void initQueue();

  void analyzeOne();
  void saveTo(ArmAnalysis& a);

  MLIRContext* ctx;
  std::vector<std::vector<Operation*>>& blockOpStorage;

  std::vector<QueueEntry> searchQueue;
  llvm::SetVector<ArmInfo, std::vector<ArmInfo>, DenseSet<ArmInfo, UniqueArmMapInfo>> singleOps;
  std::vector<ArmInfo> multiOps;

  llvm::MapVector<StringAttr, Type> args;
  llvm::SmallVector<Type> resultTypes;
};

void ArmAnalysisImpl::addBlock(Block* block) {
  auto r = llvm::make_pointer_range(*block);
  blockOpStorage.emplace_back(r.begin(), r.end());

  for (Operation* op : blockOpStorage.back())
    if (llvm::isa<func::ReturnOp>(op))
      mergeReturnValues(resultTypes, op->getOperandTypes());

  if (auto funcOp = llvm::dyn_cast<func::FuncOp>(block->getParentOp())) {
    for (BlockArgument arg : funcOp.getArguments()) {
      args[getBlockArgName(arg)] = arg.getType();
    }
  }
}

void ArmAnalysisImpl::initQueue() {
  SmallVector<std::unique_ptr<CandidateTrace>> initRuns;
  for (ArrayRef<Operation*> ops : blockOpStorage) {
    for (size_t i = 0; i < ops.size(); ++i) {
      initRuns.emplace_back(std::make_unique<CandidateTrace>(ops.slice(i)));
    }
  }
  QueueEntry entry(std::move(initRuns));
  if (entry.runs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No initial runs found\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Initial queue entry with " << entry.runs.size()
                            << " runs: " << entry.getArmInfo() << "\n");
    searchQueue.emplace_back(std::move(entry));
  }
}

void ArmAnalysisImpl::saveTo(ArmAnalysis& a) {
  a.distinctOps = singleOps.takeVector();
  a.multiOpArms = std::move(multiOps);
  a.funcType = FunctionType::get(ctx, llvm::to_vector(llvm::make_second_range(args)), resultTypes);
  a.argNames = llvm::to_vector(llvm::make_first_range(args));
}

void ArmAnalysisImpl::analyzeOne() {
  assert(!searchQueue.empty());

  QueueEntry top = std::move(searchQueue.back());
  searchQueue.pop_back();
  std::pop_heap(searchQueue.begin(), searchQueue.end());
  // Save this for later since it can get overwritten when we advance.
  ArmInfo topArmInfo = top.getArmInfo();

  LLVM_DEBUG(llvm::dbgs() << "queue size=" << searchQueue.size() << ", analyzing " << topArmInfo
                          << "\n");

  DenseSet<Operation*> poison;
  llvm::MapVector<ArmInfo,
                  SmallVector<std::unique_ptr<CandidateTrace>>,
                  DenseMap<ArmInfo, unsigned, UniqueArmMapInfo>>
      subRuns;
  bool countThis = false;

  SetVector<Location> locs;

  for (std::unique_ptr<CandidateTrace>& run : top.runs) {
    // Deeply uniquify locations since otherwise we end up with an
    // excessive number of things like call sites.
    if (locs.empty()) {
      for (Location loc : run->locs) {
        loc->walk([&](Location subLoc) {
          if (llvm::isa<FileLineColRange>(subLoc))
            locs.insert(subLoc);
          return WalkResult::advance();
        });
      }
    }

    if (run->advance(poison).failed()) {
      LLVM_DEBUG(llvm::dbgs() << "advancing failed\n");
      countThis = true;
      continue;
    }

    const ArmInfo& key = run->key;
    subRuns[key].push_back(std::move(run));
  }

  if (topArmInfo.getOps().size() == 1) {
    // Always save single-op runs
    topArmInfo.loc = FusedLoc::get(ctx, locs.getArrayRef());
    singleOps.insert(topArmInfo);
  }

  if (subRuns.size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << subRuns.size() << "subruns found\n");
    countThis = true;
  }

  for (auto& [subKey, subRun] : subRuns) {
    LLVM_DEBUG(llvm::dbgs() << "subrun: " << subKey << "\n");
    auto it = llvm::remove_if(subRun, [&](auto& r) { return r->isPoisoned(poison); });
    if (it != subRun.end()) {
      LLVM_DEBUG(llvm::dbgs() << "poisoned\n");
      countThis = true;
      subRun.erase(it, subRun.end());
      continue;
    }
    if (subRun.size() < kMinArmUseCount && subKey.getOps().size() > 1) {
      LLVM_DEBUG(llvm::dbgs() << "not enough used\n");
      countThis = true;
      continue;
    }

    searchQueue.push_back(QueueEntry(std::move(subRun)));
    std::push_heap(searchQueue.begin(), searchQueue.end());
  }

  if (countThis && topArmInfo.getOps().size() > 1) {
    topArmInfo.loc = FusedLoc::get(ctx, locs.getArrayRef());
    multiOps.emplace_back(std::move(topArmInfo));
  }
}

ArmAnalysis::ArmAnalysis(mlir::Operation* op) {
  ArmAnalysisImpl impl(op->getContext(), blockOpStorage);
  op->walk([&](func::FuncOp funcOp) {
    LLVM_DEBUG(llvm::dbgs() << "Found function " << funcOp << "\n");
    for (Block& block : funcOp.getBody()) {
      LLVM_DEBUG(llvm::dbgs() << "Found a block\n");

      impl.addBlock(&block);
    }
  });
  impl.initQueue();
  while (!impl.searchQueue.empty())
    impl.analyzeOne();
  impl.saveTo(*this);
}

} // namespace zirgen::ByteCode
