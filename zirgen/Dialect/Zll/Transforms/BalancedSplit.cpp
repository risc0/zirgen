#include "zirgen/Dialect/Zll/Transforms/BalancedSplit.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

constexpr size_t kNumParts = 2;

struct OpInfo {
  SmallVector<Operation*> sources;
  SmallVector<Operation*> sinks;
};

bool isInexpensive(Operation* op) {
  return llvm::isa<ConstOp, GetOp, GetGlobalOp>(op);
}

struct Splitter {
  Splitter(std::array<Block*, kNumParts> blocks);

  std::optional<size_t> getPartIndex(Operation* op) {
    for (const auto& [idx, part] : llvm::enumerate(parts)) {
      if (part.contains(op))
        return idx;
    }
    return std::nullopt;
  }

  // Finds the top owner within one of the blocks we manage
  Operation* getTopOwner(Operation* op);

  template <typename Range> std::array<size_t, kNumParts> getPartCounts(const Range& r) {
    std::array<size_t, kNumParts> counts{};
    for (Operation* op : r) {
      counts[*getPartIndex(op)]++;
    }
    return counts;
  }

  // Returns true if progress has been made
  bool balance();

  // List of values in and out of each operation we manage
  DenseMap<Operation*, OpInfo> opInfo;

  std::array<DenseSet<Operation*>, kNumParts> parts;
  std::array<Block*, kNumParts> blocks;
};

Splitter::Splitter(std::array<Block*, kNumParts> blocks) : blocks(blocks) {
  for (auto [idx, block] : llvm::enumerate(blocks)) {
    auto ops = llvm::make_pointer_range(blocks[idx]->getOperations());
    parts[idx].insert(ops.begin(), ops.end());
  }

  for (Block* b : blocks) {
    for (Operation* op : llvm::make_pointer_range(b->getOperations())) {
      if (isInexpensive(op))
        continue;
      OpInfo i;
      op->walk([&](Operation* subOp) {
        for (Value val : subOp->getOperands()) {
          Operation* sourceOp = getTopOwner(val.getDefiningOp());
          if (sourceOp && !isInexpensive(sourceOp))
            i.sources.push_back(sourceOp);
        }
      });

      for (Value val : op->getResults()) {
        for (Operation* subOp : val.getUsers()) {
          Operation* sinkOp = getTopOwner(subOp);
          if (sinkOp && !isInexpensive(sinkOp))
            i.sinks.push_back(sinkOp);
        }
      }
      opInfo[op] = i;
    }
  }
}

bool Splitter::balance() {
  ssize_t bestChange = 0;
  Operation* bestToMove = nullptr;

  for (auto& [op, i] : opInfo) {
    size_t opFromPart = *getPartIndex(op);
    size_t opToPart = 1 - opFromPart;

    auto sourceCounts = getPartCounts(i.sources);
    auto sinkCounts = getPartCounts(i.sinks);
    bool canMove = true;
    ssize_t improvement = 0;

    for (auto idx : llvm::seq(kNumParts)) {
      if (idx > opToPart && sourceCounts[idx]) {
        // Ops above the cut cannot source things from ops below the cut.
        canMove = false;
        break;
      }
      if (idx < opToPart && sinkCounts[idx]) {
        // Cannot output to things above the cut if w'ere below the cut
        canMove = false;
        break;
      }

      if (idx == opFromPart) {
        improvement -= (sourceCounts[idx] + sinkCounts[idx]);
      }

      if (idx == opToPart) {
        improvement += (sourceCounts[idx] + sinkCounts[idx]);
      }
    }

    if (!canMove)
      continue;
    if (improvement > bestChange) {
      bestChange = improvement;
      bestToMove = op;
    }
  }

  if (!bestToMove)
    return false;

  size_t opFromPart = *getPartIndex(bestToMove);
  size_t opToPart = 1 - opFromPart;
  //  llvm::errs() << "moving " << *bestToMove << " from " << opFromPart << " to " << opToPart
  //               << " for  great " << bestChange << "\n";

  assert(parts[opFromPart].erase(bestToMove));
  assert(parts[opToPart].insert(bestToMove).second);

  if (opFromPart < opToPart) {
    bestToMove->moveBefore(blocks[opToPart], blocks[opToPart]->begin());
  } else {
    bestToMove->moveBefore(blocks[opToPart], blocks[opToPart]->end());
  }
  return true;
}

Operation* Splitter::getTopOwner(Operation* op) {
  Operation* origOp = 0;
  while (op) {
    for (auto [idx, b] : llvm::enumerate(blocks)) {
      if (op->getBlock() == b)
        return op;
    }
    op = op->getParentOp();
  }

  if (origOp) {
    llvm::errs() << "Original operation: " << *origOp << "\n";
  }
  return nullptr;
}

} // namespace

Block* balancedSplitBlock(Block* orig) {
  auto endIt = orig->getOperations().end();
  auto advance = [&](auto& it) {
    while (it != endIt) {
      ++it;
      if (it == endIt)
        break;
      if (!isInexpensive(&*it))
        return true;
    }
    return false;
  };

  auto it = orig->getOperations().begin();
  auto halfIt = it;

  while (advance(it) && advance(it)) {
    bool didAdvance = advance(halfIt);
    assert(didAdvance);
  }

  Block* result = orig->splitBlock(halfIt);
  if (!result)
    return nullptr;

  Splitter splitter({orig, result});
  while (splitter.balance()) {
  }

  return result;
}

void balancedSplitFunc(mlir::func::FuncOp func) {
  Block* splitResult = balancedSplitBlock(&func.getBody().front());
  if (!splitResult) {
    return;
  }

  auto mod = func->getParentOfType<ModuleOp>();
  unsigned int uniqueIndex = 0;
  auto newName = SymbolTable::generateSymbolName<64>(
      func.getName(), [&](auto sym) -> bool { return mod.lookupSymbol(sym); }, uniqueIndex);

  IRRewriter builder(func.getContext());
  // Apparently makeRegionIsolatedFromAbove only correctly detects captures when run on a descendant
  // region, so start the new function out inside the old one.
  builder.setInsertionPointToEnd(&func.getBody().front());
  auto newFuncOp = builder.create<func::FuncOp>(
      func.getLoc(),
      newName,
      FunctionType::get(func.getContext(), /*inputs=*/{}, func.getResultTypes()));
  SymbolTable::setSymbolVisibility(newFuncOp, SymbolTable::Visibility::Private);
  splitResult->moveBefore(&newFuncOp.getBody(), newFuncOp.getBody().end());
  auto passValues = makeRegionIsolatedFromAbove(
      builder,
      newFuncOp.getBody(),
      /*clone into region?=*/[&](Operation* op) { return isInexpensive(op); });

  newFuncOp.setFunctionType(
      FunctionType::get(func.getContext(),
                        /*inputs=*/newFuncOp.getBody().front().getArgumentTypes(),
                        /*outputs=*/func.getResultTypes()));

  auto callResult = builder.create<func::CallOp>(func.getLoc(), newFuncOp, passValues);
  builder.create<func::ReturnOp>(func.getLoc(), callResult->getResults());

  newFuncOp->moveBefore(func);
}

struct BalancedSplitPass : public BalancedSplitBase<BalancedSplitPass> {
  BalancedSplitPass(size_t maxOps) : maxOps(maxOps) {}

  void runOnOperation() override {
    for (;;) {
      func::FuncOp biggest;
      size_t biggestOps = 0;

      getOperation()->walk([&](func::FuncOp funcOp) {
        size_t numOps = funcOp.getBody().front().getOperations().size();
        if (numOps > biggestOps) {
          biggestOps = numOps;
          biggest = funcOp;
        }
      });

      if (biggestOps <= maxOps)
        // Everything is small enough; we're done here
        return;

      // Split the biggest function, then go again.
      balancedSplitFunc(biggest);

      //      llvm::errs() << "One balanced split done, current state:\n";
      //      llvm::errs() << *getOperation() << "\n";
    }
  }

  size_t maxOps;
};

std::unique_ptr<Pass> createBalancedSplitPass() {
  return std::make_unique<BalancedSplitPass>(2000);
}

std::unique_ptr<Pass> createBalancedSplitPass(size_t maxOps) {
  return std::make_unique<BalancedSplitPass>(maxOps);
}

} // namespace zirgen::Zll
