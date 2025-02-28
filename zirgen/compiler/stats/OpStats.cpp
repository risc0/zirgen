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

#include "zirgen/compiler/stats/OpStats.h"

#include "mlir/IR/AsmState.h"
#include "risc0/fp/fp.h"
#include "risc0/fp/fpext.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Timer.h"
#include <random>

#define DEBUG_TYPE "op-stats"

using namespace risc0;
using namespace zirgen::Zll;
using namespace mlir;
namespace cl = llvm::cl;

namespace zirgen {

namespace {

enum class SortOrder { Disabled, Max, Inner, Outer, Flat, Any };

struct OpStatsCLOptions {
  cl::opt<enum SortOrder> sortOrder{
      "op-stats",
      cl::desc("Produce operation statistics with this order"),
      cl::init(SortOrder::Disabled),
      cl::values(
          clEnumValN(SortOrder::Disabled, "disabled", "Do not calculate operation statistics"),
          clEnumValN(SortOrder::Inner, "inner", "Combined across callees"),
          clEnumValN(SortOrder::Outer, "outer", "Combined across callers"),
          clEnumValN(SortOrder::Flat, "flat", "Uncombined"),
          clEnumValN(SortOrder::Any, "any", "Combined occurrences anywhere in call stack"),
          clEnumValN(SortOrder::Max, "max", "Maximum of all the metrics"))};
};

llvm::ManagedStatic<OpStatsCLOptions> clOpts;

// Number of times to run profiling loop
constexpr size_t kNumIter = 1000 * 1000;

// Prevent a value from being optimized for use in profiling loop.  Adapted from
// DoNotOptimize in Google's benchmark.h.
template <typename FpT> FpT blackBox(FpT value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
  return value;
}

// Returns a function to run a single iteration of profiling for OpT.
template <typename OpT> auto runOp();

template <> auto runOp<AddOp>() {
  return [](auto inVal) { return blackBox(inVal) + blackBox(inVal); };
}

template <> auto runOp<SubOp>() {
  return [](auto inVal) { return blackBox(inVal) - blackBox(inVal); };
}

template <> auto runOp<MulOp>() {
  return [](auto inVal) { return blackBox(inVal) * blackBox(inVal); };
}

template <> auto runOp<AndEqzOp>() {
  return [](auto inVal) {
    return /*in=*/blackBox(inVal) +
           /*constraint=*/blackBox(inVal.elems[0]) * /*mix=*/blackBox(inVal);
  };
}

template <> auto runOp<AndCondOp>() {
  return [](auto inVal) {
    return /*in=*/blackBox(inVal) +
           /*cond=*/blackBox(inVal.elems[0]) * /*inside=*/blackBox(inVal) * /*mix=*/blackBox(inVal);
  };
}

template <typename FpT> FpT getRandVal(const std::function<uint64_t(void)>& randFunc);

template <> Fp getRandVal<Fp>(const std::function<uint64_t(void)>& randFunc) {
  return Fp(randFunc() % Fp::P);
}

template <> FpExt getRandVal<FpExt>(const std::function<uint64_t(void)>& randFunc) {
  return FpExt{getRandVal<Fp>(randFunc),
               getRandVal<Fp>(randFunc),
               getRandVal<Fp>(randFunc),
               getRandVal<Fp>(randFunc)};
}

struct LocStat {
  double flat = 0;

  // Cumulative statistics across many matching locations.
  double insideCum = 0;
  double outsideCum = 0;
  double anyCum = 0;

  // Returns the metric to sort on
  double sortStat() const {
    switch (clOpts->sortOrder) {
    case SortOrder::Max:
      return *llvm::max_element(
          std::initializer_list<double>({flat, insideCum, outsideCum, anyCum}));
    case SortOrder::Inner:
      return insideCum;
    case SortOrder::Outer:
      return outsideCum;
    case SortOrder::Flat:
      return flat;
    case SortOrder::Any:
      return anyCum;
    default:
      llvm_unreachable("Unhandled sort order");
    }
  }
  bool operator<(const LocStat& rhs) const { return sortStat() < rhs.sortStat(); }
};

class LocStats {
public:
  void countLoc(Location loc, double c) {
    seenFlat.clear();
    seenInside.clear();
    seenOutside.clear();
    seenAny.clear();

    // Accumulate for any `FileLineColLoc` or `NameLoc`s that occur anywhere in the call chain
    loc->walk([&](Location subLoc) {
      if (llvm::isa<FileLineColLoc>(subLoc)) {
        if (seenAny.insert(subLoc).second)
          locs[subLoc].anyCum += c;
      } else if (auto nameLoc = llvm::dyn_cast<NameLoc>(subLoc)) {
        // Strip out inner location, and just count name.
        nameLoc = NameLoc::get(nameLoc.getName());
        if (seenAny.insert(nameLoc).second)
          locs[nameLoc].anyCum += c;
      }
      return WalkResult::advance();
    });

    // Expand any `FusedLoc`s in the call chain so we can count them all.
    SmallVector<Location> workList, locList;
    workList.push_back(loc);

    while (!workList.empty()) {
      Location workLoc = workList.pop_back_val();

      if (auto fusedLoc = workLoc->findInstanceOf<FusedLoc>()) {
        for (auto subLoc : fusedLoc.getLocations()) {
          AttrTypeReplacer replacer;
          replacer.addReplacement([&](LocationAttr replaceLoc) -> Attribute {
            if (replaceLoc == fusedLoc)
              return subLoc;
            return replaceLoc;
          });
          workList.push_back(llvm::cast<Location>(replacer.replace(subLoc)));
        }
      } else {
        locList.push_back(workLoc);
      }
    }

    // Divide up the flat accounting between all locations that are
    // fused together, to ensure that the elements of the `flat`
    // column always sum up to 100%.
    double flatC = c / locList.size();
    for (Location loc : locList) {
      countLocImpl(loc, c, flatC);
    }
  }

  SmallVector<std::pair<Location, LocStat>> toVector() const {
    return llvm::to_vector_of<std::pair<Location, LocStat>>(locs);
  }

private:
  // Deconstruct a recursive CallSiteLoc into a linear vector of inner locations.
  void getCallSiteChain(CallSiteLoc loc, SmallVector<Location>& chain) {
    if (auto callerLoc = dyn_cast<CallSiteLoc>(loc.getCaller()))
      getCallSiteChain(callerLoc, chain);
    else
      chain.push_back(loc.getCaller());
    if (auto calleeLoc = dyn_cast<CallSiteLoc>(loc.getCallee()))
      getCallSiteChain(calleeLoc, chain);
    else
      chain.push_back(loc.getCallee());
  }

  // Construct a recursive CallSiteLoc from a linear vector of locations.
  Location makeCallChain(ArrayRef<Location> inChain) {
    LocationAttr outLoc;
    for (auto loc : inChain) {
      if (outLoc) {
        outLoc = CallSiteLoc::get(loc, outLoc);
      } else
        outLoc = loc;
    }
    return outLoc;
  }

  void countLocImpl(Location loc, double c, double flatC) {
    if (seenFlat.insert(loc).second)
      locs[loc].flat += flatC;
    if (auto callSiteLoc = llvm::dyn_cast<CallSiteLoc>(loc)) {
      SmallVector<Location> chain;
      getCallSiteChain(callSiteLoc, chain);

      for (auto idx : llvm::seq<size_t>(1, chain.size())) {
        auto insideLoc = makeCallChain(ArrayRef(chain).slice(0, idx));
        if (seenInside.insert(insideLoc).second)
          locs[insideLoc].insideCum += c;
        auto outsideLoc = makeCallChain(ArrayRef(chain).slice(chain.size() - idx));
        if (seenOutside.insert(outsideLoc).second)
          locs[outsideLoc].outsideCum += c;
      }
    } else {
      loc->walk([&](Location subLoc) {
        if (loc == subLoc)
          return WalkResult::advance();
        countLocImpl(subLoc, c, flatC);
        return WalkResult::skip();
      });
    }
  }

  DenseMap<Location, LocStat> locs;

  // Deduplication so we only count each operation once, even if an accumulation location occurs
  // more than once in its location tree.
  DenseSet<Location> seenFlat, seenInside, seenOutside, seenAny;
};

} // namespace

// Calculate the number of bogocycles used for OpT with the given extension field.
template <typename OpT, size_t K, typename FpT> double BogoCycleAnalysis::getOrCalcBogoCycles() {
  std::pair<mlir::StringLiteral, size_t> id = std::make_pair(OpT::getOperationName(), K);
  auto it = bogoCycles.find(id);
  if (it != bogoCycles.end()) {
    return it->second;
  }

  std::random_device rndDev;
  std::mt19937 rnd(rndDev());
  FpT val = getRandVal<FpT>(rnd);
  auto f = runOp<OpT>();

  llvm::TimeRecord beginTime = llvm::TimeRecord::getCurrentTime(/*start=*/true);
  for (size_t i = 0; i != kNumIter; ++i) {
    val = f(val);
  }
  llvm::TimeRecord totTime = llvm::TimeRecord::getCurrentTime(/*start=*/false);
  totTime -= beginTime;

  double cycles = totTime.getUserTime() * 100;
  bogoCycles[id] = cycles;
  LLVM_DEBUG({
    llvm::dbgs() << "Bogo cycles for " << OpT::getOperationName() << " (k=" << K
                 << "): " << llvm::format("%.3f", cycles) << "\n";
  });
  return cycles;
}

double BogoCycleAnalysis::getBogoCycles(Operation* op) {
  // Relative costs of operations, in bogocycles.
  return TypeSwitch<Operation*, double>(op)
      .Case<AddOp, SubOp, MulOp>([&](auto op) {
        if (op.getType().getFieldK() == 4) {
          return getOrCalcBogoCycles<decltype(op), 4, FpExt>();
        } else {
          assert(op.getType().getFieldK() == 1);
          return getOrCalcBogoCycles<decltype(op), 1, Fp>();
        }
      })
      .Case<AndEqzOp, AndCondOp>(
          [&](auto op) { return getOrCalcBogoCycles<decltype(op), 4, FpExt>(); })
      .Default([](auto) { return 0; });
}

void BogoCycleAnalysis::printStatsIfRequired(Operation* topOp, llvm::raw_ostream& os) {
  if (!clOpts.isConstructed()) {
    throw(std::runtime_error("op-stats command line options must be registered"));
  }

  if (clOpts->sortOrder == SortOrder::Disabled) {
    LLVM_DEBUG({ llvm::dbgs() << "BogoCycle operation statistics not requested"; });
    return;
  }

  double totCycles = 0;
  LocStats locStats;

  topOp->walk([&](Operation* op) {
    double c = getBogoCycles(op);
    if (!c)
      return;

    totCycles += c;
    locStats.countLoc(op->getLoc(), c);
  });

  auto totLocStats = locStats.toVector();

  llvm::sort(totLocStats, llvm::less_second());

  os << llvm::format("%10s %10s %10s %10s %-100s\n",
                     (const char*)"Flat",
                     (const char*)"Inside",
                     (const char*)"Outside",
                     (const char*)"Any",
                     (const char*)"Location");

  // Set us up to be able to
  OpPrintingFlags flags;
  flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/true);
  AsmState asmState(topOp, flags);

  for (auto& [loc, stat] : llvm::reverse(totLocStats)) {
    if (stat.flat)
      os << llvm::format("%9.5f%% ", stat.flat * 100. / totCycles);
    else
      os << llvm::indent(11);
    if (stat.insideCum)
      os << llvm::format("%9.5f%% ", stat.insideCum * 100. / totCycles);
    else
      os << llvm::indent(11);

    if (stat.outsideCum)
      os << llvm::format("%9.5f%% ", stat.outsideCum * 100. / totCycles);
    else
      os << llvm::indent(11);

    if (stat.anyCum)
      os << llvm::format("%9.5f%% ", stat.anyCum * 100. / totCycles);
    else
      os << llvm::indent(11);

    // We want to indent the " at ..." call site messages so they're easier to read
    std::string s;
    llvm::raw_string_ostream sos(s);
    loc->print(sos, asmState);
    llvm::interleave(
        llvm::split(s, '\n'),
        [&](StringRef line) { os << line; },
        [&]() { os << "\n"
                   << llvm::indent(11 * 4); });
    os << "\n\n";
  }
}

void registerOpStatsCLOptions() {
  *clOpts;
}

} // namespace zirgen
