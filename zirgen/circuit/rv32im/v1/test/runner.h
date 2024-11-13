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

#pragma once

#include <set>

#include "zirgen/circuit/rv32im/v1/edsl/top.h"

namespace zirgen::rv32im_v1 {

enum IncludeDir {
  Read = 1,
  Write = 2,
  Both = 3,
};

struct PageFaultInfo {
  PageTableInfo info;
  std::set<uint32_t> reads;
  std::set<uint32_t> writes;
  bool forceFlush = false;

  void include(uint32_t addr, IncludeDir dir = IncludeDir::Read);
  void dump();
};

using Polynomial = llvm::SmallVector<uint64_t, 4>;

// TODO: Highly redundant with stuff in Exec.cpp
struct BytePolynomial {
  BytePolynomial();
  static BytePolynomial zero();
  static BytePolynomial one();
  BytePolynomial shift() const;
  BytePolynomial operator*(int x) const;
  BytePolynomial operator+(const BytePolynomial& rhs) const;
  BytePolynomial operator*(const BytePolynomial& rhs) const;
  std::vector<int32_t> coeffs;
};

struct Runner : public RamExternHandler {
  Runner(size_t maxCycles, std::map<uint32_t, uint32_t> elfImage, uint32_t entryPoint);

  // Run all the stages
  void run();

  void runAgain();

  // Clear things when done, return true if program halted
  bool done();

  void setInput(const std::vector<uint32_t>& input);

  void runStage(size_t stage);
  void setMix();

private:
  void generateCircuit();
  Digest initMemoryImage();

  void storePageEntry(uint32_t pgidx, const Digest& digest);

  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const Zll::InterpVal*> args,
                                                size_t outCount) override;

  PageFaultInfo getPageFaultInfo(uint32_t pc, uint32_t inst);

  // Determine if a flush is required because there won't be enough cycles to complete the current
  // instruction assuming that paging out all dirty pages up to this point require a certain amount
  // of cycles.
  bool needsFlush(uint32_t cycle);

private:
  Module module;

  std::vector<Polynomial> code;
  size_t cycles;
  std::vector<Polynomial> out;
  std::vector<Polynomial> data;
  std::vector<Polynomial> mix;
  std::vector<Polynomial> accum;
  std::vector<llvm::MutableArrayRef<Polynomial>> args;

  bool isHalted = false;
  bool userMode = false;

  // When the machine is in a flushing state, then no new dirty pages will be recorded and the next
  // dirty page will be reported in a 'pageInfo' extern.
  bool isFlushing = false;

  // Tracks pages that have already been paged in.
  std::set<uint32_t> finishedPageReads;

  // Tracks which pages are dirty and need to be paged out in a subsquent flush
  std::set<uint32_t> dirtyPages;

  // This is just for diagnostics: tracks which words have been paged in.
  std::set<uint32_t> residentWords;

  uint32_t lastPc;

  std::vector<uint32_t> input;
  std::deque<uint64_t> syscallPending;
  uint32_t syscallA0Out;
  uint32_t syscallA1Out;

  void computePolyWitness(mlir::func::FuncOp func);

  bool inCarry;
  BytePolynomial poly;
  BytePolynomial term;
  BytePolynomial total;
  BytePolynomial totCarry;

public: // TODO: Hack
  std::map<uint32_t, uint32_t> polyWitness;
};

} // namespace zirgen::rv32im_v1
