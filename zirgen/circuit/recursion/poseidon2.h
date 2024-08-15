// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/wom.h"

namespace zirgen::recursion {

namespace poseidon2 {
#include "zirgen/compiler/zkp/poseidon2_consts.h"

constexpr size_t CELLS_RATE = 16;
constexpr size_t CELLS_OUT = 8;
} // namespace poseidon2

using Poseidon2Cells = std::array<Reg, poseidon2::CELLS>;

// Load data from WOM in preparation for Poseidon2
class Poseidon2LoadImpl : public CompImpl<Poseidon2LoadImpl> {
public:
  Poseidon2LoadImpl(Code code, WomHeader header);
  void set(Code code, Val writeAddr);

private:
  // This take 85 elements (9 * 5 + 8 * 5)
  WomBody body;
  // These are the loads, and don't take space
  std::vector<WomReg> ios;
  // The final state @85
  Poseidon2Cells output;
};
using Poseidon2Load = Comp<Poseidon2LoadImpl>;

// Do a full round
class Poseidon2FullImpl : public CompImpl<Poseidon2FullImpl> {
public:
  Poseidon2FullImpl(Code code, WomHeader header);
  void set(Code code, Val writeAddr);

private:
  // pass takes 0 cells
  WomPass pass;
  // 72 cells = 3 * 24
  Poseidon2Cells pre1;
  Poseidon2Cells post1;
  Poseidon2Cells pre2;
  // Lucky # 13 to make output land in the same spot
  std::array<Reg, 13> padding;
  // The final state @85
  Poseidon2Cells output;
};
using Poseidon2Full = Comp<Poseidon2FullImpl>;

// Do all the partial rounds
class Poseidon2PartialImpl : public CompImpl<Poseidon2PartialImpl> {
public:
  Poseidon2PartialImpl(Code code, WomHeader header);
  void set(Code code, Val writeAddr);

private:
  // pass takes 0 cells
  WomPass pass;
  // 72 cells = 3 * 24
  Poseidon2Cells sboxIn;
  Poseidon2Cells tmp;
  Poseidon2Cells pre2;
  // Lucky # 13 to make output land in the same spot
  std::array<Reg, 13> padding;
  // The final state @85
  Poseidon2Cells output;
};
using Poseidon2Partial = Comp<Poseidon2PartialImpl>;

// Load the results of Poseidon2 data
class Poseidon2StoreImpl : public CompImpl<Poseidon2StoreImpl> {
public:
  Poseidon2StoreImpl(Code code, WomHeader header);
  void set(Code code, Val writeAddr);

private:
  // This take 85 elements (9 * 5 + 8 * 5)
  WomBody body;
  // These are the stores, and don't take space
  std::vector<WomReg> ios;
  // The final state @85
  Poseidon2Cells output;
};
using Poseidon2Store = Comp<Poseidon2StoreImpl>;

// Declare some things used in CheckedBytes as well
using CellVals = std::array<Val, poseidon2::CELLS>;
CellVals mulMExt(CellVals in);

} // namespace zirgen::recursion
