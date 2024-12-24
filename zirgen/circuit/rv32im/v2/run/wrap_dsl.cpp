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

#include "zirgen/circuit/rv32im/v2/run/wrap_dsl.h"
#include "zirgen/circuit/rv32im/v2/platform/constants.h"

#include <array>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "risc0/core/util.h"

namespace zirgen::rv32im_v2 {

namespace impl {

using Val = risc0::Fp;
using ExtVal = risc0::FpExt;

size_t to_size_t(Val v) {
  return v.asUInt32();
}

ExtVal operator+(const Val& lhs, const ExtVal& rhs) {
  return FpExt(lhs) + rhs;
}
ExtVal operator-(const ExtVal& lhs, const Val& rhs) {
  return lhs - FpExt(rhs);
}

struct ExecContext {
public:
  ExecContext(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle)
      : stepHandler(stepHandler), trace(trace), cycle(cycle) {}

  StepHandler& stepHandler;
  ExecutionTrace& trace;
  size_t cycle;
};

// Setup the basic field stuff
#define SET_FIELD(x) /**/

constexpr size_t EXT_SIZE = 4;

// Built in field operations
Val isz(Val x) {
  return Val(x == Val(0));
}
Val neg_0(Val x) {
  return -x;
}
Val inv_0(Val x) {
  return inv(x);
}
ExtVal inv_0(ExtVal x) {
  return inv(x);
}
Val bitAnd(Val a, Val b) {
  return Val(a.asUInt32() & b.asUInt32());
}
Val mod(Val a, Val b) {
  return Val(a.asUInt32() % b.asUInt32());
}
Val inRange(Val low, Val mid, Val high) {
  assert(low <= high);
  return Val(low <= mid && mid < high);
}
void eqz(Val a, const char* loc) {
  if (a.asUInt32()) {
    std::cerr << "eqz failure at: " << loc << "\n";
    throw std::runtime_error("eqz failure");
  }
}
void eqz(ExtVal a, const char* loc) {
  for (size_t i = 0; i < EXT_SIZE; i++) {
    eqz(a.elems[i], loc);
  }
}

// Define index type (used in back)
using Index = size_t;

struct Reg {
  constexpr Reg(size_t col) : col(col) {}
  size_t col;
};

struct BufferObj {
  virtual Val load(size_t col, size_t back) = 0;
  virtual void store(size_t col, Val val) = 0;
};

struct MutableBufObj : public BufferObj {
  MutableBufObj(ExecContext& ctx, TraceGroup& group) : ctx(ctx), group(group) {}
  Val load(size_t col, size_t back) override {
    size_t backRow = (group.getRows() + ctx.cycle - back) % group.getRows();
    return group.get(backRow, col);
  }
  void store(size_t col, Val val) override { return group.set(ctx.cycle, col, val); }
  ExecContext& ctx;
  TraceGroup& group;
};

using MutableBuf = MutableBufObj*;

struct GlobalBufObj : public BufferObj {
  GlobalBufObj(ExecContext& ctx, GlobalTraceGroup& group) : ctx(ctx), group(group) {}
  Val load(size_t col, size_t back) override {
    assert(back == 0);
    return group.get(col);
  }
  void store(size_t col, Val val) override { return group.set(col, val); }
  ExecContext& ctx;
  GlobalTraceGroup& group;
};

using GlobalBuf = GlobalBufObj*;

template <typename T> struct BoundLayout {
  BoundLayout(const T& layout, BufferObj* buf) : layout(&layout), buf(buf) {}
  BoundLayout() = default;
  BoundLayout(const BoundLayout&) = default;

  const T* layout = nullptr;
  BufferObj* buf = nullptr;
};

#define BIND_LAYOUT(orig, buf) BoundLayout(orig, buf)
#define LAYOUT_LOOKUP(orig, elem) BoundLayout(orig.layout->elem, orig.buf)
#define LAYOUT_SUBSCRIPT(orig, index) BoundLayout((*orig.layout)[index], orig.buf)
#define EQZ(val, loc) eqz(val, loc)

void store(ExecContext& ctx, BoundLayout<Reg> reg, Val val) {
  reg.buf->store(reg.layout->col, val);
}

void storeExt(ExecContext& ctx, BoundLayout<Reg> reg, ExtVal val) {
  for (size_t i = 0; i < EXT_SIZE; i++) {
    reg.buf->store(reg.layout->col + i, val.elems[i]);
  }
}

Val load(ExecContext& ctx, BoundLayout<Reg> reg, size_t back) {
  return reg.buf->load(reg.layout->col, back);
}

ExtVal loadExt(ExecContext& ctx, BoundLayout<Reg> reg, size_t back) {
  std::array<Fp, EXT_SIZE> elems;
  for (size_t i = 0; i < EXT_SIZE; i++) {
    elems[i] = reg.buf->load(reg.layout->col + i, back);
  }
  return FpExt(elems[0], elems[1], elems[2], elems[3]);
}

#define LOAD(reg, back) load(ctx, reg, back)
#define LOAD_EXT(reg, back) loadExt(ctx, reg, back)
#define STORE(reg, val) store(ctx, reg, val)
#define STORE_EXT(reg, val) storeExt(ctx, reg, val)

// Map + reduce support
template <typename T1, typename F, size_t N> auto map(std::array<T1, N> a, F f) {
  std::array<decltype(f(a[0])), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i]);
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
auto map(std::array<T1, N> a, std::array<T2, N> b, F f) {
  std::array<decltype(f(a[0], b[0])), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i], b[i]);
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
auto map(std::array<T1, N> a, const BoundLayout<T2>& b, F f) {
  std::array<decltype(f(a[0], BoundLayout((*b.layout)[0], b.buf))), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i], BoundLayout((*b.layout)[i], b.buf));
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
auto reduce(std::array<T1, N> elems, T2 start, F f) {
  T2 cur = start;
  for (size_t i = 0; i < N; i++) {
    cur = f(cur, elems[i]);
  }
  return cur;
}

template <typename T1, typename T2, typename T3, typename F, size_t N>
auto reduce(std::array<T1, N> elems, T2 start, const BoundLayout<T3>& b, F f) {
  T2 cur = start;
  for (size_t i = 0; i < N; i++) {
    cur = f(cur, elems[i], BoundLayout((*b.layout)[i], b.buf));
  }
  return cur;
}

// All the extern handling
#define INVOKE_EXTERN(ctx, name, ...) extern_##name(ctx, ##__VA_ARGS__)

std::array<Val, 5> extern_getMemoryTxn(ExecContext& ctx, Val addr) {
  auto txn = ctx.stepHandler.getMemoryTxn(addr.asUInt32());
  return {txn.prevCycle, txn.prevVal & 0xffff, txn.prevVal >> 16, txn.val & 0xffff, txn.val >> 16};
}

void extern_lookupDelta(ExecContext& ctx, Val table, Val index, Val count) {
  ctx.stepHandler.lookupDelta(table, index, count);
}

Val extern_lookupCurrent(ExecContext& ctx, Val table, Val index) {
  return ctx.stepHandler.lookupCurrent(table, index);
}

void extern_memoryDelta(
    ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count) {
  ctx.stepHandler.memoryDelta(
      addr.asUInt32(), cycle.asUInt32(), dataLow.asUInt32() | (dataHigh.asUInt32() << 16), count);
}

uint32_t extern_getDiffCount(ExecContext& ctx, Val cycle) {
  return ctx.stepHandler.getDiffCount(cycle.asUInt32());
}

Val extern_isFirstCycle_0(ExecContext& ctx) {
  return ctx.cycle == 0;
}

std::ostream& hex_word(std::ostream& os, uint32_t word) {
  std::cout << "0x"                                          //
            << std::hex << std::setw(8) << std::setfill('0') //
            << word                                          //
            << std::dec << std::setw(0);
  return os;
}

void extern_log(ExecContext& ctx, const std::string& message, std::vector<Val> vals) {
  std::cout << "LOG: '" << message << "': ";
  for (size_t i = 0; i < vals.size(); i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    hex_word(std::cout, vals[i].asUInt32());
  }
  std::cout << "\n";
}

std::array<Val, 4> extern_divide(
    ExecContext& ctx, Val numerLow, Val numerHigh, Val denomLow, Val denomHigh, Val signType) {
  uint32_t numer = numerLow.asUInt32() | (numerHigh.asUInt32() << 16);
  uint32_t denom = denomLow.asUInt32() | (denomHigh.asUInt32() << 16);
  auto [quot, rem] = risc0::divide_rv32im(numer, denom, signType.asUInt32());
  std::array<Val, 4> ret;
  ret[0] = quot & 0xffff;
  ret[1] = quot >> 16;
  ret[2] = rem & 0xffff;
  ret[3] = rem >> 16;
  return ret;
}

// TODO: logging
void extern_print(ExecContext& ctx, Val v) {
  std::cout << "LOG: " << v.asUInt32() << "\n";
}

std::array<Val, 2> extern_getMajorMinor(ExecContext& ctx) {
  auto majorMinor = ctx.stepHandler.getMajorMinor();
  return {majorMinor.first, majorMinor.second};
}

Val extern_hostReadPrepare(ExecContext& ctx, Val fp, Val len) {
  return ctx.stepHandler.readPrepare(fp.asUInt32(), len.asUInt32());
}

Val extern_hostWrite(ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal) {
  uint32_t fd = fdVal.asUInt32();
  uint32_t addr = addrLow.asUInt32() | (addrHigh.asUInt32() << 16);
  uint32_t len = lenVal.asUInt32();
  return ctx.stepHandler.write(fd, addr, len);
}

std::array<Val, 2> extern_nextPagingIdx(ExecContext& ctx) {
  auto ret = ctx.stepHandler.nextPagingIdx();
  return {ret[0], ret[1]};
}

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "zirgen/circuit/rv32im/v2/dsl/rv32im.cpp.inc"

} // namespace impl

CircuitParams getDslParams() {
  CircuitParams ret;
  ret.dataCols = impl::kRegCountData;
  ret.globalCols = impl::kRegCountGlobal;
  ret.accumCols = impl::kRegCountAccum;
  ret.mixCols = impl::kRegCountMix;
  return ret;
}

size_t getCycleCol() {
  return impl::kLayout_Top.cycle._super.col;
}

size_t getTopStateCol() {
  return impl::kLayout_Top.nextPcLow._super.col;
}

size_t getEcall0StateCol() {
  return impl::kLayout_Top.instResult.arm8.s0._super.col;
}

size_t getPoseidonStateCol() {
  return impl::kLayout_Top.instResult.arm9.state.hasState._super.col;
}

size_t getShaStateCol() {
  return impl::kLayout_Top.instResult.arm11.state.stateInAddr._super.col;
}

void DslStep(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle) {
  impl::ExecContext ctx(stepHandler, trace, cycle);
  impl::MutableBufObj data(ctx, trace.data);
  impl::GlobalBufObj global(ctx, trace.global);
  step_Top(ctx, &data, &global);
}

void DslStepAccum(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle) {
  impl::ExecContext ctx(stepHandler, trace, cycle);
  impl::MutableBufObj data(ctx, trace.data);
  impl::MutableBufObj accum(ctx, trace.accum);
  // Global is required when using user-accum
  // impl::GlobalBufObj global(ctx, trace.global);
  impl::GlobalBufObj mix(ctx, trace.mix);
  step_TopAccum(ctx, &accum, &data, /*&global, */ &mix);
}

} // namespace zirgen::rv32im_v2
