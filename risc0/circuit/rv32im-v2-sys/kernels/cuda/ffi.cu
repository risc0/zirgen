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

#include "buffers.h"
#include "cuda.h"
#include "fp.h"
#include "fpext.h"
#include "preflight.h"
#include "tables.h"

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cuda/std/array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <vector>

namespace risc0 {

struct ExecBuffers {
  Buffer global;
  Buffer data;
};

struct AccumBuffers {
  Buffer data;
  Buffer accum;
  Buffer mix;
};

__device__ cuda::std::array<uint32_t, 2>
divide_rv32im(uint32_t numer, uint32_t denom, uint32_t signType) {
  uint32_t onesComp = (signType == 2);
  bool negNumer = signType && int32_t(numer) < 0;
  bool negDenom = signType == 1 && int32_t(denom) < 0;
  if (negNumer) {
    numer = -numer - onesComp;
  }
  if (negDenom) {
    denom = -denom - onesComp;
  }
  uint32_t quot;
  uint32_t rem;
  if (denom == 0) {
    quot = 0xffffffff;
    rem = numer;
  } else {
    quot = numer / denom;
    rem = numer % denom;
  }
  uint32_t quotNegOut = (negNumer ^ negDenom) - ((denom == 0) * negNumer);
  uint32_t remNegOut = negNumer;
  if (quotNegOut) {
    quot = -quot - onesComp;
  }
  if (remNegOut) {
    rem = -rem - onesComp;
  }
  return {quot, rem};
}

namespace impl {

using Val = Fp;
using ExtVal = FpExt;

__device__ size_t to_size_t(Fp v) {
  return v.asUInt32();
}

// Setup the basic field stuff
#define SET_FIELD(x) /**/

constexpr size_t EXT_SIZE = sizeof(ExtVal::elems) / sizeof(Val);
static_assert(EXT_SIZE == 4);

struct ExecContext {
  __device__ constexpr ExecContext(PreflightTrace& preflight, LookupTables& tables, size_t cycle)
      : preflight(preflight), tables(tables), cycle(cycle) {}
  PreflightTrace& preflight;
  LookupTables& tables;
  size_t cycle;
};

// Built in field operations
__device__ Val isz(Val x) {
  return Val(x == Val(0));
}

__device__ Val neg_0(Val x) {
  return -x;
}

__device__ Val inv_0(Val x) {
  return inv(x);
}

__device__ ExtVal inv_0(ExtVal x) {
  return inv(x);
}

__device__ Val bitAnd(Val a, Val b) {
  return Val(a.asUInt32() & b.asUInt32());
}

__device__ Val inRange(Val low, Val mid, Val high) {
  assert(low <= high);
  return Val(low <= mid && mid < high);
}

__device__ void eqz(Val a, const char* loc) {
  if (a.asUInt32()) {
    printf("eqz failure at: %s\n", loc);
    assert(false && "eqz failure");
  }
}

__device__ void eqz(ExtVal a, const char* loc) {
  for (size_t i = 0; i < EXT_SIZE; i++) {
    eqz(a.elems[i], loc);
  }
}

// Define index type (used in back)
using Index = size_t;

struct Reg {
  __device__ constexpr Reg(size_t col) : col(col) {}
  size_t col;
};

struct BufferObj {
  __device__ virtual Val load(size_t col, size_t back) = 0;
  __device__ virtual void store(size_t col, Val val) = 0;
};

struct MutableBufObj : public BufferObj {
  __device__ constexpr MutableBufObj(ExecContext& ctx, Buffer& buf) : ctx(ctx), buf(buf) {}

  __device__ Val load(size_t col, size_t back) override {
    if (back > ctx.cycle) {
      // std::cerr << "Going back too far, back: " << back << ", cycle: " << ctx.cycle << "\n";
      return 0;
    }
    return buf.get(ctx.cycle - back, col);
  }

  __device__ void store(size_t col, Val val) override { return buf.set(ctx.cycle, col, val); }

  ExecContext& ctx;
  Buffer& buf;
};

using MutableBuf = MutableBufObj*;

struct GlobalBufObj : public BufferObj {
  __device__ constexpr GlobalBufObj(ExecContext& ctx, Buffer& buf) : ctx(ctx), buf(buf) {}

  __device__ Val load(size_t col, size_t back) override {
    assert(back == 0);
    return buf.get(0, col);
  }

  __device__ void store(size_t col, Val val) override { return buf.set(0, col, val); }

  ExecContext& ctx;
  Buffer& buf;
};

using GlobalBuf = GlobalBufObj*;

template <typename T> struct BoundLayout {
  __device__ constexpr BoundLayout(const T& layout, BufferObj* buf) : layout(layout), buf(buf) {}
  const T& layout;
  BufferObj* buf;
};

#define BIND_LAYOUT(orig, buf) BoundLayout(orig, buf)
#define LAYOUT_LOOKUP(orig, elem) BoundLayout(orig.layout.elem, orig.buf)
#define LAYOUT_SUBSCRIPT(orig, index) BoundLayout(orig.layout[index], orig.buf)
#define EQZ(val, loc) eqz(val, loc)

__device__ void store(ExecContext& ctx, BoundLayout<Reg> reg, Val val) {
  reg.buf->store(reg.layout.col, val);
}

__device__ void storeExt(ExecContext& ctx, BoundLayout<Reg> reg, ExtVal val) {
  for (size_t i = 0; i < EXT_SIZE; i++) {
    reg.buf->store(reg.layout.col + i, val.elems[i]);
  }
}

__device__ Val load(ExecContext& ctx, BoundLayout<Reg> reg, size_t back) {
  return reg.buf->load(reg.layout.col, back);
}

__device__ ExtVal loadExt(ExecContext& ctx, BoundLayout<Reg> reg, size_t back) {
  cuda::std::array<Fp, EXT_SIZE> elems;
  for (size_t i = 0; i < EXT_SIZE; i++) {
    elems[i] = reg.buf->load(reg.layout.col + i, back);
  }
  return FpExt(elems[0], elems[1], elems[2], elems[3]);
}

#define LOAD(reg, back) load(ctx, reg, back)
#define LOAD_EXT(reg, back) loadExt(ctx, reg, back)
#define STORE(reg, val) store(ctx, reg, val)
#define STORE_EXT(reg, val) storeExt(ctx, reg, val)

// Map + reduce support
template <typename T1, typename F, size_t N> __device__ auto map(cuda::std::array<T1, N> a, F f) {
  cuda::std::array<decltype(f(a[0])), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i]);
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
auto __device__ map(cuda::std::array<T1, N> a, cuda::std::array<T2, N> b, F f) {
  cuda::std::array<decltype(f(a[0], b[0])), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i], b[i]);
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
auto __device__ map(cuda::std::array<T1, N> a, const BoundLayout<T2>& b, F f) {
  cuda::std::array<decltype(f(a[0], BoundLayout(b.layout[0], b.buf))), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i], BoundLayout(b.layout[i], b.buf));
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
auto __device__ reduce(cuda::std::array<T1, N> elems, T2 start, F f) {
  T2 cur = start;
  for (size_t i = 0; i < N; i++) {
    cur = f(cur, elems[i]);
  }
  return cur;
}

template <typename T1, typename T2, typename T3, typename F, size_t N>
auto __device__ reduce(cuda::std::array<T1, N> elems, T2 start, const BoundLayout<T3>& b, F f) {
  T2 cur = start;
  for (size_t i = 0; i < N; i++) {
    cur = f(cur, elems[i], BoundLayout(b.layout[i], b.buf));
  }
  return cur;
}

// All the extern handling
#define INVOKE_EXTERN(ctx, name, ...) extern_##name(ctx, ##__VA_ARGS__)

__device__ cuda::std::array<Val, 5> extern_getMemoryTxn(ExecContext& ctx, Val addrElem) {
  uint32_t addr = addrElem.asUInt32();
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx++;
  const MemoryTransaction& txn = ctx.preflight.txns[txnIdx];
  // printf("getMemoryTxn(%lu, 0x%08x): txn(%u, 0x%08x, 0x%08x)\n",
  //        ctx.cycle,
  //        addr,
  //        txn.cycle,
  //        txn.addr,
  //        txn.word);

  if (txn.cycle != ctx.cycle) {
    printf("txn.cycle: %u, ctx.cycle: %zu\n", txn.cycle, ctx.cycle);
    assert(false && "txn cycle mismatch");
  }

  if (txn.addr != addr) {
    printf("txn.addr: 0x%08x, addr: 0x%08x\n", txn.addr, addr);
    assert(false && "memory peek not in preflight");
  }
  return {
      txn.prevCycle,
      txn.prevWord & 0xffff,
      txn.prevWord >> 16,
      txn.word & 0xffff,
      txn.word >> 16,
  };
}

__device__ void extern_lookupDelta(ExecContext& ctx, Val table, Val index, Val count) {
  // printf("lookupDelta(table: %u, index: %u, count: %u, P: %u)\n",
  //        table.asUInt32(),
  //        index.asUInt32(),
  //        count.asUInt32(),
  //        Fp::P);
  ctx.tables.lookupDelta(table, index, count);
}

__device__ Val extern_lookupCurrent(ExecContext& ctx, Val table, Val index) {
  Val ret = ctx.tables.lookupCurrent(table, index);
  // printf("lookupCurrent(table: %u, index: %u): %u\n",
  //        table.asUInt32(),
  //        index.asUInt32(),
  //        ret.asUInt32());
  return ret;
}

__device__ void
extern_memoryDelta(ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count) {
  // printf("memoryDelta\n");
  // ctx.tables.memoryDelta(
  //     addr.asUInt32(), cycle.asUInt32(), dataLow.asUInt32() | (dataHigh.asUInt32() << 16),
  //     count);
}

__device__ uint32_t extern_getDiffCount(ExecContext& ctx, Val cycle) {
  // printf("getDiffCount\n");
  return ctx.preflight.cycles[cycle.asUInt32()].diffCount;
}

__device__ Val extern_isFirstCycle_0(ExecContext& ctx) {
  return ctx.cycle == 0;
}

__device__ Val extern_getCycle(ExecContext& ctx) {
  return ctx.cycle;
}

// std::ostream& hex_word(std::ostream& os, uint32_t word) {
//   std::cout << "0x"                                          //
//             << std::hex << std::setw(8) << std::setfill('0') //
//             << word                                          //
//             << std::dec << std::setw(0);
//   return os;
// }

template <typename T> __device__ void extern_log(ExecContext& ctx, const char* message, T vals) {
  // std::cout << "LOG: '" << message << "': ";
  // for (size_t i = 0; i < vals.size(); i++) {
  //   if (i != 0) {
  //     std::cout << ", ";
  //   }
  //   hex_word(std::cout, vals[i].asUInt32());
  // }
  // std::cout << "\n";
}

__device__ cuda::std::array<Val, 4> extern_divide(
    ExecContext& ctx, Val numerLow, Val numerHigh, Val denomLow, Val denomHigh, Val signType) {
  printf("divide\n");
  uint32_t numer = numerLow.asUInt32() | (numerHigh.asUInt32() << 16);
  uint32_t denom = denomLow.asUInt32() | (denomHigh.asUInt32() << 16);
  auto [quot, rem] = divide_rv32im(numer, denom, signType.asUInt32());
  cuda::std::array<Val, 4> ret;
  ret[0] = quot & 0xffff;
  ret[1] = quot >> 16;
  ret[2] = rem & 0xffff;
  ret[3] = rem >> 16;
  return ret;
}

// TODO: logging
__device__ void extern_print(ExecContext& ctx, Val v) {
  printf("LOG: %u\n", v.asUInt32());
}

__device__ cuda::std::array<Val, 2> extern_getMajorMinor(ExecContext& ctx) {
  uint8_t major = ctx.preflight.cycles[ctx.cycle].major;
  uint8_t minor = ctx.preflight.cycles[ctx.cycle].minor;
  return {major, minor};
}

__device__ Val extern_hostReadPrepare(ExecContext& ctx, Val fp, Val len) {
  printf("hostReadPrepare\n");
  assert(false && "extern_hostReadPrepare");
  // return ctx.stepHandler.readPrepare(fp.asUInt32(), len.asUInt32());
  // return 0;
}

__device__ Val
extern_hostWrite(ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal) {
  printf("hostWrite\n");
  assert(false && "extern_hostWrite");
  // uint32_t fd = fdVal.asUInt32();
  // uint32_t addr = addrLow.asUInt32() | (addrHigh.asUInt32() << 16);
  // uint32_t len = lenVal.asUInt32();
  // return ctx.stepHandler.write(fd, addr, len);
  // return 0;
}

__device__ cuda::std::array<Val, 2> extern_nextPagingIdx(ExecContext& ctx) {
  uint32_t pagingIdx = ctx.preflight.cycles[ctx.cycle].pagingIdx;
  uint32_t machineMode = ctx.preflight.cycles[ctx.cycle].machineMode;
  // printf("nextPagingIdx: (0x%05x, %u)\n", pagingIdx, machineMode);
  return {pagingIdx, machineMode};
}

#include "defs.cu.inc"

#include "types.cuh.inc"

#include "layout.cu.inc"

#include "steps.cu.inc"

} // namespace impl

__device__ void
stepExec(ExecBuffers& buffers, PreflightTrace& preflight, LookupTables& tables, size_t cycle) {
  impl::ExecContext ctx(preflight, tables, cycle);
  impl::MutableBufObj data(ctx, buffers.data);
  impl::GlobalBufObj global(ctx, buffers.global);
  step_Top(ctx, &data, &global);
}

__device__ void
stepAccum(AccumBuffers& buffers, PreflightTrace& preflight, LookupTables& tables, size_t cycle) {
  impl::ExecContext ctx(preflight, tables, cycle);
  impl::MutableBufObj data(ctx, buffers.data);
  impl::MutableBufObj accum(ctx, buffers.accum);
  impl::GlobalBufObj mix(ctx, buffers.mix);
  step_TopAccum(ctx, &accum, &data, &mix);
}

} // namespace risc0

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

extern "C" {

using namespace risc0;

const char* risc0_circuit_rv32im_v2_cuda_witgen(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* preflight,
                                                uint32_t lastCycle) {
  // LookupTables tables;
  // size_t split = preflight->tableSplitCycle;
  // try {
  //   switch (mode) {
  //   case kStepModeParallel: {
  //     auto begin1 = poolstl::iota_iter<uint32_t>(0);
  //     auto end1 = poolstl::iota_iter<uint32_t>(split);
  //     std::for_each(poolstl::par, begin1, end1, [&](uint32_t cycle) {
  //       stepExec(*buffers, *preflight, tables, cycle);
  //     });

  //     auto begin2 = poolstl::iota_iter<uint32_t>(split);
  //     auto end2 = poolstl::iota_iter<uint32_t>(lastCycle);
  //     std::for_each(poolstl::par, begin2, end2, [&](uint32_t cycle) {
  //       stepExec(*buffers, *preflight, tables, cycle);
  //     });
  //   } break;
  //   case kStepModeSeqForward:
  //     for (size_t cycle = 0; cycle < lastCycle; cycle++) {
  //       stepExec(*buffers, *preflight, tables, cycle);
  //     }
  //     break;
  //   case kStepModeSeqReverse: {
  //     for (size_t i = split; i-- > 0;) {
  //       // printf("stepExec: %zu\n", i);
  //       stepExec(*buffers, *preflight, tables, i);
  //     }
  //     for (size_t i = lastCycle; i-- > split;) {
  //       // printf("stepExec: %zu\n", i);
  //       stepExec(*buffers, *preflight, tables, i);
  //     }
  //   } break;
  //   }
  // } catch (const std::exception& err) {
  //   return strdup(err.what());
  // }
  return nullptr;
}

const char* risc0_circuit_rv32im_v2_cuda_accum(AccumBuffers* buffers,
                                               PreflightTrace* preflight,
                                               uint32_t lastCycle) {
  // try {
  //   LookupTables tables;
  //   for (size_t cycle = 0; cycle < lastCycle; cycle++) {
  //     stepAccum(*buffers, *preflight, tables, cycle);
  //   }
  // } catch (const std::exception& err) {
  //   return strdup(err.what());
  // }
  return nullptr;
}

} // extern "C"
