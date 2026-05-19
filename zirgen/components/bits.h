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

#include "zirgen/components/reg.h"

namespace zirgen {

/**
 * @brief Tag used to construct a BitImpl that reuses an existing Reg.
 *
 * Pass ShareBitWithRegister{} to the two-argument BitImpl constructor when a
 * Reg has already been allocated and you only need the bit-constraint attached.
 */
struct ShareBitWithRegister {};

/**
 * @brief Circuit component constraining a witness register to {0, 1}.
 *
 * Allocates one register in the "data" pool and registers a verify callback
 * that enforces:  reg * (1 - reg) = 0
 *
 * Prefer the Bit alias (Comp<BitImpl>) over using BitImpl directly.
 */
struct BitImpl : CompImpl<BitImpl> {
  BitImpl(llvm::StringRef source = "data") : reg(Label("bit"), source) {
    this->registerCallback("_builtin_verify", &BitImpl::onVerify);
  }

  // Constructs from an existing register; caller is responsible for setting labels.
  BitImpl(ShareBitWithRegister, Reg reg) : reg(reg) {
    this->registerCallback("_builtin_verify", &BitImpl::onVerify);
  }

  // The following constraint checks that in a BitImpl, either reg = 0 or reg = 1
  void onVerify() { eqz(reg * (1 - reg)); }

  Val get() { return reg; }
  void set(Val val) { reg->set(val); }

  Reg reg;
};

using Bit = Comp<BitImpl>;

/**
 * @brief Allocator token for a two-bit (twit) register drawn from the shared pool.
 *
 * TwitPrepareImpl fills the pool; TwitImpl draws from it.
 */
struct TwitAlloc : public AllocatableBase {
  TwitAlloc(Buffer buf, size_t id = 0) : AllocatableBase(id), buf(buf) {}
  Buffer buf;
  void finalize() override { return buf[0] = 0; }
  void saveLabel(llvm::StringRef label) override { CompContext::saveLabel(buf, label); }
};

/**
 * @brief Pre-allocates `size` twit registers and adds them to the "twit" pool.
 *
 * Must be instantiated before any TwitImpl so that the pool is populated.
 * Each register is range-checked to {0, 1, 2, 3} in the verify callback.
 */
template <size_t size> struct TwitPrepareImpl : public CompImpl<TwitPrepareImpl<size>> {
  TwitPrepareImpl() {
    for (size_t i = 0; i < size; i++) {
      Buffer buf = CompContext::allocateFromPool<RegAlloc>("data")->buf;
      elems.push_back(buf);
      CompContext::addToPool("twit", std::make_shared<TwitAlloc>(buf));
    }
    this->registerCallback("_builtin_verify", &TwitPrepareImpl::onVerify);
  }

  void onVerify() {
    for (size_t i = 0; i < size; i++) {
      Val x = elems[i][0];
      // The following constraint checks that x is either 0, 1, 2, or 3.
      eqz(x * (1 - x) * (2 - x) * (3 - x));
    }
  }

  std::vector<Buffer> elems;
};

template <size_t size> using TwitPrepare = Comp<TwitPrepareImpl<size>>;

/**
 * @brief Circuit component holding a two-bit value constrained to {0, 1, 2, 3}.
 *
 * Draws one register from the "twit" pool populated by TwitPrepareImpl.
 * Prefer the Twit alias (Comp<TwitImpl>) over using TwitImpl directly.
 */
class TwitImpl : public CompImpl<TwitImpl> {
public:
  TwitImpl() : reg(Label("twit"), CompContext::allocateFromPool<TwitAlloc>("twit")->buf) {}
  Val get() { return reg; }
  void set(Val val) { reg->set(val); }

  Reg reg;
};

using Twit = Comp<TwitImpl>;

} // namespace zirgen
