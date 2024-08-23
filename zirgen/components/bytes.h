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

#include "zirgen/components/plonk.h"

namespace zirgen {

namespace impl {

// Bytes are always made in pairs of two, which means the table needs 2^16 entries
// but the accumulations are twice as fast and take half the space.
struct BytesPlonkElementImpl : public CompImpl<BytesPlonkElementImpl> {
  static constexpr size_t rawSize() { return 2; }
  std::vector<Val> toVals();
  void setFromVals(std::vector<Val> vals);
  void setInit();
  void setFini();
  void setNext(std::vector<Val> vals);

  Reg high;
  Reg low;
};

using BytesPlonkElement = Comp<BytesPlonkElementImpl>;

struct BytesPlonkVerifierImpl : public CompImpl<BytesPlonkVerifierImpl> {
  void verify(BytesPlonkElement a,
              BytesPlonkElement b,
              Comp<BytesPlonkVerifierImpl> prevVerifier,
              size_t back,
              Val checkDirty);
  void setInit() {}
  bool hasState() { return false; }
  std::vector<Val> toVals() { return {}; }
  void setFromVals(llvm::ArrayRef<Val> vals) {}
};

using BytesPlonkVerifier = Comp<BytesPlonkVerifierImpl>;

struct ByteAlloc : public AllocatableBase {
  ByteAlloc(Reg reg, size_t id) : AllocatableBase(id), reg(reg) {}
  void finalize() override { reg->set(0); }
  void saveLabel(llvm::StringRef label) override { reg->saveLabel(label); }
  Reg reg;
};

} // namespace impl

struct BytesHeaderImpl : public PlonkHeaderBase<impl::BytesPlonkElement, impl::BytesPlonkVerifier>,
                         public CompImpl<BytesHeaderImpl> {
  BytesHeaderImpl();
  Val getCheckDirty() { return 0; }
};

using BytesHeader = Comp<BytesHeaderImpl>;
using BytesInit = PlonkInit<BytesHeader>;
using BytesPass = PlonkPass<BytesHeader>;
using BytesFini = PlonkFini<impl::BytesPlonkElement, BytesHeader>;

class ByteRegImpl : public CompImpl<ByteRegImpl> {
public:
  ByteRegImpl();
  // Gets the value (always a valid byte)
  Val get();
  // Sets the value to the lower 8 bits
  // Returns the rest of the input shifted down
  Val set(Val in);
  // Sets the value to a number which must be 8 bits
  void setExact(Val in);

  Reg reg;
};

using ByteReg = Comp<ByteRegImpl>;

class BytesSetupImpl : public CompImpl<BytesSetupImpl> {
public:
  // Calculate how many setup cycles we need for a given setup row size
  static size_t setupCount(size_t useRegs);
  // Construct a setup that uses up to 'useRegs' registers.
  BytesSetupImpl(BytesHeader header, size_t useRegs);

  // Actually set the setup values
  void set(Val isFirst, Val isLast);

private:
  size_t pairCount;
  PlonkBody<impl::BytesPlonkElement, impl::BytesPlonkVerifier, BytesHeader> body;
};

using BytesSetup = Comp<BytesSetupImpl>;

class BytesBodyImpl : public CompImpl<BytesBodyImpl> {
public:
  // Make a table capable of holding 'count' elements, which get added to allocator
  BytesBodyImpl(BytesHeader header, size_t count);

private:
  size_t pairCount;
  PlonkBody<impl::BytesPlonkElement, impl::BytesPlonkVerifier, BytesHeader> body;
};

using BytesBody = Comp<BytesBodyImpl>;

} // namespace zirgen
