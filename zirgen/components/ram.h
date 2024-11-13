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

#include "zirgen/components/u32.h"

namespace zirgen {

namespace MemoryOpType {

constexpr size_t kPageIo = 0;
constexpr size_t kRead = 1;
constexpr size_t kWrite = 2;

} // namespace MemoryOpType

namespace impl {

struct RamPlonkElementImpl : public CompImpl<RamPlonkElementImpl> {
  RamPlonkElementImpl()
      : addr(Label("addr")), cycle(Label("cycle")), memOp(Label("mem_op")), data(Label("data")) {}
  static constexpr size_t rawSize() { return 3 + U32RegImpl::rawSize(); }
  void setInit();
  void setFini();
  std::vector<Val> toVals();
  void setFromVals(llvm::ArrayRef<Val> vals);
  void setNOP();

  // In verification order
  Reg addr;
  Reg cycle;
  // The memory operation to perform, allowed values: see MemoryOpType.
  Reg memOp;
  U32Reg data;
};
using RamPlonkElement = Comp<RamPlonkElementImpl>;

struct RamPlonkVerifierImpl : public CompImpl<RamPlonkVerifierImpl> {
  RamPlonkVerifierImpl()
      : isNewAddr(Label("is_new_addr"))
      , dirty(Label("dirty"))
      , diff(Label("diff"))
      , extra(Label("extra")) {}
  void verify(RamPlonkElement a,
              RamPlonkElement b,
              Comp<RamPlonkVerifierImpl> prevVerifier,
              size_t back,
              Val checkDirty);
  void setInit();
  bool hasState() { return true; }
  std::vector<Val> toVals();
  void setFromVals(llvm::ArrayRef<Val> vals);

  Bit isNewAddr;
  Bit dirty;
  std::array<ByteReg, 3> diff;
  Twit extra;
};

using RamPlonkVerifier = Comp<RamPlonkVerifierImpl>;

struct RamAlloc : public AllocatableBase {
  RamAlloc(RamPlonkElement elem, size_t id = 0) : AllocatableBase(id), elem(elem) {}
  void finalize() override { elem->setNOP(); }
  void saveLabel(llvm::StringRef label) override { elem->saveLabel(label); }
  RamPlonkElement elem;
};

struct RamHeaderImpl : public PlonkHeaderBase<RamPlonkElement, RamPlonkVerifier>,
                       public CompImpl<RamHeaderImpl> {
  RamHeaderImpl(Reg checkDirty);
  Val getCheckDirty() { return checkDirty; }

  Reg checkDirty;
};

} // namespace impl

using RamHeader = Comp<impl::RamHeaderImpl>;
using RamInit = PlonkInit<RamHeader>;
using RamPass = PlonkPass<RamHeader>;
using RamFini = PlonkFini<impl::RamPlonkElement, RamHeader>;

class RamRegImpl : public CompImpl<RamRegImpl> {
public:
  RamRegImpl();

  U32Val doRead(Val cycle, Val addr, Val op = MemoryOpType::kRead);
  void doWrite(Val cycle, Val addr, U32Val data, Val op = MemoryOpType::kWrite);
  void doPreLoad(Val cycle, Val addr, U32Val data);
  void doNOP();

  void set(Val addr, Val cycle, Val memOp, U32Val data);
  Val addr();
  Val cycle();
  Val memOp();
  U32Val data();

private:
  impl::RamPlonkElement elem;
};

using RamReg = Comp<RamRegImpl>;

class RamBodyImpl : public CompImpl<RamBodyImpl> {
public:
  // Make a table capable of holding 'count' memory IOs
  RamBodyImpl(RamHeader header, size_t count);

private:
  PlonkBody<impl::RamPlonkElement, impl::RamPlonkVerifier, RamHeader> body;
};

using RamBody = Comp<RamBodyImpl>;

class RamExternHandler : public PlonkExternHandler {
public:
  RamExternHandler();
  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const Zll::InterpVal*> args,
                                                size_t outCount) override;

  uint32_t loadU32(uint32_t addr);
  void storeU32(uint32_t addr, uint32_t value);

protected:
  std::vector<uint32_t> image;
};

U32Val ramPeek(Val addr);

} // namespace zirgen
