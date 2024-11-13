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

#include "zirgen/components/ram.h"

#include "zirgen/components/test/test_runner.h"

#include <deque>
#include <gtest/gtest.h>

using namespace zirgen::Zll;

namespace zirgen {

namespace {

namespace InstType {

constexpr size_t BYTES_INIT = 0;
constexpr size_t SETUP = 1;
constexpr size_t RAM_INIT = 2;
constexpr size_t CHECK = 3;
constexpr size_t RAM_FINI = 4;
constexpr size_t BYTES_FINI = 5;
constexpr size_t COUNT = 6;

} // namespace InstType

struct TopImpl;
using Top = Comp<TopImpl>;

struct BytesInitStepImpl : public CompImpl<BytesInitStepImpl> {
  BytesInitStepImpl(BytesHeader bytesHeader) : bytes(bytesHeader) {}
  void set(Top top) {}

  BytesInit bytes;
};

using BytesInitStep = Comp<BytesInitStepImpl>;

struct SetupStepImpl : public CompImpl<SetupStepImpl> {
  SetupStepImpl(BytesHeader bytesHeader) : bytes(bytesHeader, 40) {}
  void set(Top top);

  BytesSetup bytes;
};

using SetupStep = Comp<SetupStepImpl>;

struct RamInitStepImpl : public CompImpl<RamInitStepImpl> {
  RamInitStepImpl(BytesHeader bytesHeader)
      : bytes(bytesHeader, 3), ramHeader(checkDirty), ram(ramHeader) {}

  void set(Top top) { checkDirty->set(0); }

  Reg checkDirty;
  BytesBody bytes;
  RamHeader ramHeader;
  RamInit ram;
};

using RamInitStep = Comp<RamInitStepImpl>;

struct CheckStepImpl : public CompImpl<CheckStepImpl> {
  CheckStepImpl(BytesHeader bytesHeader)
      : bytes(bytesHeader, 3), ramHeader(checkDirty), ram(ramHeader, 1) {}
  void set(Top top);

  Reg checkDirty;
  BytesBody bytes;
  RamHeader ramHeader;
  RamBody ram;
  RamReg io;
};

using CheckStep = Comp<CheckStepImpl>;

struct RamFiniStepImpl : public CompImpl<RamFiniStepImpl> {
  RamFiniStepImpl(BytesHeader bytesHeader)
      : bytes(bytesHeader, 3), ramHeader(checkDirty), ram(ramHeader) {}
  void set(Top top) { checkDirty->set(0); }

  Reg checkDirty;
  BytesBody bytes;
  RamHeader ramHeader;
  RamFini ram;
};

using RamFiniStep = Comp<RamFiniStepImpl>;

struct BytesFiniStepImpl : public CompImpl<BytesFiniStepImpl> {
  BytesFiniStepImpl(BytesHeader bytesHeader) : bytes(bytesHeader) {}
  void set(Top top) {}
  BytesFini bytes;
};

using BytesFiniStep = Comp<BytesFiniStepImpl>;

struct TopImpl : public CompImpl<TopImpl> {
  TopImpl() : code("code", false), lastSetup("code"), mux(code, bytesHeader) {}

  void set() {
    mux->doMux([&](auto inner) { inner->set(asComp()); });
  }

  // Code
  OneHot<InstType::COUNT> code;
  Reg lastSetup;

  // Data
  TwitPrepareImpl<1> twits;
  BytesHeader bytesHeader;
  Mux<BytesInitStep, SetupStep, RamInitStep, CheckStep, RamFiniStep, BytesFiniStep> mux;
};

void SetupStepImpl::set(Top top) {
  Val isFirst = 1 - BACK(1, top->code->at(InstType::SETUP));
  Val isLast = top->lastSetup;
  bytes->set(isFirst, isLast);
}

void CheckStepImpl::set(Top top) {
  NONDET {
    auto vals = doExtern("getTestData", "", 4, {});
    Val cycle = vals[0];
    Val addr = vals[1];
    Val data = vals[2];
    Val memOp = vals[3];
    U32Val dataU32 = {data, data + 1, data + 2, data + 3};
    XLOG("cycle: %u, addr: %10x, data: %x, memOp: %u, dataU32: %w",
         cycle,
         addr,
         data,
         memOp,
         dataU32);
    io->set(addr, cycle, memOp, dataU32);
  }
  checkDirty->set(0);
}

class RamTestExternHandler : public RamExternHandler {
public:
  RamTestExternHandler(std::deque<std::vector<uint64_t>> data) : data(data) {}

  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const InterpVal*> args,
                                                size_t outCount) override {
    if (name == "getTestData") {
      assert(outCount == 4);
      if (data.size() == 0) {
        return std::vector<uint64_t>{0, 0, 0, 0};
      }
      auto ret = data.front();
      data.pop_front();
      return ret;
    }
    return PlonkExternHandler::doExtern(name, extra, args, outCount);
  }

  std::deque<std::vector<uint64_t>> data;
};

} // namespace

void runTest(std::deque<std::vector<uint64_t>> testData) {
  size_t codeSize = InstType::COUNT + 1;
  TestRunner runner(
      /*stages=*/5,
      codeSize,
      /*outSize=*/1,
      /*dataSize=*/100,
      /*mixSize=*/9 * kExtSize,
      /*accumSize=*/6 * kExtSize,
      [](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        CompContext::init({"_ram_finalize",
                           "ram_verify",
                           "_bytes_finalize",
                           "bytes_verify",
                           "compute_accum",
                           "verify_accum"});

        CompContext::addBuffer("code", code);
        CompContext::addBuffer("out", out);
        CompContext::addBuffer("data", data);
        CompContext::addBuffer("mix", mix);
        CompContext::addBuffer("accum", accum);

        Top top;
        top->set();

        CompContext::fini();
      });

  // runner.module.dumpStage(true);

  // Gernerate code
  size_t setupCount = BytesSetupImpl::setupCount(40);
  size_t checkCount = testData.size();
  size_t cycles = 1 + setupCount + 1 + checkCount + 2;
  std::vector<uint64_t> code(codeSize * cycles);
  code[0 * codeSize + InstType::BYTES_INIT] = 1;
  for (size_t i = 0; i < setupCount; i++) {
    code[(i + 1) * codeSize + InstType::SETUP] = 1;
    if (i == setupCount - 1) {
      code[(i + 1) * codeSize + InstType::COUNT] = 1;
    }
  }
  code[(setupCount + 1) * codeSize + InstType::RAM_INIT] = 1;
  for (size_t i = 0; i < checkCount; i++) {
    code[(i + setupCount + 2) * codeSize + InstType::CHECK] = 1;
  }
  code[(cycles - 2) * codeSize + InstType::RAM_FINI] = 1;
  code[(cycles - 1) * codeSize + InstType::BYTES_FINI] = 1;

  RamTestExternHandler handler(testData);
  runner.setup(&handler, code);
  runner.runStage(0);
  handler.sort("ram");
  runner.runStage(1);
  handler.sort("bytes");
  runner.runStage(2);
  runner.setMix();
  runner.runStage(3);
  handler.calcPrefixProducts(ExtensionField(kFieldPrimeDefault, kExtSize));
  runner.runStage(4);
  runner.done();
}

// Check a normal write/read works
TEST(Ram, Basic) {
  runTest({
      {0, 3, 0x17, MemoryOpType::kPageIo},
      {1, 3, 0x17, MemoryOpType::kRead},
      {2, 3, 0x00, MemoryOpType::kWrite},
  });
}

// Check pageIn followed by invalid read fails
TEST(Ram, InvalidRead) {
  EXPECT_THROW(runTest({
                   {0, 3, 0x17, MemoryOpType::kPageIo},
                   {1, 5, 0x13, MemoryOpType::kPageIo},
                   {2, 3, 0x00, MemoryOpType::kRead},
                   {3, 5, 0x00, MemoryOpType::kRead},
                   {4, 3, 0x00, MemoryOpType::kWrite},
                   {5, 5, 0x00, MemoryOpType::kWrite},
               }),
               std::runtime_error);
}

// Check a normal write/read works
TEST(Ram, ReadWrite) {
  runTest({
      {0, 3, 0x00, MemoryOpType::kPageIo},
      {1, 5, 0x00, MemoryOpType::kPageIo},
      {2, 3, 0x17, MemoryOpType::kWrite},
      {3, 5, 0x13, MemoryOpType::kWrite},
      {4, 3, 0x17, MemoryOpType::kRead},
      {5, 5, 0x13, MemoryOpType::kRead},
      {6, 3, 0x17, MemoryOpType::kPageIo},
      {7, 5, 0x13, MemoryOpType::kPageIo},
  });
}

// Check an invalid write/read fails
TEST(Ram, InvalidWrite) {
  EXPECT_THROW(runTest({
                   {0, 3, 0x00, MemoryOpType::kPageIo},
                   {1, 5, 0x00, MemoryOpType::kPageIo},
                   {2, 3, 0x17, MemoryOpType::kWrite},
                   {3, 5, 0x13, MemoryOpType::kWrite},
                   {4, 3, 0x17, MemoryOpType::kRead},
                   {5, 5, 0x12, MemoryOpType::kRead},
               }),
               std::runtime_error);
}

} // namespace zirgen
