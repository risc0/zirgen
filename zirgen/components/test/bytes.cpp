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

#include "zirgen/components/bytes.h"

#include "zirgen/components/test/test_runner.h"

#include <deque>
#include <gtest/gtest.h>

using namespace zirgen::Zll;

namespace zirgen {

namespace {

namespace InstType {
constexpr size_t INIT = 0;
constexpr size_t SETUP = 1;
constexpr size_t CHECK = 2;
constexpr size_t FINI = 3;
constexpr size_t COUNT = 4;
} // namespace InstType

class TopImpl;
using Top = Comp<TopImpl>;

class InitStepImpl : public CompImpl<InitStepImpl> {
public:
  InitStepImpl(BytesHeader header) : bytes(header) {}
  void set(Top top);
  BytesInit bytes;
};

using InitStep = Comp<InitStepImpl>;

class SetupStepImpl : public CompImpl<SetupStepImpl> {
public:
  SetupStepImpl(BytesHeader header) : bytes(header, 40) {}
  void set(Top top);
  BytesSetup bytes;
};

using SetupStep = Comp<SetupStepImpl>;

class CheckStepImpl : public CompImpl<CheckStepImpl> {
public:
  CheckStepImpl(BytesHeader header) : bytesPrepare(header, 20) { bytes.resize(20); }
  void set(Top top);
  BytesBody bytesPrepare;
  std::vector<ByteReg> bytes;
};

using CheckStep = Comp<CheckStepImpl>;

class FiniStepImpl : public CompImpl<FiniStepImpl> {
public:
  FiniStepImpl(BytesHeader header) : bytes(header) {}
  void set(Top top);
  BytesFini bytes;
};

using FiniStep = Comp<FiniStepImpl>;

class TopImpl : public CompImpl<TopImpl> {
public:
  TopImpl() : code("code", false), lastSetup("code"), mux(code, header) {}

  void set() {
    mux->doMux([&](auto inner) { inner->set(asComp()); });
  }

  // Code
  OneHot<InstType::COUNT> code;
  Reg lastSetup;

  // Data
  BytesHeader header;
  Mux<InitStep, SetupStep, CheckStep, FiniStep> mux;
};

void InitStepImpl::set(Top top) {}

void SetupStepImpl::set(Top top) {
  Val isFirst = 1 - BACK(1, top->code->at(InstType::SETUP));
  Val isLast = top->lastSetup;
  bytes->set(isFirst, isLast);
}

void CheckStepImpl::set(Top top) {
  for (size_t i = 0; i < 20; i++) {
    bytes[i]->setExact(doExtern("getTestData", "", 1, {})[0]);
  }
}

void FiniStepImpl::set(Top top) {}

class BytesTestExternHandler : public PlonkExternHandler {
public:
  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const InterpVal*> args,
                                                size_t outCount) override {
    if (name == "getTestData") {
      assert(outCount == 1);
      assert(args.size() == 0);
      std::vector<uint64_t> ret;
      if (fault) {
        ret.push_back(rand() % 300);
      } else {
        ret.push_back(rand() % 256);
      }
      return ret;
    }
    return PlonkExternHandler::doExtern(name, extra, args, outCount);
  }
  bool fault = false;
};

} // namespace

constexpr size_t kNumGroups = 4;

TEST(Bytes, Basic) {
  size_t codeSize = 5;
  TestRunner runner(
      /*stages=*/4,
      codeSize,
      /*outSize=*/1,
      /*dataSize=*/42,
      /*mixSize=*/2 * kExtSize,
      /*accumSize=*/kNumGroups * kExtSize,
      [](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        CompContext::init({"_bytes_finalize", "bytes_verify", "compute_accum", "verify_accum"});

        CompContext::addBuffer("code", code);
        CompContext::addBuffer("out", out);
        CompContext::addBuffer("data", data);
        CompContext::addBuffer("mix", mix);
        CompContext::addBuffer("accum", accum);

        Top top;
        top->set();

        CompContext::fini();
      });

  // Gernerate code
  size_t setupCount = BytesSetupImpl::setupCount(40);
  size_t checkCount = 40;
  size_t cycles = 1 + setupCount + checkCount + 1;
  std::vector<uint64_t> code(codeSize * cycles);
  code[0 * codeSize + InstType::INIT] = 1;
  for (size_t i = 0; i < setupCount; i++) {
    code[(i + 1) * codeSize + InstType::SETUP] = 1;
    if (i == setupCount - 1) {
      code[(i + 1) * codeSize + 4] = 1;
    }
  }
  for (size_t i = 0; i < checkCount; i++) {
    code[(i + 1 + setupCount) * codeSize + InstType::CHECK] = 1;
  }
  code[(cycles - 1) * codeSize + InstType::FINI] = 1;

  // Make sure normal case works
  {
    BytesTestExternHandler handler;
    runner.setup(&handler, code);
    runner.runStage(0);
    handler.sort("bytes");
    runner.runStage(1);
    runner.setMix();
    runner.runStage(2);
    handler.calcPrefixProducts(ExtensionField(kFieldPrimeDefault, kExtSize));
    runner.runStage(3);
  }
  // Inject an invalid value into the table and make sure we fail
  {
    BytesTestExternHandler handler;
    handler.fault = true;
    runner.setup(&handler, code);
    runner.runStage(0);
    handler.sort("bytes");
    EXPECT_THROW(runner.runStage(1), std::runtime_error);
  }
}

} // namespace zirgen
