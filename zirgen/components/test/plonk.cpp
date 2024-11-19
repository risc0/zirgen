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

#include "zirgen/components/plonk.h"

#include "zirgen/components/reg.h"
#include "zirgen/components/test/test_runner.h"

#include <deque>
#include <gtest/gtest.h>

using namespace zirgen::Zll;

namespace zirgen {

namespace {

// An element that must be [0, 10] due to plookup
class SimplePlookupImpl : public CompImpl<SimplePlookupImpl> {
public:
  // Normal methods
  void set(Val val) { reg->set(val); }
  Val get() { return reg; }

  static constexpr size_t rawSize() { return 1; }
  std::vector<Val> toVals() { return {reg}; }
  void setFromVals(std::vector<Val> vals) { reg->set(vals[0]); }
  void setInit() { reg->set(Val(0)); }
  void setFini() { reg->set(Val(9)); }

private:
  Reg reg;
};

using SimplePlookup = Comp<SimplePlookupImpl>;

// Plookup callbacks
class SimplePlookupVerifierImpl : public CompImpl<SimplePlookupVerifierImpl> {
public:
  void verify(SimplePlookup a,
              SimplePlookup b,
              Comp<SimplePlookupVerifierImpl> prevVerifier,
              size_t back,
              Val checkDirty) {
    Val diff = b->get() - BACK(back, a->get());
    eqz(diff * (1 - diff));
  }

  void setInit() {}
};

using SimplePlookupVerifier = Comp<SimplePlookupVerifierImpl>;

struct SimpleHeaderImpl : public PlonkHeaderBase<SimplePlookup, SimplePlookupVerifier>,
                          public CompImpl<SimpleHeaderImpl> {
  SimpleHeaderImpl()
      : PlonkHeaderBase("SimplePlookup",
                        "_plonk_finalize",
                        "plonk_verify",
                        "plonk_compute_accum",
                        "plonk_verify_accum") {}
  Val getCheckDirty() { return 0; }
};

using SimpleHeader = Comp<SimpleHeaderImpl>;

namespace InstType {

constexpr size_t INIT = 0;
constexpr size_t SETUP = 1;
constexpr size_t CHECK = 2;
constexpr size_t FINI = 3;
constexpr size_t COUNT = 4;

} // namespace InstType

class InitStepImpl : public CompImpl<InitStepImpl> {
public:
  InitStepImpl(SimpleHeader header) : plonk(header) {}

  void set(OneHot<InstType::COUNT> code) {}

  PlonkInit<SimpleHeader> plonk;
};

using InitStep = Comp<InitStepImpl>;

class SetupStepImpl : public CompImpl<SetupStepImpl> {
public:
  SetupStepImpl(SimpleHeader header) : plonk(header, 5, 4) {}

  void set(OneHot<InstType::COUNT> code) {
    IF(1 - BACK(1, code->at(InstType::SETUP))) {
      for (size_t i = 0; i < 5; i++) {
        plonk->at(i)->set(i);
      }
    }
    IF(BACK(1, code->at(InstType::SETUP))) {
      for (size_t i = 0; i < 5; i++) {
        plonk->at(i)->set(5 + BACK(1, plonk->at(i)->get()));
      }
    }
  }

  PlonkBody<SimplePlookup, SimplePlookupVerifier, SimpleHeader> plonk;
};

using SetupStep = Comp<SetupStepImpl>;

class CheckStepImpl : public CompImpl<CheckStepImpl> {
public:
  CheckStepImpl(SimpleHeader header) : plonk(header, 5, 4) {}

  void set(OneHot<InstType::COUNT> code) {
    for (size_t i = 0; i < 5; i++) {
      plonk->at(i)->set(doExtern("getTestData", "", 1, {})[0]);
    }
  }

  PlonkBody<SimplePlookup, SimplePlookupVerifier, SimpleHeader> plonk;
};

using CheckStep = Comp<CheckStepImpl>;

class FiniStepImpl : public CompImpl<FiniStepImpl> {
public:
  FiniStepImpl(SimpleHeader header) : plonk(header) {}

  void set(OneHot<InstType::COUNT> code) {}

  PlonkFini<SimplePlookup, SimpleHeader> plonk;
};

using FiniStep = Comp<FiniStepImpl>;

class TopImpl : public CompImpl<TopImpl> {
public:
  TopImpl() : code("code", false), mux(code, header) {}

  void set() {
    mux->doMux([&](auto inner) { inner->set(code); });
  }

  OneHot<InstType::COUNT> code;
  SimpleHeader header;
  Mux<InitStep, SetupStep, CheckStep, FiniStep> mux;
};

using Top = Comp<TopImpl>;

class PlonkTestExternHandler : public PlonkExternHandler {
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
        ret.push_back(rand() % 11);
      } else {
        ret.push_back(rand() % 10);
      }
      return ret;
    }
    return PlonkExternHandler::doExtern(name, extra, args, outCount);
  }
  bool fault = false;
};

} // namespace

TEST(Plonk, Basic) {
  TestRunner runner(
      /*stages=*/4,
      /*codeSize=*/4,
      /*outSize=*/1,
      /*dataSize=*/10,
      /*mixSize=*/kExtSize,
      /*accumSize=*/2 * kExtSize,
      [](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        CompContext::init(
            {"_plonk_finalize", "plonk_verify", "plonk_compute_accum", "plonk_verify_accum"});

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
  size_t setupCount = 2;
  size_t checkCount = 10;
  size_t cycles = 1 + setupCount + checkCount + 1;
  std::vector<uint64_t> code(4 * cycles);
  code[0 * 4 + InstType::INIT] = 1;
  for (size_t i = 0; i < setupCount; i++) {
    code[(i + 1) * 4 + InstType::SETUP] = 1;
  }
  for (size_t i = 0; i < checkCount; i++) {
    code[(i + 1 + setupCount) * 4 + InstType::CHECK] = 1;
  }
  code[(cycles - 1) * 4 + InstType::FINI] = 1;

  // Make sure normal case works
  {
    PlonkTestExternHandler handler;
    runner.setup(&handler, code);
    runner.runStage(0);
    handler.sort("SimplePlookup");
    runner.runStage(1);
    runner.setMix();
    runner.runStage(2);
    handler.calcPrefixProducts(ExtensionField(kFieldPrimeDefault, kExtSize));
    runner.runStage(3);
  }
  // Inject an invalid value into the table and make sure we fail
  {
    PlonkTestExternHandler handler;
    handler.fault = true;
    runner.setup(&handler, code);
    runner.runStage(0);
    handler.sort("SimplePlookup");
    EXPECT_THROW(runner.runStage(1), std::runtime_error);
  }

  // Make the plonk permutation not work
  {
    PlonkTestExternHandler handler;
    runner.setup(&handler, code);
    runner.runStage(0);
    handler.sort("SimplePlookup");
    runner.runStage(1);
    runner.setMix();
    runner.data[53][0] = runner.data[53][0] + 1;
    runner.runStage(2);
    handler.calcPrefixProducts(ExtensionField(kFieldPrimeDefault, kExtSize));
    EXPECT_THROW(runner.runStage(3), std::runtime_error);
  }
}

} // namespace zirgen
