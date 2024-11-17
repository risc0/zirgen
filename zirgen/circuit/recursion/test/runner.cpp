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

#include "zirgen/circuit/recursion/test/runner.h"

#include "zirgen/compiler/zkp/baby_bear.h"

#include <random>

size_t kMaxDegree = 5;

namespace zirgen::recursion {

using namespace Zll;

struct Runner::RecursionExternHandler : public WomExternHandler {
  RecursionExternHandler(llvm::ArrayRef<uint32_t> proof)
      : proof(proof.begin(), proof.end()), offset(0) {}

  std::vector<uint32_t> proof;
  size_t offset;
  std::deque<llvm::SmallVector<uint64_t, 4>> body;

  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const InterpVal*> args,
                                                size_t outCount) override {
    auto fpArgs = asFpArray(args);
    // TODO: this probably breaks log externs
    if (name == "readIOPHeader") {
      size_t count = fpArgs[0];
      size_t k = fpArgs[1] / 2;
      bool flip = fpArgs[1] % 2;
      auto itStart = proof.begin() + offset;
      if (k != 2) {
        std::vector<uint32_t> arr(itStart, itStart + k * count);
        offset += k * count;
        for (size_t i = 0; i < arr.size(); i++) {
          arr[i] = (uint64_t(arr[i]) * kBabyBearFromMontgomery) % kBabyBearP;
        }
        for (size_t i = 0; i < count; i++) {
          llvm::SmallVector<uint64_t, 4> poly(k);
          for (size_t j = 0; j < k; j++) {
            if (flip) {
              poly[j] = arr[i * k + j];
            } else {
              poly[j] = arr[j * count + i];
            }
          }
          body.push_back(poly);
        }
      } else {
        std::vector<uint32_t> arr;
        for (size_t i = 0; i < count; i++) {
          arr.push_back(proof[offset + i] & 0xffff);
          arr.push_back(proof[offset + i] >> 16);
        }
        offset += count;
        for (size_t i = 0; i < count; i++) {
          llvm::SmallVector<uint64_t, 4> poly(2);
          for (size_t j = 0; j < 2; j++) {
            poly[j] = uint64_t(arr[2 * i + j]);
          }
          body.push_back(poly);
        }
      }
      return std::vector<uint64_t>{};
    }
    if (name == "readIOPBody") {
      auto front = body.front();
      body.pop_front();
      front.resize(kBabyBearExtSize);
      if (fpArgs[2] == 1) {
        for (size_t i = 0; i < 4; i++) {
          front[i] = (front[i] * kBabyBearToMontgomery) % kBabyBearP;
        }
      }
      return std::vector<uint64_t>(front.begin(), front.end());
    }
    return WomExternHandler::doExtern(name, extra, args, outCount);
  }
};

Runner::Runner() {
  module.addFunc<5>(
      "recursion",
      {cbuf(kCodeSize), gbuf(kOutSize), mbuf(kDataSize), gbuf(kMixSize), mbuf(kAccumSize)},
      [](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        CompContext::init(
            {"_wom_finalize", "wom_verify", "do_nothing", "compute_accum", "verify_accum"});

        CompContext::addBuffer("code", code);
        CompContext::addBuffer("out", out);
        CompContext::addBuffer("data", data);
        CompContext::addBuffer("mix", mix);
        CompContext::addBuffer("accum", accum);

        Top top;
        top->set();

        CompContext::fini();
      });
  module.optimize();

  if (module.computeMaxDegree("recursion") > kMaxDegree) {
    llvm::errs() << "Degree exceeded max degree " << kMaxDegree << "\n";
    module.dumpPoly("recursion");
    throw(std::runtime_error("Maximum degree exceeeded"));
  }

  // module.dump();
  // module.dumpPoly("recursion");
  // exit(1);

  module.optimize(5);
  // module.dumpStage(0);
}

void Runner::setup(llvm::ArrayRef<uint32_t> code, llvm::ArrayRef<uint32_t> proof) {
  handler = std::make_unique<RecursionExternHandler>(proof);
  module.setExternHandler(handler.get());
  cycles = code.size() / kCodeSize;
  this->code = std::vector<Polynomial>(code.size());
  for (size_t i = 0; i < this->code.size(); i++) {
    this->code[i] = {code[i]};
  }
  out = std::vector<Polynomial>(kOutSize, Polynomial(1, kFieldInvalid));
  data = std::vector<Polynomial>(kDataSize * cycles, Polynomial(1, kFieldInvalid));
  mix = std::vector<Polynomial>(kMixSize, Polynomial(1, kFieldInvalid));
  accum = std::vector<Polynomial>(kAccumSize * cycles, Polynomial(1, kFieldInvalid));
  args = {this->code, out, data, mix, accum};
}

void Runner::run() {
  runStage(0);
  handler->sort("wom");
  runStage(1);
  runStage(2);
  setMix();
  runStage(3);
  handler->calcPrefixProducts(ExtensionField(kFieldPrimeDefault, kExtSize));
  runStage(4);
}

void Runner::done() {
  module.setExternHandler(nullptr);
  handler.reset();
  code.clear();
  out.clear();
  data.clear();
  mix.clear();
  accum.clear();
  args.clear();
}

void Runner::runStage(size_t stage) {
  module.runStage(stage, "recursion", args, 0, cycles);
}

void Runner::setMix() {
  // Not cryptographic randomness, this runner is only for testing
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, kFieldPrimeDefault - 1);
  for (size_t i = 0; i < mix.size(); i++) {
    mix[i] = {static_cast<uint64_t>(distribution(generator))};
  }
}

} // namespace zirgen::recursion
