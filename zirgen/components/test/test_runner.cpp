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

#include "zirgen/components/test/test_runner.h"

namespace zirgen {

using namespace Zll;

void TestRunner::setup(ExternHandler* handler, std::vector<uint64_t> code) {
  steps = code.size() / codeSize;
  this->code = std::vector<Polynomial>(code.size());
  for (size_t i = 0; i < this->code.size(); i++) {
    this->code[i] = {code[i]};
  }
  module.setExternHandler(handler);
  out = std::vector<Polynomial>(outSize, {kFieldInvalid});
  data = std::vector<Polynomial>(dataSize * steps, {kFieldInvalid});
  mix = std::vector<Polynomial>(mixSize, {kFieldInvalid});
  accum = std::vector<Polynomial>(accumSize * steps, {kFieldInvalid});
  args = {this->code, out, data, mix, accum};
}

void TestRunner::runStage(size_t stage) {
  module.runStage(stage, "test_func", args, 0, steps);
}

void TestRunner::setMix() {
  for (size_t i = 0; i < mix.size(); i++) {
    // For tests, don't bother with real randomness
    mix[i][0] = 1 + i * i;
  }
}

void TestRunner::dump() {
  for (size_t i = 0; i < steps; i++) {
    for (size_t j = 0; j < dataSize; j++) {
      Interpreter::PolynomialRef polynomial = data[i * dataSize + j];
      if (isInvalid(polynomial)) {
        llvm::errs() << "? ";
      } else {
        llvm::errs() << "{";
        for (size_t k = 0; k < polynomial.size(); k++) {
          llvm::errs() << polynomial[k] << ", ";
        }
        llvm::errs() << "} ";
      }
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n";
}

void TestRunner::done() {
  module.setExternHandler(nullptr);
  code.clear();
  out.clear();
  data.clear();
  mix.clear();
  accum.clear();
  args.clear();
}

} // namespace zirgen
