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

#include "zirgen/components/bytes.h"
#include "zirgen/components/test/test_runner.h"

namespace zirgen {

struct TestWithBytes {
  template <typename Func>
  TestWithBytes(size_t userIn, size_t userOut, size_t bytesRequired, size_t regsRequired, Func func)
      : codeSize(userIn)
      , outSize(userOut)
      , dataSize(regsRequired + ceilDiv(bytesRequired, 2) * 4)
      , mixSize(2 * kExtSize)
      , accumSize(ceilDiv(ceilDiv(bytesRequired, 2), 3) * kExtSize)
      , runner(
            3,
            codeSize,
            outSize,
            dataSize,
            mixSize,
            accumSize,
            [bytesRequired, func](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
              CompContext::init(
                  {"_bytes_finalize", "bytes_verify", "compute_accum", "verify_accum"});

              CompContext::addBuffer("code", code);
              CompContext::addBuffer("out", out);
              CompContext::addBuffer("data", data);
              CompContext::addBuffer("mix", mix);
              CompContext::addBuffer("accum", accum);

              BytesHeader header;
              BytesBody body(header, bytesRequired);

              func();

              CompContext::fini();
            }) {}

  Zll::Interpreter::Buffer run(std::vector<uint64_t> in, bool dump = false) {
    PlonkExternHandler handler;
    runner.setup(&handler, in);
    runner.runStage(0);
    Zll::Interpreter::Buffer out = runner.out;
    if (dump) {
      runner.dump();
    }
    runner.done();
    return out;
  }

  size_t codeSize;
  size_t outSize;
  size_t dataSize;
  size_t mixSize;
  size_t accumSize;
  TestRunner runner;
};

} // namespace zirgen
