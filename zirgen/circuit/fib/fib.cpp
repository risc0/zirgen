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

#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"
#include "zirgen/compiler/edsl/edsl.h"

using namespace zirgen;

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  registerCodegenCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "fib edsl");

  Module module;
  auto f = module.addFunc<5>( //
      "fib",
      {cbuf(3, "code"), gbuf(1, "out"), mbuf(1, "data"), gbuf(1, "mix"), mbuf(1, "accum")},
      [](Buffer control, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        // Normal execution
        Register val = data[0];
        IF(control[0]) { val = 1; }
        IF(control[1]) { val = BACK(1, Val(val)) + BACK(2, Val(val)); }
        IF(control[2]) {
          // TODO: Fix register equality via BufAccess
          out[0] = CaptureVal(val);
        }
        barrier(1 - control[2]);
        barrier(1);
        barrier(1);
        barrier(1);
        IF(control[0] + control[1] + control[2]) { accum[0] = 1; }
        barrier(1);
      });

  module.setPhases(
      f,
      /*phases=*/{{"exec", "verify_mem", "verify_bytes", "compute_accum", "verify_accum"}});
  module.setProtocolInfo(FIBONACCI_CIRCUIT_INFO);
  emitCode(module.getModule());
}
