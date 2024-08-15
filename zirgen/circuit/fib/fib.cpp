// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
  // clang-format off
  module.addFunc<5>(
    "fib",
    {cbuf(3), gbuf(1), mbuf(1), gbuf(1), mbuf(1)},
    [](Buffer control, Buffer out, Buffer data, Buffer mix, Buffer accum) {
    // Normal execution
    Register val = data[0];
    IF(control[0]) {
      val = 1;
    }
    IF(control[1]) {
      val = BACK(1, Val(val)) + BACK(2, Val(val));
    }
    IF(control[2]) {
      // TODO: Fix register equality via BufAccess
      out[0] = CaptureVal(val);
    }
    barrier(1 - control[2]);
    barrier(1);
    barrier(1);
    barrier(1);
    IF(control[0] + control[1] + control[2]) {
      accum[0] = 1;
    }
    barrier(1);
  });
  // clang-format on
  // module.dump();

  EmitCodeOptions opts = {
      .info = FIBONACCI_CIRCUIT_INFO,
      .stages = {{"exec", "verify_mem", "verify_bytes", "compute_accum", "verify_accum"}}};
  emitCode(module.getModule(), opts);
}
