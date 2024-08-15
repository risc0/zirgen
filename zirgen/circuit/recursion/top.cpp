// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/recursion/top.h"

namespace zirgen::recursion {

TopImpl::TopImpl()
    : mux(Label("mux"),
          Labels({"micro_ops",
                  "macro_ops",
                  "poseidon2_load",
                  "poseidon2_full",
                  "poseidon2_partial",
                  "poseidon2_store",
                  "checked_bytes"}),
          code->select,
          code,
          womHeader) {}

void TopImpl::set() {
  mux->doMux([&](auto inner) { inner->set(code, code->writeAddr->get()); });
}

} // namespace zirgen::recursion
