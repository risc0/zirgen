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
