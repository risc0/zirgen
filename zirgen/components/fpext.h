// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/components/reg.h"

namespace zirgen {

#if GOLDILOCKS
constexpr size_t kExtSize = 2;
#else
constexpr size_t kExtSize = kBabyBearExtSize;
#endif

class CaptureFpExt;

class FpExt;

class FpExtRegImpl : public CompImpl<FpExtRegImpl> {
public:
  FpExtRegImpl(llvm::StringRef source = "data");
  FpExt get(mlir::Location loc = currentLoc());
  void set(CaptureFpExt rhs);
  Val elem(size_t i) { return elems[i]; }

private:
  std::vector<Reg> elems;
};

using FpExtReg = Comp<FpExtRegImpl>;

class FpExt {
public:
  FpExt() = default;
  FpExt(Val x, mlir::Location loc = currentLoc());
  FpExt(std::array<Val, kExtSize> elems, mlir::Location loc = currentLoc());
  FpExt(FpExtReg reg, mlir::Location loc = currentLoc());
  Val elem(size_t i) { return elems[i]; }
  std::array<Val, kExtSize> getElems() { return elems; }
  std::vector<Val> toVals() { return std::vector<Val>(elems.begin(), elems.end()); }
  static FpExt fromVals(llvm::ArrayRef<Val> vals, mlir::Location loc = currentLoc());

private:
  std::array<Val, kExtSize> elems;
};

class CaptureFpExt {
public:
  CaptureFpExt(FpExt ext, mlir::Location loc = currentLoc()) : ext(ext), loc(loc) {}
  CaptureFpExt(FpExtReg ext, mlir::Location loc = currentLoc()) : ext(ext, loc), loc(loc) {}
  FpExt ext;
  mlir::Location loc;
};

FpExt operator+(CaptureFpExt a, CaptureFpExt b);
FpExt operator-(CaptureFpExt a, CaptureFpExt b);
FpExt operator*(CaptureFpExt a, CaptureFpExt b);
void eq(CaptureFpExt a, CaptureFpExt b);
FpExt inv(CaptureFpExt a);

template <> struct LogPrep<FpExt> {
  static void toLogVec(std::vector<Val>& out, FpExt x) {
    for (size_t i = 0; i < kExtSize; i++) {
      out.push_back(x.elem(i));
    }
  }
};

} // namespace zirgen
