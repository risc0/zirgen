// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/verify/verify.h"

namespace zirgen::verify {

struct MixState {
  Val tot;
  Val mul;
};

class CircuitBase : public CircuitInterface {
public:
  virtual void add_taps() = 0;
  virtual MixState
  poly_edsl(llvm::ArrayRef<Val> u, llvm::ArrayRef<Val> out, llvm::ArrayRef<Val> mix) const = 0;

  const Zll::TapSet& get_taps() const override { return tapSet; }

  Val compute_poly(llvm::ArrayRef<Val> u,
                   llvm::ArrayRef<Val> out,
                   llvm::ArrayRef<Val> accumMix,
                   Val polyMix) const override {
    this->polyMix = polyMix;
    auto ms = poly_edsl(u, out, accumMix);
    return ms.tot;
  }

  Val poly_edsl_const(llvm::ArrayRef<uint64_t> coeffs, const char* loc) const {
    assert(coeffs.size() == 1);
    // XLOG("CONST: %u", coeffs[0]);
    return Val(coeffs[0]);
  }
  Val poly_edsl_get(llvm::ArrayRef<Val> buf, size_t idx, const char* loc) const {
    // llvm::errs() << "@ " << loc << ", getting idx = " << idx << ", buf size = " << buf.size() <<
    // "\n"; XLOG("GET: %w", buf[idx]);
    return buf[idx];
  }
  Val poly_edsl_get_global(llvm::ArrayRef<Val> buf, size_t idx, const char* loc) const {
    // XLOG("GET_GLOBAL: [%u, 0, 0, 0]", buf[idx]);
    return buf[idx];
  }
  Val poly_edsl_add(Val a, Val b, const char* loc) const {
    // XLOG("ADD: %w", a + b);
    return a + b;
  }
  Val poly_edsl_sub(Val a, Val b, const char* loc) const {
    // XLOG("SUB: %w", a - b);
    return a - b;
  }
  Val poly_edsl_mul(Val a, Val b, const char* loc) const {
    // XLOG("MUL: %w", a * b);
    return a * b;
  }
  MixState poly_edsl_true(const char* loc) const {
    // XLOG("TRUE: 0");
    return MixState{0, 1};
  }
  MixState poly_edsl_and_eqz(MixState in, Val cond, const char* loc) const {
    // XLOG("AND_EQZ: %w, %w", in.tot + in.mul * cond, in.mul * polyMix);
    return MixState{in.tot + in.mul * cond, in.mul * polyMix};
  }
  MixState poly_edsl_and_cond(MixState in, Val cond, MixState inner, const char* loc) const {
    // XLOG("AND_COND: %w, %w", in.tot + cond * inner.tot * in.mul, in.mul * inner.mul);
    return MixState{in.tot + cond * inner.tot * in.mul, in.mul * inner.mul};
  }

protected:
  Zll::TapSet tapSet;
  mutable Val polyMix;
};

} // namespace zirgen::verify
