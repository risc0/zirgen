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

#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/components/fpext.h"
#include "zirgen/components/mux.h"

#include <deque>

namespace zirgen {

template <typename Element, typename Verifier> struct PlonkHeaderBase {
  PlonkHeaderBase(
      llvm::StringRef name,              // Name of this plonk group for use in extern calls
      llvm::StringRef finalizePhase,     // The phase in which all user txns are complete
      llvm::StringRef verifyPhase,       // The phase in which to do post-sort verification
      llvm::StringRef computeAccumPhase, // The phase in which to compute accumulation factors
      llvm::StringRef verifyAccumPhase)  // The phase in which to verify that accumulation was
                                         // calculated correctly
      : name(name)
      , finalizePhase(finalizePhase)
      , verifyPhase(verifyPhase)
      , computeAccumPhase(computeAccumPhase)
      , verifyAccumPhase(verifyAccumPhase)
      , element(Label("element"))
      , verifier(Label("verifier"))
      , accum(Label("accum"), "accum") {
    for (size_t i = 0; i < Element::InnerType::rawSize(); i++) {
      mixers.emplace_back(Label("mix", i), "mix");
    }
  }

  std::string name;
  std::string finalizePhase;
  std::string verifyPhase;
  std::string computeAccumPhase;
  std::string verifyAccumPhase;
  Element element;
  Verifier verifier;
  FpExtReg accum;
  std::vector<FpExtReg> mixers;
};

// PlonkInit should be activated on the first cycle.
template <typename Header> class PlonkInitImpl : public CompImpl<PlonkInitImpl<Header>> {
public:
  PlonkInitImpl(Header header) : header(header) {
    this->registerCallback(header->verifyPhase, &PlonkInitImpl::onVerify);
    this->registerCallback(header->verifyAccumPhase, &PlonkInitImpl::onVerifyAccum);
  }

  void onVerify() {
    header->element->setInit();
    header->verifier->setInit();
  }

  void onVerifyAccum() { header->accum->set(FpExt(Val(1))); }

  Header header;
};

template <typename Header> using PlonkInit = Comp<PlonkInitImpl<Header>>;

// When a plonk component doesn't actually do anything during a
// cycle, for instance the RAM component during a "setup" step, it can
// "pass" on filling in any data by using a PlonkPass.
template <typename Header> class PlonkPassImpl : public CompImpl<PlonkPassImpl<Header>> {
public:
  PlonkPassImpl(Header header) : header(header) {
    this->registerCallback(header->verifyPhase, &PlonkPassImpl::onVerify);
    this->registerCallback(header->computeAccumPhase, &PlonkPassImpl::onComputeAccum);
    this->registerCallback(header->verifyAccumPhase, &PlonkPassImpl::onVerifyAccum);
  }

  void onVerify() {
    auto elementVals = BACK(1, header->element->toVals());
    header->element->setFromVals(elementVals);

    if (header->verifier->hasState()) {
      auto verifierVals = BACK(1, header->verifier->toVals());
      header->verifier->setFromVals(verifierVals);
    }
  }

  void onComputeAccum() {
    NONDET {
      // Write the multiplicative identity.
      doExtern("plonkWriteAccum", header->name, 0, {1, 0, 0, 0});
    }
  }

  void onVerifyAccum() {
    NONDET {
      // This should read the same value as last time, since we wrote the multiplicative identity.
      std::vector<Val> externAccums = doExtern("plonkReadAccum", header->name, kExtSize, {});
      FpExt fpext = FpExt(std::array<Val, kExtSize>{
          externAccums[0], externAccums[1], externAccums[2], externAccums[3]});
      header->accum->set(fpext);
    }
    eq(header->accum->get(), BACK(1, header->accum->get()));
  }
  Header header;
};

template <typename Header> using PlonkPass = Comp<PlonkPassImpl<Header>>;

// PlonkFini should be activated on the last cycle.
template <typename Element, typename Header>
class PlonkFiniImpl : public CompImpl<PlonkFiniImpl<Element, Header>> {
public:
  PlonkFiniImpl(Header header_) : header(header_) {
    this->registerCallback(header->verifyPhase, &PlonkFiniImpl::onVerify);
    this->registerCallback(header->verifyAccumPhase, &PlonkFiniImpl::onVerifyAccum);
  }

  void onVerify() {
    element->setFini();
    header->verifier->verify(header->element, element, header->verifier, 1, 0);
  }

  void onVerifyAccum() { eq(BACK(1, header->accum->get()), FpExt(Val(1))); }

  Header header;
  Element element;
};

template <typename Element, typename Verifier>
using PlonkFini = Comp<PlonkFiniImpl<Element, Verifier>>;

// Construct a 'PLONK' argument.
// Basically each row/cycle considers a `count` of elements that the user
// specifies.  Each of these is later reordered into some canonical form,
// at which point we want to verify that 1) The reordered versions are
// a permuation of the originals 2) The reordered versions has some sort
// of local consistency (i.e. x' = x or x' = x + 1 for range check).
// We template on the type of element
template <typename Element, typename Verifier, typename Header>
class PlonkBodyImpl : public CompImpl<PlonkBodyImpl<Element, Verifier, Header>> {
public:
  PlonkBodyImpl(Header header_, size_t count, size_t maxDeg)
      : header(header_)
      , count(count)
      , maxDeg(maxDeg)
      , elemsPer(maxDeg - 1)
      , groups(ceilDiv(count, elemsPer)) {
    // Make components for
    // 1) The user specified data (called lhs here)
    // 2) The reordered data (called rhs)
    // 3) Any state verifiers need
    lhs.resize(count);
    // The final RHS element is the one in the header so that
    // different cycles with different counts are compatible
    rhs.resize(count - 1);
    rhs.push_back(header->element);
    verifiers.resize(count - 1);
    verifiers.push_back(header->verifier);
    // Now allocate any overflow for accumulation
    // Note, since we have one accum in the header, we only needs groups - 1
    // The final accum comes from the header
    for (size_t i = 0; i < groups - 1; i++) {
      accums.emplace_back("accum");
    }
    accums.emplace_back(header->accum);
    this->registerCallback(header->finalizePhase, &PlonkBodyImpl::onFinalize);
    this->registerCallback(header->verifyPhase, &PlonkBodyImpl::onVerify);
    this->registerCallback(header->computeAccumPhase, &PlonkBodyImpl::onComputeAccum);
    this->registerCallback(header->verifyAccumPhase, &PlonkBodyImpl::onVerifyAccum);
  }

  // Access a lhs element to use for whatever
  Element at(size_t idx) {
    assert(idx < count);
    return lhs[idx];
  }

  void onFinalize() {
    NONDET {
      for (size_t i = 0; i < count; i++) {
        doExtern("plonkWrite", header->name, 0, lhs[i]->toVals());
      }
    }
  }

  void onVerify() {
    // When we have computed all cycles once, we go back and load
    // the reordered RHS elements
    NONDET {
      for (size_t i = 0; i < count; i++) {
        auto loaded = doExtern("plonkRead", header->name, Element::InnerType::rawSize(), {});
        rhs[i]->setFromVals(loaded);
      }
    }
    // Then we verify they all are locally consistent.
    // We need to special case the 0'th element to read from the prior cycle
    Val checkDirty = header->getCheckDirty();
    verifiers[0]->verify(header->element, rhs[0], header->verifier, 1, checkDirty);
    for (size_t i = 1; i < count; i++) {
      verifiers[i]->verify(rhs[i - 1], rhs[i], verifiers[i - 1], 0, checkDirty);
    }
  }

  // Precompute lhs and rhs group products since we need them for both
  // computing and verifying.
  std::vector<FpExt> makeGroupProducts(std::vector<Element>& elems) {
    std::vector<FpExt> groupProds;
    for (size_t i = 0; i < groups; i++) {
      FpExt prod(Val(1));
      for (size_t j = 0; j < elemsPer; j++) {
        // Skip extra elements on final group as needed
        if (i * elemsPer + j >= count) {
          continue;
        }
        // Generate linear combinations over mixCoeffs
        auto elemVals = elems[i * elemsPer + j]->toVals();
        FpExt tot(Val(1));
        for (size_t k = 0; k < Element::InnerType::rawSize(); k++) {
          tot = tot + header->mixers[k] * FpExt(elemVals[k]);
        }
        prod = prod * tot;
      }
      groupProds.push_back(prod);
    }
    return groupProds;
  };

  // To do the PLONK argument, we want to accumulate all of the data,
  // multiplying by a mixed version of the LHS, and dividing by the RHS.  We
  // need to do this in 'batches' however due to degree limits.  Let's say we
  // wanted to multiply in 4 element (A, B, C, D) and divide out 4 elements (W,
  // X, Y, Z).  Let's say we have max degree 3, so we can't have any equations
  // of degree higher than 3.  Calling 'in' the original value, and 'out' the
  // computed value, we have:
  //
  // out = in * A * B * C * D * W^-1 * X^-1 * Y^-1 * Z^-1
  //
  // Via associativity and commutativity, we have:
  //
  // out = C * D * A * B * in * W^-1 * X^-1 * Y^-1 * Z^-1
  //
  // Now, we can introduce a tempory to get:
  //
  // tmp = A * B * in * W^-1 * X^-1 out = C * D * tmp * Y^-1 * Z^-1
  //
  // Finally, we multiply by WX and YZ to remove the inverses:
  //
  // W * X * tmp = C * D * in; Y * Z * out = A * B * tmp
  //
  // At this point, we have two degree 3 polynomials and a temporary which
  // together imply the original equation.  Basically we will
  // nondeterministically compute tmp + out given in and A, B, C, D, W, X, Y, Z,
  // and then verify via the equations about.
  void onComputeAccum() {
    NONDET {
      // Actually make the two groups
      auto lhsGroups = makeGroupProducts(lhs);
      auto rhsGroups = makeGroupProducts(rhs);

      // TODO: Change this to FpExt instead of Val when we support extension fields in Vals.
      FpExt prod = FpExt(Val(1));
      for (size_t i = 0; i < groups; i++) {
        prod = prod * lhsGroups[i] * inv(rhsGroups[i]);
      }
      doExtern("plonkWriteAccum", header->name, 0, prod.toVals());
    }
  }

  void onVerifyAccum() {
    auto lhsGroups = makeGroupProducts(lhs);
    auto rhsGroups = makeGroupProducts(rhs);

    NONDET {
      // TODO: Change this to use FpExt natively when we support extension fields in Vals.
      std::vector<Val> externAccums = doExtern("plonkReadAccum", header->name, kExtSize, {});
      // The saved value is from *after* we multiplied this row, so work backwards.
      FpExt prod = FpExt::fromVals(externAccums);
      for (int i = groups - 1; i >= 0; i--) {
        accums[i]->set(prod);
        prod = prod * inv(lhsGroups[i]) * rhsGroups[i];
      }
    }

    // Now verify that things are correct.
    FpExt cur = BACK(1, header->accum->get());
    for (size_t i = 0; i < groups; i++) {
      FpExt next = accums[i];
      eq(cur * lhsGroups[i], next * rhsGroups[i]);
      cur = next;
    }
  }

  Header header;
  size_t count;
  size_t maxDeg;
  size_t elemsPer;
  size_t groups;
  std::vector<Element> lhs;
  std::vector<Element> rhs;
  std::vector<Verifier> verifiers;
  std::vector<FpExtReg> accums;
};

template <typename Element, typename Verifier, typename Header>
using PlonkBody = Comp<PlonkBodyImpl<Element, Verifier, Header>>;

class PlonkExternHandler : public Zll::ExternHandler {
public:
  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const Zll::InterpVal*> args,
                                                size_t outCount) override;
  void sort(llvm::StringRef name);
  void calcPrefixProducts(Zll::ExtensionField f);

private:
  using FpVec = std::vector<uint64_t>;
  using FpMat = std::deque<FpVec>;
  std::map<std::string, FpMat> plonkRows;
  std::map<std::string, FpMat> plonkAccumRows;
};

} // namespace zirgen
