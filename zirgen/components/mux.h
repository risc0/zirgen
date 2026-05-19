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

#include "zirgen/components/onehot.h"

/**
 * @file mux.h
 * @brief Template multiplexer component for selecting between circuit arms.
 *
 * MuxImpl<Arm0, Arm1, …> takes a OneHot<N> selector and instantiates one
 * component of each arm type.  Only the arm whose bit is set contributes
 * constraints; the others are skipped via CompContext::enterArm/leaveArm.
 *
 * Usage:
 * @code
 *   OneHot<2> sel = ...;
 *   Mux<Comp<ArmA>, Comp<ArmB>> m(sel, ...constructor_args...);
 *   m->at<0>()   // access the ArmA instance
 *   m->doMux([](auto& arm) { arm->doSomething(); });
 * @endcode
 *
 * Labels can optionally be provided to name the arms in the witness layout.
 *
 * Use the Mux<…> alias (Comp<MuxImpl<…>>) rather than MuxImpl directly.
 */

namespace zirgen {

namespace impl {

template <typename... InnerTypes> struct MuxData;

template <> struct MuxData<> {
  MuxData() = default;
  template <size_t size, typename... Args>
  void
  construct(OneHot<size> select, llvm::ArrayRef<const char*> labels, size_t which, Args... args) {}
  template <size_t size, typename Func> void apply(OneHot<size> select, size_t which, Func func) {}
};

template <typename First, typename... Rest> struct MuxData<Comp<First>, Rest...> {
  MuxData() = default;

  template <size_t size, typename... Args>
  void
  construct(OneHot<size> select, llvm::ArrayRef<const char*> labels, size_t which, Args... args) {
    // Push the arm number onto the construct path to let us easily identify different arms if
    // there's no labels.
    std::string label = labels.empty() ? std::to_string(which) : labels.front();
    CompContext::pushConstruct(label, typeid(First).name());
    CompContext::enterArm(select->atRaw(which));
    first = std::make_shared<First>(args...);

    // Register any callbacks for the component of the mux.
    for (auto kvp : first->callbackHelper) {
      CompContext::registerCallbackRaw(
          kvp.first, [first = this->first, method = kvp.second]() { (first.get()->*method)(); });
    }
    first->callbackHelper.clear();

    CompContext::leaveArm();
    CompContext::popConstruct();
    rest.construct(select, labels.empty() ? labels : labels.drop_front(), which + 1, args...);
  }

  template <size_t size, typename Func> void apply(OneHot<size> select, size_t which, Func func) {
    IF(select->at(which)) { func(first->asComp()); }
    rest.apply(select, which + 1, func);
  }

  std::shared_ptr<First> first;
  MuxData<Rest...> rest;
};

template <size_t idx, typename Inner> struct GetElement {
  static auto& get(Inner& inner) {
    return GetElement<idx - 1, decltype(inner.rest)>::get(inner.rest);
  }
};

template <typename Inner> struct GetElement<0, Inner> {
  static auto& get(Inner& inner) { return inner.first; }
};

} // namespace impl

template <typename... InnerTypes> class MuxImpl : public CompImpl<MuxImpl<InnerTypes...>> {
public:
  template <typename... Args>
  MuxImpl(std::vector<const char*> labels, OneHot<sizeof...(InnerTypes)> select, Args... args)
      : select(select) {
    CompContext::enterMux();
    assert(labels.size() == 0 || labels.size() == sizeof...(InnerTypes));
    inner.construct(select, llvm::ArrayRef(labels), 0, args...);
    CompContext::leaveMux();
  }
  template <typename... Args>
  MuxImpl(OneHot<sizeof...(InnerTypes)> select, Args... args)
      : MuxImpl(/*labels=*/Labels({}), select, args...) {}

  template <typename Func> void doMux(Func func) { inner.apply(select, 0, func); }

  template <size_t idx> auto& at() {
    return impl::GetElement<idx, impl::MuxData<InnerTypes...>>::get(inner);
  }

  OneHot<sizeof...(InnerTypes)> select;
  impl::MuxData<InnerTypes...> inner;
};

template <typename... InnerTypes> using Mux = Comp<MuxImpl<InnerTypes...>>;

} // namespace zirgen
