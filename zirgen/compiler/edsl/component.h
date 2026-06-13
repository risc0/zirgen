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

#include <utility>

#include "zirgen/compiler/edsl/edsl.h"

namespace zirgen {

/// Abstract base for anything that can be placed in a CompContext allocation
/// pool. Concrete subtypes (e.g. `RegAlloc`) supply buffer storage that can be
/// claimed during circuit construction.
struct AllocatableBase {
  AllocatableBase(size_t id) : id(id) {}
  virtual ~AllocatableBase() {}
  /// Called once after the containing component's constructor returns.
  virtual void finalize() = 0;
  /// Record the pool name under which this item was allocated.
  virtual void saveLabel(llvm::StringRef label) = 0;
  size_t id;
};

/// An allocatable register backed by a Buffer element. Circuits that use
/// shared register pools (e.g. multiplexed arms) obtain these from
/// `CompContext::allocateFromPool`.
struct RegAlloc : public AllocatableBase {
  RegAlloc(Buffer buf, size_t id = 0) : AllocatableBase(id), buf(buf) {}
  Buffer buf;
  void finalize() override {}
  void saveLabel(llvm::StringRef label) override;
};

/// Debug / layout metadata collected while constructing a component tree.
/// Accumulates sub-component identity and source location so that the layout
/// IR (GlobalConstOp) can be emitted after construction.
struct ConstructInfo {
  std::string desc;
  std::map<std::string, Buffer> labels;
  std::string typeName;
  std::map<std::string /* ident */, std::shared_ptr<ConstructInfo>> subcomponents;
  SourceLoc loc;
};

/// Thread-local singleton that mediates component construction. Maintains the
/// stack of active `ConstructInfo` nodes, the buffer registry, and the
/// allocation pool. All `Comp<T>` constructors interact with this context.
class CompContext {
public:
  /// @name Lifecycle
  /// @{
  /// Initialize the context with the ordered list of execution phase names.
  static void init(std::vector<llvm::StringRef> phases);
  /// Register a named buffer so it is visible to all components.
  static void addBuffer(llvm::StringRef name, Buffer buf);
  /// Finalize the context; emits the top-level return value `ret`.
  static void fini(Val ret = 0);
  /// @}

  /// @name Mux support (called during component construction)
  /// @{
  static void enterMux();
  static void enterArm(Buffer cond);
  static void leaveArm();
  static void leaveMux();
  /// @}

  /// @name Debug / layout tracking
  /// @{
  /// Push a new ConstructInfo node for a component being built.
  static void pushConstruct(llvm::StringRef ident, llvm::StringRef ty, SourceLoc loc = current());
  /// Pop the current ConstructInfo node after the component constructor returns.
  static void popConstruct();
  /// Return the ConstructInfo for the component being built right now.
  static std::shared_ptr<ConstructInfo> getCurConstruct();
  static void saveLabel(Buffer buf, llvm::StringRef label);

  /// Emit a GlobalConstOp describing the layout of the top-level component.
  template <typename Comp> static void emitLayout(Comp top) {
    emitLayoutInternal(top->constructInfo);
  }

  /// Return a slash-separated string identifying the current construction path,
  /// suitable for use as a unique identifier in generated IR.
  static std::string getCurConstructPath();
  /// @}

  /// @name Allocation pool
  /// @{
  /// Add an allocatable item to the named pool.
  static void addToPool(llvm::StringRef name, std::shared_ptr<AllocatableBase> item);
  static std::shared_ptr<AllocatableBase> allocateFromPoolBase(llvm::StringRef name);
  /// Claim the next item of type T from the named pool.
  template <typename T> static std::shared_ptr<T> allocateFromPool(llvm::StringRef name) {
    auto downcast = std::dynamic_pointer_cast<T>(allocateFromPoolBase(name));
    assert(downcast);
    downcast->saveLabel(name);
    return downcast;
  }
  /// @}

  /// Register a named callback to be invoked by the execution engine at a
  /// specific phase. Callbacks are bound after component construction so that
  /// `shared_from_this` is safe to use inside them.
  static void registerCallbackRaw(llvm::StringRef name, std::function<void()> func);
};

/// A named label used to identify a component in the layout IR. Carries a
/// source location so that debug information is attributed to the declaration
/// site. Implicitly converts to `StringRef` for use as a `CompContext` label.
///
/// When cast to `std::array<Comp, N>` it expands to N indexed labels
/// (e.g. "reg[0]", "reg[1]", ...) suitable for initializing arrays of
/// sub-components.
class Label {
public:
  Label(SourceLoc loc = current()) : loc(loc) {}

  /* implicit */ Label(llvm::StringRef label, SourceLoc loc = current()) : label(label), loc(loc) {}

  /// Numbered instance: produces a label like "name[index]".
  Label(llvm::StringRef label, size_t index, SourceLoc loc = current())
      : label((label + "[" + std::to_string(index) + "]").str()), loc(loc) {}

  operator llvm::StringRef() const { return label; }
  operator const std::string&() const { return label; }

  /// Expand to an array of N indexed labels; used to initialize `std::array<Comp, N>`.
  template <typename Comp, size_t N> operator std::array<Comp, N>() const {
    auto seq = std::make_index_sequence<N>();
    return genArray<Comp, N>(seq);
  }

  SourceLoc getLoc() { return loc; }

private:
  template <typename Comp, size_t N, size_t... Is>
  std::array<Comp, N> genArray(std::index_sequence<Is...>) const {
    return std::array<Comp, N>{Label(label, Is)...};
  }

  std::string label;
  SourceLoc loc;
};

inline std::vector<const char*> Labels(std::initializer_list<const char*> labels) {
  return std::vector<const char*>(labels.begin(), labels.end());
}

/// Smart-pointer wrapper that owns a circuit component of type T. Combines
/// `shared_ptr` semantics with automatic `CompContext` registration so that
/// every component participates in layout tracking.
///
/// Construction pushes a `ConstructInfo` node, allocates T (forwarding all
/// args), wires pending callbacks via `shared_from_this`, then pops the node.
///
/// If T has a `get()` method returning `Val`, `Comp<T>` transparently
/// participates in arithmetic expressions.
template <typename T> class Comp {
private:
  template <class U> static auto try_get(U obj, int) -> decltype(obj.get()) { return obj.get(); }
  template <class U> static int try_get(U obj, long) { return 0; }

public:
  Comp(std::shared_ptr<T> inner) : inner(inner) {}

  /// Construct and register the component with the given label.
  template <typename... Args> Comp(Label label, Args... args) {
    CompContext::pushConstruct(label, typeid(T).name(), label.getLoc());
    inner = std::make_shared<T>(args...);
    for (auto kvp : inner->callbackHelper) {
      CompContext::registerCallbackRaw(
          kvp.first, [inner = this->inner, method = kvp.second]() { (inner.get()->*method)(); });
    }
    inner->callbackHelper.clear();
    inner->constructInfo = CompContext::getCurConstruct();
    inner->constructInfo->loc = label.getLoc();
    CompContext::popConstruct();
  }
  /// Construct without an explicit label (uses an empty label).
  Comp() : Comp(Label{}) {}

  template <typename Arg,
            typename... Args,
            typename = std::enable_if_t<!std::is_same_v<Arg, Label>>>
  Comp(Arg arg, Args... args) : Comp(Label{}, arg, args...) {}

  T& operator*() { return *inner; }
  T* operator->() { return inner.get(); }

  /// Allow this component to act as a Val in expressions when T::get() exists.
  operator Val() { return inner->get(); }

  using InnerType = T;

private:
  std::shared_ptr<T> inner;
};

/// CRTP base that every user-defined component type should inherit. Provides:
/// - `shared_from_this` support (via `enable_shared_from_this`)
/// - `registerCallback` for deferred callback wiring (safe after construction)
/// - `saveLabel` for inserting this component into its parent's layout tree
/// - `constructInfo` pointer populated by `Comp<T>` after construction
///
/// Callbacks registered during the constructor body are deferred because
/// `shared_from_this` is not valid until the shared_ptr is fully constructed.
/// `Comp<T>` handles the deferred wiring.
template <typename T> struct CompImpl : public std::enable_shared_from_this<T> {
  typedef void (T::*MethodPtr)();
  std::vector<std::pair<llvm::StringRef, MethodPtr>> callbackHelper;

  /// Wrap this object in a Comp<T> smart pointer (only valid post-construction).
  Comp<T> asComp() { return Comp<T>(static_cast<T*>(this)->shared_from_this()); }

  /// Queue a member function to be registered as a named callback. Must be
  /// called from the constructor; wiring happens in `Comp<T>`.
  void registerCallback(llvm::StringRef name, MethodPtr method) {
    callbackHelper.emplace_back(name, method);
  }

  std::shared_ptr<ConstructInfo> constructInfo;

  /// Insert this component under `label` in the current parent's
  /// ConstructInfo tree, for allocatable components created away from their
  /// ultimate use site.
  void saveLabel(llvm::StringRef label) {
    if (label.empty()) {
      return;
    }
    auto prev = CompContext::getCurConstruct();
    if (prev->subcomponents.count(label.str())) {
      llvm::errs() << "Duplicate label being copied to a new component: " << label << "\n";
      return;
    }
    prev->subcomponents.emplace(label, constructInfo);
  }
};

} // namespace zirgen
