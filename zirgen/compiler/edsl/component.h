// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <utility>

#include "zirgen/compiler/edsl/edsl.h"

namespace zirgen {

// The base class for 'allocatable' things
struct AllocatableBase {
  AllocatableBase(size_t id) : id(id) {}
  virtual ~AllocatableBase() {}
  virtual void finalize() = 0;
  virtual void saveLabel(llvm::StringRef label) = 0;
  size_t id;
};

// A register that is available for allocation
struct RegAlloc : public AllocatableBase {
  RegAlloc(Buffer buf, size_t id = 0) : AllocatableBase(id), buf(buf) {}
  Buffer buf;
  void finalize() override {}
  void saveLabel(llvm::StringRef label) override;
};

// Information on a constructed component
struct ConstructInfo {
  std::string desc;
  std::map<std::string, Buffer> labels;
  std::string typeName;
  std::map<std::string /* ident */, std::shared_ptr<ConstructInfo>> subcomponents;
  SourceLoc loc;
};

// A context singleton used during component constructon
class CompContext {
public:
  // System API
  static void init(std::vector<llvm::StringRef> phases);
  static void addBuffer(llvm::StringRef name, Buffer buf);
  static void fini(Val ret = 0);

  // Mux API (called during construction)
  static void enterMux();
  static void enterArm(Buffer cond);
  static void leaveArm();
  static void leaveMux();

  // Debug info
  static void pushConstruct(llvm::StringRef ident, llvm::StringRef ty, SourceLoc loc = current());
  static void popConstruct();
  static std::shared_ptr<ConstructInfo> getCurConstruct();
  static void saveLabel(Buffer buf, llvm::StringRef label);

  // Emits a GlobalConstOp describing the layout of our module.
  template <typename Comp> static void emitLayout(Comp top) {
    emitLayoutInternal(top->constructInfo);
  }

  // Gets a unique identifier for this construct path.
  static std::string getCurConstructPath();

  // Allocation API
  static void addToPool(llvm::StringRef name, std::shared_ptr<AllocatableBase> item);
  static std::shared_ptr<AllocatableBase> allocateFromPoolBase(llvm::StringRef name);
  template <typename T> static std::shared_ptr<T> allocateFromPool(llvm::StringRef name) {
    auto downcast = std::dynamic_pointer_cast<T>(allocateFromPoolBase(name));
    assert(downcast);
    downcast->saveLabel(name);
    return downcast;
  }

  // Callback API
  static void registerCallbackRaw(llvm::StringRef name, std::function<void()> func);
};

// A label for a component in the layout.
class Label {
public:
  Label(SourceLoc loc = current()) : loc(loc) {}

  /* implicit */ Label(llvm::StringRef label, SourceLoc loc = current()) : label(label), loc(loc) {}

  // Numbered instance of something
  Label(llvm::StringRef label, size_t index, SourceLoc loc = current())
      : label((label + "[" + std::to_string(index) + "]").str()), loc(loc) {}

  // Convert to a singular label
  operator llvm::StringRef() const { return label; }
  operator const std::string&() const { return label; }

  // Label all elements of an array
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

// A smart-pointer style wrapper class for components
template <typename T> class Comp {
private:
  template <class U> static auto try_get(U obj, int) -> decltype(obj.get()) { return obj.get(); }
  template <class U> static int try_get(U obj, long) { return 0; }

public:
  Comp(std::shared_ptr<T> inner) : inner(inner) {}

  // Forward construction
  // If a label is specified, label this component in its parent's context.
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
  // If no label is specified, use a blank label.
  Comp() : Comp(Label{}) {}

  template <typename Arg,
            typename... Args,
            typename = std::enable_if_t<!std::is_same_v<Arg, Label>>>
  Comp(Arg arg, Args... args) : Comp(Label{}, arg, args...) {}

  // Dereference
  T& operator*() { return *inner; }
  T* operator->() { return inner.get(); }

  // Allow a component to act as a value in expressions if it support get()
  operator Val() { return inner->get(); }

  using InnerType = T;

private:
  std::shared_ptr<T> inner;
};

template <typename T> struct CompImpl : public std::enable_shared_from_this<T> {
  // C++ lifetime is very annoying.  We register callbacks during constructions, but we can't
  // use shared_from_this yet (since it's invalid until after construction), so we keep the list
  // of callbacks and then make them into lambdas that hold a copy of the object after
  // construction
  typedef void (T::*MethodPtr)();
  std::vector<std::pair<llvm::StringRef, MethodPtr>> callbackHelper;
  Comp<T> asComp() { return Comp<T>(static_cast<T*>(this)->shared_from_this()); }

  void registerCallback(llvm::StringRef name, MethodPtr method) {
    callbackHelper.emplace_back(name, method);
  }

  std::shared_ptr<ConstructInfo> constructInfo;
  // Saves this component as the given label in the current context,
  // for cases like allocatable components that might not be used
  // where they're defined.
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
