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

#include "zirgen/compiler/edsl/component.h"
#include "llvm/ADT/StringExtras.h"

#include <cxxabi.h>
#include <deque>
#include <set>
#include <typeinfo>

namespace zirgen {

namespace {

struct AllocData {
  std::map<std::string, std::deque<std::shared_ptr<AllocatableBase>>> pools;
};

struct AllocState {
  bool needsFlush;
  std::vector<AllocData> stack;
};

struct CallbackArm;

struct CallbackBlock {
  std::vector<std::function<void()>> normal;
  std::vector<std::unique_ptr<CallbackArm>> arms;
  void emit();
};

struct CallbackArm {
  CallbackArm(Buffer cond) : cond(cond) {}
  Buffer cond;
  CallbackBlock inner;
  void emit() {
    IF(cond[0]) { inner.emit(); }
  }
};

void CallbackBlock::emit() {
  for (auto& f : normal) {
    f();
  }
  for (auto& arm : arms) {
    arm->emit();
  }
}

struct CallbackStack {
  CallbackStack(llvm::StringRef name) : name(name) { stack.push_back(&root); }
  void emit() { root.emit(); }
  void push(Buffer cond) {
    stack.back()->arms.push_back(std::make_unique<CallbackArm>(cond));
    stack.push_back(&stack.back()->arms.back()->inner);
  }
  void pop() { stack.pop_back(); }
  std::string name;
  CallbackBlock root;
  std::vector<CallbackBlock*> stack;
};

struct CallbackState {
  std::deque<CallbackStack> phases;
};

AllocState* gAllocState = nullptr;
CallbackState* gCallbackState = nullptr;

} // namespace

void RegAlloc::saveLabel(llvm::StringRef label) {
  CompContext::saveLabel(buf, label);
}

void CompContext::init(std::vector<llvm::StringRef> phases) {
  assert(gAllocState == nullptr);
  assert(gCallbackState == nullptr);
  gAllocState = new AllocState;
  gAllocState->needsFlush = true;
  gAllocState->stack.emplace_back();
  gCallbackState = new CallbackState;
  gCallbackState->phases.emplace_back("_alloc_finalize");
  for (auto str : phases) {
    gCallbackState->phases.emplace_back(str);
  }
  gCallbackState->phases.emplace_back("_builtin_verify");
}

void CompContext::addBuffer(llvm::StringRef name, Buffer buf) {
  assert(gAllocState);
  auto& queue = gAllocState->stack[0].pools[name.str()];
  for (size_t i = 0; i < buf.size(); i++) {
    queue.emplace_back(std::make_shared<RegAlloc>(buf.slice(i, 1)));
  }

  auto bufVal = buf.getBuf();
  auto arg = mlir::cast<mlir::BlockArgument>(bufVal);
  auto argNum = arg.getArgNumber();
  auto funcOp = mlir::cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
  funcOp.setArgAttr(argNum, "zirgen.argName", mlir::StringAttr::get(bufVal.getContext(), name));
}

void CompContext::fini(Val ret) {
  assert(gAllocState);
  assert(gCallbackState);
  if (gAllocState->needsFlush) {
    registerCallbackRaw("_alloc_finalize", [finalPools = gAllocState->stack.back().pools]() {
      // This is a rather lame way to do this, as it only works when 'mux' is always
      // the final component of the parent component.
      NONDET {
        for (auto& kvp : finalPools) {
          for (auto& val : kvp.second) {
            val->finalize();
          }
        }
      }
    });
  }
  for (auto& phase : gCallbackState->phases) {
    if (phase.name[0] != '_') {
      barrier(ret);
      ret = 0;
    }
    phase.emit();
  }
  barrier(ret);
  delete gAllocState;
  gAllocState = nullptr;
  delete gCallbackState;
  gCallbackState = nullptr;
}

void CompContext::enterMux() {
  assert(gAllocState);
}

void CompContext::enterArm(Buffer cond) {
  assert(gAllocState);
  assert(gCallbackState);
  gAllocState->needsFlush = true;
  gAllocState->stack.push_back(gAllocState->stack.back());
  for (auto& phase : gCallbackState->phases) {
    phase.push(cond);
  }
}

void CompContext::leaveArm() {
  assert(gAllocState);
  assert(gCallbackState);
  if (gAllocState->needsFlush) {
    registerCallbackRaw("_alloc_finalize", [finalPools = gAllocState->stack.back().pools]() {
      // This is a rather lame way to do this, as it only works when 'mux' is always
      // the final component of the parent component.
      NONDET {
        for (auto& kvp : finalPools) {
          for (auto& val : kvp.second) {
            val->finalize();
          }
        }
      }
    });
  }
  for (auto& phase : gCallbackState->phases) {
    phase.pop();
  }
  gAllocState->stack.pop_back();
}

void CompContext::leaveMux() {
  assert(gAllocState);
  gAllocState->needsFlush = false;
}

namespace {

std::vector<std::shared_ptr<ConstructInfo>> curConstructs;

std::string demangle(std::string ident) {
  int status;
  char* demangled = abi::__cxa_demangle(ident.c_str(), 0, 0, &status);
  if (status == 0) {
    std::string result = demangled;
    free(demangled);
    return result;
  }
  return ident;
}

} // namespace

void CompContext::pushConstruct(llvm::StringRef ident, llvm::StringRef mangledTy, SourceLoc loc) {
  std::string demangledTy = demangle(mangledTy.str());

  // Generate something a bit more readable out of excessive things like this:
  // zirgen::MuxImpl<zirgen::Comp<zirgen::riscv::InitStepImpl>,
  // zirgen::Comp<zirgen::riscv::SetupStepImpl>, zirgen::Comp<zirgen::riscv::RamLoadStepImpl>,
  // zirgen::Comp<zirgen::riscv::ResetStepImpl>, zirgen::Comp<zirgen::riscv::BodyStepImpl>,
  // zirgen::Comp<zirgen::riscv::RamFiniStepImpl>, zirgen::Comp<zirgen::riscv::BytesFiniStepImpl> >,
  // zirgen::BytesSetupImpl,
  // zirgen::PlonkBodyImpl<zirgen::Comp<zirgen::impl::BytesPlonkElementImpl>,
  // zirgen::Comp<zirgen::impl::BytesPlonkVerifierImpl> >

  llvm::StringRef tyIdent = demangledTy;
  tyIdent.consume_front("zirgen::");
  tyIdent.consume_front("riscv::");
  tyIdent.consume_front("impl::");
  tyIdent.consume_front("rv32im_v1::");

  size_t tmplPos = tyIdent.find('<');
  if (tmplPos != llvm::StringRef::npos) {
    tyIdent = tyIdent.substr(0, tmplPos);
  }
  tyIdent.consume_back("Impl");
  // The type mess above will now be just "Mux".

  auto newConstruct = std::make_shared<ConstructInfo>();
  newConstruct->typeName = tyIdent;
  if (ident.empty()) {
    newConstruct->desc = tyIdent;
  } else {
    newConstruct->desc = (ident + "(" + tyIdent + ")").str();
  }

  if (!curConstructs.empty()) {
    auto prev = curConstructs.back();
    if (!ident.empty()) {
      if (prev->subcomponents.count(ident.str())) {
        llvm::errs() << "Warning: duplicate path in layout: " << getCurConstructPath() << "\n";
        ident = {};
      } else {
        prev->subcomponents.emplace(ident, newConstruct);
      }
    }
  }

  curConstructs.emplace_back(newConstruct);
}

std::shared_ptr<ConstructInfo> CompContext::getCurConstruct() {
  return curConstructs.back();
}

void CompContext::popConstruct() {
  auto cur = curConstructs.back();
  curConstructs.pop_back();
  if (cur->labels.size() == 1 && cur->subcomponents.empty() && !curConstructs.empty()) {
    // Component with only one thing inside it; replace the
    // subcomponent in the parent with just the buffer label.
    auto parent = curConstructs.back();

    for (auto [ident, info] : parent->subcomponents) {
      if (info != cur) {
        continue;
      }
      if (parent->labels.count(ident)) {
        continue;
      }
      parent->labels.emplace(ident, cur->labels.begin()->second);
      parent->subcomponents.erase(ident);
      break;
    }
  }
}

void CompContext::saveLabel(Buffer buf, llvm::StringRef label) {
  assert(!curConstructs.empty());
  auto cur = curConstructs.back();
  if (cur->labels.count(label.str())) {
    llvm::errs() << "Duplicate label for buffer: " << label << "\n";
    label = {};
  }

  if (label.empty()) {
    size_t idx = 0;
    while (cur->labels.count(std::to_string(idx))) {
      ++idx;
    }
    cur->labels.emplace(std::to_string(idx), buf);
  }
}

std::string CompContext::getCurConstructPath() {
  return llvm::join(llvm::map_range(curConstructs,
                                    [](const std::shared_ptr<ConstructInfo>& info) -> std::string {
                                      return info->desc;
                                    }),
                    "/");
}

void CompContext::addToPool(llvm::StringRef name, std::shared_ptr<AllocatableBase> item) {
  assert(gAllocState);
  gAllocState->stack.back().pools[name.str()].push_back(item);
}

std::shared_ptr<AllocatableBase> CompContext::allocateFromPoolBase(llvm::StringRef name) {
  assert(gAllocState);
  // llvm::errs() << getCurConstructPath() << ": " << name;
  auto& items = gAllocState->stack.back().pools[name.str()];
  if (items.empty()) {
    // llvm::errs() << "\n";
    throw std::runtime_error(("Out of space allocating from pool: " + name).str());
  }
  auto ptr = items.front();
  // llvm::errs() << ": " << ptr->id << "\n";
  items.pop_front();
  return ptr;
}

void CompContext::registerCallbackRaw(llvm::StringRef name, std::function<void()> func) {
  assert(gCallbackState);
  assert(gCallbackState);
  for (auto& phase : gCallbackState->phases) {
    if (phase.name == name) {
      phase.stack.back()->normal.push_back(func);
      return;
    }
  }
  throw std::runtime_error(("Unknown phase: " + name).str());
}

} // namespace zirgen
