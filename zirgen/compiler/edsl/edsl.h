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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/InitLLVM.h"

#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"
#include "zirgen/compiler/edsl/source_loc.h"

namespace zirgen {

using risc0::SourceLoc;

SourceLoc checkCurrentLoc(SourceLoc loc);

struct OverrideLocation {
  OverrideLocation(SourceLoc loc);
  ~OverrideLocation();
};

inline SourceLoc current(SourceLoc loc = SourceLoc::current()) {
  return checkCurrentLoc(loc);
}

class Val;
class DigestVal;
class Register;
class Buffer;
struct CaptureVal;
class Module;
class NondetGuard;
class IfGuard;
struct ConstructInfo;

class Val {
public:
  Val() = default;
  Val(mlir::Value value) : value(value) {}
  Val(uint64_t val, SourceLoc loc = current());
  Val(llvm::ArrayRef<uint64_t> coeffs, SourceLoc loc = current());
  Val(Register reg, SourceLoc loc = current());

  mlir::Value getValue() const { return value; }

private:
  mlir::Value value;
};

class Register {
  friend class Val;
  friend class Buffer;
  friend struct CaptureVal;

public:
  void operator=(const Register& x) = delete;
  void operator=(CaptureVal x);

private:
  Register(mlir::Value buf, llvm::StringRef ident = {}) : buf(buf), ident(ident) {}
  mlir::Value buf;
  std::string ident;
};

struct CaptureIdx {
  CaptureIdx(size_t idx, SourceLoc loc = current()) : idx(idx), loc(loc) {}
  size_t idx;

  SourceLoc loc;
  mlir::Location getLoc();
};

class Buffer {
  template <typename T, size_t N> friend class std::array;
  friend Module;

public:
  Buffer(mlir::Value buf) : buf(buf) {}
  size_t size() { return mlir::cast<Zll::BufferType>(buf.getType()).getSize(); }
  Val get(size_t idx, llvm::StringRef ident, SourceLoc loc = current());
  void set(size_t idx, Val x, llvm::StringRef ident, SourceLoc loc = current());
  void setDigest(size_t idx, DigestVal x, llvm::StringRef ident, SourceLoc loc = current());
  Buffer slice(size_t offset, size_t size, SourceLoc loc = current());
  Register getRegister(size_t idx, llvm::StringRef ident = {}, SourceLoc loc = current());
  Register operator[](CaptureIdx idx) { return getRegister(idx.idx, {}, idx.loc); }
  void labelLayout(llvm::ArrayRef<std::string> labels, SourceLoc loc = current()) const;
  mlir::Value getBuf() { return buf; }

private:
  Buffer() {}
  mlir::Value buf;
};

template <typename T> class Comp;

struct CaptureVal {
private:
  // This is some helper functions for some nasty SFINAE nonsense to allow
  // 'Comp<T>' types to be used in expressions *IF* the type T implements a
  // 'get()' methods that returns a Val
  template <class U> static auto try_get(U obj, int) -> decltype(obj.get()) { return obj.get(); }
  template <class U> static int try_get(U obj, long) { return 0; }

public:
  CaptureVal(uint64_t val, SourceLoc loc = current()) : val(val, loc), loc(loc) {}
  CaptureVal(Val val, SourceLoc loc = current()) : val(val), loc(loc) {}
  CaptureVal(Register val, SourceLoc loc = current()) : val(val, loc), loc(loc), ident(val.ident) {}

  template <typename T,
            typename std::enable_if<
                std::is_same<Val, decltype(try_get(*static_cast<T*>(nullptr), 0))>::value,
                int>::type = 0>
  CaptureVal(Comp<T> comp, SourceLoc loc = current()) : val(comp->get()), loc(loc) {}

  Val val;
  SourceLoc loc;
  mlir::Value getValue() { return val.getValue(); }
  mlir::Location getLoc();
  std::string ident;
};

enum class ArgumentType {
  BUFFER,
  IOP,
};

struct ArgumentInfo {
  ArgumentType type;
  Zll::BufferKind kind;
  size_t size;
  std::string name;
  size_t degree;
};

inline ArgumentInfo cbuf(size_t size, std::string name = {}, size_t degree = 1) {
  return ArgumentInfo{ArgumentType::BUFFER, Zll::BufferKind::Constant, size, name, degree};
}
inline ArgumentInfo cbuf(size_t size, size_t degree) {
  return cbuf(size, {}, degree);
}

inline ArgumentInfo mbuf(size_t size, std::string name = {}, size_t degree = 1) {
  return ArgumentInfo{ArgumentType::BUFFER, Zll::BufferKind::Mutable, size, name, degree};
}
inline ArgumentInfo mbuf(size_t size, size_t degree) {
  return mbuf(size, {}, degree);
}

inline ArgumentInfo gbuf(size_t size, std::string name = {}, size_t degree = 1) {
  return ArgumentInfo{ArgumentType::BUFFER, Zll::BufferKind::Global, size, name, degree};
}
inline ArgumentInfo gbuf(size_t size, size_t degree) {
  return gbuf(size, {}, degree);
}

inline ArgumentInfo ioparg(std::string name = {}) {
  return ArgumentInfo{ArgumentType::IOP, Zll::BufferKind::Mutable, 0, name, 0};
}

class Module {
  friend NondetGuard;
  friend IfGuard;

public:
  Module();
  template <size_t N, typename F>
  inline mlir::func::FuncOp addFunc(const std::string& name,
                                    std::array<ArgumentInfo, N> args,
                                    F func,
                                    SourceLoc loc = current()) {
    beginFunc(name, std::vector<ArgumentInfo>(args.begin(), args.end()), loc);
    std::array<mlir::Value, N> vargs;
    for (size_t i = 0; i < N; i++) {
      vargs[i] = builder.getBlock()->getArgument(i);
    }
    std::apply(func, vargs);
    auto f = endFunc(loc);

    for (size_t i = 0; i < N; i++) {
      std::string argName = args[i].name;
      if (!argName.empty()) {
        f.setArgAttr(i, "zirgen.argName", builder.getStringAttr(argName));
      }
    }
    return f;
  }

  void addOptimizationPasses(mlir::PassManager& pm);
  void optimize(size_t stageCount = 0);
  void setExternHandler(Zll::ExternHandler* handler);
  void runFunc(llvm::StringRef name,
               llvm::ArrayRef<Zll::Interpreter::BufferRef> bufs,
               size_t startCycle = 0,
               size_t cycleCount = 1);
  void runStage(size_t stage,
                llvm::StringRef name,
                llvm::ArrayRef<Zll::Interpreter::BufferRef> bufs,
                size_t startCycle = 0,
                size_t cycleCount = 1);

  void dump(bool debug = false);
  size_t computeMaxDegree(llvm::StringRef name);
  void dumpPoly(llvm::StringRef name);
  void dumpStage(size_t stage, bool debug = false);

  mlir::MLIRContext* getCtx() { return &ctx; }
  mlir::OpBuilder& getBuilder() { return builder; }
  mlir::ModuleOp getModule() { return *module; }

  static Module* getCurModule();

  void setPhases(mlir::func::FuncOp funcOp, llvm::ArrayRef<std::string> phases);
  void setProtocolInfo(ProtocolInfo info);

private:
  void beginFunc(const std::string& name, const std::vector<ArgumentInfo>& args, SourceLoc loc);
  mlir::func::FuncOp endFunc(SourceLoc loc);
  void pushIP(mlir::Block* block);
  void popIP();
  void runFunc(mlir::func::FuncOp func,
               llvm::ArrayRef<Zll::Interpreter::BufferRef> bufs,
               size_t startCycle,
               size_t cycleCount);

  mlir::MLIRContext ctx;
  using ModOwner = mlir::OwningOpRef<mlir::ModuleOp>;
  ModOwner module;
  Zll::ExternHandler* handler;
  std::vector<ModOwner> stages;
  mlir::OpBuilder builder;
  std::vector<mlir::OpBuilder::InsertPoint> ipStack;
};

Val operator+(CaptureVal a, CaptureVal b);
Val operator-(CaptureVal a, CaptureVal b);
Val operator-(CaptureVal a);
Val operator*(CaptureVal a, CaptureVal b);
Val operator&(CaptureVal a, CaptureVal b);
Val operator/(CaptureVal a, CaptureVal b);
Val inv(CaptureVal a);
Val raisepow(CaptureVal a, size_t exp);
Val isz(CaptureVal a);
std::pair<DigestVal, std::vector<Val>> hashCheckedBytes(CaptureVal pt, size_t count);

void eqz(CaptureVal a);
void eq(CaptureVal a, CaptureVal b);
void barrier(CaptureVal a);

void emitLayoutInternal(std::shared_ptr<ConstructInfo> info);
void transformLayout(mlir::MLIRContext* ctx,
                     std::shared_ptr<ConstructInfo> info,
                     llvm::DenseMap</*bufName=*/mlir::StringAttr, mlir::Type>& layoutType,
                     llvm::DenseMap</*bufName=*/mlir::StringAttr, mlir::Attribute>& layoutAttr);

std::vector<Val> doExtern(const std::string& name,
                          const std::string& extra,
                          size_t outSize,
                          llvm::ArrayRef<Val> in,
                          SourceLoc loc = current());

class NondetGuard {
public:
  NondetGuard(SourceLoc loc = current());
  ~NondetGuard();
  operator bool() { return true; }
};

class IfGuard {
public:
  IfGuard(Val cond, SourceLoc loc = current());
  ~IfGuard();
  operator bool() { return true; }
};

void beginBack(size_t dist, bool unchecked = false);
void endBack();

#define NONDET if (auto nondetGuard = NondetGuard())
#define IF(cond) if (auto ifGuard = IfGuard(cond))
namespace impl {
template <typename T> T endBackHelper(T x) {
  endBack();
  return x;
}
} // namespace impl
#define BACK(dist, expr) ::zirgen::impl::endBackHelper((beginBack(dist), expr))
#define UNCHECKED_BACK(dist, expr) ::zirgen::impl::endBackHelper((beginBack(dist, true), expr))

// Template override to write objects to log vector, default is to cast it as Val
template <typename T> struct LogPrep {
  static void toLogVec(std::vector<Val>& out, T x) { out.push_back(x); }
};

template <typename T, size_t N> struct LogPrep<std::array<T, N>> {
  static void toLogVec(std::vector<Val>& out, std::array<T, N> x) {
    for (size_t i = 0; i < N; i++) {
      LogPrep<T>::toLogVec(out, x[i]);
    }
  }
};

inline void toLogVec(std::vector<Val>& out) {}

template <typename T, typename... Args> void toLogVec(std::vector<Val>& out, T obj, Args... args) {
  LogPrep<T>::toLogVec(out, obj);
  toLogVec(out, args...);
}

template <typename... Args> void XLOG(std::string fmt, Args... args) {
  std::vector<Val> out;
  toLogVec(out, args...);
  doExtern("log", fmt, 0, out);
}

class DigestVal {
public:
  DigestVal() = default;
  DigestVal(mlir::Value value) : value(value) {}
  mlir::Value getValue() const { return value; }

private:
  mlir::Value value;
};

// Pretend the DigestVal is a plain old Val to pass it to the extern.
// TODO: This seems kludgy; do we want to convert the log vector to a vector of mlir::Value instead
// of Val?
template <> struct LogPrep<DigestVal> {
  static void toLogVec(std::vector<Val>& out, DigestVal x) { out.push_back(Val(x.getValue())); }
};

DigestVal hash(llvm::ArrayRef<Val> inputs, bool flip = false, SourceLoc loc = current());
DigestVal intoDigest(llvm::ArrayRef<Val> inputs,
                     Zll::DigestKind kind = Zll::DigestKind::Default,
                     SourceLoc loc = current());
std::vector<Val> fromDigest(DigestVal digest, size_t size, SourceLoc loc = current());
DigestVal fold(DigestVal lhs, DigestVal rhs, SourceLoc loc = current());
DigestVal taggedStruct(llvm::StringRef tag,
                       llvm::ArrayRef<DigestVal> digests,
                       llvm::ArrayRef<Val> vals,
                       SourceLoc loc = current());
DigestVal
taggedListCons(llvm::StringRef tag, DigestVal head, DigestVal tail, SourceLoc loc = current());
void assert_eq(DigestVal lhs, DigestVal rhs, SourceLoc loc = current());

class ReadIopVal {
public:
  ReadIopVal(mlir::Value value) : value(value) {}
  mlir::Value getValue() const { return value; }

  std::vector<Val> readBaseVals(size_t count, bool flip = false, SourceLoc loc = current());
  std::vector<Val> readExtVals(size_t count, bool flip = false, SourceLoc loc = current());

  // Read digests of the DigestKind::Default from the IOP stream.
  std::vector<DigestVal> readDigests(size_t count, SourceLoc loc = current());
  void commit(DigestVal digest, SourceLoc loc = current());
  Val rngBits(uint32_t bits, SourceLoc loc = current());
  Val rngBaseVal(SourceLoc loc = current());
  Val rngExtVal(SourceLoc loc = current());

private:
  mlir::Value value;
};

Val select(Val idx, llvm::ArrayRef<Val> in, SourceLoc loc = current());
DigestVal select(Val idx, llvm::ArrayRef<DigestVal> in, SourceLoc loc = current());

Val normalize(Val in, SourceLoc loc = current());

struct HashCheckedPublicOutput {
  DigestVal poseidon;
  DigestVal sha;
  std::vector<Val> vals;
};
HashCheckedPublicOutput hashCheckedBytesPublic(CaptureVal pt, size_t count);

// Registers common command line options that EDSL users will probably want.
void registerEdslCLOptions();

} // namespace zirgen
