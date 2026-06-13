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

/// RAII guard that overrides the source location used for all eDSL ops created
/// within its lifetime. Useful for macros that emit ops on behalf of user code.
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

/// A field element value in the eDSL IR. Wraps an `mlir::Value` of `ValType`
/// and participates in arithmetic operator overloads. Can be constructed from
/// a constant, an extension-field polynomial (via coefficient array), or by
/// reading a `Register`.
class Val {
public:
  Val() = default;
  Val(mlir::Value value) : value(value) {}
  /// Construct a constant Val from an integer literal.
  Val(uint64_t val, SourceLoc loc = current());
  /// Construct an extension-field element from raw base-field coefficients.
  Val(llvm::ArrayRef<uint64_t> coeffs, SourceLoc loc = current());
  /// Load the value currently stored in a Register.
  Val(Register reg, SourceLoc loc = current());

  /// Return the underlying MLIR value.
  mlir::Value getValue() const { return value; }

private:
  mlir::Value value;
};

/// A mutable cell inside a Buffer. Obtain one via `Buffer::getRegister` or
/// `Buffer::operator[]`. Write to it with `operator=(CaptureVal)`.
class Register {
  friend class Val;
  friend class Buffer;
  friend struct CaptureVal;

public:
  void operator=(const Register& x) = delete;
  /// Emit a store op that writes `x` into this register's buffer slot.
  void operator=(CaptureVal x);

private:
  Register(mlir::Value buf, llvm::StringRef ident = {}) : buf(buf), ident(ident) {}
  mlir::Value buf;
  std::string ident;
};

/// An index into a Buffer, paired with a source location. Used by
/// `Buffer::operator[]` so that the source location of the subscript
/// expression is recorded rather than the location of the operator itself.
struct CaptureIdx {
  CaptureIdx(size_t idx, SourceLoc loc = current()) : idx(idx), loc(loc) {}
  size_t idx;

  SourceLoc loc;
  mlir::Location getLoc();
};

/// A contiguous slice of a ZK witness / constraint buffer. Buffers can be
/// constant (read-only), mutable (witness), or global. Use `get`/`set` for
/// individual element access and `slice` to create sub-buffers.
class Buffer {
  template <typename T, size_t N> friend class std::array;
  friend Module;

public:
  Buffer(mlir::Value buf) : buf(buf) {}
  /// Return the number of elements in this buffer.
  size_t size() { return mlir::cast<Zll::BufferType>(buf.getType()).getSize(); }
  /// Emit a load op reading element `idx` and return it as a Val.
  Val get(size_t idx, llvm::StringRef ident, SourceLoc loc = current());
  /// Emit a store op writing Val `x` to element `idx`.
  void set(size_t idx, Val x, llvm::StringRef ident, SourceLoc loc = current());
  /// Store a DigestVal (hash) at element `idx` (occupies multiple base-field slots).
  void setDigest(size_t idx, DigestVal x, llvm::StringRef ident, SourceLoc loc = current());
  /// Return a sub-buffer starting at `offset` with the given `size`.
  Buffer slice(size_t offset, size_t size, SourceLoc loc = current());
  /// Return a Register handle for element `idx` so it can be assigned via `=`.
  Register getRegister(size_t idx, llvm::StringRef ident = {}, SourceLoc loc = current());
  /// Subscript operator — returns a Register to allow assignment syntax.
  Register operator[](CaptureIdx idx) { return getRegister(idx.idx, {}, idx.loc); }
  /// Attach human-readable field names to this buffer's layout for debug output.
  void labelLayout(llvm::ArrayRef<std::string> labels, SourceLoc loc = current()) const;
  /// Return the underlying MLIR buffer value.
  mlir::Value getBuf() { return buf; }

private:
  Buffer() {}
  mlir::Value buf;
};

template <typename T> class Comp;

/// Implicit conversion wrapper used as the parameter type for arithmetic
/// operators and constraint-emitting helpers. Accepts `uint64_t`, `Val`,
/// `Register`, and any `Comp<T>` whose inner type exposes a `get()` returning
/// a `Val`. The source location is captured at each call site so that emitted
/// ops are attributed to the user's code rather than the operator overload.
struct CaptureVal {
private:
  // SFINAE helpers: detect whether T has a get() returning Val.
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

/// Discriminates between buffer and IOP arguments in a circuit function signature.
enum class ArgumentType {
  BUFFER,
  IOP,
};

/// Describes one argument (buffer or IOP) passed to a circuit function.
/// Use the `cbuf`, `mbuf`, `gbuf`, and `ioparg` helpers to construct these.
struct ArgumentInfo {
  ArgumentType type;
  Zll::BufferKind kind;
  size_t size;
  std::string name;
  size_t degree;
};

/// Declare a constant (read-only) buffer argument with the given size, optional
/// name, and polynomial degree bound.
inline ArgumentInfo cbuf(size_t size, std::string name = {}, size_t degree = 1) {
  return ArgumentInfo{ArgumentType::BUFFER, Zll::BufferKind::Constant, size, name, degree};
}
inline ArgumentInfo cbuf(size_t size, size_t degree) {
  return cbuf(size, {}, degree);
}

/// Declare a mutable (witness) buffer argument.
inline ArgumentInfo mbuf(size_t size, std::string name = {}, size_t degree = 1) {
  return ArgumentInfo{ArgumentType::BUFFER, Zll::BufferKind::Mutable, size, name, degree};
}
inline ArgumentInfo mbuf(size_t size, size_t degree) {
  return mbuf(size, {}, degree);
}

/// Declare a global buffer argument (shared across cycles).
inline ArgumentInfo gbuf(size_t size, std::string name = {}, size_t degree = 1) {
  return ArgumentInfo{ArgumentType::BUFFER, Zll::BufferKind::Global, size, name, degree};
}
inline ArgumentInfo gbuf(size_t size, size_t degree) {
  return gbuf(size, {}, degree);
}

/// Declare an IOP (Interactive Oracle Proof) stream argument.
inline ArgumentInfo ioparg(std::string name = {}) {
  return ArgumentInfo{ArgumentType::IOP, Zll::BufferKind::Mutable, 0, name, 0};
}

/// Top-level container for an eDSL circuit. Owns the `MLIRContext`, the
/// `ModuleOp`, and an `OpBuilder` used by all eDSL primitives. Exactly one
/// Module is active at a time (accessed via `getCurModule()`).
///
/// Typical usage:
/// ```cpp
/// Module m;
/// m.addFunc<2>("step", {mbuf(N), cbuf(M)}, [](mlir::Value wit, mlir::Value con) {
///   Buffer witness(wit), constant(con);
///   // ... eDSL ops ...
/// });
/// m.optimize();
/// ```
class Module {
  friend NondetGuard;
  friend IfGuard;

public:
  Module();
  /// Add a circuit function to the module. `args` describes the buffer/IOP
  /// parameters; `func` is a C++ lambda that is called immediately to emit the
  /// function body using the eDSL operators. Returns the resulting FuncOp.
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

  /// Populate `pm` with the standard eDSL optimization passes (inlining,
  /// constant folding, dead-code elimination, etc.).
  void addOptimizationPasses(mlir::PassManager& pm);
  /// Run the full optimization pipeline. `stageCount` > 0 splits the module
  /// into that many execution stages before optimizing.
  void optimize(size_t stageCount = 0);
  /// Set the handler invoked when an `extern` op is interpreted at runtime.
  void setExternHandler(Zll::ExternHandler* handler);
  /// Interpret a named function from this module against the given buffers.
  void runFunc(llvm::StringRef name,
               llvm::ArrayRef<Zll::Interpreter::BufferRef> bufs,
               size_t startCycle = 0,
               size_t cycleCount = 1);
  /// Interpret one execution stage of a multi-stage circuit.
  void runStage(size_t stage,
                llvm::StringRef name,
                llvm::ArrayRef<Zll::Interpreter::BufferRef> bufs,
                size_t startCycle = 0,
                size_t cycleCount = 1);

  /// Print the module IR to stderr.
  void dump(bool debug = false);
  /// Return the maximum polynomial degree of the named function.
  size_t computeMaxDegree(llvm::StringRef name);
  /// Print the polynomial constraints for the named function.
  void dumpPoly(llvm::StringRef name);
  /// Print the IR for a specific execution stage.
  void dumpStage(size_t stage, bool debug = false);

  mlir::MLIRContext* getCtx() { return &ctx; }
  mlir::OpBuilder& getBuilder() { return builder; }
  mlir::ModuleOp getModule() { return *module; }

  /// Return the Module currently being built (thread-local singleton).
  static Module* getCurModule();

  /// Annotate `funcOp` with the ordered list of execution phase names.
  void setPhases(mlir::func::FuncOp funcOp, llvm::ArrayRef<std::string> phases);
  /// Attach protocol metadata (e.g. hash function selection) to the module.
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

/// @name Arithmetic operators over field elements
/// @{
Val operator+(CaptureVal a, CaptureVal b);
Val operator-(CaptureVal a, CaptureVal b);
Val operator-(CaptureVal a);
Val operator*(CaptureVal a, CaptureVal b);
/// Bitwise AND — valid only over Boolean (0/1) values.
Val operator&(CaptureVal a, CaptureVal b);
Val operator/(CaptureVal a, CaptureVal b);
/// Multiplicative inverse in the field; undefined for zero.
Val inv(CaptureVal a);
/// Raise `a` to the integer power `exp` using repeated squaring.
Val raisepow(CaptureVal a, size_t exp);
/// Return 1 if `a` is zero, else 0.
Val isz(CaptureVal a);
/// Hash `count` bytes starting at buffer pointer `pt`; returns the digest
/// and the decoded byte values.
std::pair<DigestVal, std::vector<Val>> hashCheckedBytes(CaptureVal pt, size_t count);
/// @}

/// Emit a constraint asserting `a == 0`.
void eqz(CaptureVal a);
/// Emit a constraint asserting `a == b`.
void eq(CaptureVal a, CaptureVal b);
/// Emit a barrier constraint used for cycle-ordering.
void barrier(CaptureVal a);

void emitLayoutInternal(std::shared_ptr<ConstructInfo> info);
void transformLayout(mlir::MLIRContext* ctx,
                     std::shared_ptr<ConstructInfo> info,
                     llvm::DenseMap</*bufName=*/mlir::StringAttr, mlir::Type>& layoutType,
                     llvm::DenseMap</*bufName=*/mlir::StringAttr, mlir::Attribute>& layoutAttr);

/// Emit a call to an external (host-provided) function named `name`.
/// `extra` is an opaque string tag forwarded to the handler; `outSize` is the
/// number of return values; `in` are the input field elements.
std::vector<Val> doExtern(const std::string& name,
                          const std::string& extra,
                          size_t outSize,
                          llvm::ArrayRef<Val> in,
                          SourceLoc loc = current());

/// RAII guard for a non-deterministic region. Ops emitted inside a `NONDET`
/// block are not constrained and may use values computed outside the ZK proof.
class NondetGuard {
public:
  NondetGuard(SourceLoc loc = current());
  ~NondetGuard();
  operator bool() { return true; }
};

/// RAII guard for a conditional region. Ops emitted inside an `IF(cond)` block
/// are multiplied by the condition value, making them active only when `cond`
/// is non-zero.
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

/// A hash digest value in the eDSL IR. Wraps an `mlir::Value` of `DigestType`
/// and is produced/consumed by hashing primitives such as `hash`, `fold`, and
/// `taggedStruct`. Cannot participate directly in arithmetic; use `fromDigest`
/// to extract component field elements.
class DigestVal {
public:
  DigestVal() = default;
  DigestVal(mlir::Value value) : value(value) {}
  /// Return the underlying MLIR value.
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

/// Hash an array of field elements using the circuit's default hash function.
/// If `flip` is true, the input endianness is reversed.
DigestVal hash(llvm::ArrayRef<Val> inputs, bool flip = false, SourceLoc loc = current());
/// Pack field elements into a DigestVal of the specified kind.
DigestVal intoDigest(llvm::ArrayRef<Val> inputs,
                     Zll::DigestKind kind = Zll::DigestKind::Default,
                     SourceLoc loc = current());
/// Unpack `size` field elements out of a DigestVal.
std::vector<Val> fromDigest(DigestVal digest, size_t size, SourceLoc loc = current());
/// Combine two DigestVals into one using the hash-fold operation.
DigestVal fold(DigestVal lhs, DigestVal rhs, SourceLoc loc = current());
/// Construct a tagged-struct digest from a string tag, child digests, and
/// additional field-element inputs.
DigestVal taggedStruct(llvm::StringRef tag,
                       llvm::ArrayRef<DigestVal> digests,
                       llvm::ArrayRef<Val> vals,
                       SourceLoc loc = current());
/// Construct a tagged Merkle-list cons digest from a tag, head, and tail.
DigestVal
taggedListCons(llvm::StringRef tag, DigestVal head, DigestVal tail, SourceLoc loc = current());
/// Emit a constraint asserting that two DigestVals are equal.
void assert_eq(DigestVal lhs, DigestVal rhs, SourceLoc loc = current());

/// Handle to an IOP (Interactive Oracle Proof) stream argument. Used to read
/// prover-provided values and to mix verifier challenges into the Fiat-Shamir
/// transcript.
class ReadIopVal {
public:
  ReadIopVal(mlir::Value value) : value(value) {}
  mlir::Value getValue() const { return value; }

  /// Read `count` base-field elements from the prover.
  std::vector<Val> readBaseVals(size_t count, bool flip = false, SourceLoc loc = current());
  /// Read `count` extension-field elements from the prover.
  std::vector<Val> readExtVals(size_t count, bool flip = false, SourceLoc loc = current());

  /// Read `count` digests of `DigestKind::Default` from the IOP stream.
  std::vector<DigestVal> readDigests(size_t count, SourceLoc loc = current());
  /// Absorb a digest into the Fiat-Shamir transcript.
  void commit(DigestVal digest, SourceLoc loc = current());
  /// Sample `bits` bits of verifier randomness from the transcript.
  Val rngBits(uint32_t bits, SourceLoc loc = current());
  /// Sample one base-field verifier challenge.
  Val rngBaseVal(SourceLoc loc = current());
  /// Sample one extension-field verifier challenge.
  Val rngExtVal(SourceLoc loc = current());

private:
  mlir::Value value;
};

/// Multiplexer: return `in[idx]` without branches.
Val select(Val idx, llvm::ArrayRef<Val> in, SourceLoc loc = current());
/// Multiplexer over DigestVals: return `in[idx]`.
DigestVal select(Val idx, llvm::ArrayRef<DigestVal> in, SourceLoc loc = current());

/// Reduce a field element modulo the characteristic so that it fits in
/// canonical representation.
Val normalize(Val in, SourceLoc loc = current());

/// Outputs produced by `hashCheckedBytesPublic`: both hash variants plus the
/// decoded byte values as field elements.
struct HashCheckedPublicOutput {
  DigestVal poseidon;
  DigestVal sha;
  std::vector<Val> vals;
};
/// Like `hashCheckedBytes` but exposes both Poseidon and SHA digests as public
/// outputs for cross-validation in the recursion layer.
HashCheckedPublicOutput hashCheckedBytesPublic(CaptureVal pt, size_t count);

/// Register common command-line options for eDSL-based circuit binaries
/// (e.g. output directory, validity split count).
void registerEdslCLOptions();

} // namespace zirgen
