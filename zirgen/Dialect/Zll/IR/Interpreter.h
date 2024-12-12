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

#include <deque>
#include <random>

#include "mlir/IR/AsmState.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/compiler/zkp/hash.h"
#include "zirgen/compiler/zkp/read_iop.h"

namespace zirgen::Zll {

// A value used by the interpreter that is allocation-free for simple types.
class InterpVal {
public:
  using Polynomial = llvm::SmallVector<uint64_t, 4>;
  using PolynomialRef = llvm::ArrayRef<uint64_t>;
  using BufferRef = llvm::MutableArrayRef<Polynomial>;

  template <typename T = mlir::Attribute> T getAttr(mlir::MLIRContext* ctx = nullptr) const {
    if (std::holds_alternative<Polynomial>(storage)) {
      // Allow field elements to be accessed as attributes.
      assert(ctx && "Context must be specified to construct a PolynomialAttr");
      return llvm::cast<T>(PolynomialAttr::get(ctx, std::get<Polynomial>(storage)));
    }
    return llvm::cast<T>(std::get<mlir::Attribute>(storage));
  }
  mlir::MutableArrayRef<Polynomial> getBuf() const { return std::get<BufferRef>(storage); }
  PolynomialRef getVal() const { return std::get<Polynomial>(storage); }
  const Digest& getDigest() const { return std::get<Digest>(storage); }
  ReadIop* getIop() const { return std::get<ReadIop*>(storage); }

  uint64_t getBaseFieldVal() const {
    auto poly = getVal();
    assert(poly.size() == 1 && "a base field element must have extension degree 1");
    return poly[0];
  }

  InterpVal() = default;
  InterpVal(const InterpVal&) = delete;

  void setAttr(mlir::Attribute attr) {
    if (auto polynomialAttr = llvm::dyn_cast<PolynomialAttr>(attr)) {
      // Store field elements as SmallVectors instead of attributes.
      storage.emplace<Polynomial>(polynomialAttr.asArrayRef());
    } else {
      storage.emplace<mlir::Attribute>(attr);
    }
  }
  void setBuf(BufferRef buffer) { storage.emplace<BufferRef>(buffer); }
  void setDigest(const Digest& digest) { storage.emplace<Digest>(digest); }
  void setIop(ReadIop* iop) { storage.emplace<ReadIop*>(iop); }

  // Automatically convert vals from common types
  void setVal(PolynomialRef elems) { storage.emplace<Polynomial>(elems); }
  void setVal(uint64_t val) { storage.emplace<Polynomial>({val}); }
  void setVal(uint32_t val) { setVal(static_cast<uint64_t>(val)); }
  void setVal(int val) { setVal(static_cast<uint64_t>(val)); }

  // Copy from another interpreter value, regardless of type.
  void setInterpVal(const InterpVal* rhs) { storage = rhs->storage; }

  void print(llvm::raw_ostream& os) const;
  void print(llvm::raw_ostream& os, mlir::AsmState& asmState) const;

  bool operator==(const InterpVal& other) const { return storage == other.storage; }
  bool operator!=(const InterpVal& other) const { return !(*this == other); }

private:
  std::variant<mlir::Attribute, BufferRef, Polynomial, Digest, ReadIop*> storage;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const InterpVal& val) {
  val.print(os);
  return os;
}

std::vector<uint64_t> asFpArray(llvm::ArrayRef<const Zll::InterpVal*> array);

class ExternHandler {
public:
  virtual ~ExternHandler() {}
  virtual std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                        llvm::StringRef extra,
                                                        llvm::ArrayRef<const InterpVal*> arg,
                                                        size_t outCount);

  // Add input data bytes available through the readInput extern.
  void addInput(llvm::StringRef inputName, llvm::StringRef inputBytes);
  // If no input name is specifeid, use default input stream.
  void addInput(llvm::StringRef inputBytes) { addInput("", inputBytes); }

protected:
  std::mt19937 coeffPRNG;

  // Byte stream inputs for readInput per input stream
  mlir::DenseMap<llvm::StringRef, std::deque<uint8_t>> input;
  // Configured number of bytes per input stream set by configureInput.
  mlir::DenseMap<llvm::StringRef, size_t> inputBytesPerElem;
};

class OpEvaluator {
public:
  OpEvaluator(mlir::Operation* op,
              llvm::ArrayRef<InterpVal*> outputs,
              llvm::ArrayRef<const InterpVal*> inputs)
      : op(op), outputs(outputs), inputs(inputs) {}
  virtual llvm::StringRef getStrategy() = 0;
  virtual mlir::LogicalResult evaluate(Interpreter* interp) = 0;
  virtual ~OpEvaluator() {}

  mlir::Operation* op;
  llvm::SmallVector<InterpVal*> outputs;
  llvm::SmallVector<const InterpVal*> inputs;
};

class Interpreter {
public:
  using Polynomial = InterpVal::Polynomial;
  using PolynomialRef = InterpVal::PolynomialRef;
  using Buffer = std::vector<Polynomial>;
  using BufferRef = InterpVal::BufferRef;

  Interpreter(mlir::MLIRContext* ctx, std::unique_ptr<IHashSuite> hashSuite = shaHashSuite());
  ~Interpreter();

  void setCycle(size_t cycle);
  size_t getCycle();
  void setTotCycles(size_t totCycles);
  size_t getTotCycles();

  // Calculates a wrapping (current cycle - back) modulo totCycles
  size_t getBackCycle(size_t backDistance);

  const IHashSuite& getHashSuite();
  void setExternHandler(ExternHandler* handler);
  ExternHandler* getExternHandler();
  BufferRef makeBuf(mlir::Value buffer, size_t size, BufferKind kind);

  // A size of 0 means a global buffer that doesn't have separate values per cycle.
  void setNamedBuf(llvm::StringRef name, BufferRef val, size_t size = 0);
  BufferRef getNamedBuf(mlir::StringRef name);
  bool hasNamedBuf(mlir::StringRef name);
  size_t getNamedBufSize(mlir::StringRef name);

  void setSilenceErrors(bool silence) { silenceErrors = silence; }
  bool getSilenceErrors() { return silenceErrors; }

  // Accessors for interpreted values of various types.
  PolynomialRef getVal(mlir::Value v) { return vals.at(v)->getVal(); }
  template <typename T = mlir::Attribute> T getAttr(mlir::Value v) {
    return vals.at(v)->getAttr<T>(ctx);
  }
  ReadIop* getIop(mlir::Value v) { return vals.at(v)->getIop(); }
  BufferRef getBuf(mlir::Value v) { return vals.at(v)->getBuf(); }
  const Digest& getDigest(mlir::Value v) { return vals.at(v)->getDigest(); }
  bool hasVal(mlir::Value v) { return vals.contains(v); }
  const InterpVal* getInterpVal(mlir::Value v) { return vals.at(v); }

  template <typename T> void setVal(mlir::Value v, T val) { getOrCreateInterpVal(v)->setVal(val); }
  void setAttr(mlir::Value v, mlir::Attribute val) { getOrCreateInterpVal(v)->setAttr(val); }
  void setIop(mlir::Value v, ReadIop* val) { getOrCreateInterpVal(v)->setIop(val); }
  void setBuf(mlir::Value v, BufferRef val) { getOrCreateInterpVal(v)->setBuf(val); }
  void setDigest(mlir::Value v, const Digest& val) { getOrCreateInterpVal(v)->setDigest(val); }
  void setInterpVal(mlir::Value v, const InterpVal* val) {
    getOrCreateInterpVal(v)->setInterpVal(val);
  }

  /// Attempts to interpret the value and everything it depends on,
  /// and returns the evaluated constant as an attribute.  Returns nullptr
  /// on failure.
  mlir::Attribute evaluateConstant(mlir::Value);

  template <typename T> T evaluateConstantOfType(mlir::Value val) {
    auto res = evaluateConstant(val);
    if (!res)
      return T();
    if (!llvm::isa<T>(res)) {
      mlir::emitError(val.getLoc()) << val << " evaluated as a constant " << res << "; expecting a "
                                    << llvm::getTypeName<T>();
      return T();
    }
    return llvm::cast<T>(res);
  }

  /// Interprets the whole block and returns the return value if any.
  mlir::FailureOr<llvm::SmallVector<mlir::Attribute>> runBlock(mlir::Block& block);

  // Adaptor for operations that binds everyhting possible beforehand.
  template <typename ConcreteOp>
  OpEvaluator* getEvaluatorForEvalOp(ConcreteOp op,
                                     llvm::ArrayRef<InterpVal*> outputs,
                                     llvm::ArrayRef<const InterpVal*> inputs) {
    return new (interpAlloc.Allocate<EvalOpEvaluator<ConcreteOp>>())
        EvalOpEvaluator<ConcreteOp>(this, op, outputs, inputs);
  }
  template <typename ConcreteOp>
  OpEvaluator* getEvaluatorForEvalOp(ConcreteOp op,
                                     llvm::ArrayRef<InterpVal*> outputs,
                                     llvm::ArrayRef<const InterpVal*> inputs,
                                     ExtensionField field) {
    return new (interpAlloc.Allocate<EvalOpFieldEvaluator<ConcreteOp>>())
        EvalOpFieldEvaluator<ConcreteOp>(this, op, outputs, inputs, field);
  }

  mlir::LogicalResult evalCallConstant(mlir::CallOpInterface callOp,
                                       llvm::ArrayRef<InterpVal*> outputs,
                                       llvm::ArrayRef<const InterpVal*> inputs);

  // Specify a return value to return from the current lbock
  void setResultValues(llvm::ArrayRef<mlir::Attribute> resultValues);

  mlir::MLIRContext* getContext() { return ctx; }

private:
  template <typename ConcreteOp> struct EvalOpFieldEvaluator : public OpEvaluator {
    EvalOpFieldEvaluator(Interpreter* interp,
                         ConcreteOp op,
                         llvm::ArrayRef<InterpVal*> outputs,
                         llvm::ArrayRef<const InterpVal*> inputsArg,
                         ExtensionField field)
        : OpEvaluator(op, outputs, inputsArg), op(op), adaptor(inputs, op), field(field) {}

    llvm::StringRef getStrategy() override { return "Adaptor+Field"; }
    mlir::LogicalResult evaluate(Interpreter* interp) override {
      return op.evaluate(*interp, outputs, adaptor, field);
    }

    ConcreteOp op;
    typename ConcreteOp::EvalAdaptor adaptor;
    ExtensionField field;
  };

  template <typename ConcreteOp> struct EvalOpEvaluator : public OpEvaluator {
    EvalOpEvaluator(Interpreter* interp,
                    ConcreteOp op,
                    llvm::ArrayRef<InterpVal*> outputs,
                    llvm::ArrayRef<const InterpVal*> inputsArg)
        : OpEvaluator(op, outputs, inputsArg), op(op), adaptor(inputs, op) {}

    llvm::StringRef getStrategy() override { return "Adaptor"; }
    mlir::LogicalResult evaluate(Interpreter* interp) override {
      return op.evaluate(*interp, outputs, adaptor);
    }

    ConcreteOp op;
    typename ConcreteOp::EvalAdaptor adaptor;
  };

  using BlockEvaluators = llvm::SmallVector<OpEvaluator*>;

  InterpVal* getOrCreateInterpVal(mlir::Value val);

  void setAttribute(mlir::Value value, mlir::Attribute attr);
  mlir::FailureOr<mlir::Attribute> getAsAttribute(mlir::Value value);

  // Returns a function to evaluate the operation with all the
  // interfaces resolved for performance.
  OpEvaluator* getOpEvaluator(mlir::Operation* op);
  BlockEvaluators* getBlockEvaluators(mlir::Block& block);

  mlir::LogicalResult evaluate(OpEvaluator* eval);

  size_t cycle = 0;
  size_t totCycles = 0;
  std::unique_ptr<IHashSuite> hashSuite;
  ExternHandler* handler;
  llvm::SmallVector<llvm::SmallVector<Polynomial>> allocBufs;

  mlir::DenseMap<mlir::Block*, BlockEvaluators*> blockEvaluators;

  // All stored values.
  mlir::DenseMap<mlir::Value, InterpVal*> vals;

  // Return value being returned.
  llvm::SmallVector<mlir::Attribute> resultValues;

  mlir::DenseMap<llvm::StringRef, std::pair<BufferRef, /*size=*/size_t>> namedBufs;

  bool silenceErrors;
  // Operation dumping state when running a debugging trace.
  std::optional<mlir::AsmState> asmState;

  llvm::BumpPtrAllocator interpAlloc;
  mlir::MLIRContext* ctx;
};

} // namespace zirgen::Zll
