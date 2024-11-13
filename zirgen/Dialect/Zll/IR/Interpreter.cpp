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

#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <deque>

#define DEBUG_TYPE "interpreter"

using namespace mlir;

namespace zirgen::Zll {

namespace {

// Formats as either a base field element or an extension field element.
template <typename T> void formatFieldElem(const InterpVal* interpVal, llvm::raw_ostream& os, T f) {
  auto val = interpVal->getVal();
  if (val.size() == 1) {
    f(val[0]);
  } else {
    os << "[";
    interleaveComma(val, os, f);
    os << "]";
  }
}

} // namespace

std::optional<std::vector<uint64_t>> ExternHandler::doExtern(llvm::StringRef name,
                                                             llvm::StringRef extra,
                                                             llvm::ArrayRef<const InterpVal*> args,
                                                             size_t outCount) {
  if (name == "readCoefficients") {
    // TODO: Migrate users of readCoefficients to use readInput, or
    // move readCoefficients to a circuit-specific extern handler.

    // Produce 'fake' coefficients for now via a PRNG
    assert(outCount == 16);
    std::vector<uint64_t> ret;
    for (size_t i = 0; i < 16; i++) {
      ret.push_back(coeffPRNG() & 0xff);
    }
    return ret;
  }
  if (name == "configureInput") {
    // Usage: configureInput(/*extra=*/inputName, bytesPerElem)
    //
    // where:
    //   extra = named input source (default = "")
    //   bytesPerElem = number of bytes of input per element
    auto fpArgs = asFpArray(args);
    if (fpArgs.size() != 1)
      throw std::runtime_error("wrong number of arguments to configureInput");
    size_t bytesPerElem = fpArgs[0];
    inputBytesPerElem[extra] = bytesPerElem;
    return std::vector<uint64_t>{};
  }
  if (name == "readInput") {
    // Usage: readInput(/*extra=*/inputName)
    //
    // Reads a number of elements from input, specified by the number
    // of elements returned.  configureInput must be called first to
    // configure the format of the input.

    size_t bytesPerElem = inputBytesPerElem[extra];
    if (!bytesPerElem || bytesPerElem > sizeof(uint64_t)) {
      throw std::runtime_error("invalid bytesPerElem");
    }

    auto& inputBytes = input[extra];

    auto fpArgs = asFpArray(args);
    if (fpArgs.size() != 0)
      throw std::runtime_error("wrong number of arguments to readInput");

    std::vector<uint64_t> ret;

    for (size_t i = 0; i != outCount; ++i) {
      uint64_t elem = 0;
      for (size_t byteIdx = 0; byteIdx != bytesPerElem; byteIdx++) {
        if (inputBytes.empty())
          throw std::runtime_error("readInput: input underrun");
        elem |= inputBytes.front() << (byteIdx * 8);
        inputBytes.pop_front();
      }
      ret.push_back(elem);
    }
    return ret;
  }
  if (name == "log") {
    auto& os = llvm::outs();
    size_t argNum = 0;
    auto nextArg = [&]() {
      if (argNum >= args.size()) {
        throw std::runtime_error(("Ran out of arguments processing " + extra).str());
      }
      return args[argNum++];
    };
    std::string extraStr = extra.str();
    const char* p = extraStr.c_str();
    while (*p) {
      if (*p == '%') {
        p++;
        int len = 0;
        while (*p >= '0' && *p <= '9') {
          len *= 10;
          len += *p - '0';
          p++;
        }
        if (*p == '%') {
          os << "%";
          p++;
        } else if (*p == 'x') {
          formatFieldElem(nextArg(), os, [&](auto elem) { os << llvm::format_hex(elem, len); });
          p++;
        } else if (*p == 'u') {
          formatFieldElem(nextArg(), os, [&](auto elem) { os << llvm::format_decimal(elem, len); });
          p++;
        } else if (*p == 'p') {
          auto poly = nextArg()->getVal();
          os << "[";
          for (size_t i = 0; i < poly.size(); i++) {
            if (i) {
              os << ", ";
            }
            os << poly[i];
          }
          os << "]";
          p++;
        } else if (*p == 'w') {
          uint64_t vals[sizeof(uint32_t)];
          uint32_t u32val = 0;
          for (auto& val : vals) {
            val = nextArg()->getBaseFieldVal();
            u32val >>= 8;
            u32val |= val << 24;
          }
          os << llvm::format_hex(u32val, len);
          p++;
        } else if (*p == 'h') {
          os << nextArg()->getDigest();
          p++;
        } else if (*p == 'e') {
          uint64_t vals[kBabyBearExtSize];
          for (auto& val : vals) {
            val = nextArg()->getBaseFieldVal();
          }
          os << "[";
          for (size_t i = 0; i < kBabyBearExtSize; ++i) {
            if (i) {
              os << ", ";
            }
            os << vals[i];
          }
          os << "]";
          p++;
        }
      } else {
        os << *p;
        p++;
      }
    }
    if (argNum != args.size()) {
      throw std::runtime_error(("Unused arguments in format " + extra).str());
    }
    os << "\n";
    return std::vector<uint64_t>{};
  }
  throw std::runtime_error(("Unknown extern: " + name).str());
}

void ExternHandler::addInput(llvm::StringRef inputName, StringRef newBytes) {
  auto& bytes = input[inputName];
  bytes.insert(bytes.end(), newBytes.begin(), newBytes.end());
}

Interpreter::Interpreter(MLIRContext* ctx, std::unique_ptr<IHashSuite> hashSuite)
    : cycle(0), hashSuite(std::move(hashSuite)), handler(nullptr), silenceErrors(false), ctx(ctx) {}
Interpreter::~Interpreter() {}

void Interpreter::setCycle(size_t cycle) {
  LLVM_DEBUG({ llvm::dbgs() << "Starting cycle " << cycle << "\n"; });
  this->cycle = cycle;
}

size_t Interpreter::getCycle() {
  return cycle;
}

void Interpreter::setTotCycles(size_t totCycles) {
  this->totCycles = totCycles;
}

size_t Interpreter::getTotCycles() {
  return totCycles;
}

size_t Interpreter::getBackCycle(size_t backDistance) {
  size_t cycle = getCycle();
  if (totCycles && cycle < backDistance) {
    cycle += totCycles;
  }
  if (backDistance > cycle) {
    llvm::errs() << "Back distance " << backDistance << " too far, with no totCycles specified\n";
    abort();
  }
  return cycle - backDistance;
}

const IHashSuite& Interpreter::getHashSuite() {
  return *hashSuite;
}

void Interpreter::setExternHandler(ExternHandler* handler) {
  this->handler = handler;
}

ExternHandler* Interpreter::getExternHandler() {
  return handler;
}

Interpreter::BufferRef Interpreter::makeBuf(mlir::Value buffer, size_t size, BufferKind kind) {
  allocBufs.emplace_back(size, Polynomial(1, kFieldInvalid));
  BufferRef ref = allocBufs.back();
  setBuf(buffer, ref);
  return ref;
}

Interpreter::BufferRef Interpreter::getNamedBuf(llvm::StringRef name) {
  if (!namedBufs.count(name)) {
    llvm::errs() << "Undefined buffer " << name << "\n";
  }
  assert(namedBufs.count(name));
  return namedBufs[name].first;
}

bool Interpreter::hasNamedBuf(llvm::StringRef name) {
  return namedBufs.count(name);
}

size_t Interpreter::getNamedBufSize(llvm::StringRef name) {
  if (!namedBufs.count(name)) {
    llvm::errs() << "Undefined buffer " << name << "\n";
  }
  assert(namedBufs.count(name));
  return namedBufs[name].second;
}
void Interpreter::setNamedBuf(llvm::StringRef name, Interpreter::BufferRef val, size_t size) {
  assert(val.size() != 0 && (size == 0 || (val.size() % size == 0)));
  namedBufs[name] = std::make_pair(val, size);
}

mlir::Attribute Interpreter::evaluateConstant(mlir::Value value) {
  ScopedDiagnosticHandler handler(ctx, [&](Diagnostic& diag) {
    diag.attachNote(value.getLoc()) << "While attempting to evaluate " << value << " as a constant";
    return failure();
  });

  assert(value);
  std::deque<mlir::Value> workList;
  workList.push_back(value);

  while (!workList.empty()) {
    mlir::Value work = workList.front();
    workList.pop_front();

    if (vals.contains(work)) {
      // Already got this value.
      continue;
    }

    bool hasAllOperands = true;
    assert(work);
    auto op = work.getDefiningOp();
    if (!op) {
      emitError(value.getLoc()) << "Unable to find defining op for value " << work << " needed for "
                                << value << "\n";
      return {};
    }
    for (mlir::Value operand : op->getOperands()) {
      assert(operand);
      if (!vals.contains(operand)) {
        workList.push_back(operand);
        hasAllOperands = false;
      }
    }

    if (!hasAllOperands) {
      // Try back later when we've finished evaluating operands.
      assert(work);
      workList.push_back(work);
      continue;
    }

    // All values it depends on have been evaluted; we can now try to evaluate this operation.
    ScopedDiagnosticHandler handler(ctx, [&](Diagnostic& diag) {
      if (getContext()->shouldPrintOpOnDiagnostic()) {
        diag.attachNote(op->getLoc())
            .append("see current operation: ")
            .appendOp(*op, OpPrintingFlags().printGenericOpForm());
      } else {
        diag.attachNote(op->getLoc()) << "while evaluating this";
      }

      return failure();
    });

    try {
      auto eval = getOpEvaluator(op);
      if (auto callOp = dyn_cast<CallOpInterface>(op)) {
        // Special case for evaluating constants: evaluate only the
        // return value in a subinterpreter so we don't have to worry
        // about side effects.
        if (failed(evalCallConstant(callOp, eval->outputs, eval->inputs))) {
          op->emitError() << "Call evaluation failed";
          return {};
        }
      } else {
        if (failed(evaluate(eval))) {
          op->emitError() << "Evaluation failed";
          return {};
        }
      }
    } catch (const std::runtime_error& err) {
      op->emitError() << "Evaluation failed with exception: " << err.what();
      return {};
    }

    if (!vals.contains(work)) {
      op->emitError() << "Evaluation did not produce a value";
      return {};
    }
  }
  return getAttr(value);
}

void InterpVal::print(llvm::raw_ostream& os) const {
  if (std::holds_alternative<Polynomial>(storage)) {
    formatFieldElem(this, os, [&](auto elem) { os << elem; });
  } else if (std::holds_alternative<Attribute>(storage)) {
    os << std::get<Attribute>(storage);
  } else if (std::holds_alternative<BufferRef>(storage)) {
    os << "Buffer";
  } else if (std::holds_alternative<Digest>(storage)) {
    os << std::get<Digest>(storage);
  } else if (std::holds_alternative<ReadIop*>(storage)) {
    os << "ReadIop";
  } else {
    os << "(UNKNOWN INTERP VAL)";
  }
}

void InterpVal::print(llvm::raw_ostream& os, AsmState& asmState) const {
  if (std::holds_alternative<Polynomial>(storage)) {
    os << "Fp{";
    interleaveComma(std::get<Polynomial>(storage), os);
    os << "}";
  } else if (std::holds_alternative<Attribute>(storage)) {
    std::get<Attribute>(storage).print(os, asmState);
  } else if (std::holds_alternative<BufferRef>(storage)) {
    os << "Buffer";
  } else if (std::holds_alternative<Digest>(storage)) {
    os << std::get<Digest>(storage);
  } else if (std::holds_alternative<ReadIop*>(storage)) {
    os << "ReadIop";
  } else {
    os << "(UNKNOWN INTERP VAL)";
  }
}

std::vector<uint64_t> asFpArray(llvm::ArrayRef<const Zll::InterpVal*> array) {
  std::vector<uint64_t> out;
  for (const auto* val : array) {
    out.push_back(val->getBaseFieldVal());
  }
  return out;
}

mlir::LogicalResult Interpreter::evaluate(OpEvaluator* eval) {
  static size_t evaluateTraceCount = 0;
  size_t thisEvaluateTraceCount{};
  LLVM_DEBUG({
    thisEvaluateTraceCount = ++evaluateTraceCount;

    if (!asmState) {
      auto module = eval->op->getParentOfType<mlir::ModuleOp>();
      asmState.emplace(
          module,
          OpPrintingFlags().assumeVerified().printGenericOpForm(false).elideLargeElementsAttrs());

      // Generate type aliases so printing operations is less verbose.
      module->print(llvm::nulls(), *asmState);
    }
    eval->op->print(llvm::dbgs(), *asmState);
    llvm::dbgs() << " (" << eval->getStrategy() << ") at " << eval->op->getLoc() << "\n";
    if (!eval->inputs.empty()) {
      llvm::dbgs() << "  Inputs:\n";
      for (auto input : eval->inputs) {
        llvm::dbgs() << "    ";
        input->print(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
      }
    }
  });

  mlir::LogicalResult res = eval->evaluate(this);

  LLVM_DEBUG({
    if (thisEvaluateTraceCount != evaluateTraceCount) {
      // If any other evaluation traces occured while we were executing
      // the operation, indicate which operation is currently finishing.
      llvm::dbgs() << "Finished evaluating: ";
      eval->op->print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    }
    if (!eval->outputs.empty()) {
      llvm::dbgs() << "  Output:\n";
      for (auto output : eval->outputs) {
        llvm::dbgs() << "    ";
        output->print(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
      };

      if (failed(res))
        llvm::dbgs() << "FAILED\n";
    }
    llvm::errs() << "\n";
  });
  return res;
}

LogicalResult Interpreter::evalCallConstant(CallOpInterface callOp,
                                            ArrayRef<InterpVal*> outputs,
                                            ArrayRef<const InterpVal*> inputs) {
  LLVM_DEBUG({ llvm::dbgs() << "Calling " << callOp << " in constant context\n"; });
  auto calleeName = llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee());

  auto calleeOp = SymbolTable::lookupNearestSymbolFrom<CallableOpInterface>(callOp, calleeName);
  if (!calleeOp) {
    callOp->emitError() << "Unable to find callee " << calleeName << "\n";
    return failure();
  }

  bool gotErrorMsg = false;
  ScopedDiagnosticHandler handler(ctx, [&](Diagnostic& diag) {
    gotErrorMsg = true;
    auto& note = diag.attachNote(callOp.getLoc()) << "While calling " << calleeName << " from ";
    note.appendOp(*callOp, OpPrintingFlags().printGenericOpForm());
    return failure();
  });

  auto region = calleeOp.getCallableRegion();

  Interpreter callInterp(callOp.getContext());
  for (auto [arg, blockArg] : zip(inputs, region->getArguments())) {
    callInterp.setInterpVal(blockArg, arg);
  }

  auto terminator = region->front().getTerminator();
  assert(terminator->hasTrait<OpTrait::ReturnLike>());
  assert(terminator->getNumOperands() == callOp->getNumResults());

  for (auto [result, calleeResult] : llvm::zip(outputs, terminator->getOperands())) {
    LLVM_DEBUG({ llvm::dbgs() << "Evaluating inside callee\n"; });
    auto res = callInterp.evaluateConstant(calleeResult);
    if (!res) {
      LLVM_DEBUG({ llvm::dbgs() << "Unable to evaluate terminator " << calleeResult << "\n"; });
      if (!gotErrorMsg && !getSilenceErrors())
        callOp->emitError() << "Unknown evaluation error occured";
      return failure();
    }
    LLVM_DEBUG({ llvm::dbgs() << "Done evaluating inside callee\n"; });
    result->setAttr(res);
  }
  return success();
}

FailureOr<SmallVector<Attribute>> Interpreter::runBlock(mlir::Block& block) {
  auto* blockEvaluators = getBlockEvaluators(block);

  OpEvaluator* evaluator;
  bool gotErrorMsg = false;
  ScopedDiagnosticHandler handler(
      ctx, DiagnosticEngine::HandlerTy([&](Diagnostic& diag) {
        gotErrorMsg = true;
        auto& note = diag.attachNote(evaluator->op->getLoc()) << "While attempting to evaluate ";
        note.appendOp(*evaluator->op, OpPrintingFlags().printGenericOpForm());
        return failure();
      }));

  for (auto* eval : *blockEvaluators) {
    evaluator = eval;
    if (failed(evaluate(evaluator))) {
      if (!gotErrorMsg && !getSilenceErrors())
        eval->op->emitError() << "Evaluation error occured";
      return failure();
    }
  }

  return std::move(resultValues);
}

namespace {

struct CallOpEvaluator : public OpEvaluator {
  CallOpEvaluator(Interpreter* interp,
                  mlir::CallOpInterface op,
                  llvm::ArrayRef<InterpVal*> outputs,
                  llvm::ArrayRef<const InterpVal*> inputs,
                  Region* region)
      : OpEvaluator(op, outputs, inputs), op(op), region(region) {}

  llvm::StringRef getStrategy() override { return "Call"; }
  mlir::LogicalResult evaluate(Interpreter* interp) override {
    for (auto [arg, blockArg] : zip_equal(inputs, region->getArguments())) {
      interp->setInterpVal(blockArg, arg);
    }

    auto results = interp->runBlock(region->front());
    if (failed(results))
      return failure();
    assert(results->size() == outputs.size());
    for (auto [result, output] : zip_equal(*results, outputs)) {
      output->setAttr(result);
    }

    return success();
  }
  CallOpInterface op;
  Region* region;
};

struct TerminatorOpEvaluator : public OpEvaluator {
  TerminatorOpEvaluator(Interpreter* interp,
                        Operation* op,
                        llvm::ArrayRef<InterpVal*> outputs,
                        llvm::ArrayRef<const InterpVal*> inputs)
      : OpEvaluator(op, outputs, inputs) {}
  llvm::StringRef getStrategy() override { return "Terminator"; }
  mlir::LogicalResult evaluate(Interpreter* interp) override {
    auto resultAttrs = llvm::to_vector(
        llvm::map_range(inputs, [&](auto input) { return input->getAttr(interp->getContext()); }));
    interp->setResultValues(resultAttrs);
    return success();
  }
};

struct FoldOpEvaluator : public OpEvaluator {
  FoldOpEvaluator(Interpreter* interp,
                  Operation* op,
                  llvm::ArrayRef<InterpVal*> outputs,
                  llvm::ArrayRef<const InterpVal*> inputs)
      : OpEvaluator(op, outputs, inputs), op(op) {}

  llvm::StringRef getStrategy() override { return "Fold"; }
  mlir::LogicalResult evaluate(Interpreter* interp) override {
    // Otherwise, try to use constant folding to evaluate this.
    auto inputRange =
        llvm::map_range(inputs, [&](auto operand) { return operand->getAttr(op->getContext()); });

    operands.assign(inputRange.begin(), inputRange.end());

    SmallVector<OpFoldResult, 4> foldResults;
    if (succeeded(op->fold(operands, foldResults)) && !foldResults.empty()) {
      for (auto [foldResult, opResultPart] : llvm::zip(foldResults, outputs)) {
        // Apparently "structured bindings" are only capturable in lambdas
        // in c++20?  TODO: Remove this indirecion at some point.
        auto opResult = opResultPart;
        TypeSwitch<OpFoldResult>(foldResult)
            .Case<mlir::Value>(
                [&](mlir::Value value) { opResult->setInterpVal(interp->getInterpVal(value)); })
            .Case<mlir::Attribute>([&](mlir::Attribute attr) { opResult->setAttr(attr); });
      }
      return success();
    }

    LLVM_DEBUG({ llvm::dbgs() << "Operation unimplemented: " << *op << "\n"; });
    return failure();
  }
  Operation* op;
  SmallVector<Attribute> operands;
};

} // namespace

OpEvaluator* Interpreter::getOpEvaluator(Operation* op) {
  // Analyze the operation for the proper interface to use and the
  // input and output InterpVals once, then save it so we don't have
  // to do it again each time we execute.

  SmallVector<InterpVal*> outputs = llvm::to_vector(llvm::map_range(
      op->getResults(), [&](auto val) -> InterpVal* { return getOrCreateInterpVal(val); }));
  SmallVector<const InterpVal*> inputs = llvm::to_vector(llvm::map_range(
      op->getOperands(), [&](auto val) -> const InterpVal* { return getOrCreateInterpVal(val); }));

  if (auto evalOp = dyn_cast<EvalOp>(op)) {
    return evalOp.getOpEvaluator(this, outputs, inputs);
  }

  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
    auto calleeName = llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee());
    auto calleeOp = SymbolTable::lookupNearestSymbolFrom<CallableOpInterface>(callOp, calleeName);
    return new (interpAlloc.Allocate<CallOpEvaluator>())
        CallOpEvaluator(this, callOp, outputs, inputs, calleeOp.getCallableRegion());
  }

  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return new (interpAlloc.Allocate<TerminatorOpEvaluator>())
        TerminatorOpEvaluator(this, op, outputs, inputs);
  }

  return new (interpAlloc.Allocate<FoldOpEvaluator>()) FoldOpEvaluator(this, op, outputs, inputs);
}

InterpVal* Interpreter::getOrCreateInterpVal(mlir::Value val) {
  auto& v = vals[val];
  if (!v) {
    v = new (interpAlloc.Allocate<InterpVal>()) InterpVal();
  }
  return v;
}

Interpreter::BlockEvaluators* Interpreter::getBlockEvaluators(mlir::Block& block) {
  auto [it, isNewEntry] = blockEvaluators.try_emplace(&block);
  auto& blockEvals = it->second;
  if (isNewEntry) {
    blockEvals = new (interpAlloc.Allocate<BlockEvaluators>()) BlockEvaluators();

    // First allocate InterpVals for each input to aid in cache locality.
    for (Operation& op : block) {
      for (auto input : op.getOperands()) {
        getOrCreateInterpVal(input);
      }
    }

    // Now, generate evaluators for each operation.
    for (Operation& op : block) {
      blockEvals->emplace_back(getOpEvaluator(&op));
    }
  }

  return blockEvals;
}

void Interpreter::setResultValues(llvm::ArrayRef<mlir::Attribute> newResultValues) {
  assert(resultValues.empty() && "Only one operation may set return values during a block");
  resultValues = llvm::to_vector(newResultValues);
}

} // namespace zirgen::Zll
