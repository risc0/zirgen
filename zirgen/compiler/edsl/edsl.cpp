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

#include "zirgen/compiler/edsl/edsl.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/Analysis/DegreeAnalysis.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/compiler/edsl/component.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen {

namespace {

size_t gBackDist = 0;
bool gBackUnchecked = false;
bool gBackUsed = false;

Module* curModule = nullptr;

OpBuilder& getBuilder() {
  assert(curModule);
  return curModule->getBuilder();
}

MLIRContext* getCtx() {
  assert(curModule);
  return curModule->getCtx();
}

Location toLoc(SourceLoc loc, StringRef ident = {}) {
  LocationAttr inner;
  if (loc.filename) {
    auto id = StringAttr::get(getCtx(), loc.filename);
    inner = FileLineColLoc::get(id, loc.line, loc.column);
  } else {
    inner = UnknownLoc::get(getCtx());
  }

  if (ident.empty()) {
    return inner;
  } else {
    return NameLoc::get(StringAttr::get(getCtx(), ident), inner);
  }
}

std::vector<SourceLoc>& getLocStack() {
  static std::vector<SourceLoc> stack;
  return stack;
}

} // namespace

Module* Module::getCurModule() {
  return curModule;
}

OverrideLocation::OverrideLocation(SourceLoc loc) {
  getLocStack().push_back(loc);
}

OverrideLocation::~OverrideLocation() {
  getLocStack().pop_back();
}

SourceLoc checkCurrentLoc(SourceLoc loc) {
  if (getLocStack().empty()) {
    return loc;
  }
  return getLocStack().back();
}

Val::Val(uint64_t val, SourceLoc loc) {
  Type ty = ValType::getBaseType(getBuilder().getContext());
  value = getBuilder().create<ConstOp>(toLoc(loc), ty, val);
}

Val::Val(llvm::ArrayRef<uint64_t> val, SourceLoc loc) {
  assert(val.size() == kBabyBearExtSize);
  Type ty = ValType::get(getBuilder().getContext(), kFieldPrimeDefault, val.size());
  value = getBuilder().create<ConstOp>(toLoc(loc), ty, val);
}

Val::Val(Register reg, SourceLoc loc) {
  if (cast<BufferType>(reg.buf.getType()).getKind() == BufferKind::Global) {
    value = getBuilder().create<GetGlobalOp>(toLoc(loc, reg.ident), reg.buf, 0);
  } else {
    auto getOp =
        getBuilder().create<GetOp>(toLoc(loc, reg.ident), reg.buf, 0, gBackDist, IntegerAttr());
    if (gBackUnchecked) {
      getOp->setAttr("unchecked", UnitAttr::get(getOp.getContext()));
    }
    value = getOp;
    gBackUsed = true;
  }
}

void Register::operator=(CaptureVal x) {
  if (cast<BufferType>(buf.getType()).getKind() == BufferKind::Global) {
    getBuilder().create<SetGlobalOp>(toLoc(x.loc, x.ident), buf, 0, x.getValue());
  } else {
    getBuilder().create<SetOp>(toLoc(x.loc, x.ident), buf, 0, x.getValue());
  }
}

mlir::Location CaptureIdx::getLoc() {
  return toLoc(loc);
}

Val Buffer::get(size_t idx, StringRef ident, SourceLoc loc) {
  if (cast<BufferType>(buf.getType()).getKind() == BufferKind::Global) {
    return Val(getBuilder().create<GetGlobalOp>(toLoc(loc, ident), buf, 0));
  } else {
    return Val(getBuilder().create<GetOp>(toLoc(loc, ident), buf, idx, 0, IntegerAttr()));
  }
}

void Buffer::set(size_t idx, Val x, StringRef ident, SourceLoc loc) {
  if (cast<BufferType>(buf.getType()).getKind() == BufferKind::Global) {
    getBuilder().create<SetGlobalOp>(toLoc(loc, ident), buf, idx, x.getValue());
  } else {
    getBuilder().create<SetOp>(toLoc(loc, ident), buf, idx, x.getValue());
  }
}

void Buffer::setDigest(size_t idx, DigestVal x, StringRef ident, SourceLoc loc) {
  if (cast<BufferType>(buf.getType()).getKind() != BufferKind::Global) {
    throw(std::runtime_error("Currently digests can only be stored in globals"));
  }
  getBuilder().create<SetGlobalDigestOp>(toLoc(loc, ident), buf, idx, x.getValue());
}

Buffer Buffer::slice(size_t offset, size_t size, SourceLoc loc) {
  return Buffer(getBuilder().create<SliceOp>(toLoc(loc), buf, offset, size));
}

Register Buffer::getRegister(size_t idx, StringRef ident, SourceLoc loc) {
  if (idx >= cast<BufferType>(buf.getType()).getSize()) {
    llvm::errs() << "Out of bounds index: " << loc.filename << ":" << loc.line << "\n";
    throw std::runtime_error("OOB Index");
  }
  return Register(getBuilder().create<SliceOp>(toLoc(loc), buf, idx, 1), ident);
}

mlir::Location CaptureVal::getLoc() {
  return toLoc(loc);
}

Module::Module() : builder(&ctx) {
  ctx.getOrLoadDialect<ZllDialect>();
  ctx.getOrLoadDialect<Iop::IopDialect>();
  ctx.getOrLoadDialect<ZStruct::ZStructDialect>();
  module = ModuleOp::create(UnknownLoc::get(&ctx));
  builder.setInsertionPointToEnd(&module->getBodyRegion().front());
}

void Module::addOptimizationPasses(PassManager& pm) {
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(createSortForReproducibilityPass());
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
}

void Module::optimize(size_t stageCount) {
  PassManager pm(module->getContext());
  addOptimizationPasses(pm);
  if (failed(pm.run(*module))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }
  if (stageCount > 0) {
    for (size_t i = 0; i < stageCount; i++) {
      stages.emplace_back(module->clone());
      PassManager pm(&ctx);
      OpPassManager& opm = pm.nest<func::FuncOp>();
      opm.addPass(createSplitStagePass(i));
      opm.addPass(createCanonicalizerPass());
      opm.addPass(createCSEPass());
      if (failed(pm.run(*stages[i]))) {
        throw std::runtime_error("Failed to apply stage1 passes");
      }
    }
  }
}

void Module::setExternHandler(ExternHandler* handler) {
  this->handler = handler;
}

void Module::runFunc(StringRef name,
                     llvm::ArrayRef<Interpreter::BufferRef> bufs,
                     size_t startCycle,
                     size_t cycleCount) {
  auto func = module->lookupSymbol<func::FuncOp>(name);
  if (!func) {
    throw std::runtime_error(("Unable to find function: " + name).str());
  }
  runFunc(func, bufs, startCycle, cycleCount);
}

void Module::runStage(size_t stage,
                      StringRef name,
                      llvm::ArrayRef<Interpreter::BufferRef> bufs,
                      size_t startCycle,
                      size_t cycleCount) {
  assert(stage < stages.size());
  auto func = stages[stage]->lookupSymbol<func::FuncOp>(name);
  if (!func) {
    throw std::runtime_error(("Unable to find function: " + name).str());
  }
  runFunc(func, bufs, startCycle, cycleCount);
}

void Module::dump(bool debug) {
  module->print(llvm::errs(), OpPrintingFlags().enableDebugInfo(debug));
}

size_t Module::computeMaxDegree(StringRef name) {
  auto moduleCopy = dyn_cast<ModuleOp>(module->clone());
  PassManager pm(moduleCopy->getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(createMakePolynomialPass());

  DataFlowSolver solver;
  solver.load<Zll::DegreeAnalysis>();
  if (failed(pm.run(moduleCopy)) || failed(solver.initializeAndRun(moduleCopy)))
    throw std::runtime_error("Failed to calculate degree");

  size_t max = 0;
  moduleCopy.walk([&](func::ReturnOp op) {
    auto point = solver.getProgramPointAfter(op);
    Zll::Degree degree = solver.lookupState<DegreeLattice>(point)->getValue();
    max = std::max<size_t>(max, degree.get());
  });
  if (max == 0) {
    throw(std::runtime_error("Nonsensical maximum degree 0 encountered in circuit"));
  }
  return max;
}

void Module::dumpPoly(StringRef name) {
  PassManager pm(module->getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(createMakePolynomialPass());
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  opm.addPass(createComputeTapsPass());
  if (failed(pm.run(*module))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }
  auto func = module->lookupSymbol<func::FuncOp>(name);
  if (!func) {
    throw std::runtime_error(("Unable to find function: " + name).str());
  }

  DataFlowSolver solver;
  solver.load<Zll::DegreeAnalysis>();
  if (failed(solver.initializeAndRun(func)))
    throw std::runtime_error("Failed to calculate degree");

  Block* block = &func.front();
  Operation* cur = block->getTerminator();
  auto point = solver.getProgramPointAfter(cur);
  size_t degree = solver.lookupState<DegreeLattice>(point)->getValue().get();
  llvm::errs() << "Degree = " << degree << "\n";
  while (true) {
    point = solver.getProgramPointAfter(cur);
    cur->print(llvm::errs(), OpPrintingFlags().enableDebugInfo(true));
    size_t curDeg = solver.lookupState<DegreeLattice>(point)->getValue().get();
    llvm::errs() << "\n";
    if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(cur)) {
      cur = retOp.getOperands()[0].getDefiningOp();
    } else if (auto op = mlir::dyn_cast<AndCondOp>(cur)) {
      Value in = op.getIn();
      size_t inDeg = solver.lookupState<DegreeLattice>(in)->getValue().get();
      if (inDeg == curDeg) {
        cur = in.getDefiningOp();
      } else {
        cur = op.getInner().getDefiningOp();
      }
    } else if (auto op = mlir::dyn_cast<AndEqzOp>(cur)) {
      Value in = op.getIn();
      size_t inDeg = solver.lookupState<DegreeLattice>(in)->getValue().get();
      if (inDeg == curDeg) {
        cur = in.getDefiningOp();
      } else {
        cur = op.getVal().getDefiningOp();
      }
    } else if (auto op = mlir::dyn_cast<SubOp>(cur)) {
      Value in = op.getLhs();
      size_t inDeg = solver.lookupState<DegreeLattice>(in)->getValue().get();
      if (inDeg == curDeg) {
        cur = in.getDefiningOp();
      } else {
        cur = op.getRhs().getDefiningOp();
      }
    } else if (auto op = mlir::dyn_cast<MulOp>(cur)) {
      Value in = op.getLhs();
      size_t inDeg = solver.lookupState<DegreeLattice>(in)->getValue().get();
      if (inDeg == curDeg) {
        cur = in.getDefiningOp();
      } else {
        cur = op.getRhs().getDefiningOp();
      }
    } else {
      break;
    }
  }

  // module->print(llvm::errs(), OpPrintingFlags().enableDebugInfo(true));
}

void Module::dumpStage(size_t stage, bool debug) {
  assert(stage <= stages.size());
  stages[stage]->print(llvm::errs(), OpPrintingFlags().enableDebugInfo(debug));
}

void Module::beginFunc(const std::string& name,
                       const std::vector<ArgumentInfo>& args,
                       SourceLoc loc) {
  curModule = this;
  std::vector<Type> inTypes;
  for (auto ai : args) {
    if (ai.type == ArgumentType::IOP) {
      inTypes.push_back(Iop::IOPType::get(&ctx));
    } else {
      inTypes.push_back(BufferType::get(
          &ctx, ValType::get(&ctx, kFieldPrimeDefault, ai.degree), ai.size, ai.kind));
    }
  }
  auto funcType = FunctionType::get(&ctx, inTypes, {});
  auto func = builder.create<func::FuncOp>(toLoc(loc), name, funcType);
  pushIP(func.addEntryBlock());
}

void Module::setPhases(mlir::func::FuncOp funcOp, llvm::ArrayRef<std::string> phases) {
  // First find all the buffers that are tap buffers.  These need to be sorted alphabetically
  llvm::SmallVector<std::string> tapBufs;
  for (auto [argIdx, buf] : llvm::enumerate(funcOp.getArguments())) {
    auto argName =
        llvm::dyn_cast_if_present<mlir::StringAttr>(funcOp.getArgAttr(argIdx, "zirgen.argName"));
    if (!argName)
      argName = builder.getStringAttr("arg" + std::to_string(argIdx));
    auto arg = funcOp.getArgument(argIdx);
    auto ty = llvm::cast<BufferType>(arg.getType());
    if (ty.getKind() != BufferKind::Global) {
      tapBufs.push_back(argName.str());
    }
  }

  // Now we can calculate reg group ids and construct our BufferDescAttrs.
  llvm::sort(tapBufs);
  llvm::StringMap<std::optional<size_t>> regGroupIds;
  for (auto [idx, name] : llvm::enumerate(tapBufs)) {
    regGroupIds[name] = idx;
  }

  llvm::SmallVector<BufferDescAttr> buffers;
  for (auto [argIdx, buf] : llvm::enumerate(funcOp.getArguments())) {
    auto argName =
        llvm::dyn_cast_if_present<mlir::StringAttr>(funcOp.getArgAttr(argIdx, "zirgen.argName"));
    if (!argName)
      argName = builder.getStringAttr("arg" + std::to_string(argIdx));
    auto arg = funcOp.getArgument(argIdx);
    auto ty = llvm::cast<BufferType>(arg.getType());

    buffers.push_back(builder.getAttr<BufferDescAttr>(argName, ty, regGroupIds[argName]));
  }

  auto& builder = getBuilder();
  setModuleAttr(funcOp, builder.getAttr<BuffersAttr>(buffers));

  llvm::SmallVector<mlir::StringAttr> steps;
  for (auto phase : phases) {
    steps.push_back(builder.getAttr<mlir::StringAttr>(phase));
  }
  setModuleAttr(funcOp, builder.getAttr<StepsAttr>(steps));
  setModuleAttr(funcOp, builder.getAttr<CircuitNameAttr>(funcOp.getName()));
}

void Module::setProtocolInfo(ProtocolInfo info) {
  setModuleAttr(getModule(), getBuilder().getAttr<ProtocolInfoAttr>(info));
}

mlir::func::FuncOp Module::endFunc(SourceLoc loc) {
  auto returnOp = builder.create<func::ReturnOp>(toLoc(loc));
  popIP();
  curModule = nullptr;
  return returnOp->getParentOfType<mlir::func::FuncOp>();
}

void Module::pushIP(Block* block) {
  ipStack.push_back(builder.saveInsertionPoint());
  builder.setInsertionPointToStart(block);
}

void Module::popIP() {
  builder.restoreInsertionPoint(ipStack.back());
  ipStack.pop_back();
}

void Module::runFunc(func::FuncOp func,
                     llvm::ArrayRef<Interpreter::BufferRef> bufs,
                     size_t startCycle,
                     size_t cycleCount) {
  if (func.getNumArguments() != bufs.size()) {
    throw std::runtime_error("Invalid number of argument, got " + std::to_string(bufs.size()) +
                             ", expected " + std::to_string(func.getNumArguments()));
  }
  // Load the argument into the interpreter
  Interpreter interpreter(getCtx());
  interpreter.setExternHandler(handler);
  for (size_t i = 0; i < bufs.size(); i++) {
    Value arg = func.getArgument(i);
    auto type = dyn_cast<BufferType>(arg.getType());
    if (!type) {
      throw std::runtime_error("Function has non-buffer types");
    }
    if (bufs[i].empty()) {
      (void)interpreter.makeBuf(arg, type.getSize(), type.getKind());
      continue;
    }
    interpreter.setBuf(arg, bufs[i]);
  }
  // Run the function
  for (size_t i = 0; i < cycleCount; i++) {
    interpreter.setCycle(startCycle + i);
    // try {
    if (failed(interpreter.runBlock(func.front())))
      throw(std::runtime_error("Evaluating block failed"));
    /*
    } catch (const std::exception& ex) {
      llvm::errs() << "Failed on cycle " << i << "\n";
      interpreter.setDebug(true);
      interpreter.runBlock(func.front());
    }
    */
  }
}

Val operator+(CaptureVal a, CaptureVal b) {
  return Val(getBuilder().createOrFold<AddOp>(a.getLoc(), a.getValue(), b.getValue()));
}

Val operator-(CaptureVal a, CaptureVal b) {
  return Val(getBuilder().createOrFold<SubOp>(a.getLoc(), a.getValue(), b.getValue()));
}

Val operator-(CaptureVal a) {
  return Val(getBuilder().createOrFold<NegOp>(a.getLoc(), a.getValue()));
}

Val operator*(CaptureVal a, CaptureVal b) {
  return Val(getBuilder().createOrFold<MulOp>(a.getLoc(), a.getValue(), b.getValue()));
}

Val operator&(CaptureVal a, CaptureVal b) {
  return Val(getBuilder().createOrFold<BitAndOp>(a.getLoc(), a.getValue(), b.getValue()));
}

Val operator/(CaptureVal a, CaptureVal b) {
  Value invB = getBuilder().createOrFold<InvOp>(b.getLoc(), b.getValue());
  return Val(getBuilder().createOrFold<MulOp>(a.getLoc(), a.getValue(), invB));
}

Val inv(CaptureVal a) {
  return Val(getBuilder().createOrFold<InvOp>(a.getLoc(), a.getValue()));
}

std::pair<DigestVal, std::vector<Val>> hashCheckedBytes(CaptureVal pt, size_t count) {
  llvm::SmallVector<Value, 2> outputs;
  getBuilder().createOrFold<HashCheckedBytesOp>(outputs, pt.getLoc(), pt.getValue(), count);
  DigestVal digest = outputs[0];
  std::vector<Val> evals;
  for (size_t i = 1; i < outputs.size(); i++) {
    evals.emplace_back(outputs[i]);
  }
  return {digest, evals};
}

HashCheckedPublicOutput hashCheckedBytesPublic(CaptureVal pt, size_t count) {
  HashCheckedPublicOutput out;
  llvm::SmallVector<Value, 2> outputs;
  getBuilder().createOrFold<HashCheckedBytesPublicOp>(outputs, pt.getLoc(), pt.getValue(), count);
  out.poseidon = outputs[0];
  out.sha = outputs[1];
  for (size_t i = 2; i < outputs.size(); i++) {
    out.vals.emplace_back(outputs[i]);
  }
  return out;
}

Val raisepow(CaptureVal a, size_t exp) {
  if (exp == 0) {
    // TODO: There seems to be some problem with DCE and constant
    // folding if we allow ExpandPowPattern to create a ConstOp.
    return 1;
  }
  return Val(getBuilder().createOrFold<PowOp>(a.getLoc(), a.getValue(), exp));
}

Val isz(CaptureVal a) {
  return Val(getBuilder().createOrFold<IsZeroOp>(a.getLoc(), a.getValue()));
}

void eqz(CaptureVal a) {
  getBuilder().create<EqualZeroOp>(a.getLoc(), a.getValue());
}

void eq(CaptureVal a, CaptureVal b) {
  Value diff = getBuilder().createOrFold<SubOp>(a.getLoc(), a.getValue(), b.getValue());
  getBuilder().create<EqualZeroOp>(a.getLoc(), diff);
}

void barrier(CaptureVal a) {
  getBuilder().create<BarrierOp>(a.getLoc(), a.getValue());
}

std::vector<Val> doExtern(const std::string& name,
                          const std::string& extra,
                          size_t outSize,
                          llvm::ArrayRef<Val> in,
                          SourceLoc loc) {
  MLIRContext* ctx = getBuilder().getContext();
  std::vector<Type> outTypes;
  for (size_t i = 0; i < outSize; i++) {
    outTypes.push_back(ValType::getBaseType(ctx));
  }
  std::vector<Value> inValues;
  for (auto& val : in) {
    inValues.push_back(val.getValue());
  }
  auto op = getBuilder().create<ExternOp>(toLoc(loc), outTypes, inValues, name, extra);
  std::vector<Val> outs;
  for (size_t i = 0; i < outSize; i++) {
    outs.emplace_back(op.getResult(i));
  }
  return outs;
}

void emitLayoutInternal(std::shared_ptr<ConstructInfo> info) {
  auto& builder = getBuilder();
  DenseMap<StringAttr, mlir::Type> layoutTypes;
  DenseMap<StringAttr, mlir::Attribute> layoutAttrs;
  transformLayout(getCtx(), info, layoutTypes, layoutAttrs);

  SmallVector<StringAttr> bufNames =
      llvm::map_to_vector(layoutAttrs, [](auto kv) { return kv.first; });
  // Make sure the order of buffer names is deterministic
  llvm::sort(bufNames, [&](auto a, auto b) { return a.strref() < b.strref(); });

  for (auto bufName : bufNames) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(curModule->getModule().getBody());
    builder.create<ZStruct::GlobalConstOp>(builder.getUnknownLoc(),
                                           bufName.str() + "_layout",
                                           layoutTypes.at(bufName),
                                           layoutAttrs.at(bufName));
  }
}

NondetGuard::NondetGuard(SourceLoc loc) {
  assert(curModule);
  auto nondetOp = getBuilder().create<NondetOp>(toLoc(loc));
  Block* innerBlock = new Block();
  nondetOp.getInner().push_back(innerBlock);
  curModule->pushIP(innerBlock);
}

NondetGuard::~NondetGuard() {
  assert(curModule);
  getBuilder().create<TerminateOp>(toLoc(SourceLoc()));
  curModule->popIP();
}

IfGuard::IfGuard(Val cond, SourceLoc loc) {
  assert(curModule);
  Block* innerBlock = new Block();
  auto ifOp = getBuilder().create<IfOp>(toLoc(loc), cond.getValue());
  ifOp.getInner().push_back(innerBlock);
  curModule->pushIP(innerBlock);
}

IfGuard::~IfGuard() {
  assert(curModule);
  getBuilder().create<TerminateOp>(toLoc(SourceLoc()));
  curModule->popIP();
}

void beginBack(size_t back, bool unchecked) {
  gBackDist = back;
  gBackUnchecked = unchecked;
  gBackUsed = false;
}

void endBack() {
  assert(gBackUsed && "No operations present inside BACK; perhaps try calling get() inside?");
  gBackDist = 0;
  gBackUnchecked = false;
}

DigestVal hash(llvm::ArrayRef<Val> inputs, bool flip, SourceLoc loc) {
  std::vector<Value> vals;
  for (const auto& in : inputs) {
    vals.push_back(in.getValue());
  }
  Type digestType = DigestType::get(getBuilder().getContext(), DigestKind::Default);
  Value out = getBuilder().create<HashOp>(toLoc(loc), digestType, flip, vals);
  return DigestVal(out);
}

DigestVal intoDigest(llvm::ArrayRef<Val> inputs, DigestKind kind, SourceLoc loc) {
  std::vector<Value> vals;
  for (const auto& in : inputs) {
    vals.push_back(in.getValue());
  }
  auto digestType = DigestType::get(getBuilder().getContext(), kind);
  Value out = getBuilder().create<IntoDigestOp>(toLoc(loc), digestType, vals);
  return DigestVal(out);
}

std::vector<Val> fromDigest(DigestVal digest, size_t size, SourceLoc loc) {
  auto& builder = getBuilder();
  Type valType = ValType::getBaseType(builder.getContext());
  std::vector<Type> types(size, valType);
  auto fromOp = builder.create<FromDigestOp>(toLoc(loc), types, digest.getValue());
  std::vector<Val> vals;
  for (Value out : fromOp.getOut()) {
    vals.push_back(Val(out));
  }
  return vals;
}

DigestVal fold(DigestVal lhs, DigestVal rhs, SourceLoc loc) {
  Value out = getBuilder().create<HashFoldOp>(toLoc(loc), lhs.getValue(), rhs.getValue());
  return DigestVal(out);
}

DigestVal taggedStruct(llvm::StringRef tag,
                       llvm::ArrayRef<DigestVal> digests,
                       llvm::ArrayRef<Val> vals,
                       SourceLoc loc) {
  std::vector<Value> digestVals;
  for (const auto& in : digests) {
    digestVals.push_back(in.getValue());
  }
  std::vector<Value> valsVals;
  for (const auto& in : vals) {
    valsVals.push_back(in.getValue());
  }

  Value out = getBuilder().create<TaggedStructOp>(toLoc(loc), tag, digestVals, valsVals);
  return DigestVal(out);
}

DigestVal taggedListCons(llvm::StringRef tag, DigestVal head, DigestVal tail, SourceLoc loc) {
  return taggedStruct(tag, {head, tail}, {}, loc);
}

void assert_eq(DigestVal lhs, DigestVal rhs, SourceLoc loc) {
  getBuilder().create<HashAssertEqOp>(toLoc(loc), lhs.getValue(), rhs.getValue());
}

std::vector<Val> ReadIopVal::readBaseVals(size_t count, bool flip, SourceLoc sloc) {
  auto& builder = getBuilder();
  Type valType = ValType::getBaseType(builder.getContext());
  std::vector<Type> types(count, valType);
  auto readOp = builder.create<Iop::ReadOp>(toLoc(sloc), types, getValue(), flip);
  std::vector<Val> out;
  for (size_t i = 0; i < count; i++) {
    out.emplace_back(readOp.getOuts()[i]);
  }
  return out;
}

std::vector<Val> ReadIopVal::readExtVals(size_t count, bool flip, SourceLoc sloc) {
  auto& builder = getBuilder();
  Type valType = ValType::getExtensionType(builder.getContext());
  std::vector<Type> types(count, valType);
  auto readOp = builder.create<Iop::ReadOp>(toLoc(sloc), types, getValue(), flip);
  std::vector<Val> out;
  for (size_t i = 0; i < count; i++) {
    out.emplace_back(readOp.getOuts()[i]);
  }
  return out;
}

std::vector<DigestVal> ReadIopVal::readDigests(size_t count, SourceLoc sloc) {
  auto& builder = getBuilder();
  Type digestType = DigestType::get(builder.getContext(), DigestKind::Default);
  std::vector<Type> types(count, digestType);
  auto readOp = builder.create<Iop::ReadOp>(toLoc(sloc), types, getValue(), false);
  std::vector<DigestVal> out;
  for (size_t i = 0; i < count; i++) {
    out.emplace_back(readOp.getOuts()[i]);
  }
  return out;
}

void ReadIopVal::commit(DigestVal digest, SourceLoc loc) {
  getBuilder().create<Iop::CommitOp>(toLoc(loc), getValue(), digest.getValue());
}

Val ReadIopVal::rngBits(uint32_t bits, SourceLoc loc) {
  auto& builder = getBuilder();
  Type valType = ValType::getBaseType(builder.getContext());
  return Val(builder.create<Iop::RngBitsOp>(toLoc(loc), valType, getValue(), bits));
}

Val ReadIopVal::rngBaseVal(SourceLoc loc) {
  auto& builder = getBuilder();
  Type valType = ValType::getBaseType(builder.getContext());
  return Val(builder.create<Iop::RngValOp>(toLoc(loc), valType, getValue()));
}

Val ReadIopVal::rngExtVal(SourceLoc loc) {
  auto& builder = getBuilder();
  Type valType = ValType::getExtensionType(builder.getContext());
  return Val(builder.create<Iop::RngValOp>(toLoc(loc), valType, getValue()));
}

Val select(Val idx, llvm::ArrayRef<Val> inputs, SourceLoc loc) {
  auto& builder = getBuilder();
  std::vector<Value> vals;
  for (const auto& in : inputs) {
    vals.push_back(in.getValue());
  }
  Value out = builder.create<SelectOp>(toLoc(loc), vals[0].getType(), idx.getValue(), vals);
  return out;
}

DigestVal select(Val idx, llvm::ArrayRef<DigestVal> inputs, SourceLoc loc) {
  auto& builder = getBuilder();
  std::vector<Value> vals;
  for (const auto& in : inputs) {
    vals.push_back(in.getValue());
  }
  Value out = builder.create<SelectOp>(toLoc(loc), vals[0].getType(), idx.getValue(), vals);
  return out;
}

Val normalize(Val in, SourceLoc loc) {
  auto& builder = getBuilder();
  Value out = builder.create<NormalizeOp>(toLoc(loc), in.getValue(), 0, "");
  return out;
}

void registerEdslCLOptions() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
}

} // namespace zirgen
