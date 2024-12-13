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

#include "zirgen/Dialect/Zll/IR/IR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/Interfaces.cpp.inc"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/compiler/zkp/baby_bear.h"
#include "zirgen/compiler/zkp/sha256.h"

mlir::ParseResult parseArrType(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::Type>& out) {
  mlir::Type oneType;
  size_t count;
  if (parser.parseType(oneType) || parser.parseStar() || parser.parseInteger<size_t>(count)) {
    return mlir::failure();
  }
  for (size_t i = 0; i < count; i++) {
    out.push_back(oneType);
  }
  return mlir::success();
}

void printArrType(mlir::OpAsmPrinter& p, mlir::Operation* op, mlir::TypeRange types) {
  if (types.size() == 0) {
    p.getStream() << "!zirgen.val<BabyBear>";
  } else {
    p.printType(types[0]);
  }
  p.getStream() << " * " << types.size();
}

mlir::ParseResult parseSelectType(mlir::OpAsmParser& parser,
                                  mlir::Type& out,
                                  mlir::Type& idx,
                                  llvm::SmallVectorImpl<mlir::Type>& in) {
  if (parseArrType(parser, in)) {
    return mlir::failure();
  }
  out = in[0];
  idx = in[0];
  return mlir::success();
}

void printSelectType(mlir::OpAsmPrinter& p,
                     mlir::Operation* op,
                     mlir::Type out,
                     mlir::Type idx,
                     mlir::TypeRange types) {
  printArrType(p, op, types);
}

#define GET_OP_CLASSES
#include "zirgen/Dialect/Zll/IR/Ops.cpp.inc"

using namespace mlir;

namespace zirgen::Zll {

// Folding

static bool matchVal(Attribute operand, uint64_t val) {
  auto op = dyn_cast_or_null<PolynomialAttr>(operand);
  return op && op.size() == 1 && op[0] == val;
}

static ExtensionField getExtensionField(Type ty) {
  return cast<ValType>(ty).getExtensionField();
}

static uint64_t integerFromAttr(Attribute attr) {
  auto polyAttr = dyn_cast<PolynomialAttr>(attr);
  if (polyAttr && polyAttr.size() == 1) {
    return polyAttr[0];
  }
  auto intAttr = cast<IntegerAttr>(attr);
  return intAttr.getUInt();
}

static ArrayRef<uint64_t> polynomialFromAttr(Attribute attr) {
  // TODO: Play less fast and loose with types here so we can just use cast<PolynomialAttr> instead
  // of having to allocate a new PolynomialAttr if we get an IntegerAttr.
  auto polyAttr = dyn_cast<PolynomialAttr>(attr);
  if (!polyAttr) {
    auto intAttr = cast<IntegerAttr>(attr);
    polyAttr = PolynomialAttr::get(attr.getContext(), {intAttr.getUInt()});
  }
  return polyAttr.asArrayRef();
}

template <typename Func> static OpFoldResult tryFold1(Attribute operand, Func func) {
  if (!operand) {
    return OpFoldResult();
  }
  auto in = polynomialFromAttr(operand);
  auto out = func(in);
  return OpFoldResult(PolynomialAttr::get(operand.getContext(), out));
}

template <typename Func> static OpFoldResult tryFold2(Attribute lhs, Attribute rhs, Func func) {
  if (!lhs || !rhs) {
    return OpFoldResult();
  }
  auto lhsPoly = polynomialFromAttr(lhs);
  auto rhsPoly = polynomialFromAttr(rhs);
  auto out = func(lhsPoly, rhsPoly);
  return OpFoldResult(PolynomialAttr::get(lhs.getContext(), out));
}

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) {
  return getCoefficientsAttr();
}

OpFoldResult StringOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

OpFoldResult IsZeroOp::fold(FoldAdaptor adaptor) {
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold1(adaptor.getIn(),
                  [f](ArrayRef<uint64_t> in) { return isZero(in) ? f.One() : f.Zero(); });
}

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold1(adaptor.getIn(), [f](ArrayRef<uint64_t> in) { return f.Neg(in); });
}

OpFoldResult InvOp::fold(FoldAdaptor adaptor) {
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold1(adaptor.getIn(), [f](ArrayRef<uint64_t> in) { return f.Inv(in); });
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  if (matchVal(adaptor.getRhs(), 0) && getLhs().getType() == getType()) {
    return getLhs();
  }
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold2(adaptor.getLhs(),
                  adaptor.getRhs(),
                  [f](ArrayRef<uint64_t> lhs, ArrayRef<uint64_t> rhs) { return f.Add(lhs, rhs); });
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  if (matchVal(adaptor.getRhs(), 0) && getLhs().getType() == getType()) {
    return getLhs();
  }
  if (getLhs() == getRhs()) {
    return OpFoldResult(PolynomialAttr::get(getLhs().getContext(), 0));
  }
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold2(adaptor.getLhs(),
                  adaptor.getRhs(),
                  [f](ArrayRef<uint64_t> lhs, ArrayRef<uint64_t> rhs) { return f.Sub(lhs, rhs); });
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  if (matchVal(adaptor.getRhs(), 1) && getLhs().getType() == getType()) {
    return getLhs();
  }
  if (matchVal(adaptor.getRhs(), 0)) {
    return OpFoldResult(PolynomialAttr::get(getContext(), 0));
  }
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold2(adaptor.getLhs(),
                  adaptor.getRhs(),
                  [f](ArrayRef<uint64_t> lhs, ArrayRef<uint64_t> rhs) { return f.Mul(lhs, rhs); });
}

OpFoldResult BitAndOp::fold(FoldAdaptor adaptor) {
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold2(
      adaptor.getLhs(), adaptor.getRhs(), [f](ArrayRef<uint64_t> lhs, ArrayRef<uint64_t> rhs) {
        return f.BitAnd(lhs, rhs);
      });
}

OpFoldResult ModOp::fold(FoldAdaptor adaptor) {
  ExtensionField f = getExtensionField(getOut().getType());
  return tryFold2(adaptor.getLhs(),
                  adaptor.getRhs(),
                  [f](ArrayRef<uint64_t> lhs, ArrayRef<uint64_t> rhs) { return f.Mod(lhs, rhs); });
}

OpFoldResult InRangeOp::fold(FoldAdaptor adaptor) {
  if (!adaptor.getLow() || !adaptor.getMid() || !adaptor.getHigh())
    return OpFoldResult();

  uint64_t low = integerFromAttr(adaptor.getLow());
  uint64_t mid = integerFromAttr(adaptor.getMid());
  uint64_t high = integerFromAttr(adaptor.getHigh());
  if (low > high)
    return {};

  ExtensionField f = getExtensionField(getOut().getType());
  auto out = (low <= mid && mid < high) ? f.One() : f.Zero();
  return OpFoldResult(PolynomialAttr::get(getContext(), out));
}

// Evaluation

LogicalResult ConstOp::evaluate(Interpreter& interp,
                                llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                EvalAdaptor& adaptor) {
  outs[0]->setVal(getCoefficients());
  return success();
}

LogicalResult NondetOp::evaluate(Interpreter& interp,
                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                 EvalAdaptor& adaptor) {
  return interp.runBlock(getInner().front());
}

LogicalResult IfOp::evaluate(Interpreter& interp,
                             llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                             EvalAdaptor& adaptor) {
  Interpreter::PolynomialRef condVal = adaptor.getCond()->getVal();
  if (!isZero(condVal)) {
    return interp.runBlock(getInner().front());
  }
  return success();
}

LogicalResult GetOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor) {
  if (getBack() > interp.getCycle() && !interp.getTotCycles()) {
    // TODO: Change this back to a throw once the DSL works enough that we can
    // avoid reading back too far.
    // throw std::runtime_error("Attempt to read back too far");
    llvm::errs() << "WARNING: attempt to read back too far\n";
    outs[0]->setVal(0);
    return success();
  }
  auto buf = adaptor.getBuf()->getBuf();
  size_t size = cast<BufferType>(getBuf().getType()).getSize();
  size_t totOffset = size * interp.getBackCycle(getBack()) + getOffset();
  if (totOffset >= buf.size()) {
    return emitError() << "Attempting to get out of bounds index " << totOffset
                       << " from buffer of size " << buf.size();
  }
  Interpreter::PolynomialRef val = buf[totOffset];
  if (isInvalid(val)) {
    if (!getOperation()->hasAttr("unchecked")) {
      auto diag = emitError() << "GetOp: Read before write";
      Operation* op = getOperation()->getParentOp();
      while (op) {
        diag.attachNote(op->getLoc()) << "contained here";
        op = op->getParentOp();
      }
      return diag;
    }
    outs[0]->setVal(0);
  } else {
    outs[0]->setVal(val);
  }

  return success();
}

LogicalResult SetOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor) {
  Interpreter::BufferRef vec = adaptor.getBuf()->getBuf();
  size_t size = dyn_cast<BufferType>(getBuf().getType()).getSize();
  size_t totOffset = size * interp.getCycle() + getOffset();
  if (totOffset >= vec.size()) {
    return emitError() << "Attempting to set out of bounds index " << totOffset
                       << " in buffer of size " << vec.size();
  }
  Interpreter::Polynomial& val = vec[totOffset];
  Interpreter::PolynomialRef newVal = adaptor.getIn()->getVal();
  if (!isInvalid(val) && val != newVal) {
    return emitError() << "SetOp: Invalid set, cur=" << val << ", new = " << newVal;
  }
  val = Interpreter::Polynomial(newVal.begin(), newVal.end());
  return success();
}

LogicalResult GetGlobalOp::evaluate(Interpreter& interp,
                                    llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                    EvalAdaptor& adaptor) {
  Interpreter::PolynomialRef val = adaptor.getBuf()->getBuf()[getOffset()];
  if (isInvalid(val)) {
    return emitError() << "GetGlobalOp: Read before write";
  }
  outs[0]->setVal(val);
  return success();
}

LogicalResult SetGlobalOp::evaluate(Interpreter& interp,
                                    llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                    EvalAdaptor& adaptor) {
  Interpreter::Polynomial& val = adaptor.getBuf()->getBuf()[getOffset()];
  Interpreter::PolynomialRef newVal = adaptor.getIn()->getVal();
  if (!isInvalid(val) && val != newVal) {
    return emitError() << "SetGlobalOp: Invalid set, cur=" << val << ", new = " << newVal;
  }
  val = Interpreter::Polynomial(newVal.begin(), newVal.end());
  return success();
}

LogicalResult SetGlobalDigestOp::evaluate(Interpreter& interp,
                                          llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                          EvalAdaptor& adaptor) {
  const Digest& digest = adaptor.getIn()->getDigest();
  auto buf = adaptor.getBuf()->getBuf();
  std::vector<uint32_t> encoded;
  switch (cast<DigestType>(getIn().getType()).getKind()) {
  case DigestKind::Default:
    encoded = interp.getHashSuite().encode(digest);
    break;
  case DigestKind::Sha256:
    encoded = shaHashSuite()->encode(digest);
    break;
  case DigestKind::Poseidon2:
    encoded = poseidon2HashSuite()->encode(digest);
    break;
  }
  for (size_t i = 0; i < encoded.size(); i++) {
    Interpreter::Polynomial& val = buf[getOffset() * kEncodedDigestSize + i];
    if (!isInvalid(val) && val[0] != encoded[i]) {
      return emitError() << "SetGlobalDigestOp: Invalid set; old = " << val
                         << ", new = " << encoded[i];
    }
    val[0] = encoded[i];
  }
  return success();
}

LogicalResult EqualZeroOp::evaluate(Interpreter& interp,
                                    llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                    EvalAdaptor& adaptor) {
  if (!isZero(adaptor.getIn()->getVal())) {
    if (interp.getSilenceErrors()) {
      return failure();
    } else {
      return emitError() << "EqualZeroOp: Not zero: " << adaptor.getIn()->getVal();
    }
  }
  return success();
}

LogicalResult BarrierOp::evaluate(Interpreter& interp,
                                  llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                  EvalAdaptor& adaptor) {
  // Do nothing
  return success();
}

LogicalResult IsZeroOp::evaluate(Interpreter& interp,
                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                 EvalAdaptor& adaptor,
                                 ExtensionField& field) {
  outs[0]->setVal(isZero(adaptor.getIn()->getVal()) ? field.One() : field.Zero());
  return success();
}

LogicalResult NegOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor,
                              ExtensionField& field) {
  outs[0]->setVal(field.Sub(field.Zero(), adaptor.getIn()->getVal()));
  return success();
}

LogicalResult InvOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor,
                              ExtensionField& field) {
  outs[0]->setVal(field.Inv(adaptor.getIn()->getVal()));
  return success();
}

LogicalResult AddOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor,
                              ExtensionField& field) {
  outs[0]->setVal(field.Add(adaptor.getLhs()->getVal(), adaptor.getRhs()->getVal()));
  return success();
}

LogicalResult SubOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor,
                              ExtensionField& field) {
  outs[0]->setVal(field.Sub(adaptor.getLhs()->getVal(), adaptor.getRhs()->getVal()));
  return success();
}

LogicalResult MulOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor,
                              ExtensionField& field) {
  outs[0]->setVal(field.Mul(adaptor.getLhs()->getVal(), adaptor.getRhs()->getVal()));
  return success();
}

LogicalResult PowOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor,
                              ExtensionField& field) {
  Interpreter::PolynomialRef base = adaptor.getIn()->getVal();
  Interpreter::Polynomial result = field.One();
  uint32_t exp = getExponent();
  for (size_t i = 0; i != exp; ++i) {
    result = field.Mul(result, base);
  }
  outs[0]->setVal(result);
  return success();
}

LogicalResult BitAndOp::evaluate(Interpreter& interp,
                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                 EvalAdaptor& adaptor,
                                 ExtensionField& field) {
  outs[0]->setVal(field.BitAnd(adaptor.getLhs()->getVal(), adaptor.getRhs()->getVal()));
  return success();
}

LogicalResult ModOp::evaluate(Interpreter& interp,
                              llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                              EvalAdaptor& adaptor,
                              ExtensionField& field) {
  outs[0]->setVal(field.Mod(adaptor.getLhs()->getVal(), adaptor.getRhs()->getVal()));
  return success();
}

LogicalResult VariadicPackOp::evaluate(Interpreter& interp,
                                       llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                       EvalAdaptor& adaptor) {
  MLIRContext* ctx = getContext();
  auto ins =
      llvm::to_vector(llvm::map_range(adaptor.getIn(), [&](auto in) { return in->getAttr(ctx); }));
  outs[0]->setAttr(ArrayAttr::get(ctx, ins));
  return success();
}

LogicalResult ExternOp::evaluate(Interpreter& interp,
                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                 EvalAdaptor& adaptor) {
  ExternHandler* handler = interp.getExternHandler();
  if (!handler) {
    return emitError() << "No extern handler set";
  }
  // TODO: We used to flatten extension field elements here... is that necessary?
  size_t outCount = getNumResults();
  std::optional<std::vector<uint64_t>> outFp =
      handler->doExtern(getName(), getExtra(), adaptor.getIn(), outCount);
  if (!outFp)
    return failure();
  assert(outFp->size() == outCount);
  for (size_t i = 0; i < getNumResults(); i++) {
    outs[i]->setVal((*outFp)[i]);
  }
  return success();
}

LogicalResult HashOp::evaluate(Interpreter& interp,
                               llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                               EvalAdaptor& adaptor) {
  size_t k = 0;
  if (!getIn().empty()) {
    k = cast<ValType>(getIn()[0].getType()).getFieldK();
  }
  std::vector<uint32_t> vals(k * getIn().size());
  for (size_t i = 0; i < getIn().size(); i++) {
    auto poly = adaptor.getIn()[i]->getVal();
    for (size_t j = 0; j < k; j++) {
      if (getFlip()) {
        vals[i * k + j] = poly[j];
      } else {
        vals[j * getIn().size() + i] = poly[j];
      }
    }
  }
  auto hashVal = interp.getHashSuite().hash(vals.data(), vals.size());
  outs[0]->setDigest(hashVal);
  return success();
}

LogicalResult IntoDigestOp::evaluate(Interpreter& interp,
                                     llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                     EvalAdaptor& adaptor) {
  std::vector<uint32_t> encoded;
  for (auto val : adaptor.getIn()) {
    auto poly = val->getVal();
    encoded.push_back(poly[0]);
  }
  Digest out;
  switch (cast<DigestType>(getOut().getType()).getKind()) {
  case DigestKind::Default:
    out = interp.getHashSuite().decode(encoded);
    break;
  case DigestKind::Sha256:
    out = shaHashSuite()->decode(encoded);
    break;
  case DigestKind::Poseidon2:
    out = poseidon2HashSuite()->decode(encoded);
    break;
  }
  outs[0]->setDigest(out);
  return success();
}

LogicalResult FromDigestOp::evaluate(Interpreter& interp,
                                     llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                     EvalAdaptor& adaptor) {
  const Digest& digest = adaptor.getIn()->getDigest();
  std::vector<uint32_t> encoded;
  switch (cast<DigestType>(getIn().getType()).getKind()) {
  case DigestKind::Default:
    encoded = interp.getHashSuite().encode(digest);
    break;
  case DigestKind::Sha256:
    encoded = shaHashSuite()->encode(digest, getOut().size());
    break;
  case DigestKind::Poseidon2:
    encoded = poseidon2HashSuite()->encode(digest, getOut().size());
    break;
  }
  for (size_t i = 0; i < encoded.size(); i++) {
    outs[i]->setVal(encoded[i]);
  }
  return success();
}

LogicalResult HashFoldOp::evaluate(Interpreter& interp,
                                   llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                   EvalAdaptor& adaptor) {
  outs[0]->setDigest(
      interp.getHashSuite().hashPair(adaptor.getLhs()->getDigest(), adaptor.getRhs()->getDigest()));
  return success();
}

LogicalResult TaggedStructOp::evaluate(Interpreter& interp,
                                       llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                       EvalAdaptor& adaptor) {
  std::vector<uint32_t> words;

  // Add the hashed tag to the front of the buffer.
  Digest tag = shaHash(getTag().str());
  words.insert(words.end(), tag.words, tag.words + 8);

  // Add digests to the buffer.
  for (auto val : adaptor.getDigests()) {
    Digest digest = val->getDigest();
    words.insert(words.end(), digest.words, digest.words + 8);
  }

  // Add vals to the buffer.
  for (auto val : adaptor.getVals()) {
    auto poly = val->getVal();
    assert(poly.size() == 1);
    words.push_back(poly[0]);
  }

  // Count bits of data, plus 16 bits for hash count
  size_t bitCount = words.size() * 32 + 16;
  // Specify the number of hashes + pad
  words.push_back(adaptor.getDigests().size() | 0x800000);
  if (words.size() % 16 == 15) {
    // Not enough room to add size, need at least 2 words
    // Add one more word to go to next block
    words.push_back(0);
  }
  // Now push zeros until we hit the end of the block
  while (words.size() % 16 != 15) {
    words.push_back(0);
  }
  bitCount = (bitCount & 0x0000FFFF) << 16 | (bitCount & 0xFFFF0000) >> 16;
  bitCount = (bitCount & 0x00FF00FF) << 8 | (bitCount & 0xFF00FF00) >> 8;
  words.push_back(bitCount);
  outs[0]->setDigest(shaHash(words.data(), words.size()));
  return success();
}

LogicalResult HashAssertEqOp::evaluate(Interpreter& interp,
                                       llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                       EvalAdaptor& adaptor) {
  auto lhs = adaptor.getLhs()->getDigest();
  auto rhs = adaptor.getRhs()->getDigest();
  if (lhs != rhs) {
    emitError() << "HashAssertEqOp: Mismatch, lhs=" << lhs << " rhs=" << rhs;
    return failure();
  }
  return success();
}

LogicalResult SelectOp::evaluate(Interpreter& interp,
                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                 EvalAdaptor& adaptor) {
  uint64_t idxNum = adaptor.getIdx()->getVal()[0];
  if (idxNum >= getElems().size()) {
    emitError() << "Select index out of range: size = " << getElems().size()
                << ", idx = " << idxNum;
    return failure();
  }
  outs[0]->setInterpVal(adaptor.getElems()[idxNum]);
  return success();
}

LogicalResult HashCheckedBytesOp::evaluate(Interpreter& interp,
                                           llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                           EvalAdaptor& adaptor) {
  auto psuite = poseidon2HashSuite();
  // Special case for 0 outputs
  if (getOperation()->getResults().size() == 1) {
    auto hashVal = psuite->hash(nullptr, 0);
    outs[0]->setDigest(hashVal);
    return success();
  }

  auto field = llvm::cast<ValType>(getOperation()->getResult(1).getType()).getExtensionField();

  ExternHandler* handler = interp.getExternHandler();
  if (!handler) {
    return emitError() << "No extern handler set";
  }

  auto evalPt = adaptor.getEvalPt()->getVal();
  std::vector<uint32_t> coeffs;
  std::vector<uint32_t> accumCoeffs(16, 0);
  size_t countAccumed = 0;
  for (size_t i = 0; i < adaptor.getEvalsCount(); i++) {
    std::optional<std::vector<uint64_t>> newCoeffs =
        handler->doExtern("readCoefficients", "", {}, 16);
    assert(newCoeffs && "readCoefficients shouldn't fail");
    auto result = field.Zero();
    auto currentPower = field.One();
    for (size_t j = 0; j < 16; j++) {
      if ((*newCoeffs)[j] > 255) {
        throw std::runtime_error("Coefficient fails range check");
      }
      result = field.Add(result, field.Mul((*newCoeffs)[j], currentPower));
      currentPower = field.Mul(currentPower, evalPt);
      accumCoeffs[j] *= 256;
      accumCoeffs[j] += (*newCoeffs)[j];
    }
    outs[1 + i]->setVal(result);
    countAccumed++;
    if (countAccumed == 3) {
      coeffs.insert(coeffs.end(), accumCoeffs.begin(), accumCoeffs.end());
      std::fill(accumCoeffs.begin(), accumCoeffs.end(), 0);
      countAccumed = 0;
    }
  }
  if (countAccumed != 0) {
    coeffs.insert(coeffs.end(), accumCoeffs.begin(), accumCoeffs.end());
  }
  auto hashVal = psuite->hash(coeffs.data(), coeffs.size());
  outs[0]->setDigest(hashVal);
  return success();
}

LogicalResult HashCheckedBytesPublicOp::evaluate(Interpreter& interp,
                                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                                 EvalAdaptor& adaptor) {
  auto psuite = poseidon2HashSuite();

  if (adaptor.getEvalsCount() == 0) {
    return emitError() << "Can't do a empty HashCheckedBytesPublicOp";
  }

  auto field = llvm::cast<ValType>(getOperation()->getResult(2).getType()).getExtensionField();

  ExternHandler* handler = interp.getExternHandler();
  if (!handler) {
    return emitError() << "No extern handler set";
  }

  auto evalPt = adaptor.getEvalPt()->getVal();
  std::vector<uint32_t> coeffs;
  for (size_t i = 0; i < adaptor.getEvalsCount(); i++) {
    std::optional<std::vector<uint64_t>> newCoeffs =
        handler->doExtern("readCoefficients", "", {}, 16);
    assert(newCoeffs && "readCoefficients shouldn't fail");
    auto result = field.Zero();
    auto currentPower = field.One();
    for (size_t j = 0; j < 16; j++) {
      if ((*newCoeffs)[j] > 255) {
        throw std::runtime_error("Coefficient fails range check");
      }
      result = field.Add(result, field.Mul((*newCoeffs)[j], currentPower));
      currentPower = field.Mul(currentPower, evalPt);
    }
    outs[2 + i]->setVal(result);
    coeffs.insert(coeffs.end(), (*newCoeffs).begin(), (*newCoeffs).end());
  }
  auto hashVal1 = psuite->hash(coeffs.data(), coeffs.size());
  outs[0]->setDigest(hashVal1);
  assert(coeffs.size() % 4 == 0);
  std::vector<uint32_t> coeffs2(coeffs.size() / 4);
  for (size_t i = 0; i < coeffs2.size(); i++) {
    for (size_t j = 0; j < 4; j++) {
      coeffs2[i] |= coeffs[4 * i + j] << (8 * j);
    }
  }
  while (coeffs2.size() % 16 != 0) {
    coeffs2.push_back(0);
  }
  auto hashVal2 = shaHash(coeffs2.data(), coeffs2.size());
  outs[1]->setDigest(hashVal2);
  return success();
}

LogicalResult NormalizeOp::evaluate(Interpreter& interp,
                                    llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                    EvalAdaptor& adaptor) {
  outs[0]->setVal(adaptor.getIn()->getVal());
  return success();
}

// Reduction support

#define GET(x) get(ranges, x)
#define SET(x, y) set(ranges, x, y)

bool ConstOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  int64_t val = getCoefficients()[0];
  SET(getOut(), BigIntRange(val));
  return true;
}

bool SetGlobalOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  return GET(getIn()).inRangeP();
}

bool SetGlobalDigestOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  return GET(getIn()).inRangeP();
}

bool EqualZeroOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  return GET(getIn()).inRangeP();
}

bool IsZeroOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), BigIntRange(0, 1));
  return true;
}

bool NegOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), -GET(getIn()));
  return true;
}

bool InvOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), BigIntRange::rangeP());
  return GET(getIn()).inRangeP();
}

bool AddOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), GET(getLhs()) + GET(getRhs()));
  return GET(getOut()).inRangeMax();
}

bool SubOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), GET(getLhs()) - GET(getRhs()));
  return GET(getOut()).inRangeMax();
}

bool MulOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), GET(getLhs()) * GET(getRhs()));
  return ranges[getOut()].inRangeMax();
}

bool BitAndOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), BigIntRange::rangeP());
  return GET(getLhs()).inRangeP() && GET(getRhs()).inRangeP();
}

bool ModOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  // TODO: This is pessemistic, but also, should never get called.
  SET(getOut(), BigIntRange::rangeP());
  return GET(getLhs()).inRangeP() && GET(getRhs()).inRangeP();
}

bool HashOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  bool allGood = true;
  for (Value v : getIn()) {
    if (!GET(v).inRangeP()) {
      allGood = false;
      break;
    }
  }
  return allGood;
}

bool HashFoldOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  return true;
}

bool HashAssertEqOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  return true;
}

bool SelectOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  if (!GET(getIdx()).inRangeP()) {
    return false;
  }
  if (!mlir::isa<ValType>(getElems()[0].getType())) {
    return true;
  }
  BigInt low = GET(getElems()[0]).getLow();
  BigInt high = GET(getElems()[0]).getHigh();
  for (size_t i = 1; i < getElems().size(); i++) {
    low = std::min(low, GET(getElems()[i]).getLow());
    high = std::max(high, GET(getElems()[i]).getHigh());
  }
  BigIntRange out(low, high);
  SET(getOut(), out);
  return out.inRangeMax();
}

bool NormalizeOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), BigIntRange::rangeP());
  return true;
}

#undef GET
#undef SET

// Canonicalizers

template <class Op> struct RemoveEmptyBody : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const override {
    if (isa<TerminateOp>(op.getInner().front().front())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

void NondetOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<RemoveEmptyBody<NondetOp>>(context);
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<RemoveEmptyBody<IfOp>>(context);
}

class CanonicalizeEqualZeroOp : public RewritePattern {
public:
  CanonicalizeEqualZeroOp(MLIRContext* context)
      : RewritePattern(EqualZeroOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation* opBase, PatternRewriter& rewriter) const override {
    auto op = cast<EqualZeroOp>(opBase);
    // Check to see if the input is a constant
    if (auto constOp = op.getIn().getDefiningOp<ConstOp>()) {
      if (isZero(constOp.getCoefficients())) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

void EqualZeroOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {
  patterns.add<CanonicalizeEqualZeroOp>(context);
}

template <class Op> struct RemoveSlicePattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const override {
    auto buf = op.getBuf();
    if (!buf) {
      return failure();
    }
    auto sliceOp = dyn_cast_or_null<SliceOp>(buf.getDefiningOp());
    if (!sliceOp) {
      return failure();
    }
    rewriter.modifyOpInPlace(op.getOperation(), [&]() {
      op->setOperand(0, sliceOp.getIn());
      op->setAttr("offset", rewriter.getUI32IntegerAttr(op.getOffset() + sliceOp.getOffset()));
    });
    return success();
  }
};

template <class Op> struct LiftPattern : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IfOp op, PatternRewriter& rewriter) const override {
    bool foundAny = false;
    for (auto& region : op->getRegions()) {
      for (auto opToMove : region.getOps<Op>()) {
        foundAny = true;

        opToMove->moveBefore(op);
      }
    }
    if (foundAny) {
      return success();
    } else {
      return failure();
    }
  }
};

template <class Op> struct RemoveBackPattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const override {
    auto backOp = dyn_cast_or_null<BackOp>(op.getBuf().getDefiningOp());
    if (!backOp) {
      return failure();
    }
    rewriter.modifyOpInPlace(op.getOperation(), [&]() {
      op->setOperand(0, backOp.getIn());
      op->setAttr("back", rewriter.getUI32IntegerAttr(op.getBack() + backOp.getBack()));
    });
    return success();
  }
};

struct ExpandPowPattern : public OpRewritePattern<PowOp> {
  using OpRewritePattern<PowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op, PatternRewriter& rewriter) const override {
    auto loc = op->getLoc();
    mlir::Type ty = op.getOut().getType();
    if (op.getExponent() == 0) {
      rewriter.replaceOp(op, rewriter.create<ConstOp>(loc, ty, 1));
    } else if (op.getExponent() == 1) {
      rewriter.replaceOp(op, op.getOperand());
    } else if (op.getExponent() % 2) {
      // x^n = x * x^(n-1)
      rewriter.replaceOp(op,
                         rewriter.create<MulOp>(
                             loc,
                             ty,
                             op.getOperand(),
                             rewriter.create<PowOp>(loc, ty, op.getIn(), op.getExponent() - 1)));
    } else {
      assert((op.getExponent() % 2) == 0);
      // x^2n = x^n * x^n
      auto sqrtVal = rewriter.create<PowOp>(loc, ty, op.getIn(), op.getExponent() / 2);
      rewriter.replaceOp(op, rewriter.create<MulOp>(loc, ty, sqrtVal, sqrtVal));
    }
    return success();
  }
};

void GetOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<RemoveSlicePattern<GetOp>>(context);
  results.insert<RemoveBackPattern<GetOp>>(context);
}

void SetOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<RemoveSlicePattern<SetOp>>(context);
}

void GetGlobalOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<RemoveSlicePattern<GetGlobalOp>>(context);
}

void SetGlobalOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<RemoveSlicePattern<SetGlobalOp>>(context);
}

void SetGlobalDigestOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                    MLIRContext* context) {
  results.insert<RemoveSlicePattern<SetGlobalDigestOp>>(context);
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<ExpandPowPattern>(context);
}

// Type inference
LogicalResult BackOp::inferReturnTypes(MLIRContext* ctx,
                                       std::optional<Location> loc,
                                       Adaptor adaptor,
                                       SmallVectorImpl<Type>& out) {
  out.push_back(adaptor.getIn().getType());
  return success();
}

LogicalResult SliceOp::inferReturnTypes(MLIRContext* ctx,
                                        std::optional<Location> loc,
                                        Adaptor adaptor,
                                        SmallVectorImpl<Type>& out) {
  auto inType = cast<BufferType>(adaptor.getIn().getType());
  uint32_t offset = adaptor.getOffset();
  uint32_t size = adaptor.getSize();
  if (offset + size > inType.getSize()) {
    return failure();
  }
  auto outType = BufferType::get(ctx, inType.getElement(), size, inType.getKind());
  out.push_back(outType);
  return success();
}

static LogicalResult inferTypes(MLIRContext* ctx, ValueRange vals, SmallVectorImpl<Type>& out) {
  assert(1 <= vals.size());
  auto vt = cast<ValType>(vals[0].getType());
  auto fieldP = vt.getFieldP();
  auto fieldK = vt.getFieldK();
  for (size_t i = 1; i < vals.size(); ++i) {
    vt = cast<ValType>(vals[i].getType());
    if (vt.getFieldP() != fieldP) {
      return failure();
    }
    fieldK = std::max(fieldK, vt.getFieldK());
  }
  out.push_back(ValType::get(ctx, fieldP, fieldK));
  return success();
}

LogicalResult AddOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult BitAndOp::inferReturnTypes(MLIRContext* ctx,
                                         std::optional<Location> loc,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult ModOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult ConstOp::inferReturnTypes(MLIRContext* ctx,
                                        std::optional<Location> loc,
                                        Adaptor adaptor,
                                        SmallVectorImpl<Type>& out) {
  uint64_t size = adaptor.getCoefficients().size();
  out.push_back(ValType::get(ctx, kFieldPrimeDefault, size));
  return success();
}

LogicalResult GetGlobalOp::inferReturnTypes(MLIRContext* ctx,
                                            std::optional<Location> loc,
                                            Adaptor adaptor,
                                            SmallVectorImpl<Type>& out) {
  auto bufType = cast<BufferType>(adaptor.getBuf().getType());
  out.push_back(bufType.getElement());
  return success();
}

LogicalResult GetOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  auto bufType = cast<BufferType>(adaptor.getBuf().getType());
  out.push_back(bufType.getElement());
  return success();
}

LogicalResult PowOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult InvOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult IsZeroOp::inferReturnTypes(MLIRContext* ctx,
                                         std::optional<Location> loc,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult MulOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult NegOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult SubOp::inferReturnTypes(MLIRContext* ctx,
                                      std::optional<Location> loc,
                                      Adaptor adaptor,
                                      SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

LogicalResult HashCheckedBytesOp::inferReturnTypes(MLIRContext* ctx,
                                                   std::optional<Location> loc,
                                                   Adaptor adaptor,
                                                   SmallVectorImpl<Type>& out) {
  // HashCheckedBytes only works with Poseidon2 digests, not SHA
  out.push_back(DigestType::get(ctx, DigestKind::Poseidon2));
  for (size_t i = 0; i < adaptor.getEvalsCount(); i++) {
    out.push_back(adaptor.getEvalPt().getType());
  }
  return success();
}

LogicalResult HashCheckedBytesPublicOp::inferReturnTypes(MLIRContext* ctx,
                                                         std::optional<Location> loc,
                                                         Adaptor adaptor,
                                                         SmallVectorImpl<Type>& out) {
  // HashCheckedBytes only works with Poseidon2 digests, not SHA
  out.push_back(DigestType::get(ctx, DigestKind::Poseidon2));
  out.push_back(DigestType::get(ctx, DigestKind::Sha256));
  for (size_t i = 0; i < adaptor.getEvalsCount(); i++) {
    out.push_back(adaptor.getEvalPt().getType());
  }
  return success();
}

LogicalResult NormalizeOp::inferReturnTypes(MLIRContext* ctx,
                                            std::optional<Location> loc,
                                            Adaptor adaptor,
                                            SmallVectorImpl<Type>& out) {
  return inferTypes(ctx, adaptor.getOperands(), out);
}

// Verifiers

// TODO: We should check all the index size stuff for buffer access.

LogicalResult GetOp::verify() {
  // Verify that buffer is not global
  if (cast<BufferType>(getBuf().getType()).getKind() == BufferKind::Global) {
    return emitError() << "Wrong type of buffer";
  }
  return success();
}

LogicalResult SetOp::verify() {
  // Verify that buffer is not global
  if (cast<BufferType>(getBuf().getType()).getKind() == BufferKind::Global) {
    return emitError() << "Set may not be used on global buffers";
  }
  // Make sure the element we're storing is the same type
  auto bufType = getBuf().getType().getElement();
  auto valType = getIn().getType();
  if (bufType.getFieldK() < valType.getFieldK() || bufType.getFieldP() != valType.getFieldP())
    return emitError() << "Wrong type of field element; tring to store " << valType << " in "
                       << bufType << "\n";
  return success();
}

LogicalResult GetGlobalOp::verify() {
  // Verify that buffer is not global
  if (cast<BufferType>(getBuf().getType()).getKind() != BufferKind::Global &&
      cast<BufferType>(getBuf().getType()).getKind() != BufferKind::Temporary) {
    return failure();
  }
  return success();
}

LogicalResult SetGlobalOp::verify() {
  // Verify that buffer is not global
  if (cast<BufferType>(getBuf().getType()).getKind() != BufferKind::Global &&
      cast<BufferType>(getBuf().getType()).getKind() != BufferKind::Temporary) {
    return failure();
  }
  return success();
}

LogicalResult SetGlobalDigestOp::verify() {
  // Verify that buffer is not global
  if (cast<BufferType>(getBuf().getType()).getKind() != BufferKind::Global) {
    return failure();
  }
  return success();
}

LogicalResult IntoDigestOp::verify() {
  // Verify inputs are Fp
  if (cast<ValType>(getIn()[0].getType()).getFieldK() != 1) {
    return emitError() << "Values must be in base field";
  }
  // 16 elements is OK for either type of hash
  if (getIn().size() == 16) {
    return success();
  }
  // 32 (bytes) is also OK for Sha256
  if (cast<DigestType>(getOut().getType()).getKind() == DigestKind::Sha256 &&
      getIn().size() == 32) {
    return success();
  }
  // 8 (field elements) is also OK for Poseidon2
  if (cast<DigestType>(getOut().getType()).getKind() == DigestKind::Poseidon2 &&
      getIn().size() == 8) {
    return success();
  }
  return emitError() << "Unexpected number of values supplied for this hash type";
}

LogicalResult FromDigestOp::verify() {
  // Verify inputs are Fp
  if (cast<ValType>(getOut()[0].getType()).getFieldK() != 1) {
    return failure();
  }
  // 16 elements is OK for either type of hash
  if (getOut().size() == 16) {
    return success();
  }
  // 32 (bytes) is also OK for Sha256
  if (cast<DigestType>(getIn().getType()).getKind() == DigestKind::Sha256 &&
      getOut().size() == 32) {
    return success();
  }
  // 8 (field elements) is also OK for Poseidon2
  if (cast<DigestType>(getIn().getType()).getKind() == DigestKind::Poseidon2 &&
      getOut().size() == 8) {
    return success();
  }
  return failure();
}

LogicalResult TaggedStructOp::verify() {
  // Verify digests are Sha256
  for (auto digest : getDigests()) {
    if (!isa<DigestType>(digest.getType())) {
      return emitOpError() << "Input type is not a digest";
    }
  }
  // Verify values are Fps
  for (auto val : getVals()) {
    if (cast<ValType>(val.getType()).getFieldK() != 1) {
      return emitOpError() << "Input vals must be in the base field";
    }
  }
  // Verify there are less than 64k digest
  if (getDigests().size() > 65535) {
    return emitOpError() << "Too many digests";
  }
  return success();
}

void IfOp::emitStatement(codegen::CodegenEmitter& cg) {
  cg.emitConditional(getCond(), getInner());
}

void NondetOp::emitStatement(codegen::CodegenEmitter& cg) {
  cg.emitRegion(getInner());
}

void TrueOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(cg.getStringAttr("trivialConstraint"), {});
}

void VariadicPackOp::emitExpr(codegen::CodegenEmitter& cg) {
  using codegen::CodegenValue;
  auto elements = llvm::to_vector_of<CodegenValue>(
      llvm::map_range(getIn(), [&](auto member) { return CodegenValue(member).owned(); }));
  cg.emitArrayConstruct(getType(), getType().getElement(), elements);
}

void ExternOp::emitExpr(codegen::CodegenEmitter& cg) {
  llvm::SmallVector<codegen::EmitPart> macroParts = {
      /*extern name=*/codegen::CodegenIdent<codegen::IdentKind::Func>(getNameAttr())};
  llvm::append_range(macroParts, getOperands());
  cg.emitInvokeMacro(cg.getStringAttr("invokeExtern"), /*contextArgs=*/{"ctx"}, macroParts);
}

void EqualZeroOp::emitExpr(codegen::CodegenEmitter& cg) {
  cg.emitInvokeMacro(cg.getStringAttr("eqz"),
                     {getIn(), [&]() { cg.emitEscapedString(getLocString(getLoc())); }});
}

void AndEqzOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  if (getVal().getType().getFieldK() > 1)
    cg.emitFuncCall(cg.getStringAttr("andEqzExt"), /*contextArgs=*/{"ctx"}, {getIn(), getVal()});
  else
    cg.emitFuncCall(cg.getStringAttr("andEqz"), /*contextArgs=*/{"ctx"}, {getIn(), getVal()});
}

void AndCondOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  if (getCond().getType().getFieldK() > 1)
    cg.emitFuncCall(cg.getStringAttr("andCondExt"), {getIn(), getCond(), getInner()});
  else
    cg.emitFuncCall(cg.getStringAttr("andCond"), {getIn(), getCond(), getInner()});
}

void GetOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(
      cg.getStringAttr("get"),
      /*ContextArgs=*/{"ctx"},
      {getBuf(), cg.guessAttributeType(getOffsetAttr()), cg.guessAttributeType(getBackAttr())});
}

void SetOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(cg.getStringAttr("set"),
                  /*ContextArgs=*/{"ctx"},
                  {getBuf(), cg.guessAttributeType(getOffsetAttr()), getIn()});
}

void SetGlobalOp::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
  cg.emitFuncCall(cg.getStringAttr("set_global"),
                  /*ContextArgs=*/{"ctx"},
                  {getBuf(), cg.guessAttributeType(getOffsetAttr()), getIn()});
}

} // namespace zirgen::Zll
