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

#include "zirgen/Dialect/Zll/IR/Attrs.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

namespace mlir {

ParseResult parsePolynomialAttr(OpAsmParser& parser, PolynomialAttr& coefficientsAttr) {
  // Parse a single integer as an element of the base field. Note that this
  // doesn't require square brackets.
  uint64_t elem;
  if (parser.parseOptionalInteger(elem).has_value()) {
    coefficientsAttr = PolynomialAttr::get(parser.getContext(), {elem});
    return success();
  }

  // Parse an array of integers as elements of an extension field. Note that
  // this does require square brackets, and also that extensions of degree 0
  // are not allowed.
  if (parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalRSquare())) {
    SMLoc loc = parser.getCurrentLocation();
    parser.emitError(loc, "A field extension cannot have a degree of 0.");
    return failure();
  }
  SmallVector<uint64_t, 4> data;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseInteger(elem))
          return failure();
        data.push_back(elem);
        return success();
      })))
    return failure();
  if (parser.parseRSquare())
    return failure();
  assert(data.size() > 0);
  coefficientsAttr = PolynomialAttr::get(parser.getContext(), data);
  return success();
}

void printPolynomialAttr(OpAsmPrinter& printer,
                         const OpState&,
                         const PolynomialAttr& coefficients) {
  ArrayRef<uint64_t> array = coefficients;
  assert(array.size() > 0);
  if (array.size() == 1) {
    printer << array[0];
  } else {
    printer << "[";
    llvm::interleaveComma(array, printer, [&](uint64_t value) { printer << value; });
    printer << "]";
  }
}

} // namespace mlir

using namespace mlir;

namespace zirgen::Zll {

ParseResult parseField(AsmParser& p, FieldAttr& field) {
  std::string id;
  if (p.parseOptionalKeywordOrString(&id).failed()) {
    return p.emitError(p.getCurrentLocation(), "Expected name of field");
  }

  field = zirgen::Zll::getField(p.getContext(), id);
  if (!field) {
    return p.emitError(p.getCurrentLocation(), "Unknown field " + id);
  }
  return success();
}

void printField(AsmPrinter& printer, const FieldAttr& field) {
  printer << field.getName();
}

mlir::ParseResult parseFieldExt(mlir::AsmParser& parser, mlir::UnitAttr& extAttr) {
  if (parser.parseOptionalKeyword("ext").succeeded()) {
    extAttr = UnitAttr::get(parser.getContext());
  } else {
    extAttr = {};
  }
  return success();
}

void printFieldExt(mlir::AsmPrinter& printer, const mlir::UnitAttr& extAttr) {
  if (extAttr) {
    // TODO: This extra space seems to be sometimes automatic and sometimes not.  Track down
    // what we need to do here to get exactly one space before the "ext" in all cases.
    printer << " ";
    printer.printKeywordOrString("ext");
  }
}

Field FieldAttr::getBaseField() const {
  return Field(getPrime());
}

ExtensionField FieldAttr::getBaseExtensionField() const {
  return ExtensionField(getPrime(), 1);
}

ExtensionField FieldAttr::getExtExtensionField() const {
  return ExtensionField(getPrime(), getExtDegree());
}

LogicalResult FieldAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                StringRef name,
                                uint64_t prime,
                                size_t extDegree,
                                ArrayRef<uint64_t> polynomial) {
  if (polynomial.size() != extDegree) {
    return emitError() << "Polynomial extension degree mismatch in the " << name << " field";
  }
  return success();
}

llvm::SmallVector<BufferDescAttr> BuffersAttr::getTapBuffers() const {
  llvm::SmallVector<BufferDescAttr> tapBuffers;
  for (auto bufDesc : getBuffers()) {
    if (!bufDesc.getRegGroupId())
      continue;

    size_t regGroupId = *bufDesc.getRegGroupId();

    if (tapBuffers.size() <= regGroupId)
      tapBuffers.resize(regGroupId + 1);
    tapBuffers[regGroupId] = bufDesc;
  }
  return tapBuffers;
}

BufferDescAttr BuffersAttr::getBuffer(StringRef bufName) const {
  for (auto bufDesc : getBuffers()) {
    if (bufDesc.getName() == bufName)
      return bufDesc;
  }
  return {};
}

LogicalResult ProtocolInfoAttr::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                       StringRef protocolInfo) {
  if (protocolInfo.size() != PROTOCOL_INFO_LEN) {
    return emitError() << "Protocol info must be a fixed length of " << PROTOCOL_INFO_LEN;
  }

  return success();
}

ProtocolInfo ProtocolInfoAttr::getValue() {
  ProtocolInfo result;
  auto str = getProtocolInfo();
  assert(str.size() == PROTOCOL_INFO_LEN);
  std::copy(str.begin(), str.end(), result.begin());
  result[PROTOCOL_INFO_LEN] = '\0';
  return result;
}

ProtocolInfoAttr ProtocolInfoAttr::get(MLIRContext* ctx, ProtocolInfo info) {
  assert(info[PROTOCOL_INFO_LEN] == '\0');
  return get(ctx, StringRef(info.data()));
}

LogicalResult BuffersAttr::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                  ArrayRef<BufferDescAttr> buffers) {
  llvm::StringSet bufNames;
  SmallVector<size_t> tapGroupIds;

  for (auto buf : buffers) {
    if (bufNames.contains(buf.getName()))
      return emitError() << "Duplicate buffer name " << buf.getName() << "\n";

    if (buf.getRegGroupId())
      tapGroupIds.push_back(*buf.getRegGroupId());
  }

  llvm::sort(tapGroupIds);
  for (auto [idx, groupId] : llvm::enumerate(tapGroupIds)) {
    if (idx != groupId) {
      return emitError() << "Tap groups must start at zero and be monotonically increasing";
    }
  }

  return success();
}

} // namespace zirgen::Zll
