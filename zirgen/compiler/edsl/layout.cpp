// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/edsl/edsl.h"

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/compiler/edsl/component.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

namespace zirgen {

struct FieldList {
  enum class KeyType { REGULAR, EMPTY, TOMBSTONE };
  KeyType keyType = KeyType::REGULAR;
  SmallVector<FieldInfo> fields;
};

struct FieldListMapInfo {
  static FieldList getEmptyKey() { return FieldList{.keyType = FieldList::KeyType::EMPTY}; }

  static FieldList getTombstoneKey() { return FieldList{.keyType = FieldList::KeyType::TOMBSTONE}; }

  static unsigned getHashValue(const FieldList& val) {
    return llvm::hash_combine_range(val.fields.begin(), val.fields.end());
  }

  static bool isEqual(const FieldList& lhs, const FieldList& rhs) {
    return lhs.keyType == rhs.keyType && lhs.fields == rhs.fields;
  }
};

namespace {

class TransformLayout {
public:
  TransformLayout(MLIRContext* ctx) : builder(ctx) {}

  // Transforms the given ConstructInfo into a layout.
  std::pair<mlir::Attribute, mlir::Type> transform(std::shared_ptr<ConstructInfo> info);

private:
  // Generates a new unique type name based on origName.
  StringAttr generateTypeName(StringRef origName);

  // Generates a RefAttr describing the given slice value.
  ZStruct::RefAttr getRefAttr(mlir::Value buf);

  // Roll up arrays by parsing subcomponent names of the format ...[...]
  SmallVector<std::pair<Type, NamedAttribute>>
  rollUpArrays(SmallVector<std::pair<Type, NamedAttribute>> subAttrs);

  // Lists of fields for which we have already calculated the type names for, by type
  DenseMap<FieldList /* fields */, StringAttr /* type name */, FieldListMapInfo> types;

  // Type names we've already assigned, to avoid duplication.
  DenseSet<StringAttr> typeNamesUsed;

  Builder builder;
  size_t nextId = 0;
};

ZStruct::RefAttr TransformLayout::getRefAttr(mlir::Value buf) {
  size_t offset = 0;

  for (;;) {
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(buf)) {
      size_t argNum = blockArg.getArgNumber();
      StringAttr argName;
      if (auto funcOp =
              llvm::dyn_cast_if_present<FunctionOpInterface>(blockArg.getOwner()->getParentOp()))
        argName = funcOp.getArgAttrOfType<StringAttr>(argNum, "zirgen.argName");
      if (!argName)
        argName = builder.getStringAttr("arg" + std::to_string(argNum));

      return builder.getAttr<ZStruct::RefAttr>(
          offset,
          builder.getType<ZStruct::RefType>(/*element=*/
                                            llvm::cast<Zll::BufferType>(buf.getType()).getElement(),
                                            /*buffer=*/argName));
    }

    auto sliceOp = buf.getDefiningOp<Zll::SliceOp>();
    if (!sliceOp) {
      llvm::errs() << "Could not find slice oop for " << buf << "\n";
      abort();
    }

    offset += sliceOp.getOffset();
    buf = sliceOp.getIn();
  }
}

StringAttr TransformLayout::generateTypeName(StringRef origName) {
  std::string componentType = convertToSnakeFromCamelCase(origName);

  if (componentType.empty() || isdigit(componentType[0])) {
    // Anonymous type
    componentType = "layout_type";
  }

  std::string filtered;
  for (auto c : componentType) {
    if (c == '_' || c == '[' || c == ']' || c == ':') {
      if (filtered.empty() || filtered.back() != '_') {
        filtered.push_back('_');
      }
    } else {
      filtered.push_back(c);
    }
  }

  StringAttr filteredAttr = builder.getStringAttr(filtered);
  if (typeNamesUsed.contains(filteredAttr)) {
    filtered += std::to_string(nextId++);
    filteredAttr = builder.getStringAttr(filtered);
    assert(!typeNamesUsed.contains(filteredAttr));
  }

  typeNamesUsed.insert(filteredAttr);

  return filteredAttr;
}

// Parses the name of an array element, e.g. "foo[1]".  If it's not an
// array element, returns blank string with zero size.

std::pair<StringRef, size_t> parseArray(StringRef name) {
  auto pos = name.find('[');
  if (pos == StringRef::npos) {
    return std::make_pair(StringRef{}, 0);
  }
  StringRef rest = name.substr(pos + 1);
  size_t idx;
  bool failed = rest.consumeInteger(10, idx);
  assert(!failed);
  assert(rest == "]");
  return {name.substr(0, pos), idx};
}

SmallVector<std::pair<Type, NamedAttribute>>
TransformLayout::rollUpArrays(SmallVector<std::pair<Type, NamedAttribute>> subAttrs) {
  SmallVector<std::pair<Type, NamedAttribute>> newSubAttrs;
  DenseMap<StringRef /* array name */,
           std::pair</*elemType=*/Type, SmallVector<Attribute> /* elements */>>
      arrays;

  // Pass 1 - add non-arrays and gather array info
  for (auto [subType, subAttr] : subAttrs) {
    auto [name, offset] = parseArray(subAttr.getName());
    if (name.empty()) {
      // non-array subcomponent
      newSubAttrs.push_back(std::make_pair(subType, subAttr));
      continue;
    }

    auto& arr = arrays[name].second;
    auto& elemType = arrays[name].first;
    if (arr.size() <= offset)
      arr.resize(offset + 1);

    assert((!arr[offset]) && "Duplicate array index");
    arr[offset] = subAttr.getValue();
    if (elemType)
      assert(elemType == subType && "Type mismatch between array elements");
    else
      elemType = subType;
  }

  // Pass 2 - add arrays
  for (auto [subType, subAttr] : subAttrs) {
    auto parsed = parseArray(subAttr.getName());
    if (!arrays.contains(parsed.first))
      // non-array subcomponent, or array we've already outputted
      continue;

    auto [elemType, arrayContents] = arrays.at(parsed.first);
    arrays.erase(parsed.first);

    // Make sure no array elements are missing

    for (auto& elem : arrayContents) {
      if (!elem) {
        llvm::errs() << "Missing array element generating layout for subattr " << subAttr.getName()
                     << " of:\n";
        abort();
      }
    }

    auto arrayType = builder.getType<ZStruct::LayoutArrayType>(elemType, arrayContents.size());
    newSubAttrs.emplace_back(
        arrayType,
        NamedAttribute(builder.getStringAttr(parsed.first), builder.getArrayAttr(arrayContents)));
  }
  assert(arrays.empty() && "Some arrays left unprocessed?");

  return newSubAttrs;
}

std::pair<mlir::Attribute, mlir::Type>
TransformLayout::transform(std::shared_ptr<ConstructInfo> info) {
  SmallVector<std::pair<Type, NamedAttribute>> subAttrs;

  for (auto& [subident, buf] : info->labels) {
    auto refAttr = getRefAttr(buf.getBuf());
    if (!refAttr)
      continue;
    subAttrs.emplace_back(std::make_pair(refAttr.getType(),
                                         NamedAttribute(builder.getStringAttr(subident), refAttr)));
  }
  for (auto& [subident, subcomponent] : info->subcomponents) {
    auto [subAttr, subType] = transform(subcomponent);
    if (!subAttr)
      continue;
    subAttrs.emplace_back(
        std::make_pair(subType, NamedAttribute(builder.getStringAttr(subident), subAttr)));
  }

  subAttrs = rollUpArrays(subAttrs);

  FieldList fields;
  SmallVector<NamedAttribute> dictElems;

  for (auto [subType, dictElem] : subAttrs) {
    dictElems.push_back(dictElem);
    fields.fields.push_back(FieldInfo{.name = dictElem.getName(), .type = subType});
  }

  if (fields.fields.empty()) {
    // Ignore empty structures with no labeled subcomponents.
    return {};
  }

  if (subAttrs.size() == 1) {
    // Elide component with only single element
    const auto& subAttr = subAttrs.front();
    return {subAttr.second.getValue(), subAttr.first};
  }

  if (!types.contains(fields)) {
    types[fields] = generateTypeName(info->typeName);
  }

  StringRef typeName = types[fields];
  ZStruct::LayoutType layoutType = builder.getType<ZStruct::LayoutType>(typeName, fields.fields);
  auto layoutAttr =
      builder.getAttr<ZStruct::StructAttr>(builder.getDictionaryAttr(dictElems), layoutType);
  return {layoutAttr, layoutType};
}

} // namespace

void transformLayout(mlir::MLIRContext* ctx,
                     std::shared_ptr<ConstructInfo> info,
                     Type& layoutType,
                     Attribute& layoutAttr) {
  TransformLayout transform(ctx);
  std::tie(layoutAttr, layoutType) = transform.transform(info);
}

} // namespace zirgen
