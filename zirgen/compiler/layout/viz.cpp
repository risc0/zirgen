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

#include "zirgen/compiler/layout/viz.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Utilities/KeyPath.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <queue>
#include <string>

namespace zirgen {
namespace layout {
namespace viz {

namespace {

using dsl::KeyPath;
using Zll::ValType;
using namespace ZStruct;

std::string typeName(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, std::string>(t)
      .Case<ValType>([](ValType t) { return "Val"; })
      .Case<ArrayType, LayoutArrayType>([](auto t) { return typeName(t.getElement()); })
      .Case<StructType, UnionType, LayoutType>([](auto t) { return t.getId(); })
      .Case<RefType>([](RefType t) { return typeName(t.getElement()); });
}

template <typename T>
void tr_Record(std::string shape, T t, std::ostream& dest, std::queue<mlir::Type>& worklist) {
  std::string id(t.getId());
  dest << id << " [shape=" << shape << " label=\"";
  // emit the type's own name as the first field
  // use the type name as a port name, too, so we can point edges directly
  // at the name caption instead of randomly all over the record
  dest << "<" << id << ">" << id << "|{ |{";
  bool first = true;
  for (auto& field : t.getFields()) {
    if (!first) {
      dest << "|";
    }
    first = false;
    std::string name(field.name.getValue());
    // write the field name first as a "port name" we can use as
    // an edge origination point
    dest << "<" << name << ">";
    // now write the field name again as a label
    dest << name;
    // Instead of rendering array types directly, we'll point to the
    // underlying type and append a size subscript to the field name
    mlir::Type ft = field.type;
    if (ArrayType a_t = mlir::dyn_cast<ArrayType>(ft)) {
      dest << " \\[" + std::to_string(a_t.getSize()) + "\\]";
    } else if (mlir::isa<ValType>(ft)) {
      dest << ": Val";
    }
    worklist.push(field.type);
  }
  dest << "}}\"]\n";
  // write all the edges from the fields to their types
  // don't bother pointing to Val; everything points to Val eventually
  for (auto& field : t.getFields()) {
    mlir::Type ft = field.type;
    if (mlir::isa<ValType>(ft))
      continue;
    dest << std::string(t.getId()) << ":";
    dest << std::string(field.name.getValue()) << " -> ";
    std::string tn = typeName(ft);
    dest << tn << ":" << tn << "\n";
  }
}

void tr_Root(mlir::Type root, std::ostream& dest) {
  llvm::DenseSet<mlir::Type> emitted;
  std::queue<mlir::Type> worklist;
  for (worklist.push(root); !worklist.empty(); worklist.pop()) {
    mlir::Type t = worklist.front();
    if (emitted.find(t) != emitted.end())
      continue;
    emitted.insert(t);
    llvm::TypeSwitch<mlir::Type>(t)
        .Case<ValType>([&](ValType t) {
          // Don't print a separate shape for Val; everything points to it
        })
        .Case<ArrayType>([&](ArrayType t) { worklist.push(t.getElement()); })
        .Case<StructType>([&](StructType t) { tr_Record("record", t, dest, worklist); })
        .Case<UnionType>([&](UnionType t) { tr_Record("Mrecord", t, dest, worklist); })
        .Case<RefType>([&](RefType t) { worklist.push(t.getElement()); });
  }
}

class SN {
  size_t nodes = 0;
  size_t vals = 0;
  std::ostream& dest;

public:
  SN(std::ostream& dest);
  ~SN();
  // Each emitter writes a node to dest and returns its edge target string.
  std::string Emit(mlir::Type t);
  std::string Emit(ValType t);
  std::string Emit(ArrayType t);
  std::string Emit(StructType t);
  std::string Emit(UnionType t);
  std::string Emit(RefType t);
  std::string Emit(LayoutType t);

protected:
  void StructBody(StructType t, std::map<std::string, UnionType>& unions);
  void ArrayBody(ArrayType t, std::map<std::string, UnionType>& unions);
  void BodyElement(mlir::Type, std::map<std::string, UnionType>& unions);
  void Edge(std::string from_node, std::string from_port, std::string to);
  std::string LayoutUnion(LayoutType t);
  std::string LayoutStruct(LayoutType t);
  void LayoutBody(LayoutType t, std::map<std::string, LayoutType>& unions);
  void LayoutElement(mlir::Type, std::map<std::string, LayoutType>& unions);
};

SN::SN(std::ostream& dest) : dest(dest) {
  dest << "digraph {\n";
  dest << "rankdir=LR\n";
  dest << "compound=true\n";
}

SN::~SN() {
  dest << "}\n";
}

std::string SN::Emit(mlir::Type t) {
  // Forward to a type-specific emit method
  return llvm::TypeSwitch<mlir::Type, std::string>(t)
      .Case<ValType, ArrayType, StructType, UnionType, RefType, LayoutType>(
          [&](auto t) { return Emit(t); });
}

std::string SN::Emit(ValType t) {
  vals++;
  return std::string();
}

std::string SN::Emit(ArrayType t) {
  std::string nodeId(std::to_string(nodes++));
  std::map<std::string, UnionType> unions;
  dest << nodeId << " [shape=record label=\"";
  ArrayBody(t, unions);
  dest << "\"]\n";
  for (auto& iter : unions) {
    Edge(nodeId, iter.first, Emit(iter.second));
  }
  return nodeId;
}

std::string SN::Emit(StructType t) {
  std::string nodeId(std::to_string(nodes++));
  std::string structId(t.getId().str());
  std::map<std::string, UnionType> unions;
  // emit a struct as a record; nested structs become nested records.
  dest << nodeId << " [shape=record label=\"";
  StructBody(t, unions);
  // close the outermost record
  dest << "\"]\n";
  // emit all the unions we linked to, with edges pointing at them
  for (auto& iter : unions) {
    Edge(nodeId, iter.first, Emit(iter.second));
  }
  return nodeId;
}

void SN::StructBody(StructType t, std::map<std::string, UnionType>& unions) {
  // write out the type name
  dest << t.getId().str();
  // special case: only one field and it is a Val
  auto fields = t.getFields();
  if (0 == fields.size())
    return;
  if (1 == fields.size() && mlir::isa<ValType>(fields[0].type)) {
    dest << ": Val";
    return;
  }
  // write a sub-record with each field's type
  dest << "|{|{";
  bool first = true;
  for (auto& field : fields) {
    if (!first)
      dest << "|";
    first = false;
    mlir::Type ft = field.type;
    BodyElement(ft, unions);
  }
  // close the sub-record
  dest << "}}";
}

void SN::ArrayBody(ArrayType at, std::map<std::string, UnionType>& unions) {
  mlir::Type element = at.getElement();
  bool first = true;
  for (size_t i = 0; i < at.getSize(); ++i) {
    if (!first)
      dest << "|";
    first = false;
    dest << "\\[" << i << "\\]: ";
    BodyElement(element, unions);
  }
}

void SN::BodyElement(mlir::Type ft, std::map<std::string, UnionType>& unions) {
  llvm::TypeSwitch<mlir::Type>(ft)
      .Case<RefType>([&](RefType rt) { BodyElement(rt.getElement(), unions); })
      .Case<StructType>([&](StructType st) { StructBody(st, unions); })
      .Case<UnionType>([&](UnionType ut) {
        std::string port("p" + std::to_string(unions.size()));
        unions.insert({port, ut});
        dest << "<" << port << ">" << ut.getId().str();
      })
      .Case<ArrayType>([&](ArrayType at) { ArrayBody(at, unions); })
      .Case<ValType>([&](ValType vt) { dest << "Val"; });
}

std::string SN::Emit(UnionType t) {
  std::string id(std::to_string(nodes++));
  std::map<std::string, std::string> edges;
  dest << "subgraph " << id << " {\n";
  dest << "cluster=true\n";
  // Emit components and record edge target IDs
  size_t startingVals = vals;
  size_t maxVals = vals;
  for (auto& field : t.getFields()) {
    vals = startingVals;
    edges[field.name.str()] = Emit(field.type);
    maxVals = std::max(vals, maxVals);
  }
  vals = maxVals;
  // Try to enforce an order among the nested components
  dest << "{rank=same ";
  bool first = true;
  for (auto edge : edges) {
    if (!first)
      dest << " -> ";
    first = false;
    dest << edge.second;
  }
  dest << " [style=invis weight=10]}\n";
  // Emit a record containing all the fields, with port names
  dest << id << " [shape=Mrecord label=\"";
  first = true;
  for (auto field : t.getFields()) {
    if (!first)
      dest << "|";
    first = false;
    dest << "<" << field.name.str() << ">" << field.name.str();
  }
  dest << "\"]\n";
  // Emit all the edges
  for (auto iter : edges) {
    Edge(id, iter.first, iter.second);
  }
  dest << "}\n";
  return id;
}

std::string SN::Emit(RefType t) {
  return Emit(t.getElement());
}

std::string SN::Emit(LayoutType t) {
  switch (t.getKind()) {
  case LayoutKind::Normal:
  case LayoutKind::Argument:
    return LayoutStruct(t);
  case LayoutKind::Mux:
  case LayoutKind::MajorMux:
    return LayoutUnion(t);
  default:
    assert(false && "unsupported LayoutKind");
  }
}

std::string SN::LayoutUnion(LayoutType t) {
  std::string id(std::to_string(nodes++));
  std::map<std::string, std::string> edges;
  dest << "subgraph " << id << " {\n";
  dest << "cluster=true\n";
  // Emit components and record edge target IDs
  size_t startingVals = vals;
  size_t maxVals = vals;
  for (auto& field : t.getFields()) {
    vals = startingVals;
    edges[field.name.str()] = Emit(field.type);
    maxVals = std::max(vals, maxVals);
  }
  vals = maxVals;
  // Try to enforce an order among the nested components
  dest << "{rank=same ";
  bool first = true;
  for (auto edge : edges) {
    if (!first)
      dest << " -> ";
    first = false;
    dest << edge.second;
  }
  dest << " [style=invis weight=10]}\n";
  // Emit a record containing all the fields, with port names
  dest << id << " [shape=Mrecord label=\"";
  first = true;
  for (auto field : t.getFields()) {
    if (!first)
      dest << "|";
    first = false;
    dest << "<" << field.name.str() << ">" << field.name.str();
  }
  dest << "\"]\n";
  // Emit all the edges
  for (auto iter : edges) {
    Edge(id, iter.first, iter.second);
  }
  dest << "}\n";
  return id;
}

std::string SN::LayoutStruct(LayoutType t) {
  std::string nodeId(std::to_string(nodes++));
  std::string structId(t.getId().str());
  std::map<std::string, LayoutType> unions;
  // emit a struct as a record; nested structs become nested records.
  dest << nodeId << " [shape=record label=\"";
  LayoutBody(t, unions);
  // close the outermost record
  dest << "\"]\n";
  // emit all the unions we linked to, with edges pointing at them
  for (auto& iter : unions) {
    Edge(nodeId, iter.first, Emit(iter.second));
  }
  return nodeId;
}

void SN::LayoutBody(LayoutType t, std::map<std::string, LayoutType>& unions) {
  // write out the type name
  dest << t.getId().str();
  // special case: only one field and it is a Val
  auto fields = t.getFields();
  if (0 == fields.size())
    return;
  if (1 == fields.size() && mlir::isa<ValType>(fields[0].type)) {
    dest << ": Val";
    return;
  }
  // write a sub-record with each field's type
  dest << "|{|{";
  bool first = true;
  for (auto& field : fields) {
    if (!first)
      dest << "|";
    first = false;
    mlir::Type ft = field.type;
    LayoutElement(ft, unions);
  }
  // close the sub-record
  dest << "}}";
}

void SN::LayoutElement(mlir::Type ft, std::map<std::string, LayoutType>& unions) {
  llvm::TypeSwitch<mlir::Type>(ft)
      .Case<RefType>([&](RefType rt) { LayoutElement(rt.getElement(), unions); })
      .Case<LayoutType>([&](LayoutType lt) {
        switch (lt.getKind()) {
        case LayoutKind::Mux:
        case LayoutKind::MajorMux: {
          std::string port("p" + std::to_string(unions.size()));
          unions.insert({port, lt});
          dest << "<" << port << ">" << lt.getId().str();
        } break;
        case LayoutKind::Normal:
        case LayoutKind::Argument: {
          LayoutBody(lt, unions);
        } break;
        default:
          assert(false && "unsupported LayoutKind");
        }
      })
      .Case<ValType>([&](ValType vt) { dest << "Val"; });
}

void SN::Edge(std::string from_name, std::string from_port, std::string to) {
  if (to.empty())
    return;
  dest << from_name << ":\"" << from_port << "\" -> " << to << "\n";
}

class LS {
  std::ostream& dest;
  llvm::DenseMap<mlir::Type, size_t> sizes;
  llvm::DenseSet<LayoutType> comps;
  llvm::DenseSet<LayoutType> muxes;
  llvm::DenseSet<mlir::Type> emitted;

public:
  LS(std::ostream& dest);
  ~LS();
  void print(mlir::Type top);

protected:
  size_t measure(mlir::Type t);
  size_t measureRef(RefType rt);
  size_t measureStruct(LayoutType lt);
  size_t measureUnion(LayoutType lt);
  size_t measureArray(LayoutArrayType at);

  void Emit(mlir::Type t);
  void EmitRef(RefType rt);
  void EmitStruct(LayoutType lt);
  void EmitUnion(LayoutType lt);
  void EmitArray(LayoutArrayType at);
  std::string typeName(mlir::Type t);
  std::string linkToName(mlir::Type t);
  mlir::Type unwrap(mlir::Type);
  std::string hashcolor(mlir::Type);
};

LS::LS(std::ostream& dest) : dest(dest) {
  dest << "<html>\n"
          "<head>\n"
          "<style>\n"
          "  table {\n"
          "    border: 1px solid black;\n"
          "    padding: 2px;\n"
          "  }\n"
          "  tbody tr:nth-of-type(even) { background-color:#eee; }\n"
          "  thead tr td { background-color:#ccc; }\n"
          "  td {\n"
          "    padding-left: 10px;\n"
          "    padding-right: 10px;\n"
          "    padding-top: 2px;\n"
          "    padding-bottom: 2px;\n"
          "  }\n"
          "</style>\n"
          "</head>\n"
          "<body>\n"
          "<a href=\"#type index\">Index</a>\n"
          "<hr />\n";
}

LS::~LS() {
  dest << "<hr />\n";
  dest << "<h2 id=\"type index\">Index</h2>\n";
  dest << "<h3>Muxes</h3>\n";
  dest << "<ul>\n";
  for (auto& lt : muxes) {
    dest << "<li>" << linkToName(lt) << "</li>\n";
  }
  dest << "</ul>\n";
  dest << "<h3>Components</h3>\n";
  dest << "<ul>\n";
  for (auto& lt : comps) {
    dest << "<li>" << linkToName(lt) << "</li>\n";
  }
  dest << "</ul>\n";
  dest << "</body>\n";
  dest << "</html>\n";
}

void LS::print(mlir::Type t) {
  measure(t);
  Emit(t);
}

size_t LS::measure(mlir::Type t) {
  if (!t) {
    // shouldn't happen in correct code, but might cascade from an error
    return 0;
  }
  auto found = sizes.find(t);
  if (found != sizes.end()) {
    return found->second;
  }
  size_t regs = llvm::TypeSwitch<mlir::Type, size_t>(t)
                    .Case<RefType>([&](RefType rt) { return measureRef(rt); })
                    .Case<LayoutType>([&](LayoutType lt) {
                      switch (lt.getKind()) {
                      case LayoutKind::Normal:
                      case LayoutKind::Argument:
                        return measureStruct(lt);
                      case LayoutKind::Mux:
                      case LayoutKind::MajorMux:
                        return measureUnion(lt);
                      default:
                        assert(false && "unsupported LayoutKind");
                      }
                    })
                    .Case<LayoutArrayType>([&](auto lat) { return measureArray(lat); })
                    .Default([&](mlir::Type t) -> size_t {
                      llvm::errs() << "UNKNOWN TYPE: " << t;
                      assert(false && "unknown type");
                      return 0;
                    });
  sizes[t] = regs;
  return regs;
}

size_t LS::measureRef(RefType rt) {
  return 1;
}

size_t LS::measureUnion(LayoutType lt) {
  // union size is the max of the fields
  size_t regs = 0;
  for (auto& field : lt.getFields()) {
    regs = std::max(regs, measure(field.type));
  }
  muxes.insert(lt);
  return regs;
}

size_t LS::measureStruct(LayoutType lt) {
  size_t regs = 0;
  for (auto& field : lt.getFields()) {
    regs += measure(field.type);
  }
  comps.insert(lt);
  return regs;
}

size_t LS::measureArray(LayoutArrayType at) {
  return measure(at.getElement()) * at.getSize();
}

void LS::Emit(mlir::Type t) {
  if (!t) {
    // tolerate upstream errors and other degenerate conditions
    return;
  }
  assert(sizes.contains(t));
  if (emitted.contains(t)) {
    return;
  }
  emitted.insert(t);
  llvm::TypeSwitch<mlir::Type>(t)
      .Case<RefType>([&](RefType rt) { EmitRef(rt); })
      .Case<LayoutType>([&](LayoutType lt) {
        switch (lt.getKind()) {
        case LayoutKind::Normal:
        case LayoutKind::Argument: {
          EmitStruct(lt);
        } break;
        case LayoutKind::Mux:
        case LayoutKind::MajorMux: {
          EmitUnion(lt);
        } break;
        default:
          assert(false && "unsupported LayoutKind");
        }
      })
      .Case<LayoutArrayType>([&](auto lat) { EmitArray(lat); });
}

void LS::EmitRef(RefType rt) {
  // refs are all the same; don't bother printing anything
}

void LS::EmitUnion(LayoutType lt) {
  // Collect the unwrapped types of the mux arms.
  std::vector<mlir::Type> arms;
  for (auto& field : lt.getFields()) {
    arms.push_back(unwrap(field.type));
  }
  size_t regs = sizes[lt];
  // populate a table with field names & types, sorted by offset
  // outer vector holds a column for each mux element
  // each column element is one offset position
  std::vector<std::vector<mlir::Type>> table;
  table.resize(arms.size());
  for (size_t col = 0; col < arms.size(); ++col) {
    table[col].resize(regs);
    auto subt = mlir::dyn_cast<LayoutType>(arms[col]);
    if (!subt)
      continue;
    size_t off = 0;
    for (auto& subf : subt.getFields()) {
      table[col][off] = subf.type;
      off += sizes[subf.type];
    }
  }

  dest << "<h3 id=\"" << lt.getId().str() << "\">";
  dest << "Mux: " << lt.getId().str() << "</h3>\n";
  if (regs != 1) {
    dest << "<p>" << std::to_string(regs) << " registers</p>\n";
  } else {
    dest << "<p>1 register</p>\n";
  }
  dest << "<table>\n";
  dest << "<thead>\n";
  dest << "<tr><td>Offset</td>\n";
  // one column per mux element
  for (auto arm : arms) {
    dest << "<td>" << linkToName(arm) << "</td>\n";
  }
  dest << "</tr>\n";
  dest << "</thead>\n";
  dest << "<tbody>\n";
  // emit the offset table generated earlier
  for (size_t offset = 0; offset < regs; ++offset) {
    dest << "<tr>\n";
    dest << "<td align=\"right\">" << std::to_string(offset) << "</td>\n";
    for (size_t column = 0; column < table.size(); ++column) {
      if (offset < sizes[arms[column]]) {
        mlir::Type cell = table[column][offset];
        if (cell) {
          dest << "<td valign=\"top\" ";
          dest << "bgcolor=\"" << hashcolor(cell) << "\" ";
          dest << "rowspan=\"" << std::to_string(sizes[cell]) << "\">";
          dest << linkToName(cell);
          dest << "</td>\n";
        }
      } else {
        dest << "<td></td>\n";
      }
    }
    dest << "</tr>\n";
  }
  dest << "</tbody>\n";
  dest << "</table>\n";
  // Having printed the mux elements table, emit the arm components.
  for (auto t : arms) {
    Emit(t);
  }
}

void LS::EmitStruct(LayoutType lt) {
  size_t regs = sizes[lt];
  dest << "<h3 id=\"" << lt.getId().str() << "\">";
  dest << "Component: " << lt.getId().str() << "</h3>\n";
  if (regs != 1) {
    dest << "<p>" << std::to_string(regs) << " registers</p>\n";
  } else {
    dest << "<p>1 register</p>\n";
  }
  dest << "<table>\n";
  dest << "<thead>\n<tr>\n";
  dest << "<td>Offset</td>\n";
  dest << "<td>Name</td>\n";
  dest << "<td>Type</td>\n";
  dest << "<td>Size</td>\n";
  dest << "</tr>\n</thead>\n";
  dest << "<tbody>\n";
  size_t offset = 0;
  for (auto& field : lt.getFields()) {
    dest << "<tr>\n";
    size_t fsize = sizes[field.type];
    std::string fsizestr = std::to_string(fsize);
    std::string typecolor = hashcolor(field.type);
    dest << "<td>" << std::to_string(offset) << "</td>\n";
    dest << "<td valign=\"top\" ";
    dest << "bgcolor=\"" << typecolor << "\" ";
    dest << "rowspan=\"" << fsizestr << "\">";
    dest << field.name.str();
    dest << "</td>\n";
    dest << "<td valign=\"top\" ";
    dest << "bgcolor=\"" << typecolor << "\" ";
    dest << "rowspan=\"" << fsizestr << "\">";
    dest << linkToName(field.type);
    dest << "</td>\n";
    dest << "<td valign=\"top\" ";
    dest << "bgcolor=\"" << typecolor << "\" ";
    dest << "rowspan=\"" << fsizestr << "\">";
    dest << fsizestr;
    dest << "</td>\n";
    dest << "</tr>\n";
    // Insert spacer rows to accommodate all those rowspans
    size_t targetOffset = offset + fsize;
    for (++offset; offset < targetOffset; ++offset) {
      dest << "<tr>\n";
      dest << "<td>" << std::to_string(offset) << "</td>\n";
      dest << "</tr>\n";
    }
  }
  dest << "</tbody>\n";
  dest << "</table>\n";

  for (auto& field : lt.getFields()) {
    Emit(field.type);
  }
}

void LS::EmitArray(LayoutArrayType at) {
  // The array needs no report of its own, but we should emit the layout
  // of its element type.
  Emit(at.getElement());
}

std::string LS::typeName(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, std::string>(t)
      .Case<RefType>([&](RefType rt) { return "Ref"; })
      .Case<LayoutType>([&](LayoutType lt) { return lt.getId().str(); })
      .Case<LayoutArrayType>([&](auto lat) {
        std::string elementName = typeName(lat.getElement());
        return elementName + "[" + std::to_string(lat.getSize()) + "]";
      });
}

std::string LS::linkToName(mlir::Type t) {
  std::string link;
  std::string text;
  if (auto lat = mlir::dyn_cast<LayoutArrayType>(t)) {
    link = typeName(lat.getElement());
    text = typeName(t);
  } else if (mlir::isa<RefType>(t)) {
    return "Ref";
  } else {
    link = typeName(t);
    text = link;
  }
  return "<a href=\"#" + link + "\">" + text + "</a>";
}

mlir::Type LS::unwrap(mlir::Type t) {
  if (!t)
    return t;
  auto lt = mlir::dyn_cast<LayoutType>(t);
  if (!lt)
    return t;
  // If the layout has a single field whose name is "@super", use that
  // field's type - but unwrap it first
  if (lt.getFields().size() != 1)
    return t;
  auto& fi = lt.getFields()[0];
  if (fi.name != "@super")
    return t;
  return unwrap(fi.type);
}

std::string LS::hashcolor(mlir::Type t) {
  // Generate a consistent but randomly distributed background color for
  // table cells representing this type. Color consistency should make it
  // easier to see type alignment across mux arms. We'll set the high 2 bits
  // of each color channel to get pastels, then get 18 bits of data from
  // the type name via rotate & xor.
  unsigned hash = 0x999999;
  for (char c : typeName(t)) {
    hash = c ^ (((hash << 7) & 0x0003FFFF) | ((hash >> 11) & 0x7F));
  }
  // Spread hash value across three color channels and convert to hex.
  char hex[] = "0123456789ABCDEF";
  std::string out = "#EEEEEE";
  out[1] = hex[((hash >> 16) & 0x03) | 0xC];
  out[2] = hex[(hash >> 12) & 0x0F];
  out[3] = hex[((hash >> 10) & 0x03) | 0xC];
  out[4] = hex[(hash >> 6) & 0x0F];
  out[5] = hex[((hash >> 4) & 0x03) | 0xC];
  out[6] = hex[(hash >> 0) & 0x0F];
  return out;
}

} // namespace

void typeRelation(mlir::Type root, std::ostream& dest) {
  // Display the nesting of types which compose the root type.
  dest << "digraph types {\nrankdir=LR\n";
  tr_Root(root, dest);
  dest << "}\n";
}

void storageNest(mlir::Type root, std::ostream& dest) {
  // Visualize the composition layout for the given root type. Instead of
  // displaying the relationships between types, this displays the nesting
  // of instances.
  SN sn(dest);
  (void)sn.Emit(root);
}

void layoutSizes(mlir::Type root, std::ostream& dest) {
  // Visualize the composition layout for the given root type. Instead of
  // displaying the relationships between types, this displays the nesting
  // of instances.
  LS ls(dest);
  ls.print(root);
}

class LayoutAttrTreePrinter {
public:
  LayoutAttrTreePrinter(mlir::ModuleOp mod, std::ostream& os) : mod(mod), os(os) {}

  void print(mlir::Attribute attr) {
    llvm::TypeSwitch<mlir::Attribute>(attr)
        .Case<RefAttr, StructAttr, mlir::ArrayAttr>([&](auto attr) { print(attr); })
        .Case<BoundLayoutAttr>([&](auto attr) { print(attr.getLayout()); })
        .Case<mlir::SymbolRefAttr>([&](auto attr) {
          auto gco = mod.lookupSymbol<zirgen::ZStruct::GlobalConstOp>(attr);
          print(gco.getConstant());
        })
        .Default([&](auto) { os << "unknown attr"; });
  }

protected:
  void print(RefAttr ref) { os << ref.getIndex(); }

  void print(StructAttr str) {
    indent++;
    os << typeName(str.getType());
    for (auto field : str.getFields()) {
      os << "\n";
      printIndent();
      os << field.getName().str() << ": ";
      print(field.getValue());
    }
    indent--;
  }

  void print(mlir::ArrayAttr arr) {
    indent++;
    for (unsigned i = 0; i < arr.size(); i++) {
      os << "\n";
      printIndent();
      os << "[" << i << "]: ";
      print(arr[i]);
    }
    indent--;
  }

  void printIndent() {
    for (unsigned i = 0; i < indent; i++) {
      os << "| ";
    }
  }

  mlir::ModuleOp mod;
  std::ostream& os;
  unsigned indent = 0;
};

void layoutAttrs(mlir::ModuleOp mod, std::ostream& dest) {
  LayoutAttrTreePrinter printer(mod, dest);
  mod->walk([&](GlobalConstOp gco) {
    dest << "GlobalConstOp \"" << gco.getSymName().str() << "\": ";
    printer.print(gco.getConstant());
    dest << "\n";
  });
}

class ColumnKeyPathPrinter {
public:
  ColumnKeyPathPrinter(mlir::ModuleOp mod, std::ostream& os) : mod(mod), os(os) {}

  void print(mlir::Attribute attr, size_t index) {
    llvm::TypeSwitch<mlir::Attribute>(attr)
        .Case<RefAttr, StructAttr, mlir::ArrayAttr>([&](auto attr) { print(attr, index); })
        .Case<BoundLayoutAttr>([&](auto attr) { print(attr.getLayout(), index); })
        .Case<mlir::SymbolRefAttr>([&](auto attr) {
          auto gco = mod.lookupSymbol<zirgen::ZStruct::GlobalConstOp>(attr);
          print(gco.getConstant(), index);
        });
  }

private:
  void print(RefAttr ref, size_t index) {
    if (ref.getIndex() == index)
      os << keyPath << "\n";
  }

  void print(StructAttr str, size_t index) {
    for (auto field : str.getFields()) {
      keyPath.push_back(field.getName());
      print(field.getValue(), index);
      keyPath.pop_back();
    }
  }

  void print(mlir::ArrayAttr arr, size_t index) {
    for (size_t i = 0; i < arr.size(); i++) {
      keyPath.push_back(i);
      print(arr[i], index);
      keyPath.pop_back();
    }
  }

  mlir::ModuleOp mod;
  std::ostream& os;
  KeyPath keyPath;
};

void columnKeyPaths(mlir::ModuleOp mod, size_t column, std::ostream& dest) {
  KeyPath keyPath;
  mod->walk(
      [&](GlobalConstOp gco) { ColumnKeyPathPrinter(mod, dest).print(gco.getConstant(), column); });
}

} // namespace viz
} // namespace layout
} // namespace zirgen
