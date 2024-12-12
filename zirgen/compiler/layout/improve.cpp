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

// Potential future improvements:
// - After reordering a struct which appears within an inner union, don't
//   reorder it again if we encounter it again; instead, pin its field order
//   and use that to direct the layout of the other structs it occurs with.
// - When sorting merged columns, use a density metric instead of simple
//   popcount: that is the ratio of occupied registers to the total area.
//   Use this density metric to decide whether a merge is an improvement:
//   a good merge will increase density, a bad one will lower it.
// - After sorting by popcount, reorder columns with equal density to
//   minimize the degree of padding (or misalignment) required.
// - Instead of simply aligning arrays which have like element types, unpack
//   them (by counting each element as an individual instance), allowing
//   other fields of like type to participate in alignment, then regroup the
//   instances into their constituent arrays after gap-filling.

#include "zirgen/compiler/layout/improve.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <set>
#include <vector>

namespace zirgen {
namespace layout {

namespace {

using Zll::ValType;
using ZStruct::LayoutArrayType;
using ZStruct::LayoutKind;
using ZStruct::LayoutType;
using MemberCount = llvm::DenseMap<mlir::Type, size_t>;
using SizeMap = llvm::DenseMap<mlir::Type, unsigned>;
using BranchList = std::vector<Layout>;
using JobList = std::vector<std::vector<LayoutType>>;

struct Instance {
  Instance(mlir::Type type, unsigned size, size_t rows);
  mlir::Type type;
  unsigned size = 0;
  std::vector<bool> presence;
  size_t popcount() const;
  static bool order(const Instance& left, const Instance& right);
};

using InstanceTable = std::vector<Instance>;

struct Shape {
  bool present = false;
  unsigned size = 0;
};

struct Column {
  explicit Column(const Instance& inst);
  void merge(const Instance& inst);
  void merge(const Column& col);
  std::vector<Instance> instances;
  std::vector<Shape> shapes;
  unsigned width = 0;
  size_t popcount = 0;
  bool canAccept(const Column& right) { return canAccept(right.shapes); }
  bool canAccept(const std::vector<Shape>& gapfill);
};

using ColumnTable = std::vector<Column>;

enum class Padding { Allow, Forbid };

struct Process {
  Process(SizeMap& sizes, size_t nrows) : sizes(sizes), nrows(nrows) {}
  void align(BranchList& branches, Padding padding = Padding::Allow);
  void update(BranchList&, llvm::DenseMap<LayoutType, Layout>&);
  JobList subalignments(BranchList& branches);

protected:
  void tally(std::vector<MemberCount>& counts, InstanceTable& instances);
  void columnize(const InstanceTable& src, ColumnTable& dest);
  void fillGaps(ColumnTable& columns);
  void mergeArrays(ColumnTable& columns);
  void linearize(const ColumnTable& src, InstanceTable& dest);
  std::vector<unsigned> rowOffsets(InstanceTable& instances, size_t row);
  void pad(Layout& sl, const std::vector<unsigned>& offsets, size_t last);
  void pad(BranchList& branches, InstanceTable& instances);
  SizeMap& sizes;
  const size_t nrows;
};

std::string typeName(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, std::string>(t)
      .Case<ValType>([&](auto& vt) -> std::string { return "Val"; })
      .Case<LayoutArrayType>([&](auto& at) -> std::string {
        std::string count(std::to_string(at.getSize()));
        return typeName(at.getElement()) + "[" + count + "]";
      })
      .Case<LayoutType>([&](auto& st) -> std::string { return st.getId().str(); })
      .Case<RefType>([](RefType t) { return typeName(t.getElement()); }); // TODO
}

Instance::Instance(mlir::Type t, unsigned sz, size_t rows) : type(t), size(sz), presence(rows) {
  //
}

size_t Instance::popcount() const {
  size_t out = 0;
  for (size_t row = 0; row < presence.size(); ++row) {
    if (presence[row]) {
      ++out;
    }
  }
  return out;
}

bool Instance::order(const Instance& left, const Instance& right) {
  auto lpop = left.popcount();
  auto rpop = right.popcount();
  // first, sort by popcount
  if (lpop > rpop) {
    return true;
  }
  if (lpop < rpop) {
    return false;
  }
  // if popcounts are equal, sort by type size
  if (left.size > right.size) {
    return true;
  }
  if (left.size < right.size) {
    return false;
  }
  // If popcounts and sizes are equal, sort by name to stabilize the ordering
  if (typeName(left.type) > typeName(right.type)) {
    return true;
  }
  return false;
}

Column::Column(const Instance& inst) : instances{inst} {
  shapes.resize(inst.presence.size());
  for (size_t row = 0; row < inst.presence.size(); ++row) {
    if (inst.presence[row]) {
      shapes[row].present = true;
      shapes[row].size = inst.size;
      width = std::max(width, inst.size);
      ++popcount;
    }
  }
}

void Column::merge(const Instance& inst) {
  instances.push_back(inst);
  assert(shapes.size() == inst.presence.size());
  popcount = 0;
  for (size_t row = 0; row < shapes.size(); ++row) {
    if (inst.presence[row]) {
      shapes[row].present = true;
      shapes[row].size += inst.size;
    }
    if (shapes[row].present) {
      width = std::max(width, shapes[row].size);
      ++popcount;
    }
  }
}

void Column::merge(const Column& col) {
  for (auto& inst : col.instances) {
    merge(inst);
  }
}

bool Column::canAccept(const std::vector<Shape>& gapfill) {
  // Will a filler having the specified shape fit within the empty space
  // inside this column?
  assert(shapes.size() == gapfill.size());
  bool found = false;
  unsigned leftpos = 0;
  for (size_t row = 0; row < shapes.size(); ++row) {
    if (!gapfill[row].present) {
      continue;
    }
    // Reject any match which would require us to insert padding in order
    // to prevent the merged column from disaligning.
    if (found) {
      if (leftpos != shapes[row].size) {
        return false;
      }
    } else {
      found = true;
      leftpos = shapes[row].size;
    }
    // If this match would cause our column width to increase, reject it, on
    // the assumption that we would increase the raggedness of our right edge
    // and therefore the amount of filler required to align the next column.
    // A better heuristic would measure the empty space present on the right
    // edge and accept any merge which reduces that total, even if it would
    // increase the overall width of the column.
    if ((shapes[row].size + gapfill[row].size) > width) {
      return false;
    }
  }
  return true;
}

MemberCount countFieldTypes(Layout& sl) {
  MemberCount out;
  for (auto& field : sl.fields) {
    out[field.type]++;
  }
  return out;
}

MemberCount overallCount(std::vector<MemberCount>& branches) {
  MemberCount overall;
  for (auto& current : branches) {
    for (auto& iter : current) {
      mlir::Type t = iter.first;
      size_t c = iter.second;
      auto found = overall.find(t);
      if (found != overall.end()) {
        if (c > found->second) {
          found->second = c;
        }
      } else {
        overall.insert({t, c});
      }
    }
  }
  return overall;
}

void Process::tally(std::vector<MemberCount>& branchCounts,
                    InstanceTable& instances // table to populate
) {
  for (auto& found : overallCount(branchCounts)) {
    mlir::Type type = found.first;
    unsigned size = sizes[type];
    size_t count = found.second;
    // Create that many instances of this type
    for (size_t i = 0; i < count; ++i) {
      auto& current = instances.emplace_back(type, size, nrows);
      // Populate the presence column for each branch.
      for (size_t row = 0; row < nrows; ++row) {
        current.presence[row] = branchCounts[row][type] > i;
      }
    }
  }
}

void Process::columnize(const InstanceTable& instances, ColumnTable& cols) {
  cols.clear();
  cols.reserve(instances.size());
  for (size_t i = 0; i < instances.size(); ++i) {
    cols.emplace_back(instances[i]);
  }
}

void Process::fillGaps(ColumnTable& cols) {
  // Reorder the instance table by grouping non-overlapping instances into
  // columns, thereby filling gaps which would reduce alignment.
  // Iterate through each column, left to right, filling in the largest
  // gaps first, continuing on to plug smaller gaps.
  for (size_t i = 0; i < cols.size(); ++i) {
    auto& left = cols[i];
    for (size_t j = i + 1; j < cols.size(); ++j) {
      auto& right = cols[j];
      if (left.canAccept(right)) {
        left.merge(right);
        cols.erase(std::next(cols.begin(), j--));
      }
    }
    // If we increased this column's popcount, move it leftward to maintain
    // the sorting order before we continue examining successive rows.
    size_t dest = i;
    size_t searchpop = cols[i].popcount;
    while (dest > 0 && cols[dest - 1].popcount < searchpop) {
      dest--;
    }
    if (dest != i) {
      auto col = cols[i];
      cols.erase(std::next(cols.begin(), i));
      cols.insert(std::next(cols.begin(), dest), col);
      i = dest;
    }
  }
}

void Process::mergeArrays(ColumnTable& columns) {
  for (size_t i = 0; i < columns.size(); ++i) {
    auto& lCol = columns[i];
    for (size_t j = 0; j < lCol.instances.size(); ++j) {
      auto& lInst = lCol.instances[j];
      if (LayoutArrayType lAT = mlir::dyn_cast<LayoutArrayType>(lInst.type)) {
        // Look for later columns which contain arrays having the same
        // element type.
        mlir::Type eltype = lAT.getElement();
        size_t k = i + 1;
        while (k < columns.size()) {
          auto& rCol = columns[k];
          bool match = false;
          for (size_t l = 0; l < rCol.instances.size(); ++l) {
            auto& rInst = rCol.instances[l];
            if (LayoutArrayType rAT = mlir::dyn_cast<LayoutArrayType>(rInst.type)) {
              match |= rAT.getElement() == eltype;
            }
          }
          if (match && lCol.canAccept(rCol)) {
            lCol.merge(rCol);
            columns.erase(std::next(columns.begin(), k));
          } else {
            ++k;
          }
        }
      }
    }
  }
}

void Process::linearize(const ColumnTable& cols, InstanceTable& instances) {
  // Rebuild the instance table according to the column order.
  instances.clear();
  for (auto& c : cols) {
    instances.insert(instances.end(), c.instances.begin(), c.instances.end());
  }
}

std::vector<unsigned> Process::rowOffsets(InstanceTable& instances, size_t row) {
  std::vector<unsigned> out;
  unsigned offset = 0;
  for (auto& inst : instances) {
    if (inst.presence[row]) {
      out.push_back(offset);
    }
    offset += sizes[inst.type];
  }
  return out;
}

JobList Process::subalignments(BranchList& branches) {
  using AlignedTypes = llvm::DenseSet<LayoutType>;
  llvm::DenseMap<unsigned, AlignedTypes> alignments;
  llvm::DenseMap<LayoutType, llvm::DenseSet<unsigned>> offsets;
  llvm::DenseMap<LayoutType, std::shared_ptr<AlignedTypes>> mergesets;
  for (size_t row = 0; row < nrows; ++row) {
    unsigned offset = 0;
    for (auto& fi : branches[row].fields) {
      mlir::Type t = fi.type;
      if (auto at = mlir::dyn_cast<LayoutArrayType>(t)) {
        t = at.getElement();
      }
      if (auto st = mlir::dyn_cast<LayoutType>(t)) {
        alignments[offset].insert(st);
        offsets[st].insert(offset);
        offset += sizes[st];
        auto mergeset = std::make_shared<AlignedTypes>();
        mergeset->insert(st);
        mergesets.insert({st, mergeset});
      }
    }
  }
  // For each group of types aligned at a common offset, coalesce a single
  // shared AlignedTypes instance.
  for (auto& aIter : alignments) {
    AlignedTypes& group = aIter.second;
    if (group.size() > 1) {
      auto gIter = group.begin();
      LayoutType headtype = *gIter;
      auto dest = mergesets[headtype];
      for (++gIter; gIter != group.end(); ++gIter) {
        auto next = mergesets[*gIter];
        dest->insert(next->begin(), next->end());
        mergesets[*gIter] = dest;
      }
    }
  }
  // Extract the list of disjoint alignment sets remaining and create a
  // job queue.
  std::vector<std::vector<LayoutType>> subjobs;
  for (auto& mIter : mergesets) {
    auto group = mIter.second;
    if (group->size() > 1) {
      std::vector<LayoutType> subjob;
      subjob.reserve(group->size());
      for (LayoutType st : *group) {
        subjob.push_back(st);
      }
      subjobs.push_back(subjob);
      group->clear();
    }
  }
  return subjobs;
}

std::vector<mlir::Type> extractRow(InstanceTable& instances, size_t row) {
  // Return a sequence of types from one row in the instance table.
  std::vector<mlir::Type> out;
  for (auto& inst : instances) {
    if (inst.presence[row]) {
      out.push_back(inst.type);
    }
  }
  return out;
}

void reorder(Layout& sl, const std::vector<mlir::Type>& typeOrder) {
  // Sort the fields in this structure layout so that their types match the
  // specified sequence.
  assert(sl.fields.size() == typeOrder.size());
  for (size_t i = 0; i < typeOrder.size(); ++i) {
    size_t j = i;
    while (j < sl.fields.size() && sl.fields[j].type != typeOrder[i]) {
      ++j;
    }
    if (j > i) {
      auto field = sl.fields[j];
      sl.fields.erase(std::next(sl.fields.begin(), j));
      sl.fields.insert(std::next(sl.fields.begin(), i), field);
    }
  }
}

void reorder(BranchList& branches, InstanceTable& instances) {
  // Sort the fields in each of these structure layouts so that their types
  // match the corresponding sequence from the instance table.
  for (size_t row = 0; row < branches.size(); ++row) {
    std::vector<mlir::Type> typeOrder(extractRow(instances, row));
    reorder(branches[row], typeOrder);
  }
}

void Process::pad(Layout& sl, const std::vector<unsigned>& offsets, size_t lastAlignedCol) {
  // Insert padding until the fields in this structure occur at the
  // specified offsets.
  std::vector<std::pair<size_t, unsigned>> padding;
  unsigned current = 0;
  assert(lastAlignedCol <= sl.fields.size());
  assert(offsets.size() >= lastAlignedCol);
  for (size_t i = 0; i < lastAlignedCol; ++i) {
    size_t expected = offsets[i];
    if (expected > current) {
      // Build the padding list from back to front, so we can iterate front
      // to back and insert padding without invalidating indexes.
      padding.insert(padding.begin(), {i, expected - current});
      current = expected;
    }
    current += sizes[sl.fields[i].type];
  }
  for (auto iter : padding) {
    // Where should we insert padding?
    size_t index = iter.first;
    // How many units are required?
    unsigned size = iter.second;
    auto ctx = sl.fields[index].type.getContext();
    ValType vt = ValType::get(ctx, Zll::kFieldPrimeDefault, 1);
    mlir::Type t = RefType::get(ctx, vt);
    if (size > 1) {
      t = LayoutArrayType::get(ctx, t, size);
    }
    FieldInfo fi;
    fi.name = mlir::StringAttr::get(ctx, "@padding" + std::to_string(index));
    fi.type = t;
    sl.fields.insert(std::next(sl.fields.begin(), index), fi);
  }
}

void Process::pad(BranchList& branches, InstanceTable& instances) {
  for (size_t row = 0; row < nrows; ++row) {
    Layout& sl = branches[row];
    // Find the offset for each field in this row of the instance table.
    std::vector<unsigned> offsets(rowOffsets(instances, row));
    // Locate the last column which contains aligned fields.
    size_t lastAlignedCol = 0;
    size_t col = 0;
    for (size_t i = 0; i < instances.size(); ++i) {
      if (instances[i].presence[row]) {
        ++col;
        if (instances[i].popcount() > 1) {
          lastAlignedCol = col;
        }
      }
    }
    // Insert padding as needed to align fields to the specified offsets.
    pad(sl, offsets, lastAlignedCol);
  }
}

void Process::align(BranchList& branches, Padding padding) {
  assert(branches.size() == nrows);
  // Tally the occurrences of each type of field across all branches.
  std::vector<MemberCount> branchCounts;
  branchCounts.reserve(nrows);
  for (auto& b : branches) {
    branchCounts.emplace_back(countFieldTypes(b));
  }
  // Create a table associating type instances with each branch.
  InstanceTable instances;
  tally(branchCounts, instances);
  // Sort the instance table by presence vector popcount, descending.
  std::sort(instances.begin(), instances.end(), Instance::order);
  // Convert into column representation, so we can merge aligned instances.
  ColumnTable columns;
  columnize(instances, columns);
  // Merge arrays with like element types into single columns.
  mergeArrays(columns);
  // Fill gaps by moving complementary columns adjacent to each other.
  fillGaps(columns);
  // Flatten the columns back out into newly-ordered instances.
  linearize(columns, instances);
  // Reorder struct layouts to match the resulting presence rows.
  reorder(branches, instances);
  // Insert padding to prevent rightward columns from shifting left.
  if (padding == Padding::Allow) {
    pad(branches, instances);
  }
}

void Process::update(BranchList& branches, llvm::DenseMap<LayoutType, Layout>& structs) {
  for (auto& b : branches) {
    structs[b.original] = b;
  }
}

} // namespace

void improve(Circuit& circuit) {
  // Instead of iterating, we should perform a depth-first traversal.
  for (auto& utype : circuit.unionsInDfsPostorder) {
    Layout& ul(circuit.unions[utype]);
    std::set<std::string> typeIDs;
    BranchList branches;
    for (auto& fi : ul.fields) {
      if (auto st = mlir::dyn_cast<LayoutType>(fi.type)) {
        // Ignore non-struct members.
        if (st.getKind() == LayoutKind::Mux || st.getKind() == LayoutKind::MajorMux) {
          continue;
        }
        // Ignore structs which have no fields.
        if (st.getFields().empty()) {
          continue;
        }
        std::string id = st.getId().str();
        // A union may contain more than one instance of the same struct,
        // but we only need to sort its fields once.
        if (typeIDs.find(id) != typeIDs.end()) {
          continue;
        }
        typeIDs.insert(id);
        branches.push_back(circuit.structs[st]);
      }
    }
    Process p(circuit.sizes, branches.size());
    p.align(branches, Padding::Forbid);
    p.update(branches, circuit.structs);
    auto subjobs = p.subalignments(branches);
    while (!subjobs.empty()) {
      // Pop the first job from the queue.
      auto job = subjobs.front();
      subjobs.erase(subjobs.begin());
      // Find the structure layouts for this group of type branches.
      BranchList subbranches;
      subbranches.reserve(job.size());
      for (LayoutType st : job) {
        subbranches.push_back(circuit.structs[st]);
      }
      // Create an alignment process and reorder the group of structs.
      Process sp(circuit.sizes, subbranches.size());
      sp.align(subbranches, Padding::Forbid);
      sp.update(subbranches, circuit.structs);
      auto more = sp.subalignments(subbranches);
      // Append any new subalignment jobs to our work queue.
      subjobs.insert(subjobs.end(), more.begin(), more.end());
    }
  }
}

} // namespace layout
} // namespace zirgen
