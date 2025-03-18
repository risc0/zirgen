// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "elide-redundant-members"

using namespace mlir;
using namespace zirgen::ZStruct;

namespace zirgen::Zhlt {

template <typename StructLikeType>
StructLikeType dropMembers(StructLikeType type, ArrayRef<std::pair<size_t, size_t>> merges) {
  SmallVector<FieldInfo> fields(type.getFields());
  for (auto merge : llvm::reverse(merges)) {
    fields.erase(fields.begin() + merge.second);
  }
  return StructLikeType::get(type.getContext(), type.getId(), fields);
}

// For structure-like components, if two members are equal in the PackOp at the
// end of the constructor, those members will ultimately be equal in all other
// situations, such as when reconstructing an instance from a back and when
// zero-initializing (trivially, since both members are zeroed).
struct ElideRedundantMembersPass : public ElideRedundantMembersBase<ElideRedundantMembersPass> {
  ElideRedundantMembersPass() = default;
  ElideRedundantMembersPass(const ElideRedundantMembersPass& pass) {}

  struct Data {
    Type newType;
    SmallVector<std::pair<size_t, size_t>> merges;
  };

  void runOnOperation() override {
    DenseMap<Type, Data> mapping;
    for (auto component : getOperation().getBodyRegion().getOps<ComponentOp>()) {
      auto ret = cast<ReturnOp>(component.getBody().back().getTerminator());
      auto pack = dyn_cast_if_present<PackOp>(ret.getValue().getDefiningOp());
      if (!pack)
        continue;

      StructType type = pack.getOut().getType();
      auto members = pack.getMembers();
      SmallVector<std::pair<size_t, size_t>> merges;
      for (size_t i = 0; i < members.size(); i++) {
        for (size_t j = i + 1; j < members.size(); j++) {
          if (members[i] == members[j]) {
            LLVM_DEBUG(llvm::dbgs() << "optimizing member " << type.getFields()[j].name
                                    << " of type " << component.getName() << "\n");
            membersDeleted++;
            merges.emplace_back(i, j);
          }
        }
      }
      if (!merges.empty()) {
        llvm::sort(merges, [](auto a, auto b) { return a.second < b.second; });
        StructType newType = dropMembers(type, merges);
        mapping.insert({type, Data{newType, merges}});
      }
    }

    getOperation().walk([&](LookupOp lookup) {
      ArrayRef<FieldInfo> fields;
      if (auto str = dyn_cast<StructType>(lookup.getBase().getType())) {
        fields = str.getFields();
      } else {
        fields = cast<LayoutType>(lookup.getBase().getType()).getFields();
      }

      Type type = lookup.getBase().getType();
      if (mapping.contains(type)) {
        for (auto merge : mapping.at(type).merges) {
          if (lookup.getMember() == fields[merge.second].name) {
            lookup.setMember(fields[merge.first].name);
          }
        }
      }
    });
    getOperation().walk([&](PackOp pack) {
      if (mapping.contains(pack.getType())) {
        for (auto merge : llvm::reverse(mapping.at(pack.getType()).merges)) {
          pack->eraseOperand(merge.second);
        }
      }
    });

    auto typeReplacer = [&](AttrTypeReplacer& replacer,
                            Type T) -> AttrTypeReplacer::ReplaceFnResult<Type> {
      if (mapping.contains(T)) {
        return std::make_pair(mapping.at(T).newType, WalkResult::advance());
      } else {
        return std::nullopt;
      }
    };

    AttrTypeReplacer replacer;
    replacer.addReplacement([&](StructType t) { return typeReplacer(replacer, t); });
    replacer.addReplacement([&](LayoutType t) { return typeReplacer(replacer, t); });
    // replacer.addReplacement(attrReplacer); // TODO: I should optimize the attributes too!
    replacer.recursivelyReplaceElementsIn(getOperation(),
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }

  Statistic membersDeleted{this, "membersDeleted", "number of duplicate struct members pruned"};
};

std::unique_ptr<OperationPass<ModuleOp>> createElideRedundantMembersPass() {
  return std::make_unique<ElideRedundantMembersPass>();
}

} // namespace zirgen::Zhlt
