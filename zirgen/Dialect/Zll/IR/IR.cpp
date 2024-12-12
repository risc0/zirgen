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

#include "llvm/Support/FormatVariadic.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace zirgen::Zll {

// Trim filename to only include relative pathname
llvm::StringRef trimFilename(llvm::StringRef fn) {
  auto pos = fn.rfind("/risc0/");
  if (pos != llvm::StringRef::npos) {
    fn = fn.substr(pos + strlen("/risc0/"));
  }
  pos = fn.rfind("/zirgen/");
  if (pos != llvm::StringRef::npos) {
    fn = fn.substr(pos + strlen("/zirgen/"));
  }
  return fn;
}

std::string getLocString(mlir::Location loc) {
  std::string out;
  auto named = mlir::dyn_cast<mlir::NameLoc>(loc);
  if (named) {
    auto innerLoc = mlir::dyn_cast<mlir::FileLineColLoc>(named.getChildLoc());
    if (innerLoc) {
      out = llvm::formatv("{0}({1}:{2})",
                          named.getName(),
                          trimFilename(innerLoc.getFilename()),
                          innerLoc.getLine());
    } else {
      out = llvm::formatv("{0}", named.getName());
    }
  } else {
    auto fileLineCol = mlir::dyn_cast<mlir::FileLineColLoc>(loc);
    if (fileLineCol) {
      out =
          llvm::formatv("{0}:{1}", trimFilename(fileLineCol.getFilename()), fileLineCol.getLine());
    }
  }

  if (out.empty()) {
    llvm::raw_string_ostream rso(out);
    loc.print(rso);
  }

  for (auto& c : out) {
    if (c == '\n' || c == '"' || c == '\r')
      c = ' ';
  }
  return out;
}

void reinferReturnType(mlir::InferTypeOpInterface op) {
  llvm::SmallVector<mlir::Type> newTypes;
  if (failed(op.inferReturnTypes(op.getContext(),
                                 op->getLoc(),
                                 op->getOperands(),
                                 op->getAttrDictionary(),
                                 op->getPropertiesStorage(),
                                 op->getRegions(),
                                 newTypes)))
    return;

  if (mlir::TypeRange(newTypes) != op->getResultTypes()) {
    for (auto [newType, result] : llvm::zip_equal(newTypes, op->getResults()))
      result.setType(newType);
    for (mlir::Operation* user : op->getUsers()) {
      if (auto inferUser = llvm::dyn_cast<mlir::InferTypeOpInterface>(user))
        reinferReturnType(inferUser);
    }
  }
}

} // namespace zirgen::Zll
