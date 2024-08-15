// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
  auto named = loc->dyn_cast<mlir::NameLoc>();
  if (named) {
    auto innerLoc = named.getChildLoc()->dyn_cast<mlir::FileLineColLoc>();
    if (innerLoc) {
      return llvm::formatv("{0}({1}:{2})",
                           named.getName(),
                           trimFilename(innerLoc.getFilename()),
                           innerLoc.getLine());
    } else {
      return llvm::formatv("{0}", named.getName());
    }
  }

  auto fileLineCol = loc->dyn_cast<mlir::FileLineColLoc>();
  if (fileLineCol) {
    return llvm::formatv("{0}:{1}", trimFilename(fileLineCol.getFilename()), fileLineCol.getLine());
  }

  std::string out;
  llvm::raw_string_ostream rso(out);
  loc.print(rso);
  return out;
}

} // namespace zirgen::Zll
