// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

namespace zirgen::dsl {

using Key = std::variant<mlir::StringRef, size_t>;
using KeyPath = std::vector<Key>;

} // namespace zirgen::dsl

inline std::ostream& operator<<(std::ostream& os, const zirgen::dsl::KeyPath& keyPath) {
  for (const zirgen::dsl::Key& key : keyPath) {
    if (auto* member = std::get_if<mlir::StringRef>(&key)) {
      os << "." << member->str();
    } else {
      os << "[" << std::get<size_t>(key) << "]";
    }
  }
  return os;
}
