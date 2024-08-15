// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/zkp/baby_bear.h"
#include "llvm/ADT/APInt.h"

namespace zirgen::Zll {

constexpr size_t kBigIntSize = 512;
// Max bits is 252 because normalize_impl component cannot handle 253 bits.
constexpr size_t kMaxBits = 252;

struct BigInt {
public:
  inline BigInt() : inner(kBigIntSize, 0, true) {}
  inline BigInt(int64_t val) : inner(kBigIntSize, val, true) {}
  inline BigInt(llvm::StringRef str) : inner(kBigIntSize, str, 10) {}
  inline BigInt operator+(const BigInt& rhs) const { return BigInt(inner + rhs.inner); }
  inline BigInt operator-() const { return BigInt(-inner); }
  inline BigInt operator-(const BigInt& rhs) const { return BigInt(inner - rhs.inner); }
  inline BigInt operator*(const BigInt& rhs) const { return BigInt(inner * rhs.inner); }
  inline BigInt operator%(const BigInt& rhs) const { return BigInt(inner.urem(rhs.inner)); }
  inline BigInt operator/(const BigInt& rhs) const { return BigInt(inner.udiv(rhs.inner)); }
  inline bool operator<(const BigInt& rhs) const { return inner.slt(rhs.inner); }
  inline bool operator<=(const BigInt& rhs) const { return inner.sle(rhs.inner); }
  inline bool operator>(const BigInt& rhs) const { return inner.sgt(rhs.inner); }
  inline bool operator>=(const BigInt& rhs) const { return inner.sge(rhs.inner); }
  inline bool operator==(const BigInt& rhs) const { return inner.eq(rhs.inner); }
  inline bool operator!=(const BigInt& rhs) const { return inner.ne(rhs.inner); }
  inline size_t bits() const { return (inner == 0) ? 1 : inner.ceilLogBase2(); }
  inline std::string toStr() const {
    llvm::SmallVector<char, 100> out;
    inner.toStringSigned(out);
    return std::string(out.data(), out.size());
  }
  inline const llvm::APInt& getInner() const { return inner; }

private:
  inline BigInt(llvm::APInt inner) : inner(inner) {}
  llvm::APInt inner;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const BigInt& rhs) {
  os << rhs.getInner();
  return os;
}

struct BigIntRange {
public:
  inline BigIntRange() : low(0), high(0) {}
  inline BigIntRange(BigInt val) : low(val), high(val) {}
  inline BigIntRange(BigInt low, BigInt high) : low(low), high(high) {}
  inline static BigIntRange rangeP() { return BigIntRange(0, kBabyBearP - 1); }
  inline const BigInt& getLow() const { return low; }
  inline const BigInt& getHigh() const { return high; }
  inline BigIntRange operator+(const BigIntRange& rhs) const {
    return BigIntRange(low + rhs.low, high + rhs.high);
  }
  inline BigIntRange operator-() const { return BigIntRange(-high, -low); }
  inline BigIntRange operator-(const BigIntRange& rhs) const {
    return BigIntRange(low - rhs.high, high - rhs.low);
  }
  inline BigIntRange operator*(const BigIntRange& rhs) const {
    auto pair = std::minmax({low * rhs.low, low * rhs.high, high * rhs.low, high * rhs.high});
    return BigIntRange(pair.first, pair.second);
  }
  inline size_t bits() const { return (high - low).bits(); }
  inline bool inRangeP() { return low >= 0 && high <= kBabyBearP - 1; }
  inline bool inRangeMax() { return bits() <= kMaxBits; }

private:
  BigInt low;
  BigInt high;
};

inline BigIntRange get(mlir::DenseMap<mlir::Value, BigIntRange>& map, mlir::Value value) {
  if (!map.count(value)) {
    map[value] = BigIntRange::rangeP();
  }
  return map[value];
}

inline void
set(mlir::DenseMap<mlir::Value, BigIntRange>& map, mlir::Value value, BigIntRange range) {
  map[value] = range;
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const BigIntRange& rhs) {
  os << "[" << rhs.getLow() << ", " << rhs.getHigh() << "]";
  return os;
}

} // namespace zirgen::Zll
