// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "risc0/core/util.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace risc0 {

std::vector<uint8_t> loadFile(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  is.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  is.seekg(0, std::ios::end);
  size_t nbytes = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<uint8_t> elfContents(nbytes, 0);
  is.read(reinterpret_cast<char*>(elfContents.data()), nbytes);
  is.close();
  return elfContents;
}

std::array<uint32_t, 2> divide_rv32im(uint32_t numer, uint32_t denom, uint32_t signType) {
  uint32_t onesComp = (signType == 2);
  bool negNumer = signType && int32_t(numer) < 0;
  bool negDenom = signType == 1 && int32_t(denom) < 0;
  if (negNumer) {
    numer = -numer - onesComp;
  }
  if (negDenom) {
    denom = -denom - onesComp;
  }
  uint32_t quot;
  uint32_t rem;
  if (denom == 0) {
    quot = 0xffffffff;
    rem = numer;
  } else {
    quot = numer / denom;
    rem = numer % denom;
  }
  uint32_t quotNegOut = (negNumer ^ negDenom) - ((denom == 0) * negNumer);
  uint32_t remNegOut = negNumer;
  if (quotNegOut) {
    quot = -quot - onesComp;
  }
  if (remNegOut) {
    rem = -rem - onesComp;
  }
  return {quot, rem};
}

} // namespace risc0
