// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/r1cs/r1csfile.h"

#include "tools/cpp/runfiles/runfiles.h"
#include <fstream>
#include <gtest/gtest.h>

using bazel::tools::cpp::runfiles::Runfiles;

namespace zirgen::r1csfile {

namespace {

BigintLE big(uint32_t x) {
  BigintLE out(32);
  for (size_t i = 0; i < 32; ++i) {
    out[i] = x & 0x0FF;
    x >>= 8;
  }
  return out;
}

inline bool operator==(const BigintLE& x, const BigintLE& y) {
  if (x.size() != y.size()) {
    return false;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] != y[i]) {
      return false;
    }
  }
  return true;
}

} // namespace

TEST(r1csfile, example) {
  // read the circom example file and verify its contents
  std::string error;
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
  ASSERT_TRUE(runfiles != NULL);
  ASSERT_EQ(error, "");
  std::string path = runfiles->Rlocation("zirgen/zirgen/compiler/r1cs/test/example.r1cs");

  FILE* src = fopen(path.c_str(), "rb");
  ASSERT_TRUE(src != NULL);
  auto sys = r1csfile::read(src);
  fclose(src);

  // Verify that the header section had expected values
  EXPECT_EQ(sys->header.fieldSize, 32);
  BigintLE prime{0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9,
                 0x79, 0x48, 0xe8, 0x33, 0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45,
                 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1, 0x72, 0x4e, 0x64, 0x30};
  EXPECT_EQ(sys->header.prime, prime);
  EXPECT_EQ(sys->header.nWires, 7);
  EXPECT_EQ(sys->header.nPubOut, 1);
  EXPECT_EQ(sys->header.nPubIn, 2);
  EXPECT_EQ(sys->header.nPrvIn, 3);
  EXPECT_EQ(sys->header.nLabels, 1000);

  // Ensure that we found the expected constraints
  std::vector<Constraint> constraints{
      {
          {
              // A
              {5, big(3)},
              {6, big(8)},
          },
          {
              // B
              {0, big(2)},
              {2, big(20)},
              {3, big(12)},
          },
          {
              // C
              {0, big(5)},
              {2, big(7)},
          },
      },
      {
          {
              // A
              {1, big(4)},
              {4, big(8)},
              {5, big(3)},
          },
          {
              // B
              {3, big(44)},
              {6, big(6)},
          },
          {
              // C
          },
      },
      {
          {
              // A
              {6, big(4)},
          },
          {
              // B
              {0, big(6)},
              {2, big(11)},
              {3, big(5)},
          },
          {
              // C
              {6, big(600)},
          },
      },
  };
  EXPECT_EQ(sys->constraints.size(), constraints.size());
  for (size_t i = 0; i < constraints.size(); ++i) {
    const Constraint& val = sys->constraints[i];
    const Constraint& ref = constraints[i];
    EXPECT_EQ(val.A.size(), ref.A.size());
    for (size_t j = 0; j < val.A.size(); j++) {
      EXPECT_EQ(val.A[j].index, ref.A[j].index);
      EXPECT_EQ(val.A[j].value, ref.A[j].value);
    }
    EXPECT_EQ(val.B.size(), ref.B.size());
    for (size_t j = 0; j < val.B.size(); j++) {
      EXPECT_EQ(val.B[j].index, ref.B[j].index);
      EXPECT_EQ(val.B[j].value, ref.B[j].value);
    }
    EXPECT_EQ(val.C.size(), ref.C.size());
    for (size_t j = 0; j < val.C.size(); j++) {
      EXPECT_EQ(val.C[j].index, ref.C[j].index);
      EXPECT_EQ(val.C[j].value, ref.C[j].value);
    }
  }

  // Inspect the map section and validate its ID values
  std::vector<LabelID> map{0, 3, 10, 11, 12, 15, 324};
  EXPECT_EQ(sys->map.size(), map.size());
  for (size_t i = 0; i < map.size(); ++i) {
    EXPECT_EQ(sys->map[i], map[i]);
  }
}

} // namespace zirgen::r1csfile
