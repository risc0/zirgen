// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

// AB tests; runs the functions in both the emulator, and in the
// recursion circuit in the emulator, and compares the results to make
// sure everything works exactly the same.

#include "zirgen/circuit/recursion/test/AB.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// For hexDigitValue
#include "llvm/ADT/StringExtras.h"

namespace zirgen::recursion {

TEST(RECURSION, ArithAB) {
  doAB(HashType::SHA256, {{1, 2}}, [&](Buffer out, ReadIopVal iop) {
    Val a = iop.readBaseVals(1)[0];
    Val b = iop.readBaseVals(1)[0];
    out[0] = (a * b) + b - inv(a);
  });
}

TEST(RECURSION, IopAB) {
  doAB(HashType::SHA256, {{1, 2, 300000, 4, 5, 6, 7, 8, 9, 10}}, [&](Buffer out, ReadIopVal iop) {
    auto vals = iop.readBaseVals(2);
    auto digest = hash(vals);
    auto digest2 = iop.readDigests(1)[0];
    auto digest3 = fold(digest, digest2);
    iop.commit(digest3);
    out[0] = iop.rngBaseVal() * 17 + iop.rngBits(23);
  });
}

TEST(RECURSION, IopABMixed) {
  doAB(HashType::MIXED_POSEIDON2_SHA,
       {{1, 2, 300000, 4, 5, 6, 7, 8, 9, 10}},
       [&](Buffer out, ReadIopVal iop) {
         auto vals = iop.readBaseVals(2);
         auto digest = hash(vals);
         auto digest2 = iop.readDigests(1)[0];
         auto digest3 = fold(digest, digest2);
         iop.commit(digest3);
         out[0] = iop.rngBaseVal() * 17 + iop.rngBits(23);
       });
}

TEST(RECURSION, IopABPoseidon2Aligned) {
  doAB(HashType::POSEIDON2,
       {{1,  2,  300000, 4,  5,  6,  7,  8,  9,  10, 11, 12,
         13, 14, 15,     16, 17, 18, 19, 20, 21, 22, 23, 24}},
       [&](Buffer out, ReadIopVal iop) {
         auto vals = iop.readBaseVals(16);
         auto digest = hash(vals);
         auto digest2 = iop.readDigests(1)[0];
         auto digest3 = fold(digest, digest2);
         iop.commit(digest3);
         iop.rngBaseVal();
         iop.commit(digest3);
         out[0] = iop.rngBaseVal() * 17 + iop.rngBits(23);
       });
}

TEST(RECURSION, IopABPoseidon2Unaligned) {
  doAB(HashType::POSEIDON2,
       {{1, 2, 300000, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
       [&](Buffer out, ReadIopVal iop) {
         auto vals = iop.readBaseVals(15);
         auto digest = hash(vals);
         auto digest2 = iop.readDigests(1)[0];
         auto digest3 = fold(digest, digest2);
         iop.commit(digest3);
         iop.rngBaseVal();
         iop.commit(digest3);
         out[0] = iop.rngBaseVal() * 17 + iop.rngBits(23);
       });
}

TEST(RECURSION, IopABPoseidon2UnalignedLong) {
  doAB(HashType::POSEIDON2,
       {{1,  2,  300000, 4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
         14, 15, 16,     17, 18, 19, 20, 21, 22, 23, 24, 25}},
       [&](Buffer out, ReadIopVal iop) {
         auto vals = iop.readBaseVals(17);
         auto digest = hash(vals);
         auto digest2 = iop.readDigests(1)[0];
         auto digest3 = fold(digest, digest2);
         iop.commit(digest3);
         iop.rngBaseVal();
         iop.commit(digest3);
         out[0] = iop.rngBaseVal() * 17 + iop.rngBits(23);
       });
}

TEST(RECURSION, IopABPoseidon2AlignedLong) {
  doAB(HashType::POSEIDON2,
       {{1,  2,  300000, 4,  5,  100, 7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23,     24, 25, 26,  27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}},
       [&](Buffer out, ReadIopVal iop) {
         auto vals = iop.readBaseVals(32);
         auto digest = hash(vals);
         auto digest2 = iop.readDigests(1)[0];
         auto digest3 = fold(digest, digest2);
         iop.commit(digest3);
         iop.rngBaseVal();
         iop.commit(digest3);
         out[0] = iop.rngBaseVal() * 17 + iop.rngBits(23);
       });
}

TEST(RECURSION, CheckConsts) {
  doAB(HashType::SHA256, {{1, 2}}, [&](Buffer out, ReadIopVal iop) {
    Val bigConst(llvm::ArrayRef<uint64_t>({0, 0, 0, 1}));
    out[0] = bigConst * bigConst * bigConst * bigConst;
  });
}

TEST(RECURSION, HashCheckedBytes) {
  doAB(HashType::POSEIDON2, {{1, 2, 3, 4}}, [&](Buffer out, ReadIopVal iop) {
    Val z = iop.readExtVals(1)[0];
    auto result = hashCheckedBytes(z, 100);
    auto evalDigest = hash(result.second);
    out.setDigest(0, evalDigest, "eval_digest");
    out.setDigest(1, result.first, "result_digest");
  });
}

TEST(RECURSION, HashCheckedBytesPublic) {
  doAB(HashType::POSEIDON2, {{1, 2, 3, 4}}, [&](Buffer out, ReadIopVal iop) {
    Val z = iop.readExtVals(1)[0];
    auto result = hashCheckedBytesPublic(z, 17);
    out.setDigest(0, result.poseidon, "result_p");
    out.setDigest(1, result.sha, "result_s");
  });
}

TEST(RECURSION, MismatchedIopHashKinds) {
  EXPECT_THROW(
      {
        try {
          doAB(HashType::POSEIDON2, {{1, 2, 3, 4}}, [&](Buffer out, ReadIopVal iop) {
            Val z = iop.readExtVals(1)[0];
            auto result = hashCheckedBytesPublic(z, 1);
            iop.commit(result.poseidon);
            iop.commit(result.sha);
          });
        } catch (const std::runtime_error& e) {
          EXPECT_THAT(e.what(), testing::StartsWith("Unexpected digest kind"));
          throw;
        }
      },
      std::runtime_error);
}

TEST(RECURSION, Globals) {
  doAB(HashType::SHA256, {{1, 2}}, [&](Buffer out, ReadIopVal iop) {
    auto vals = iop.readBaseVals(2);
    auto digest1 = hash(vals);
    auto digest2 = fold(digest1, digest1);
    out.setDigest(0, digest1, "digest");
    out.setDigest(1, digest2, "digest2");
  });
}

TEST(RECURSION, GlobalsPoseidon2) {
  doAB(HashType::POSEIDON2, {{1, 2}}, [&](Buffer out, ReadIopVal iop) {
    auto vals = iop.readBaseVals(2);
    auto digest1 = hash(vals);
    auto digest2 = fold(digest1, digest1);
    out.setDigest(0, digest1, "digest1");
    out.setDigest(1, digest2, "digest2");
  });
}

// Exercises all the bit ops
TEST(RECURSION, BitOps) {
  doAB(HashType::SHA256, {{1, 2}}, [&](Buffer out, ReadIopVal iop) {
    Val x = 0;
    for (size_t i = 10; i != 6; --i) {
      x = x + (iop.rngBits(i) & x);
      iop.commit(hash(x));
    }
    out[0] = x;
  });
}

/*
The packing of a tagged struct is:
1) The tag digest
2) Any other digests
3) Any non-digest values as whole words (LE if numeric)
4) A count of digests a 16 bit LE # (to prevent reinterpretation)

The following python code:

```
import hashlib

def taggedStruct(tag, digests, vals):
    buf = hashlib.sha256(tag.encode()).digest()
    for digest in digests:
        buf = buf + digest
    for val in vals:
        buf = buf + val.to_bytes(length=4, byteorder='little')
    buf = buf + len(digests).to_bytes(length=2, byteorder='little')
    return hashlib.sha256(buf).digest()

# digest0 is a random set of words, each less than the baby bear prime, as a poseidon2 digest.
digest0_words = [0x0699544a, 0x10740194, 0x5fcfb7ec, 0x24d402b4,
                 0x2c917c8c, 0x58576ff6, 0x6e6063c6, 0x3fa4a82d]
digest0 = b''.join(struct.pack('<I', x) for x in digest0_words)
digest1 = taggedStruct("digest1", [], [1, 2013265920, 3])
digest2 = taggedStruct("digest2", [digest1, digest1], [2013265920, 5])
digest3 = taggedStruct("digest3", [digest1, digest2, digest1], [6, 7, 2013265920, 9, 10])
digest4 = taggedStruct("digest4", [digest3, digest0, digest2], [6, 2013265920, 9, 10])

print(digest4.hex())
```

prints:
```
566573df13310440113eb4a81e9cb8ab7c1a96aa8ed7450885d06a0ac06b956a
```

So we verify that the taggedStruct uses below produce the same result
*/

TEST(RECURSION, taggedStruct) {
  using namespace llvm;
  std::string goal = "566573df13310440113eb4a81e9cb8ab7c1a96aa8ed7450885d06a0ac06b956a";
  doAB(
      HashType::POSEIDON2,
      {{0x0699544a,
        0x10740194,
        0x5fcfb7ec,
        0x24d402b4,
        0x2c917c8c,
        0x58576ff6,
        0x6e6063c6,
        0x3fa4a82d}},
      [&](Buffer out, ReadIopVal iop) {
        auto digest0 = iop.readDigests(1)[0];
        auto digest1 = taggedStruct("digest1", {}, {1, 2013265920, 3});
        auto digest2 = taggedStruct("digest2", {digest1, digest1}, {2013265920, 5});
        auto digest3 =
            taggedStruct("digest3", {digest1, digest2, digest1}, {6, 7, 2013265920, 9, 10});
        auto digest4 = taggedStruct("digest4", {digest3, digest0, digest2}, {6, 2013265920, 9, 10});
        std::vector<Val> bytes;
        for (size_t i = 0; i < 32; i++) {
          bytes.push_back(hexDigitValue(goal[2 * i]) * 16 + hexDigitValue(goal[2 * i + 1]));
        }
        auto goal = intoDigest(bytes, Zll::DigestKind::Sha256);
        assert_eq(digest4, goal);
        out.setDigest(0, digest4, "digest");
      });
}

} // namespace zirgen::recursion
