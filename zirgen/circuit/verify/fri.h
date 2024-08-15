// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/compiler/edsl/edsl.h"
#include <functional>

namespace zirgen::verify {

constexpr size_t kFriFold = 16;
constexpr size_t kInvRate = 4;
constexpr size_t kFriMinDegree = 256;

/// Array of 'forward' roots of unity.  These are critical for computing the NTT (see NTT.h).
/// Basically, we want `fwdRou[i] ^ (2^i) == 1 (mod P)`, but `fwdRou[i] ^ (2^(i-1)) != 1 (mod P)`.
/// Also, `fwdRou[i] = fwdRou[i+1] * fwdRou[i+1] (mod P)`.  To compute these values, we begin by
/// finding the smallest number, x, such that `x^(2^(maxRouPo2 - 1)) == -1 (mod P)`.  In
/// mathematicata, we find out x = 137 via:
/// `SelectFirst[Table[{i, PowerMod[i, 2^26, P]}, {i, 2, 500}], (#[[2]] == P - 1) &][[1]]`.
/// Then we compute `Table[PowerMod[137, (2^(27 - i)), P], {i, 0, 27}]`
static constexpr uint32_t kRouFwd[] = {
    1,          2013265920, 284861408,  1801542727, 567209306,  740045640,  918899846,
    1881002012, 1453957774, 65325759,   1538055801, 515192888,  483885487,  157393079,
    1695124103, 2005211659, 1540072241, 88064245,   1542985445, 1269900459, 1461624142,
    825701067,  682402162,  1311873874, 1164520853, 352275361,  18769,      137};

/// Array of 'reverse' roots of unity.  These are just the multiplicative inverse of the forward
/// roots of unity.
static constexpr uint32_t kRouRev[] = {
    1,          2013265920, 1728404513, 1592366214, 196396260,  1253260071, 72041623,
    1091445674, 145223211,  1446820157, 1030796471, 2010749425, 1827366325, 1239938613,
    246299276,  596347512,  1893145354, 246074437,  1525739923, 1194341128, 1463599021,
    704606912,  95395244,   15672543,   647517488,  584175179,  137728885,  749463956,
};

Val fold_eval(const std::vector<Val>& values, Val x);

using InnerVerify = std::function<Val(ReadIopVal& iop, Val idx)>;

// Verify a polynomial of degree 'deg', whose values at idx are returned by the inner verifier
void friVerify(ReadIopVal& iop, size_t deg, InnerVerify inner);

Val dynamic_pow(Val in, Val pow, size_t maxPow);

} // namespace zirgen::verify
