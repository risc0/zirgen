// RUN: zirgen-opt --canonicalize %s | FileCheck %s

func.func @add_with_0(%arg : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // CHECK-LABEL: func @add_with_0
  // CHECK-NEXT: return %arg
  %0 = zll.const 0
  %1 = zll.add %0:<BabyBear>, %arg:<BabyBear>
  return %1: !zll.val<BabyBear>
}

func.func @add_const_fold() -> !zll.val<BabyBear> {
  // CHECK-LABEL: @add_const_fold
  // CHECK-NEXT: zll.const 6
  // CHECK-NEXT: return
  %0 = zll.const 2
  %1 = zll.const 4
  %2 = zll.add %0:<BabyBear>, %1:<BabyBear>
  return %2: !zll.val<BabyBear>
}

func.func @extension_add_const_fold() -> !zll.val<BabyBear ext> {
  // CHECK-LABEL: @extension_add_const_fold
  // CHECK-NEXT: zll.const [4, 6, 8, 10]
  // CHECK-NEXT: return
  %0 = zll.const [1, 2, 3, 4]
  %1 = zll.const [3, 4, 5, 6]
  %2 = zll.add %0:<BabyBear ext>, %1:<BabyBear ext>
  return %2: !zll.val<BabyBear ext>
}

func.func @sub_const_fold() -> !zll.val<BabyBear> {
  // CHECK-LABEL: @sub_const_fold
  // CHECK-NEXT: zll.const 1
  // CHECK-NEXT: return
  %0 = zll.const 4
  %1 = zll.const 3
  %2 = zll.sub %0:<BabyBear>, %1:<BabyBear>
  return %2: !zll.val<BabyBear>
}

func.func @sub_same_val(%arg0 : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // CHECK-LABEL: @sub_same_val
  // CHECK-NEXT: zll.const 0
  // CHECK-NEXT: return
  %2 = zll.sub %arg0:<BabyBear>, %arg0:<BabyBear>
  return %2: !zll.val<BabyBear>
}

func.func @extension_sub_const_fold() -> !zll.val<BabyBear ext> {
  // CHECK-LABEL: @extension_sub_const_fold
  // CHECK-NEXT: zll.const [2, 2, 2, 2]
  // CHECK-NEXT: return
  %0 = zll.const [3, 4, 5, 6]
  %1 = zll.const [1, 2, 3, 4]
  %2 = zll.sub %0:<BabyBear ext>, %1:<BabyBear ext>
  return %2: !zll.val<BabyBear ext>
}

func.func @neg_fold() -> !zll.val<BabyBear> {
  // CHECK-LABEL: @neg_fold
  // CHECK-NEXT: zll.const
  // CHECK-NEXT: return
  %0 = zll.const 1
  %1 = zll.neg %0:<BabyBear>
  return %1: !zll.val<BabyBear>
}

func.func @extension_neg_fold() -> !zll.val<BabyBear ext> {
  // CHECK-LABEL: @extension_neg_fold
  // CHECK-NEXT: zll.const
  // CHECK-NEXT: return
  %0 = zll.const [4, 5, 6, 7]
  %1 = zll.neg %0:<BabyBear ext>
  return %1: !zll.val<BabyBear ext>
}

func.func @is_zero_const_fold() -> !zll.val<BabyBear> {
  // CHECK-LABEL: @is_zero_const_fold
  // CHECK-NEXT: zll.const 1
  %0 = zll.const 0
  %1 = zll.isz %0:<BabyBear>
  return %1: !zll.val<BabyBear>
}

func.func @extension_is_zero_const_fold() -> !zll.val<BabyBear ext> {
  // CHECK-LABEL: @extension_is_zero_const_fold
  // CHECK-NEXT: zll.const [0, 0, 0, 0]
  // CHECK-NEXT: return
  %0 = zll.const [0, 1, 2, 3]
  %1 = zll.isz %0:<BabyBear ext>
  return %1: !zll.val<BabyBear ext>
}

func.func @mul_by_zero(%arg : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // CHECK-LABEL: func @mul_by_zero
  // CHECK-NEXT: zll.const 0
  // CHECK-NEXT: return
  %0 = zll.const 2
  %1 = zll.add %arg:<BabyBear>, %0:<BabyBear>
  %2 = zll.const 0
  %3 = zll.mul %1:<BabyBear>, %2:<BabyBear>
  return %3: !zll.val<BabyBear>
}

func.func @mul_const_fold() -> !zll.val<BabyBear> {
  // CHECK-LABEL: @mul_const_fold
  // CHECK-NEXT: zll.const 18
  // CHECK-NEXT: return
  %0 = zll.const 6
  %1 = zll.const 3
  %2 = zll.mul %0:<BabyBear>, %1:<BabyBear>
  return %2: !zll.val<BabyBear>
}

func.func @extension_mul_const_fold() -> !zll.val<BabyBear ext> {
  // CHECK-LABEL: @extension_mul_const_fold
  // CHECK-NEXT: zll.const [2013265877, 2013265916, 3, 4]
  // CHECK-NEXT: return
  %0 = zll.const [0, 2, 1, 0]
  %1 = zll.const [3, 0, 2, 1]
  %2 = zll.mul %0:<BabyBear ext>, %1:<BabyBear ext>
  return %2: !zll.val<BabyBear ext>
}

func.func @inv_const_fold() -> !zll.val<BabyBear> {
  // CHECK-LABEL: @inv_const_fold
  // CHECK-NEXT: zll.const 1
  // CHECK-NEXT: return
  %0 = zll.const 2
  %1 = zll.inv %0:<BabyBear>
  %2 = zll.mul %0:<BabyBear>, %1:<BabyBear>
  return %2: !zll.val<BabyBear>
}

func.func @extension_inv_const_fold() -> !zll.val<BabyBear ext> {
  // CHECK-LABEL: @extension_inv_const_fold
  // CHECK-NEXT: zll.const [1, 0, 0, 0]
  // CHECK-NEXT: return
  %0 = zll.const [4, 3, 2, 1]
  %1 = zll.inv %0:<BabyBear ext>
  %2 = zll.mul %0:<BabyBear ext>, %1:<BabyBear ext>
  return %2: !zll.val<BabyBear ext>
}

func.func @pow_expand_0(%arg : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // CHECK-LABEL: func @pow_expand_0
  // CHECK-NEXT: zll.const 1
  // CHECK-NEXT: return
  %0 = zll.pow %arg:<BabyBear>, 0
  return %0: !zll.val<BabyBear>
}

func.func @pow_expand_1(%arg : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // CHECK-LABEL: func @pow_expand_1
  // CHECK-NEXT: return %arg
  %0 = zll.pow %arg:<BabyBear>, 1
  return %0: !zll.val<BabyBear>
}

func.func @pow_expand_2(%arg : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // CHECK-LABEL: func @pow_expand_2
  // CHECK-NEXT: mul %arg0 : <BabyBear>, %arg0
  // CHECK-NEXT: return %0
  %0 = zll.pow %arg:<BabyBear>, 2
  return %0: !zll.val<BabyBear>
}

func.func @pow_expand_5(%arg : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // CHECK-LABEL: func @pow_expand_5
  // CHECK-NEXT: %0 = zll.mul %arg0 : <BabyBear>, %arg0 : <BabyBear>
  // CHECK-NEXT: %1 = zll.mul %0 : <BabyBear>, %0 : <BabyBear>
  // CHECK-NEXT: %2 = zll.mul %arg0 : <BabyBear>, %1 : <BabyBear>
  // CHECK-NEXT: return %2
  %0 = zll.pow %arg:<BabyBear>, 5
  return %0: !zll.val<BabyBear>
}

func.func @bitand_const_fold() -> !zll.val<BabyBear> {
  // CHECK-LABEL: @bitand_const_fold
  // CHECK-NEXT: zll.const 5
  %0 = zll.const 21
  %1 = zll.const 15
  %2 = zll.bit_and %0: <BabyBear>, %1: <BabyBear>
  return %2: !zll.val<BabyBear>
}
