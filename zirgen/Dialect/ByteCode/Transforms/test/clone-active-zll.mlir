// RUN: zirgen-opt -clone-active-zll="target-regs=2" %s --mlir-disable-threading | FileCheck %s

// Simple keeps it under 2 active, so nothing needs to be done.
func.func @unchanged(%arg1 : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  %0 = zll.const 1
  %1 = zll.const 2
  %2 = zll.add %0:<BabyBear>, %1:<BabyBear>
  return %2: !zll.val<BabyBear>
}

// CHECK-LABEL: func.func @unchanged
// CHECK: %0 = zll.const 1
// CHECK: %1 = zll.const 2
// CHECK: %2 = zll.add %0 {{.*}}, %1
// CHECK: return %2


// Exceeds 2 active, so should clone the first const since it's the farthest away.
func.func @simple()  {
  %0 = zll.const 1
  zll.eqz %0 : <BabyBear>
  %1 = zll.const 2
  zll.eqz %0: <BabyBear>
  zll.eqz %1: <BabyBear>
  %2 = zll.const 3
  zll.eqz %0: <BabyBear>
  zll.eqz %1: <BabyBear>
  zll.eqz %2: <BabyBear>
  return
}


// CHECK-LABEL: func.func @simple
// CHECK-NEXT: %0 = zll.const 1
// CHECK-NEXT: zll.eqz %0
// CHECK-NEXT: %1 = zll.const 2
// CHECK-NEXT: zll.eqz %0
// CHECK-NEXT: zll.eqz %1
// CHECK-NEXT: %2 = zll.const 3
// CHECK-NEXT: %3 = zll.const 1
// CHECK-NEXT: zll.eqz %3
// CHECK-NEXT: zll.eqz %1
// CHECK-NEXT: zll.eqz %2
// CHECK-NEXT: return
