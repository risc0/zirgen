// RUN: zirgen-opt -schedule-zll %s | FileCheck %s

func.func @simple() -> !zll.val<BabyBear> {
  %0 = zll.const 1
  %1 = zll.const 2
  %2 = zll.const 3
  %3 = zll.add %2:<BabyBear>, %1:<BabyBear>
  %4 = zll.add %3:<BabyBear>, %0:<BabyBear>
  return %4: !zll.val<BabyBear>
}

// CHECK-LABEL: func.func @simple
// CHECK: %0 = zll.const 3
// CHECK: %1 = zll.const 2
// CHECK: %2 = zll.add %0 {{.*}}, %1
// CHECK: %3 = zll.const 1
// CHECK: %4 = zll.add %2 {{.*}}, %3
// CHECK: return %4




