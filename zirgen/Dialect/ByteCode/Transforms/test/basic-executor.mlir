// RUN: zirgen-opt -gen-executor -gen-encoding %s --mlir-disable-threading | FileCheck %s


// -----

func.func @basic() -> !zll.val<BabyBear> {
  %0 = zll.const 1
  %1 = zll.neg %0 : <BabyBear>
  return %1 : !zll.val<BabyBear>
}

// CHECK-LABEL: func.func @basic(%arg0: !zbytecode.encoded)
// CHECK: zbytecode.execute %arg0
// CHECK-NEXT: %0:2 = zbytecode.test
// CHECK-NEXT: zbytecode.yield %0#0, %0#1 {{.*}} {intKinds = ["naive_buf", "naive_buf"]}
// CHECK-NEXT: }, {
// CHECK-NEXT: %0 = zbytecode.load "naive_buf"
// CHECK-NEXT: %1 = zbytecode.decode "zbytecode.test_0"
// CHECK-NEXT: zbytecode.operation "zbytecode.test" %0, %1
// CHECK-NEXT: zbytecode.yield  :
// CHECK-NEXT: }, {
// CHECK-NEXT: zbytecode.exit






