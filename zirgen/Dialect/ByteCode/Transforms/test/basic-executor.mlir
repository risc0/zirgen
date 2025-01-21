// RUN: zirgen-opt -gen-executor %s --mlir-disable-threading | FileCheck %s
// RUN: zirgen-opt -print-naive-bc %s --mlir-disable-threading | FileCheck %s --check-prefixes=BC-CHECK

func.func @empty() {
  return
}
// BC-CHECK-LABEL: START empty
// BC-CHECK-NEXT: INT-KIND DispatchKey bits 0
// BC-CHECK-NEXT: DispatchKey 0
// BC-CHECK-NEXT: END empty
// CHECK-LABEL: func.func @empty
// CHECK: zbytecode.execute
// CHECK-NEXT: bytecode.exit
// CHECK-NEXT: }
// CHECK-NEXT: return

func.func @basic() {
  %0, %1 = zbytecode.test [] : () -> (none, none)
  zbytecode.test [17] %1 : (none) -> ()
  return
}

// BC-CHECK-LABEL: START basic
// BC-CHECK-NEXT: INT-KIND DispatchKey bits 8
// BC-CHECK-NEXT: INT-KIND naive_buf bits 8
// BC-CHECK-NEXT: INT-KIND {{.*}}test{{.*}} bits 8
// BC-CHECK-NEXT: BUF naive_buf size 2
// BC-CHECK-NEXT: DispatchKey 0
// BC-CHECK-NEXT: naive_buf 0
// BC-CHECK-NEXT: naive_buf 1
// BC-CHECK-NEXT: DispatchKey 1
// BC-CHECK-NEXT: naive_buf 1
// BC-CHECK-NEXT: {{.*}}test{{.*}} 17
// BC-CHECK-NEXT: DispatchKey 2
// BC-CHECK-NEXT: END basic

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






