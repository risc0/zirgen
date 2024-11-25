// RUN: zirgen-translate --function=add_with_0 -cpp-codegen %s | FileCheck %s --check-prefixes=CHECK,CPP-CHECK
// RUN: zirgen-translate --function=add_with_0 -rust-codegen %s | FileCheck %s --check-prefixes=CHECK,RUST-CHECK

func.func @add_with_0(%arg : !zll.val<BabyBear>) -> !zll.val<BabyBear> {
  // RUST-CHECK-LABEL: fn add_with_0
  // RUST-CHEC-SAME: (arg0: Val) -> Result<Val> {
  // CPP-CHECK-LABEL: Val addWith_0(Val arg0) {
  %0 = zll.const 0
  %1 = zll.add %0:<BabyBear>, %arg:<BabyBear>
  %2 = zll.isz %1:<BabyBear>
  // CPP-CHECK: Val {{.*}} = isz((Val(0) + arg0))
  // RUST-CHECK: let x1 : Val = isz((Val::new(0) + arg0))
  zll.if %2 : <BabyBear> {
    // CPP-CHECK: if (to_size_t(x1)) {
    // RUST-CHECK: if is_true(x1) {
    %three = zll.const 3
    zll.eqz %three : <BabyBear>
    // CPP-CHECK: EQZ(Val(3), "Dialect/Zll/IR/test/emit-codegen.mlir:17")
    // RUST-CHECK: eqz!(Val::new(3), "Dialect/Zll/IR/test/emit-codegen.mlir:17")
  }
  // CHECK: }
  return %2: !zll.val<BabyBear>
  // CPP-CHECK: return x1
  // RUST-CHECK: return Ok(x1)
}
