// RUN: zirgen-translate -zirgen-to-rust-step --function=fib --stage=exec %s | FileCheck %s

// CHECK-LABEL: Fp step_exec(void* ctx, size_t steps, size_t cycle, Fp** args) {
func.func @fib(
  %arg0: !zll.buffer<3, constant>,
  %arg1: !zll.buffer<1, global>,
  %arg2: !zll.buffer<1, mutable>,
  %arg3: !zll.buffer<1, global>,
  %arg4: !zll.buffer<1, mutable>
) -> !zll.val<BabyBear> {
  // CHECK: Fp [[var0:x[0-9]+]](1);
  %0 = zll.const 1

  // CHECK:      auto [[var1:x[0-9]+]] = args[0][0 * steps + ((cycle - 0) & mask)];
  // CHECK-NEXT: assert([[var1]] != Fp::invalid());
  %1 = zll.get %arg0[0] back 0 : <3, constant>

  // CHECK: if ([[var1]] != 0) {
  zll.if %1 : !zll.val<BabyBear> {
    // CHECK: auto& reg = args[2][0 * steps + cycle];
    // CHECK: assert(reg == Fp::invalid() || reg == [[var0]]);
    // CHECK: reg = [[var0]];
    zll.set %arg2 : <1, mutable>[0] = %0 : <BabyBear>
  }
  // CHECK: }

  // CHECK:      auto [[var2:x[0-9]+]] = args[0][1 * steps + ((cycle - 0) & mask)];
  // CHECK-NEXT: assert([[var2]] != Fp::invalid());
  %2 = zll.get %arg0[1] back 0 : <3, constant>

  // CHECK: if ([[var2]] != 0) {
  zll.if %2 : !zll.val<BabyBear> {
    // CHECK:      auto [[var5:x[0-9]+]] = args[2][0 * steps + ((cycle - 1) & mask)];
    // CHECK-NEXT: assert([[var5]] != Fp::invalid());
    %5 = zll.get %arg2[0] back 1 : <1, mutable>

    // CHECK:      auto [[var6:x[0-9]+]] = args[2][0 * steps + ((cycle - 2) & mask)];
    // CHECK-NEXT: assert([[var6]] != Fp::invalid());
    %6 = zll.get %arg2[0] back 2 : <1, mutable>

    // CHECK: auto [[var7:x[0-9]+]] = [[var5]] + [[var6]];
    %7 = zll.add %5 : <BabyBear>, %6 : <BabyBear>

    // CHECK: auto& reg = args[2][0 * steps + cycle];
    // CHECK-NEXT: assert(reg == Fp::invalid() || reg == [[var7]]);
    // CHECK-NEXT: reg = [[var7]];
    zll.set %arg2 : <1, mutable>[0] = %7 : <BabyBear>
  }
  // CHECK: }

  // CHECK:      auto [[var3:x[0-9]+]] = args[0][2 * steps + ((cycle - 0) & mask)];
  // CHECK-NEXT: assert([[var3]] != Fp::invalid());
  %3 = zll.get %arg0[2] back 0 : <3, constant>

  // CHECK: if ([[var3]] != 0) {
  zll.if %3 : !zll.val<BabyBear> {
    // CHECK:      auto [[var5:x[0-9]+]] = args[2][0 * steps + ((cycle - 0) & mask)];
    // CHECK-NEXT: assert([[var5]] != Fp::invalid());
    %5 = zll.get %arg2[0] back 0 : <1, mutable>

    // CHECK: args[1][0] = [[var5]];
    zll.set_global %arg1 : <1, global>[0] = %5 : <BabyBear>
  }
  // CHECK: }

  // CHECK: auto [[var4:x[0-9]+]] = [[var0]] - [[var3]];
  %4 = zll.sub %0 : <BabyBear>, %3 : <BabyBear>

  // CHECK: return [[var4]];
  return %4 : !zll.val<BabyBear>
}
