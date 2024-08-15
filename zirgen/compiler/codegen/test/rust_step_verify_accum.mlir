// RUN: zirgen-translate -zirgen-to-rust-step --function=fib --stage=verify_accum %s | FileCheck %s

// CHECK-LABEL: Fp step_verify_accum(void* ctx, size_t steps, size_t cycle, Fp** args) {
func.func @fib(
  %arg0: !zll.buffer<3, constant>,
  %arg1: !zll.buffer<1, global>,
  %arg2: !zll.buffer<1, mutable>,
  %arg3: !zll.buffer<1, global>,
  %arg4: !zll.buffer<1, mutable>
) -> !zll.val<BabyBear> {
  // CHECK: Fp x0(1);
  %0 = zll.const 1

  // CHECK:      auto x1 = args[0][0 * steps + ((cycle - 0) & mask)];
  // CHECK-NEXT: assert(x1 != Fp::invalid());
  %1 = zll.get %arg0[0] back 0 : <3, constant>

  // CHECK:      auto x2 = args[0][1 * steps + ((cycle - 0) & mask)];
  // CHECK-NEXT: assert(x2 != Fp::invalid());
  %2 = zll.get %arg0[1] back 0 : <3, constant>

  // CHECK:      auto x3 = args[0][2 * steps + ((cycle - 0) & mask)];
  // CHECK-NEXT: assert(x3 != Fp::invalid());
  %3 = zll.get %arg0[2] back 0 : <3, constant>

  // CHECK: auto x4 = x1 + x2;
  %4 = zll.add %1 : <BabyBear>, %2 : <BabyBear>

  // CHECK: auto x5 = x4 + x3;
  %5 = zll.add %4 : <BabyBear>, %3 : <BabyBear>

  // CHECK: if (x5 != 0) {
  zll.if %5 : !zll.val<BabyBear> {
    // CHECK:      auto& reg = args[4][0 * steps + cycle];
    // CHECK-NEXT: assert(reg == Fp::invalid() || reg == x0);
    // CHECK-NEXT: reg = x0;
    zll.set %arg4 : <1, mutable>[0] = %0 : <BabyBear>
  }
  // CHECK: }

  // CHECK: return x0;
  return %0 : !zll.val<BabyBear>
}
