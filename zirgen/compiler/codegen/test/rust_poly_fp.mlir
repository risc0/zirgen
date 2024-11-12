// RUN: zirgen-translate -zirgen-to-rust-poly-fp --function=fib --stage=exec %s | FileCheck %s

// CHECK-LABEL: FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args) {
func.func @fib(
  %arg0: !zll.buffer<3, constant>,
  %arg1: !zll.buffer<1, global>,
  %arg2: !zll.buffer<1, mutable>,
  %arg3: !zll.buffer<1, global>,
  %arg4: !zll.buffer<1, mutable>
) -> !zll.constraint attributes {
  deg = 2 : ui32,
  taps = [
    #zll.tap<0, 0, 0>,
    #zll.tap<1, 0, 0>,
    #zll.tap<1, 1, 0>,
    #zll.tap<1, 2, 0>,
    #zll.tap<2, 0, 0>,
    #zll.tap<2, 0, 1>,
    #zll.tap<2, 0, 2>
  ]
} {
  // CHECK: size_t mask = steps - 1;
  // CHECK: Fp x0(1);
  %0 = zll.const 1 {deg = 0 : ui32}

  // CHECK: FpExt x1 = FpExt(0);
  %1 = zll.true {deg = 0 : ui32}

  // CHECK: auto x2 = args[0][0 * steps + ((cycle - kInvRate * 0) & mask)];
  %2 = zll.get %arg0[0] back 0 tap 1 : <3, constant> {deg = 1 : ui32}

  // CHECK: auto x3 = args[2][0 * steps + ((cycle - kInvRate * 0) & mask)];
  %3 = zll.get %arg2[0] back 0 tap 4 : <1, mutable> {deg = 1 : ui32}

  // CHECK: auto x4 = x3 - x0;
  %4 = zll.sub %3 : <BabyBear>, %0 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x5 = x1 + x4 * poly_mix[0];
  %5 = zll.and_eqz %1, %4 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x6 = x1 + x2 * x5 * poly_mix[0];
  %6 = zll.and_cond %1, %2 : <BabyBear>, %5 {deg = 2 : ui32}

  // CHECK: auto x7 = args[0][1 * steps + ((cycle - kInvRate * 0) & mask)];
  %7 = zll.get %arg0[1] back 0 tap 2 : <3, constant> {deg = 1 : ui32}

  // CHECK: auto x8 = args[2][0 * steps + ((cycle - kInvRate * 1) & mask)];
  %8 = zll.get %arg2[0] back 1 tap 5 : <1, mutable> {deg = 1 : ui32}

  // CHECK: auto x9 = args[2][0 * steps + ((cycle - kInvRate * 2) & mask)];
  %9 = zll.get %arg2[0] back 2 tap 6 : <1, mutable> {deg = 1 : ui32}

  // CHECK: auto x10 = x8 + x9;
  %10 = zll.add %8 : <BabyBear>, %9 : <BabyBear> {deg = 1 : ui32}

  // CHECK: auto x11 = x3 - x10;
  %11 = zll.sub %3 : <BabyBear>, %10 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x12 = x1 + x11 * poly_mix[0];
  %12 = zll.and_eqz %1, %11 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x13 = x6 + x7 * x12 * poly_mix[1];
  %13 = zll.and_cond %6, %7 : <BabyBear>, %12 {deg = 2 : ui32}

  // CHECK: auto x14 = args[0][2 * steps + ((cycle - kInvRate * 0) & mask)];
  %14 = zll.get %arg0[2] back 0 tap 3 : <3, constant> {deg = 1 : ui32}

  // CHECK: auto x15 = args[1][0];
  %15 = zll.get_global %arg1[0] : <1, global> {deg = 0 : ui32}

  // CHECK: auto x16 = x15 - x3;
  %16 = zll.sub %15 : <BabyBear>, %3 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x17 = x1 + x16 * poly_mix[0];
  %17 = zll.and_eqz %1, %16 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x18 = x13 + x14 * x17 * poly_mix[2];
  %18 = zll.and_cond %13, %14 : <BabyBear>, %17 {deg = 2 : ui32}

  // CHECK: auto x19 = x2 + x7;
  %19 = zll.add %2 : <BabyBear>, %7 : <BabyBear> {deg = 1 : ui32}

  // CHECK: auto x20 = x19 + x14;
  %20 = zll.add %19 : <BabyBear>, %14 : <BabyBear> {deg = 1 : ui32}

  // CHECK: auto x21 = args[4][0 * steps + ((cycle - kInvRate * 0) & mask)];
  %21 = zll.get %arg4[0] back 0 tap 0 : <1, mutable> {deg = 1 : ui32}

  // CHECK: auto x22 = x21 - x0;
  %22 = zll.sub %21 : <BabyBear>, %0 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x23 = x1 + x22 * poly_mix[0];
  %23 = zll.and_eqz %1, %22 : <BabyBear> {deg = 1 : ui32}

  // CHECK: FpExt x24 = x18 + x20 * x23 * poly_mix[3];
  %24 = zll.and_cond %18, %20 : <BabyBear>, %23 {deg = 2 : ui32}

  // CHECK: return x24;
  return {deg = 2 : ui32} %24 : !zll.constraint
}
