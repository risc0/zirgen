// RUN: zirgen-translate -zirgen-to-rust-poly-ext --function=fib %s | FileCheck %s

// CHECK-LABEL: const DEF: PolyExtStepDef
func.func @fib(
  %arg0: !zll.buffer<3, constant>,
  %arg1: !zll.buffer<1, global>,
  %arg2: !zll.buffer<1, mutable>,
  %arg3: !zll.buffer<1, global>,
  %arg4: !zll.buffer<1, mutable>
) -> !zll.constraint attributes {
  deg = 2 : ui32,
  zll.taps = #zll<taps
    <0, 0, 0>,
    <1, 0, 0>,
    <1, 1, 0>,
    <1, 2, 0>,
    <2, 0, 0>,
    <2, 0, 1>,
    <2, 0, 2>>
} {
  // CHECK: block: &[PolyExtStep::Const(1),
  %0 = zll.const 1 {deg = 0 : ui32}

  // CHECK: PolyExtStep::True,
  %1 = zll.true {deg = 0 : ui32}

  // CHECK: PolyExtStep::Get(1),
  %2 = zll.get %arg0[0] back 0 tap 1 : <3, constant> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Get(4),
  %3 = zll.get %arg2[0] back 0 tap 4 : <1, mutable> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Sub(2, 0),
  %4 = zll.sub %3 : <BabyBear>, %0 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndEqz(0, 3),
  %5 = zll.and_eqz %1, %4 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndCond(0, 1, 1),
  %6 = zll.and_cond %1, %2 : <BabyBear>, %5 {deg = 2 : ui32}

  // CHECK: PolyExtStep::Get(2),
  %7 = zll.get %arg0[1] back 0 tap 2 : <3, constant> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Get(5),
  %8 = zll.get %arg2[0] back 1 tap 5 : <1, mutable> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Get(6),
  %9 = zll.get %arg2[0] back 2 tap 6 : <1, mutable> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Add(5, 6),
  %10 = zll.add %8 : <BabyBear>, %9 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Sub(2, 7),
  %11 = zll.sub %3 : <BabyBear>, %10 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndEqz(0, 8),
  %12 = zll.and_eqz %1, %11 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndCond(2, 4, 3),
  %13 = zll.and_cond %6, %7 : <BabyBear>, %12 {deg = 2 : ui32}

  // CHECK: PolyExtStep::Get(3),
  %14 = zll.get %arg0[2] back 0 tap 3 : <3, constant> {deg = 1 : ui32}

  // CHECK: PolyExtStep::GetGlobal(0, 0),
  %15 = zll.get_global %arg1[0] : <1, global> {deg = 0 : ui32}

  // CHECK: PolyExtStep::Sub(10, 2),
  %16 = zll.sub %15 : <BabyBear>, %3 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndEqz(0, 11),
  %17 = zll.and_eqz %1, %16 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndCond(4, 9, 5),
  %18 = zll.and_cond %13, %14 : <BabyBear>, %17 {deg = 2 : ui32}

  // CHECK: PolyExtStep::Add(1, 4),
  %19 = zll.add %2 : <BabyBear>, %7 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Add(12, 9),
  %20 = zll.add %19 : <BabyBear>, %14 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Get(0),
  %21 = zll.get %arg4[0] back 0 tap 0 : <1, mutable> {deg = 1 : ui32}

  // CHECK: PolyExtStep::Sub(14, 0),
  %22 = zll.sub %21 : <BabyBear>, %0 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndEqz(0, 15),
  %23 = zll.and_eqz %1, %22 : <BabyBear> {deg = 1 : ui32}

  // CHECK: PolyExtStep::AndCond(6, 13, 7),
  %24 = zll.and_cond %18, %20 : <BabyBear>, %23 {deg = 2 : ui32}

  // CHECK: ret: 8
  return {deg = 2 : ui32} %24 : !zll.constraint
}
