// RUN: zirgen-translate -zirgen-to-rust-info --function=fib %s | FileCheck %s

module attributes {zll.buffers = #zll<buffers ("code", <3, constant>, 1), ("out", <1, global>), ("data", <1, mutable>, 2), ("mix", <1, global>), ("accum", <1, mutable>, 0)>, zll.protocolInfo = #zll<protocol_info "TEST____________">, zll.steps = #zll<steps "exec", "ram_verify", "verify_bytes", "compute_accum", "verify_accum">} {

func.func @fib(
  // CHECK-DAG: const CIRCUIT_INFO: ProtocolInfo = ProtocolInfo(*b"zirgen-translate");
  %arg0: !zll.buffer<3, constant>,
  // CHECK-DAG: const OUTPUT_SIZE: usize = 1;
  %arg1: !zll.buffer<1, global>,
  %arg2: !zll.buffer<1, mutable>,
  // CHECK-DAG: const MIX_SIZE: usize = 1;
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
  %0 = zll.const 1 {deg = 0 : ui32}
  %1 = zll.true {deg = 0 : ui32}
  %2 = zll.get %arg0[0] back 0 tap 1 : <3, constant> {deg = 1 : ui32}
  %3 = zll.get %arg2[0] back 0 tap 4 : <1, mutable> {deg = 1 : ui32}
  %4 = zll.sub %3 : <BabyBear>, %0 : <BabyBear> {deg = 1 : ui32}
  %5 = zll.and_eqz %1, %4 : <BabyBear> {deg = 1 : ui32}
  %6 = zll.and_cond %1, %2 : <BabyBear>, %5 {deg = 2 : ui32}
  %7 = zll.get %arg0[1] back 0 tap 2 : <3, constant> {deg = 1 : ui32}
  %8 = zll.get %arg2[0] back 1 tap 5 : <1, mutable> {deg = 1 : ui32}
  %9 = zll.get %arg2[0] back 2 tap 6 : <1, mutable> {deg = 1 : ui32}
  %10 = zll.add %8 : <BabyBear>, %9 : <BabyBear> {deg = 1 : ui32}
  %11 = zll.sub %3 : <BabyBear>, %10 : <BabyBear> {deg = 1 : ui32}
  %12 = zll.and_eqz %1, %11 : <BabyBear> {deg = 1 : ui32}
  %13 = zll.and_cond %6, %7 : <BabyBear>, %12 {deg = 2 : ui32}
  %14 = zll.get %arg0[2] back 0 tap 3 : <3, constant> {deg = 1 : ui32}
  %15 = zll.get_global %arg1[0] : <1, global> {deg = 0 : ui32}
  %16 = zll.sub %15 : <BabyBear>, %3 : <BabyBear> {deg = 1 : ui32}
  %17 = zll.and_eqz %1, %16 : <BabyBear> {deg = 1 : ui32}
  %18 = zll.and_cond %13, %14 : <BabyBear>, %17 {deg = 2 : ui32}
  %19 = zll.add %2 : <BabyBear>, %7 : <BabyBear> {deg = 1 : ui32}
  %20 = zll.add %19 : <BabyBear>, %14 : <BabyBear> {deg = 1 : ui32}
  %21 = zll.get %arg4[0] back 0 tap 0 : <1, mutable> {deg = 1 : ui32}
  %22 = zll.sub %21 : <BabyBear>, %0 : <BabyBear> {deg = 1 : ui32}
  %23 = zll.and_eqz %1, %22 : <BabyBear> {deg = 1 : ui32}
  %24 = zll.and_cond %18, %20 : <BabyBear>, %23 {deg = 2 : ui32}
  return {deg = 2 : ui32} %24 : !zll.constraint
}

}
