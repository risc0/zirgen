NondetRegStruct
back_NondetReg(ExecContext& ctx, Index distance0, BoundLayout<NondetRegLayout> layout1) {
  NondetRegStruct x2 = NondetRegStruct{._super = LOAD(LAYOUT_LOOKUP(layout1, _super), distance0)};
  return x2;
}
NondetRegStruct exec_NondetReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  BoundLayout<Reg> x2 = LAYOUT_LOOKUP(layout1, _super);
  STORE(x2, arg0);
  return NondetRegStruct{._super = LOAD(x2, 0)};
}
NondetExtRegStruct
back_NondetExtReg(ExecContext& ctx, Index distance0, BoundLayout<NondetExtRegLayout> layout1) {
  NondetExtRegStruct x2 =
      NondetExtRegStruct{._super = LOAD_EXT(LAYOUT_LOOKUP(layout1, _super), distance0)};
  return x2;
}
NondetExtRegStruct
exec_NondetExtReg(ExecContext& ctx, ExtVal arg0, BoundLayout<NondetExtRegLayout> layout1) {
  BoundLayout<Reg> x2 = LAYOUT_LOOKUP(layout1, _super);
  STORE_EXT(x2, arg0);
  return NondetExtRegStruct{._super = LOAD_EXT(x2, 0)};
}
NondetRegStruct back_Reg(ExecContext& ctx, Index distance0, BoundLayout<NondetRegLayout> layout1) {
  // Reg(<preamble>:4)
  NondetRegStruct x2 = back_NondetReg(ctx, distance0, layout1);
  return x2;
}
NondetRegStruct exec_Reg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  NondetRegStruct x2 = exec_NondetReg(ctx, arg0, layout1);
  // Reg(<preamble>:5)
  EQZ((arg0 - x2._super), "Reg(<preamble>:5)");
  return x2;
}
NondetExtRegStruct
back_ExtReg(ExecContext& ctx, Index distance0, BoundLayout<NondetExtRegLayout> layout1) {
  // ExtReg(<preamble>:10)
  NondetExtRegStruct x2 = back_NondetExtReg(ctx, distance0, layout1);
  return x2;
}
NondetExtRegStruct
exec_ExtReg(ExecContext& ctx, ExtVal arg0, BoundLayout<NondetExtRegLayout> layout1) {
  NondetExtRegStruct x2 = exec_NondetExtReg(ctx, arg0, layout1);
  // ExtReg(<preamble>:11)
  EQZ((x2._super - arg0), "loc(callsite(unknown at  ExtReg ( <preamble> :11:11)))");
  return x2;
}
NondetRegStruct
exec_NondetBitReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  // NondetBitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:11)
  NondetRegStruct x2 = exec_NondetReg(ctx, arg0, layout1);
  Val x3 = x2._super;
  // AssertBit(zirgen/circuit/rv32im/v2/dsl/bits.zir:6)
  // NondetBitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:12)
  EQZ((x3 * (Val(1) - x3)),
      "loc(callsite( AssertBit ( zirgen/circuit/rv32im/v2/dsl/bits.zir :6:20) at  NondetBitReg ( "
      "zirgen/circuit/rv32im/v2/dsl/bits.zir :12:13)))");
  return x2;
}
BitRegStruct exec_BitReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  // BitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:17)
  NondetRegStruct x2 = exec_NondetBitReg(ctx, arg0, layout1);
  // BitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:18)
  EQZ((arg0 - x2._super), "BitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:18)");
  return BitRegStruct{};
}
NondetRegStruct
exec_NondetTwitReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  // NondetTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:48)
  NondetRegStruct x2 = exec_NondetReg(ctx, arg0, layout1);
  Val x3 = x2._super;
  // AssertTwit(zirgen/circuit/rv32im/v2/dsl/bits.zir:35)
  // NondetTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:49)
  Val x4 = ((x3 * (Val(1) - x3)) * (Val(2) - x3));
  EQZ((x4 * (Val(3) - x3)),
      "loc(callsite( AssertTwit ( zirgen/circuit/rv32im/v2/dsl/bits.zir :35:44) at  NondetTwitReg "
      "( zirgen/circuit/rv32im/v2/dsl/bits.zir :49:14)))");
  return x2;
}
NondetFakeTwitRegStruct
exec_NondetFakeTwitReg(ExecContext& ctx, Val arg0, BoundLayout<NondetFakeTwitRegLayout> layout1) {
  // NondetFakeTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:55)
  NondetRegStruct x2 = exec_NondetBitReg(ctx, bitAnd(arg0, Val(1)), LAYOUT_LOOKUP(layout1, reg0));
  // NondetFakeTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:56)
  NondetRegStruct x3 = exec_NondetBitReg(
      ctx, (bitAnd(arg0, Val(2)) * Val(1006632961)), LAYOUT_LOOKUP(layout1, reg1));
  // NondetFakeTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:57)
  Val x4 = ((x3._super * Val(2)) + x2._super);
  return NondetFakeTwitRegStruct{._super = x4};
}
FakeTwitRegStruct
exec_FakeTwitReg(ExecContext& ctx, Val arg0, BoundLayout<NondetFakeTwitRegLayout> layout1) {
  // FakeTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:67)
  NondetFakeTwitRegStruct x2 = exec_NondetFakeTwitReg(ctx, arg0, layout1);
  // FakeTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:68)
  EQZ((arg0 - x2._super), "FakeTwitReg(zirgen/circuit/rv32im/v2/dsl/bits.zir:68)");
  return FakeTwitRegStruct{};
}
NondetRegStruct exec_IsZero(ExecContext& ctx, Val arg0, BoundLayout<IsZeroLayout> layout1) {
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:8)
  NondetRegStruct x2 = exec_NondetReg(ctx, isz(arg0), LAYOUT_LOOKUP(layout1, _super));
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:11)
  NondetRegStruct x3 = exec_NondetReg(ctx, inv_0(arg0), LAYOUT_LOOKUP(layout1, inv));
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:8)
  Val x4 = x2._super;
  // AssertBit(zirgen/circuit/rv32im/v2/dsl/bits.zir:6)
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:14)
  Val x5 = (Val(1) - x4);
  EQZ((x4 * x5),
      "loc(callsite( AssertBit ( zirgen/circuit/rv32im/v2/dsl/bits.zir :6:20) at  IsZero ( "
      "zirgen/circuit/rv32im/v2/dsl/is_zero.zir :14:13)))");
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:11)
  Val x6 = x3._super;
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:16)
  EQZ(((arg0 * x6) - x5), "IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:16)");
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:18)
  EQZ((x4 * arg0), "IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:18)");
  // IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:20)
  EQZ((x4 * x6), "IsZero(zirgen/circuit/rv32im/v2/dsl/is_zero.zir:20)");
  return x2;
}
ArgU8Struct exec_ArgU8(ExecContext& ctx, Val arg0, Val arg1, BoundLayout<ArgU8Layout> layout2) {
  // ArgU8(zirgen/circuit/rv32im/v2/dsl/lookups.zir:8)
  NondetRegStruct x3 = exec_NondetReg(ctx, arg0, LAYOUT_LOOKUP(layout2, count));
  // ArgU8(zirgen/circuit/rv32im/v2/dsl/lookups.zir:9)
  NondetRegStruct x4 = exec_NondetReg(ctx, arg1, LAYOUT_LOOKUP(layout2, val));
  // LookupDelta(zirgen/circuit/rv32im/v2/dsl/lookups.zir:4)
  // ArgU8(zirgen/circuit/rv32im/v2/dsl/lookups.zir:10)
  INVOKE_EXTERN(ctx, lookupDelta, Val(8), x4._super, x3._super);
  return ArgU8Struct{.count = x3, .val = x4};
}
NondetRegStruct
exec_NondetU8Reg(ExecContext& ctx, Val arg0, BoundLayout<NondetU8RegLayout> layout1) {
  // NondetU8Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:15)
  ArgU8Struct x2 = exec_ArgU8(ctx, Val(1), arg0, LAYOUT_LOOKUP(layout1, arg));
  // NondetU8Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:16)
  Val x3 = (x2.count._super - Val(1));
  EQZ(x3, "NondetU8Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:16)");
  return x2.val;
}
U8RegStruct exec_U8Reg(ExecContext& ctx, Val arg0, BoundLayout<U8RegLayout> layout1) {
  // U8Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:22)
  NondetRegStruct x2 = exec_NondetU8Reg(ctx, arg0, LAYOUT_LOOKUP(layout1, ret));
  // U8Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:23)
  EQZ((x2._super - arg0), "U8Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:23)");
  return U8RegStruct{};
}
ArgU16Struct exec_ArgU16(ExecContext& ctx, Val arg0, Val arg1, BoundLayout<ArgU16Layout> layout2) {
  // ArgU16(zirgen/circuit/rv32im/v2/dsl/lookups.zir:28)
  NondetRegStruct x3 = exec_NondetReg(ctx, arg0, LAYOUT_LOOKUP(layout2, count));
  // ArgU16(zirgen/circuit/rv32im/v2/dsl/lookups.zir:29)
  NondetRegStruct x4 = exec_NondetReg(ctx, arg1, LAYOUT_LOOKUP(layout2, val));
  // LookupDelta(zirgen/circuit/rv32im/v2/dsl/lookups.zir:4)
  // ArgU16(zirgen/circuit/rv32im/v2/dsl/lookups.zir:30)
  INVOKE_EXTERN(ctx, lookupDelta, Val(16), x4._super, x3._super);
  return ArgU16Struct{.count = x3, .val = x4};
}
NondetRegStruct
exec_NondetU16Reg(ExecContext& ctx, Val arg0, BoundLayout<NondetU16RegLayout> layout1) {
  // NondetU16Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:35)
  ArgU16Struct x2 = exec_ArgU16(ctx, Val(1), arg0, LAYOUT_LOOKUP(layout1, arg));
  // NondetU16Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:36)
  Val x3 = (x2.count._super - Val(1));
  EQZ(x3, "NondetU16Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:36)");
  return x2.val;
}
U16RegStruct exec_U16Reg(ExecContext& ctx, Val arg0, BoundLayout<U16RegLayout> layout1) {
  // U16Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:42)
  NondetRegStruct x2 = exec_NondetU16Reg(ctx, arg0, LAYOUT_LOOKUP(layout1, ret));
  // U16Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:43)
  EQZ((x2._super - arg0), "U16Reg(zirgen/circuit/rv32im/v2/dsl/lookups.zir:43)");
  return U16RegStruct{._super = arg0};
}
ToBits_5_Struct exec_ToBits_5_(ExecContext& ctx, Val arg0, BoundLayout<ToBits_5_Layout> layout1) {
  // ToBits(zirgen/circuit/rv32im/v2/dsl/po2.zir:31)
  NondetRegStruct5Array x2 =
      map(Val5Array{Val(0), Val(1), Val(2), Val(3), Val(4)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val5Array::value_type x3,
                            BoundLayout<NondetRegLayout5LayoutArray::value_type> x4) {
            // Div(<preamble>:16)
            Val x5 = inv_0(Val16Array{Val(1),
                                      Val(2),
                                      Val(4),
                                      Val(8),
                                      Val(16),
                                      Val(32),
                                      Val(64),
                                      Val(128),
                                      Val(256),
                                      Val(512),
                                      Val(1024),
                                      Val(2048),
                                      Val(4096),
                                      Val(8192),
                                      Val(16384),
                                      Val(32768)}[to_size_t(x3)]);
            // Div(<preamble>:17)
            EQZ(((x5 * Val16Array{Val(1),
                                  Val(2),
                                  Val(4),
                                  Val(8),
                                  Val(16),
                                  Val(32),
                                  Val(64),
                                  Val(128),
                                  Val(256),
                                  Val(512),
                                  Val(1024),
                                  Val(2048),
                                  Val(4096),
                                  Val(8192),
                                  Val(16384),
                                  Val(32768)}[to_size_t(x3)]) -
                 Val(1)),
                "loc(callsite( Div ( <preamble> :17:22) at  ToBits ( "
                "zirgen/circuit/rv32im/v2/dsl/po2.zir :31:43)))");
            NondetRegStruct x6 =
                exec_NondetBitReg(ctx,
                                  (x5 * bitAnd(arg0,
                                               Val16Array{Val(1),
                                                          Val(2),
                                                          Val(4),
                                                          Val(8),
                                                          Val(16),
                                                          Val(32),
                                                          Val(64),
                                                          Val(128),
                                                          Val(256),
                                                          Val(512),
                                                          Val(1024),
                                                          Val(2048),
                                                          Val(4096),
                                                          Val(8192),
                                                          Val(16384),
                                                          Val(32768)}[to_size_t(x3)])),
                                  x4);
            return x6;
          }));
  return ToBits_5_Struct{._super = x2};
}
ValU32Struct exec_DynPo2(ExecContext& ctx, Val arg0, BoundLayout<DynPo2Layout> layout1) {
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:44)
  ToBits_5_Struct x2 = exec_ToBits_5_(ctx, arg0, LAYOUT_LOOKUP(layout1, low5));
  NondetRegStruct5Array x3 = x2._super;
  Val x4 = x3[0]._super;
  Val x5 = x3[1]._super;
  Val x6 = x3[2]._super;
  Val x7 = x3[3]._super;
  Val x8 = x3[4]._super;
  // FromBits(zirgen/circuit/rv32im/v2/dsl/po2.zir:35)
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:45)
  Val x9 = (((x4 + (x5 * Val(2))) + (x6 * Val(4))) + (x7 * Val(8)));
  Val x10 = (x9 + (x8 * Val(16)));
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:46)
  NondetRegStruct x11 =
      exec_NondetU16Reg(ctx, ((arg0 - x10) * Val(1950351361)), LAYOUT_LOOKUP(layout1, checkU16));
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:47)
  Val x12 = ((x11._super * Val(32)) + x10);
  EQZ((x12 - arg0), "DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:47)");
  // CondMul(zirgen/circuit/rv32im/v2/dsl/po2.zir:39)
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:48)
  Val x13 = ((x4 * Val(2)) + (Val(1) - x4));
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:49)
  Val x14 = ((Val(1) - x5) * x13);
  Val x15 = (((x5 * x13) * Val(4)) + x14);
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:50)
  Val x16 = ((Val(1) - x6) * x15);
  Val x17 = (((x6 * x15) * Val(16)) + x16);
  NondetRegStruct x18 = exec_Reg(ctx, x17, LAYOUT_LOOKUP(layout1, b3));
  Val x19 = x18._super;
  // CondMul(zirgen/circuit/rv32im/v2/dsl/po2.zir:39)
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:51)
  Val x20 = ((Val(1) - x7) * x19);
  Val x21 = (((x7 * x19) * Val(256)) + x20);
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:52)
  Val x22 = ((Val(1) - x8) * x21);
  NondetRegStruct x23 = exec_Reg(ctx, x22, LAYOUT_LOOKUP(layout1, low));
  // DynPo2(zirgen/circuit/rv32im/v2/dsl/po2.zir:53)
  NondetRegStruct x24 = exec_Reg(ctx, (x8 * x21), LAYOUT_LOOKUP(layout1, high));
  return ValU32Struct{.low = x23._super, .high = x24._super};
}
NormalizeU32Struct exec_NormalizeU32(ExecContext& ctx,
                                     DenormedValU32Struct arg0,
                                     BoundLayout<NormalizeU32Layout> layout1) {
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:42)
  Val x2 = arg0.low;
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:44)
  NondetRegStruct x3 =
      exec_NondetU16Reg(ctx, bitAnd(x2, Val(65535)), LAYOUT_LOOKUP(layout1, low16));
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:45)
  NondetRegStruct x4 = exec_NondetBitReg(
      ctx, (bitAnd(x2, Val(65536)) * Val(2013235201)), LAYOUT_LOOKUP(layout1, lowCarry));
  Val x5 = x4._super;
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:44)
  Val x6 = x3._super;
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:46)
  EQZ((x2 - ((x5 * Val(65536)) + x6)), "NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:46)");
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:48)
  Val x7 = (arg0.high + x5);
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:50)
  NondetRegStruct x8 =
      exec_NondetU16Reg(ctx, bitAnd(x7, Val(65535)), LAYOUT_LOOKUP(layout1, high16));
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:51)
  NondetRegStruct x9 = exec_NondetBitReg(
      ctx, (bitAnd(x7, Val(65536)) * Val(2013235201)), LAYOUT_LOOKUP(layout1, highCarry));
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:50)
  Val x10 = x8._super;
  // NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:52)
  Val x11 = ((x9._super * Val(65536)) + x10);
  EQZ((x7 - x11), "NormalizeU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:52)");
  return NormalizeU32Struct{._super = ValU32Struct{.low = x6, .high = x10}, .carry = x9};
}
AddrDecomposeStruct exec_AddrDecompose(ExecContext& ctx,
                                       ValU32Struct arg0,
                                       Val arg1,
                                       BoundLayout<AddrDecomposeLayout> layout2) {
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:61)
  Val x3 = arg0.low;
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:63)
  NondetRegStruct x4 = exec_NondetTwitReg(ctx, bitAnd(x3, Val(3)), LAYOUT_LOOKUP(layout2, low2));
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:65)
  Val x5 = ((Val(1) - arg1) * Val(49151));
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:61)
  Val x6 = arg0.high;
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:65)
  Val x7 = (((arg1 * Val(65535)) + x5) - x6);
  U16RegStruct x8 = exec_U16Reg(ctx, x7, LAYOUT_LOOKUP(layout2, upperDiff));
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:67)
  NondetRegStruct x9 = exec_IsZero(ctx, x6, LAYOUT_LOOKUP(layout2, _0));
  EQZ(x9._super, "AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:67)");
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:63)
  Val x10 = x4._super;
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:69)
  NondetRegStruct x11 =
      exec_NondetU16Reg(ctx, ((x3 - x10) * Val(1509949441)), LAYOUT_LOOKUP(layout2, med14));
  Val x12 = x11._super;
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:71)
  Val x13 = (((x12 * Val(4)) + x10) - x3);
  EQZ(x13, "AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:71)");
  // AddrDecompose(zirgen/circuit/rv32im/v2/dsl/u32.zir:61)
  AddrDecomposeStruct x14 = AddrDecomposeStruct{._super = ((x6 * Val(16384)) + x12), .low2 = x4};
  return x14;
}
AddrDecomposeBitsStruct exec_AddrDecomposeBits(ExecContext& ctx,
                                               ValU32Struct arg0,
                                               Val arg1,
                                               BoundLayout<AddrDecomposeBitsLayout> layout2) {
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:79)
  Val x3 = arg0.low;
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:81)
  NondetRegStruct x4 = exec_NondetBitReg(ctx, bitAnd(x3, Val(1)), LAYOUT_LOOKUP(layout2, low0));
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:82)
  NondetRegStruct x5 =
      exec_NondetBitReg(ctx, (bitAnd(x3, Val(2)) * Val(1006632961)), LAYOUT_LOOKUP(layout2, low1));
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:83)
  Val x6 = ((x5._super * Val(2)) + x4._super);
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:85)
  Val x7 = ((Val(1) - arg1) * Val(49151));
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:79)
  Val x8 = arg0.high;
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:85)
  Val x9 = (((arg1 * Val(65535)) + x7) - x8);
  U16RegStruct x10 = exec_U16Reg(ctx, x9, LAYOUT_LOOKUP(layout2, upperDiff));
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:87)
  NondetRegStruct x11 = exec_IsZero(ctx, x8, LAYOUT_LOOKUP(layout2, _0));
  EQZ(x11._super, "AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:87)");
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:89)
  NondetRegStruct x12 =
      exec_NondetU16Reg(ctx, ((x3 - x6) * Val(1509949441)), LAYOUT_LOOKUP(layout2, med14));
  Val x13 = x12._super;
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:91)
  Val x14 = (((x13 * Val(4)) + x6) - x3);
  EQZ(x14, "AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:91)");
  // AddrDecomposeBits(zirgen/circuit/rv32im/v2/dsl/u32.zir:93)
  Val x15 = ((x8 * Val(16384)) + x13);
  return AddrDecomposeBitsStruct{._super = x15, .low0 = x4, .low1 = x5, .low2 = x6, .addr = x15};
}
CmpEqualStruct exec_CmpEqual(ExecContext& ctx,
                             ValU32Struct arg0,
                             ValU32Struct arg1,
                             BoundLayout<CmpEqualLayout> layout2) {
  // CmpEqual(zirgen/circuit/rv32im/v2/dsl/u32.zir:112)
  NondetRegStruct x3 = exec_IsZero(ctx, (arg0.low - arg1.low), LAYOUT_LOOKUP(layout2, lowSame));
  // CmpEqual(zirgen/circuit/rv32im/v2/dsl/u32.zir:113)
  NondetRegStruct x4 = exec_IsZero(ctx, (arg0.high - arg1.high), LAYOUT_LOOKUP(layout2, highSame));
  // CmpEqual(zirgen/circuit/rv32im/v2/dsl/u32.zir:114)
  NondetRegStruct x5 = exec_Reg(ctx, (x3._super * x4._super), LAYOUT_LOOKUP(layout2, isEqual));
  return CmpEqualStruct{.isEqual = x5};
}
CmpLessThanUnsignedStruct exec_CmpLessThanUnsigned(ExecContext& ctx,
                                                   ValU32Struct arg0,
                                                   ValU32Struct arg1,
                                                   BoundLayout<CmpLessThanUnsignedLayout> layout2) {
  // SubU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:33)
  // CmpLessThanUnsigned(zirgen/circuit/rv32im/v2/dsl/u32.zir:119)
  Val x3 = ((arg0.low + Val(65536)) - arg1.low);
  Val x4 = ((arg0.high + Val(65535)) - arg1.high);
  NormalizeU32Struct x5 = exec_NormalizeU32(
      ctx, DenormedValU32Struct{.low = x3, .high = x4}, LAYOUT_LOOKUP(layout2, diff));
  // CmpLessThanUnsigned(zirgen/circuit/rv32im/v2/dsl/u32.zir:120)
  Val x6 = (Val(1) - x5.carry._super);
  return CmpLessThanUnsignedStruct{.isLessThan = x6};
}
NondetRegStruct
exec_GetSignU32(ExecContext& ctx, ValU32Struct arg0, BoundLayout<GetSignU32Layout> layout1) {
  // GetSignU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:124)
  Val x2 = arg0.high;
  // GetSignU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:125)
  NondetRegStruct x3 = exec_NondetBitReg(
      ctx, (bitAnd(x2, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout1, _super));
  // GetSignU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:126)
  NondetRegStruct x4 = exec_NondetU16Reg(
      ctx, (bitAnd(x2, Val(32767)) * Val(2)), LAYOUT_LOOKUP(layout1, restTimesTwo));
  // GetSignU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:127)
  Val x5 = ((x3._super * Val(32768)) + (x4._super * Val(1006632961)));
  EQZ((x2 - x5), "GetSignU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:127)");
  return x3;
}
CmpLessThanStruct exec_CmpLessThan(ExecContext& ctx,
                                   ValU32Struct arg0,
                                   ValU32Struct arg1,
                                   BoundLayout<CmpLessThanLayout> layout2) {
  // SubU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:33)
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:133)
  Val x3 = ((arg0.low + Val(65536)) - arg1.low);
  Val x4 = ((arg0.high + Val(65535)) - arg1.high);
  NormalizeU32Struct x5 = exec_NormalizeU32(
      ctx, DenormedValU32Struct{.low = x3, .high = x4}, LAYOUT_LOOKUP(layout2, diff));
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:134)
  NondetRegStruct x6 = exec_GetSignU32(ctx, arg0, LAYOUT_LOOKUP(layout2, s1));
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:135)
  NondetRegStruct x7 = exec_GetSignU32(ctx, arg1, LAYOUT_LOOKUP(layout2, s2));
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:136)
  NondetRegStruct x8 = exec_GetSignU32(ctx, x5._super, LAYOUT_LOOKUP(layout2, s3));
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:135)
  Val x9 = x7._super;
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:134)
  Val x10 = x6._super;
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:136)
  Val x11 = x8._super;
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:138)
  Val x12 = ((x10 * (Val(1) - x9)) * (Val(1) - x11));
  Val x13 = ((Val(1) - x10) * x9);
  NondetRegStruct x14 = exec_Reg(ctx, (x12 + (x13 * x11)), LAYOUT_LOOKUP(layout2, overflow));
  Val x15 = x14._super;
  // CmpLessThan(zirgen/circuit/rv32im/v2/dsl/u32.zir:140)
  Val x16 = ((x15 + x11) - ((x15 * Val(2)) * x11));
  NondetRegStruct x17 = exec_Reg(ctx, x16, LAYOUT_LOOKUP(layout2, isLessThan));
  return CmpLessThanStruct{.isLessThan = x17};
}
ToBits_16_Struct
exec_ToBits_16_(ExecContext& ctx, Val arg0, BoundLayout<ToBits_16_Layout> layout1) {
  // ToBits(zirgen/circuit/rv32im/v2/dsl/po2.zir:31)
  NondetRegStruct16Array x2 =
      map(Val16Array{Val(0),
                     Val(1),
                     Val(2),
                     Val(3),
                     Val(4),
                     Val(5),
                     Val(6),
                     Val(7),
                     Val(8),
                     Val(9),
                     Val(10),
                     Val(11),
                     Val(12),
                     Val(13),
                     Val(14),
                     Val(15)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val16Array::value_type x3,
                            BoundLayout<NondetRegLayout16LayoutArray::value_type> x4) {
            // Div(<preamble>:16)
            Val x5 = inv_0(Val16Array{Val(1),
                                      Val(2),
                                      Val(4),
                                      Val(8),
                                      Val(16),
                                      Val(32),
                                      Val(64),
                                      Val(128),
                                      Val(256),
                                      Val(512),
                                      Val(1024),
                                      Val(2048),
                                      Val(4096),
                                      Val(8192),
                                      Val(16384),
                                      Val(32768)}[to_size_t(x3)]);
            // Div(<preamble>:17)
            EQZ(((x5 * Val16Array{Val(1),
                                  Val(2),
                                  Val(4),
                                  Val(8),
                                  Val(16),
                                  Val(32),
                                  Val(64),
                                  Val(128),
                                  Val(256),
                                  Val(512),
                                  Val(1024),
                                  Val(2048),
                                  Val(4096),
                                  Val(8192),
                                  Val(16384),
                                  Val(32768)}[to_size_t(x3)]) -
                 Val(1)),
                "loc(callsite( Div ( <preamble> :17:22) at  ToBits ( "
                "zirgen/circuit/rv32im/v2/dsl/po2.zir :31:43)))");
            NondetRegStruct x6 =
                exec_NondetBitReg(ctx,
                                  (x5 * bitAnd(arg0,
                                               Val16Array{Val(1),
                                                          Val(2),
                                                          Val(4),
                                                          Val(8),
                                                          Val(16),
                                                          Val(32),
                                                          Val(64),
                                                          Val(128),
                                                          Val(256),
                                                          Val(512),
                                                          Val(1024),
                                                          Val(2048),
                                                          Val(4096),
                                                          Val(8192),
                                                          Val(16384),
                                                          Val(32768)}[to_size_t(x3)])),
                                  x4);
            return x6;
          }));
  return ToBits_16_Struct{._super = x2};
}
FromBits_16_Struct
exec_BitwiseAndU16(ExecContext& ctx, Val arg0, Val arg1, BoundLayout<BitwiseAndU16Layout> layout2) {
  // BitwiseAndU16(zirgen/circuit/rv32im/v2/dsl/u32.zir:144)
  ToBits_16_Struct x3 = exec_ToBits_16_(ctx, arg0, LAYOUT_LOOKUP(layout2, bitsX));
  NondetRegStruct16Array x4 = x3._super;
  Val x5 = x4[0]._super;
  Val x6 = x4[1]._super;
  Val x7 = x4[2]._super;
  Val x8 = x4[3]._super;
  Val x9 = x4[4]._super;
  Val x10 = x4[5]._super;
  Val x11 = x4[6]._super;
  Val x12 = x4[7]._super;
  Val x13 = x4[8]._super;
  Val x14 = x4[9]._super;
  Val x15 = x4[10]._super;
  Val x16 = x4[11]._super;
  Val x17 = x4[12]._super;
  Val x18 = x4[13]._super;
  Val x19 = x4[14]._super;
  Val x20 = x4[15]._super;
  // FromBits(zirgen/circuit/rv32im/v2/dsl/po2.zir:35)
  // BitwiseAndU16(zirgen/circuit/rv32im/v2/dsl/u32.zir:145)
  Val x21 = (((x5 + (x6 * Val(2))) + (x7 * Val(4))) + (x8 * Val(8)));
  Val x22 = (((x21 + (x9 * Val(16))) + (x10 * Val(32))) + (x11 * Val(64)));
  Val x23 = (((x22 + (x12 * Val(128))) + (x13 * Val(256))) + (x14 * Val(512)));
  Val x24 = (((x23 + (x15 * Val(1024))) + (x16 * Val(2048))) + (x17 * Val(4096)));
  Val x25 = (((x24 + (x18 * Val(8192))) + (x19 * Val(16384))) + (x20 * Val(32768)));
  EQZ((arg0 - x25), "BitwiseAndU16(zirgen/circuit/rv32im/v2/dsl/u32.zir:145)");
  // BitwiseAndU16(zirgen/circuit/rv32im/v2/dsl/u32.zir:146)
  ToBits_16_Struct x26 = exec_ToBits_16_(ctx, arg1, LAYOUT_LOOKUP(layout2, bitsY));
  NondetRegStruct16Array x27 = x26._super;
  Val x28 = x27[0]._super;
  Val x29 = x27[1]._super;
  Val x30 = x27[2]._super;
  Val x31 = x27[3]._super;
  Val x32 = x27[4]._super;
  Val x33 = x27[5]._super;
  Val x34 = x27[6]._super;
  Val x35 = x27[7]._super;
  Val x36 = x27[8]._super;
  Val x37 = x27[9]._super;
  Val x38 = x27[10]._super;
  Val x39 = x27[11]._super;
  Val x40 = x27[12]._super;
  Val x41 = x27[13]._super;
  Val x42 = x27[14]._super;
  Val x43 = x27[15]._super;
  // FromBits(zirgen/circuit/rv32im/v2/dsl/po2.zir:35)
  // BitwiseAndU16(zirgen/circuit/rv32im/v2/dsl/u32.zir:147)
  Val x44 = (((x28 + (x29 * Val(2))) + (x30 * Val(4))) + (x31 * Val(8)));
  Val x45 = (((x44 + (x32 * Val(16))) + (x33 * Val(32))) + (x34 * Val(64)));
  Val x46 = (((x45 + (x35 * Val(128))) + (x36 * Val(256))) + (x37 * Val(512)));
  Val x47 = (((x46 + (x38 * Val(1024))) + (x39 * Val(2048))) + (x40 * Val(4096)));
  Val x48 = (((x47 + (x41 * Val(8192))) + (x42 * Val(16384))) + (x43 * Val(32768)));
  EQZ((arg1 - x48), "BitwiseAndU16(zirgen/circuit/rv32im/v2/dsl/u32.zir:147)");
  // FromBits(zirgen/circuit/rv32im/v2/dsl/po2.zir:35)
  // BitwiseAndU16(zirgen/circuit/rv32im/v2/dsl/u32.zir:149)
  Val x49 = ((x5 * x28) + ((x6 * x29) * Val(2)));
  Val x50 = ((x49 + ((x7 * x30) * Val(4))) + ((x8 * x31) * Val(8)));
  Val x51 = ((x50 + ((x9 * x32) * Val(16))) + ((x10 * x33) * Val(32)));
  Val x52 = ((x51 + ((x11 * x34) * Val(64))) + ((x12 * x35) * Val(128)));
  Val x53 = ((x52 + ((x13 * x36) * Val(256))) + ((x14 * x37) * Val(512)));
  Val x54 = ((x53 + ((x15 * x38) * Val(1024))) + ((x16 * x39) * Val(2048)));
  Val x55 = ((x54 + ((x17 * x40) * Val(4096))) + ((x18 * x41) * Val(8192)));
  Val x56 = ((x55 + ((x19 * x42) * Val(16384))) + ((x20 * x43) * Val(32768)));
  return FromBits_16_Struct{._super = x56};
}
ValU32Struct exec_BitwiseAnd(ExecContext& ctx,
                             ValU32Struct arg0,
                             ValU32Struct arg1,
                             BoundLayout<BitwiseAndLayout> layout2) {
  // BitwiseAnd(zirgen/circuit/rv32im/v2/dsl/u32.zir:155)
  FromBits_16_Struct x3 = exec_BitwiseAndU16(ctx, arg0.low, arg1.low, LAYOUT_LOOKUP(layout2, _0));
  FromBits_16_Struct x4 = exec_BitwiseAndU16(ctx, arg0.high, arg1.high, LAYOUT_LOOKUP(layout2, _1));
  return ValU32Struct{.low = x3._super, .high = x4._super};
}
ValU32Struct exec_BitwiseOr(ExecContext& ctx,
                            ValU32Struct arg0,
                            ValU32Struct arg1,
                            BoundLayout<BitwiseOrLayout> layout2) {
  // BitwiseOr(zirgen/circuit/rv32im/v2/dsl/u32.zir:159)
  ValU32Struct x3 = exec_BitwiseAnd(ctx, arg0, arg1, LAYOUT_LOOKUP(layout2, andXy));
  // BitwiseOr(zirgen/circuit/rv32im/v2/dsl/u32.zir:160)
  Val x4 = ((arg0.low + arg1.low) - x3.low);
  Val x5 = ((arg0.high + arg1.high) - x3.high);
  return ValU32Struct{.low = x4, .high = x5};
}
ValU32Struct exec_BitwiseXor(ExecContext& ctx,
                             ValU32Struct arg0,
                             ValU32Struct arg1,
                             BoundLayout<BitwiseXorLayout> layout2) {
  // BitwiseXor(zirgen/circuit/rv32im/v2/dsl/u32.zir:164)
  ValU32Struct x3 = exec_BitwiseAnd(ctx, arg0, arg1, LAYOUT_LOOKUP(layout2, andXy));
  // BitwiseXor(zirgen/circuit/rv32im/v2/dsl/u32.zir:165)
  Val x4 = ((arg0.low + arg1.low) - (x3.low * Val(2)));
  Val x5 = ((arg0.high + arg1.high) - (x3.high * Val(2)));
  return ValU32Struct{.low = x4, .high = x5};
}
DecoderStruct
exec_Decoder(ExecContext& ctx, ValU32Struct arg0, BoundLayout<DecoderLayout> layout1) {
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:9)
  Val x2 = arg0.high;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:15)
  NondetRegStruct x3 = exec_NondetBitReg(
      ctx, (bitAnd(x2, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout1, _f7_6));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:16)
  NondetRegStruct x4 = exec_NondetTwitReg(
      ctx, (bitAnd(x2, Val(24576)) * Val(2013020161)), LAYOUT_LOOKUP(layout1, _f7_45));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:17)
  NondetRegStruct x5 = exec_NondetTwitReg(
      ctx, (bitAnd(x2, Val(6144)) * Val(2012282881)), LAYOUT_LOOKUP(layout1, _f7_23));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:18)
  NondetRegStruct x6 = exec_NondetTwitReg(
      ctx, (bitAnd(x2, Val(1536)) * Val(2009333761)), LAYOUT_LOOKUP(layout1, _f7_01));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:19)
  NondetRegStruct x7 = exec_NondetTwitReg(
      ctx, (bitAnd(x2, Val(384)) * Val(1997537281)), LAYOUT_LOOKUP(layout1, _rs2_34));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:20)
  NondetRegStruct x8 = exec_NondetTwitReg(
      ctx, (bitAnd(x2, Val(96)) * Val(1950351361)), LAYOUT_LOOKUP(layout1, _rs2_12));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:21)
  NondetRegStruct x9 = exec_NondetBitReg(
      ctx, (bitAnd(x2, Val(16)) * Val(1887436801)), LAYOUT_LOOKUP(layout1, _rs2_0));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:22)
  NondetRegStruct x10 = exec_NondetTwitReg(
      ctx, (bitAnd(x2, Val(12)) * Val(1509949441)), LAYOUT_LOOKUP(layout1, _rs1_34));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:23)
  NondetRegStruct x11 =
      exec_NondetTwitReg(ctx, bitAnd(x2, Val(3)), LAYOUT_LOOKUP(layout1, _rs1_12));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:9)
  Val x12 = arg0.low;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:24)
  NondetRegStruct x13 = exec_NondetBitReg(
      ctx, (bitAnd(x12, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout1, _rs1_0));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:25)
  NondetRegStruct x14 = exec_NondetBitReg(
      ctx, (bitAnd(x12, Val(16384)) * Val(2013143041)), LAYOUT_LOOKUP(layout1, _f3_2));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:26)
  NondetRegStruct x15 = exec_NondetTwitReg(
      ctx, (bitAnd(x12, Val(12288)) * Val(2012774401)), LAYOUT_LOOKUP(layout1, _f3_01));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:27)
  NondetRegStruct x16 = exec_NondetTwitReg(
      ctx, (bitAnd(x12, Val(3072)) * Val(2011299841)), LAYOUT_LOOKUP(layout1, _rd_34));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:28)
  NondetRegStruct x17 = exec_NondetTwitReg(
      ctx, (bitAnd(x12, Val(768)) * Val(2005401601)), LAYOUT_LOOKUP(layout1, _rd_12));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:29)
  NondetRegStruct x18 = exec_NondetTwitReg(
      ctx, (bitAnd(x12, Val(128)) * Val(1997537281)), LAYOUT_LOOKUP(layout1, _rd_0));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:34)
  NondetRegStruct x19 = exec_NondetReg(ctx, bitAnd(x12, Val(127)), LAYOUT_LOOKUP(layout1, opcode));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:15)
  Val x20 = x3._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:16)
  Val x21 = x4._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:17)
  Val x22 = x5._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:38)
  Val x23 = (((x20 * Val(32768)) + (x21 * Val(8192))) + (x22 * Val(2048)));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:18)
  Val x24 = x6._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:19)
  Val x25 = x7._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:20)
  Val x26 = x8._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:41)
  Val x27 = (((x23 + (x24 * Val(512))) + (x25 * Val(128))) + (x26 * Val(32)));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:21)
  Val x28 = x9._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:22)
  Val x29 = x10._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:44)
  Val x30 = (x29 * Val(4));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:23)
  Val x31 = x11._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:44)
  Val x32 = (((x27 + (x28 * Val(16))) + x30) + x31);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:37)
  EQZ((x2 - x32), "Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:37)");
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:24)
  Val x33 = x13._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:46)
  Val x34 = (x33 * Val(32768));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:25)
  Val x35 = x14._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:26)
  Val x36 = x15._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:47)
  Val x37 = ((x34 + (x35 * Val(16384))) + (x36 * Val(4096)));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:27)
  Val x38 = x16._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:28)
  Val x39 = x17._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:29)
  Val x40 = x18._super;
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:50)
  Val x41 = (((x37 + (x38 * Val(1024))) + (x39 * Val(256))) + (x40 * Val(128)));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:46)
  EQZ((x12 - (x41 + x19._super)), "Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:46)");
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:55)
  Val x42 = (((x29 * Val(8)) + (x31 * Val(2))) + x33);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:56)
  Val x43 = (x25 * Val(8));
  Val x44 = (x26 * Val(2));
  Val x45 = ((x43 + x44) + x28);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:57)
  Val x46 = (x38 * Val(8));
  Val x47 = (x39 * Val(2));
  Val x48 = ((x46 + x47) + x40);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:58)
  Val x49 = (((x21 * Val(16)) + (x22 * Val(4))) + x24);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:59)
  Val x50 = ((x20 * Val(64)) + x49);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:60)
  Val x51 = ((x35 * Val(4)) + x36);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:66)
  Val x52 = (x20 * Val(61440));
  Val x53 = (x52 + (x50 * Val(32)));
  Val x54 = (x20 * Val(65535));
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:68)
  Val x55 = (x49 * Val(32));
  Val x56 = (((x52 + (x40 * Val(2048))) + x55) + x46);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:71)
  Val x57 = (((x34 + (x51 * Val(4096))) + (x28 * Val(2048))) + x55);
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:72)
  Val x58 = (((x20 * Val(65520)) + x30) + x31);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // Decoder(zirgen/circuit/rv32im/v2/dsl/decode.zir:70)
  ValU32Struct x59 = ValU32Struct{.low = ((x57 + x43) + x44), .high = x58};
  return DecoderStruct{.opcode = x19,
                       .rs1 = x42,
                       .rs2 = x45,
                       .rd = x48,
                       .func7 = x50,
                       .func3 = x51,
                       .immI = ValU32Struct{.low = (x53 + x45), .high = x54},
                       .immS = ValU32Struct{.low = (x53 + x48), .high = x54},
                       .immB = ValU32Struct{.low = (x56 + x47), .high = x54},
                       .immU = ValU32Struct{.low = x37, .high = x2},
                       .immJ = x59};
}
MemoryArgStruct exec_MemoryArg(ExecContext& ctx,
                               Val arg0,
                               Val arg1,
                               Val arg2,
                               ValU32Struct arg3,
                               BoundLayout<MemoryArgLayout> layout4) {
  // MemoryArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:25)
  NondetRegStruct x5 = exec_NondetReg(ctx, arg0, LAYOUT_LOOKUP(layout4, count));
  // MemoryArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:26)
  NondetRegStruct x6 = exec_NondetReg(ctx, arg1, LAYOUT_LOOKUP(layout4, addr));
  // MemoryArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:27)
  NondetRegStruct x7 = exec_NondetReg(ctx, arg2, LAYOUT_LOOKUP(layout4, cycle));
  // MemoryArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:28)
  NondetRegStruct x8 = exec_NondetReg(ctx, arg3.low, LAYOUT_LOOKUP(layout4, dataLow));
  // MemoryArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:29)
  NondetRegStruct x9 = exec_NondetReg(ctx, arg3.high, LAYOUT_LOOKUP(layout4, dataHigh));
  // MemoryDelta(zirgen/circuit/rv32im/v2/dsl/mem.zir:21)
  // MemoryArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:30)
  INVOKE_EXTERN(ctx, memoryDelta, x6._super, x7._super, x8._super, x9._super, x5._super);
  return MemoryArgStruct{.count = x5, .addr = x6, .cycle = x7, .dataLow = x8, .dataHigh = x9};
}
CycleArgStruct
exec_CycleArg(ExecContext& ctx, Val arg0, Val arg1, BoundLayout<CycleArgLayout> layout2) {
  // CycleArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:54)
  NondetRegStruct x3 = exec_NondetReg(ctx, arg0, LAYOUT_LOOKUP(layout2, count));
  // CycleArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:55)
  NondetRegStruct x4 = exec_NondetReg(ctx, arg1, LAYOUT_LOOKUP(layout2, cycle));
  // LookupDelta(zirgen/circuit/rv32im/v2/dsl/lookups.zir:4)
  // CycleArg(zirgen/circuit/rv32im/v2/dsl/mem.zir:56)
  INVOKE_EXTERN(ctx, lookupDelta, Val(0), x4._super, x3._super);
  return CycleArgStruct{.count = x3, .cycle = x4};
}
IsCycleStruct exec_IsCycle(ExecContext& ctx, Val arg0, BoundLayout<IsCycleLayout> layout1) {
  // IsCycle(zirgen/circuit/rv32im/v2/dsl/mem.zir:60)
  CycleArgStruct x2 = exec_CycleArg(ctx, Val(1), arg0, LAYOUT_LOOKUP(layout1, arg));
  // IsCycle(zirgen/circuit/rv32im/v2/dsl/mem.zir:61)
  Val x3 = (x2.count._super - Val(1));
  EQZ(x3, "IsCycle(zirgen/circuit/rv32im/v2/dsl/mem.zir:61)");
  // IsCycle(zirgen/circuit/rv32im/v2/dsl/mem.zir:62)
  Val x4 = (x2.cycle._super - arg0);
  EQZ(x4, "IsCycle(zirgen/circuit/rv32im/v2/dsl/mem.zir:62)");
  return IsCycleStruct{};
}
MemoryIOStruct exec_MemoryIO(ExecContext& ctx,
                             NondetRegStruct arg0,
                             BoundLayout<NondetRegLayout> arg1,
                             Val arg2,
                             BoundLayout<MemoryIOLayout> layout3) {
  // GetMemoryTxn(zirgen/circuit/rv32im/v2/dsl/mem.zir:51)
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:66)
  auto [x4, x5, x6, x7, x8] = INVOKE_EXTERN(ctx, getMemoryTxn, arg2);
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:67)
  MemoryArgStruct x9 = exec_MemoryArg(ctx,
                                      Val(2013265920),
                                      arg2,
                                      x4,
                                      ValU32Struct{.low = x5, .high = x6},
                                      LAYOUT_LOOKUP(layout3, oldTxn));
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:65)
  Val x10 = arg0._super;
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:68)
  MemoryArgStruct x11 = exec_MemoryArg(
      ctx, Val(1), arg2, x10, ValU32Struct{.low = x7, .high = x8}, LAYOUT_LOOKUP(layout3, newTxn));
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:69)
  Val x12 = (x9.count._super - Val(2013265920));
  EQZ(x12, "MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:69)");
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:70)
  Val x13 = (x11.count._super - Val(1));
  EQZ(x13, "MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:70)");
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:72)
  Val x14 = (x11.cycle._super - x10);
  EQZ(x14, "MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:72)");
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:68)
  Val x15 = x11.addr._super;
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:74)
  Val x16 = (x9.addr._super - x15);
  EQZ(x16, "MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:74)");
  // MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:75)
  EQZ((x15 - arg2), "MemoryIO(zirgen/circuit/rv32im/v2/dsl/mem.zir:75)");
  return MemoryIOStruct{.oldTxn = x9, .newTxn = x11};
}
IsForwardStruct
exec_IsForward(ExecContext& ctx, MemoryIOStruct arg0, BoundLayout<IsForwardLayout> layout1) {
  // IsForward(zirgen/circuit/rv32im/v2/dsl/mem.zir:83)
  Val x2 = arg0.newTxn.cycle._super;
  Val x3 = arg0.oldTxn.cycle._super;
  // IsForward(zirgen/circuit/rv32im/v2/dsl/mem.zir:84)
  IsCycleStruct x4 = exec_IsCycle(ctx, (x2 - x3), LAYOUT_LOOKUP(layout1, _0));
  return IsForwardStruct{};
}
GetDataStruct exec_MemoryRead(ExecContext& ctx,
                              NondetRegStruct arg0,
                              BoundLayout<NondetRegLayout> arg1,
                              Val arg2,
                              BoundLayout<MemoryReadLayout> layout3) {
  // MemoryRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:89)
  MemoryIOStruct x4 = exec_MemoryIO(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, io));
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:78)
  // MemoryRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:90)
  MemoryArgStruct x5 = x4.oldTxn;
  MemoryArgStruct x6 = x4.newTxn;
  Val x7 = x6.dataLow._super;
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:79)
  Val x8 = (x5.dataLow._super - x7);
  EQZ(x8,
      "loc(callsite( IsRead ( zirgen/circuit/rv32im/v2/dsl/mem.zir :79:22) at  MemoryRead ( "
      "zirgen/circuit/rv32im/v2/dsl/mem.zir :90:10)))");
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:78)
  Val x9 = x6.dataHigh._super;
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:80)
  Val x10 = (x5.dataHigh._super - x9);
  EQZ(x10,
      "loc(callsite( IsRead ( zirgen/circuit/rv32im/v2/dsl/mem.zir :80:23) at  MemoryRead ( "
      "zirgen/circuit/rv32im/v2/dsl/mem.zir :90:10)))");
  // MemoryRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:91)
  IsForwardStruct x11 = exec_IsForward(ctx, x4, LAYOUT_LOOKUP(layout3, _0));
  return GetDataStruct{
      ._super = ValU32Struct{.low = x7, .high = x9}, .diffLow = Val(0), .diffHigh = Val(1)};
}
MemoryWriteStruct exec_MemoryWrite(ExecContext& ctx,
                                   NondetRegStruct arg0,
                                   BoundLayout<NondetRegLayout> arg1,
                                   Val arg2,
                                   ValU32Struct arg3,
                                   BoundLayout<MemoryWriteLayout> layout4) {
  // MemoryWrite(zirgen/circuit/rv32im/v2/dsl/mem.zir:97)
  MemoryIOStruct x5 = exec_MemoryIO(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout4, io));
  // MemoryWrite(zirgen/circuit/rv32im/v2/dsl/mem.zir:98)
  IsForwardStruct x6 = exec_IsForward(ctx, x5, LAYOUT_LOOKUP(layout4, _0));
  // MemoryWrite(zirgen/circuit/rv32im/v2/dsl/mem.zir:97)
  MemoryArgStruct x7 = x5.newTxn;
  // MemoryWrite(zirgen/circuit/rv32im/v2/dsl/mem.zir:99)
  Val x8 = (x7.dataLow._super - arg3.low);
  EQZ(x8, "MemoryWrite(zirgen/circuit/rv32im/v2/dsl/mem.zir:99)");
  // MemoryWrite(zirgen/circuit/rv32im/v2/dsl/mem.zir:100)
  Val x9 = (x7.dataHigh._super - arg3.high);
  EQZ(x9, "MemoryWrite(zirgen/circuit/rv32im/v2/dsl/mem.zir:100)");
  return MemoryWriteStruct{};
}
MemoryWriteUnconstrainedStruct
exec_MemoryWriteUnconstrained(ExecContext& ctx,
                              NondetRegStruct arg0,
                              BoundLayout<NondetRegLayout> arg1,
                              Val arg2,
                              BoundLayout<MemoryWriteUnconstrainedLayout> layout3) {
  // MemoryWriteUnconstrained(zirgen/circuit/rv32im/v2/dsl/mem.zir:105)
  MemoryIOStruct x4 = exec_MemoryIO(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, io));
  // MemoryWriteUnconstrained(zirgen/circuit/rv32im/v2/dsl/mem.zir:106)
  IsForwardStruct x5 = exec_IsForward(ctx, x4, LAYOUT_LOOKUP(layout3, _0));
  return MemoryWriteUnconstrainedStruct{};
}
GetDataStruct exec_MemoryPageIn(ExecContext& ctx,
                                NondetRegStruct arg0,
                                BoundLayout<NondetRegLayout> arg1,
                                Val arg2,
                                BoundLayout<MemoryPageInLayout> layout3) {
  // MemoryPageIn(zirgen/circuit/rv32im/v2/dsl/mem.zir:112)
  MemoryIOStruct x4 = exec_MemoryIO(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, io));
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:78)
  // MemoryPageIn(zirgen/circuit/rv32im/v2/dsl/mem.zir:113)
  MemoryArgStruct x5 = x4.oldTxn;
  MemoryArgStruct x6 = x4.newTxn;
  Val x7 = x6.dataLow._super;
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:79)
  Val x8 = (x5.dataLow._super - x7);
  EQZ(x8,
      "loc(callsite( IsRead ( zirgen/circuit/rv32im/v2/dsl/mem.zir :79:22) at  MemoryPageIn ( "
      "zirgen/circuit/rv32im/v2/dsl/mem.zir :113:10)))");
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:78)
  Val x9 = x6.dataHigh._super;
  // IsRead(zirgen/circuit/rv32im/v2/dsl/mem.zir:80)
  Val x10 = (x5.dataHigh._super - x9);
  EQZ(x10,
      "loc(callsite( IsRead ( zirgen/circuit/rv32im/v2/dsl/mem.zir :80:23) at  MemoryPageIn ( "
      "zirgen/circuit/rv32im/v2/dsl/mem.zir :113:10)))");
  // MemoryPageIn(zirgen/circuit/rv32im/v2/dsl/mem.zir:114)
  Val x11 = (x6.cycle._super - x5.cycle._super);
  return GetDataStruct{
      ._super = ValU32Struct{.low = x7, .high = x9}, .diffLow = Val(0), .diffHigh = x11};
}
GetDataStruct exec_MemoryPageOut(ExecContext& ctx,
                                 NondetRegStruct arg0,
                                 BoundLayout<NondetRegLayout> arg1,
                                 Val arg2,
                                 BoundLayout<MemoryPageOutLayout> layout3) {
  // MemoryPageOut(zirgen/circuit/rv32im/v2/dsl/mem.zir:120)
  MemoryIOStruct x4 = exec_MemoryIO(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, io));
  // MemoryPageOut(zirgen/circuit/rv32im/v2/dsl/mem.zir:121)
  IsForwardStruct x5 = exec_IsForward(ctx, x4, LAYOUT_LOOKUP(layout3, _0));
  // MemoryPageOut(zirgen/circuit/rv32im/v2/dsl/mem.zir:120)
  MemoryArgStruct x6 = x4.oldTxn;
  MemoryArgStruct x7 = x4.newTxn;
  Val x8 = x6.dataLow._super;
  // MemoryPageOut(zirgen/circuit/rv32im/v2/dsl/mem.zir:122)
  Val x9 = (x7.dataLow._super - x8);
  // MemoryPageOut(zirgen/circuit/rv32im/v2/dsl/mem.zir:120)
  Val x10 = x6.dataHigh._super;
  // MemoryPageOut(zirgen/circuit/rv32im/v2/dsl/mem.zir:123)
  Val x11 = (x7.dataHigh._super - x10);
  return GetDataStruct{
      ._super = ValU32Struct{.low = x8, .high = x10}, .diffLow = x9, .diffHigh = x11};
}
OneHot_3_Struct exec_OneHot_3_(ExecContext& ctx, Val arg0, BoundLayout<OneHot_3_Layout> layout1) {
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:7)
  NondetRegStruct3Array x2 =
      map(Val3Array{Val(0), Val(1), Val(2)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val3Array::value_type x3,
                            BoundLayout<NondetRegLayout3LayoutArray::value_type> x4) {
            NondetRegStruct x5 = exec_NondetBitReg(ctx, isz((x3 - arg0)), x4);
            return x5;
          }));
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)
  Val x6 = x2[1]._super;
  Val x7 = (x2[0]._super + x6);
  Val x8 = x2[2]._super;
  EQZ(((x7 + x8) - Val(1)), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)");
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)
  EQZ(((x6 + (x8 * Val(2))) - arg0), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)");
  return OneHot_3_Struct{._super = x2};
}
GetDataStruct exec_MemoryGet(ExecContext& ctx,
                             NondetRegStruct arg0,
                             BoundLayout<NondetRegLayout> arg1,
                             Val arg2,
                             OneHot_3_Struct arg3,
                             BoundLayout<MemoryGetLayout> layout4) {
  // MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:128)
  BoundLayout<MemoryGet_SuperLayout> x5 = LAYOUT_LOOKUP(layout4, _super);
  // MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:127)
  NondetRegStruct3Array x6 = arg3._super;
  // MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:128)
  BoundLayout<MemoryGetArm1Layout> x7 = LAYOUT_LOOKUP(x5, arm1);
  BoundLayout<Reg> x8 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra0), count), _super);
  GetDataStruct x9;
  if (to_size_t(x6[0]._super)) {
    // MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:129)
    GetDataStruct x10 = exec_MemoryRead(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x5, arm0));
    x9 = x10;
  } else if (to_size_t(x6[1]._super)) {
    // MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:130)
    GetDataStruct x11 = exec_MemoryPageIn(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x7, _super));
    // MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:128)
    STORE(x8, Val(0));
    EQZ(LOAD(x8, 0), "MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:128)");
    x9 = x11;
  } else if (to_size_t(x6[2]._super)) {
    // MemoryGet(zirgen/circuit/rv32im/v2/dsl/mem.zir:131)
    GetDataStruct x12 = exec_MemoryPageOut(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x5, arm2));
    x9 = x12;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  return x9;
}
OneHot_8_Struct exec_OneHot_8_(ExecContext& ctx, Val arg0, BoundLayout<OneHot_8_Layout> layout1) {
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:7)
  NondetRegStruct8Array x2 =
      map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val8Array::value_type x3,
                            BoundLayout<NondetRegLayout8LayoutArray::value_type> x4) {
            NondetRegStruct x5 = exec_NondetBitReg(ctx, isz((x3 - arg0)), x4);
            return x5;
          }));
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)
  Val x6 = x2[1]._super;
  Val x7 = (x2[0]._super + x6);
  Val x8 = x2[2]._super;
  Val x9 = x2[3]._super;
  Val x10 = x2[4]._super;
  Val x11 = (((x7 + x8) + x9) + x10);
  Val x12 = x2[5]._super;
  Val x13 = x2[6]._super;
  Val x14 = x2[7]._super;
  Val x15 = (((x11 + x12) + x13) + x14);
  EQZ((x15 - Val(1)), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)");
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)
  Val x16 = (((x6 + (x8 * Val(2))) + (x9 * Val(3))) + (x10 * Val(4)));
  Val x17 = (((x16 + (x12 * Val(5))) + (x13 * Val(6))) + (x14 * Val(7)));
  EQZ((x17 - arg0), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)");
  return OneHot_8_Struct{._super = x2, .bits = x2};
}
InstInputStruct exec_InstInput(ExecContext& ctx,
                               Val arg0,
                               Val arg1,
                               Val arg2,
                               ValU32Struct arg3,
                               Val arg4,
                               Val arg5,
                               BoundLayout<InstInputLayout> layout6) {
  // InstInput(zirgen/circuit/rv32im/v2/dsl/inst.zir:15)
  OneHot_8_Struct x7 = exec_OneHot_8_(ctx, arg2, LAYOUT_LOOKUP(layout6, minorOnehot));
  return InstInputStruct{.pcU32 = arg3, .state = arg4, .mode = arg5, .minorOnehot = x7};
}
DecoderStruct exec_DecodeInst(ExecContext& ctx,
                              NondetRegStruct arg0,
                              BoundLayout<NondetRegLayout> arg1,
                              InstInputStruct arg2,
                              BoundLayout<DecodeInstLayout> layout3) {
  // DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:18)
  Val x4 = arg0._super;
  // GetDiffCount(zirgen/circuit/rv32im/v2/dsl/mem.zir:22)
  // DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:20)
  Val x5 = INVOKE_EXTERN(ctx, getDiffCount, x4);
  CycleArgStruct x6 = exec_CycleArg(ctx, neg_0(x5), x4, LAYOUT_LOOKUP(layout3, arg));
  // DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:22)
  Val x7 = (x6.cycle._super - x4);
  EQZ(x7, "DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:22)");
  // DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:24)
  AddrDecomposeStruct x8 =
      exec_AddrDecompose(ctx, arg2.pcU32, arg2.mode, LAYOUT_LOOKUP(layout3, pcAddr));
  // DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:26)
  EQZ(x8.low2._super, "DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:26)");
  // DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:28)
  GetDataStruct x9 = exec_MemoryRead(ctx, arg0, arg1, x8._super, LAYOUT_LOOKUP(layout3, loadInst));
  // DecodeInst(zirgen/circuit/rv32im/v2/dsl/inst.zir:30)
  DecoderStruct x10 = exec_Decoder(ctx, x9._super, LAYOUT_LOOKUP(layout3, _super));
  return x10;
}
GetDataStruct exec_ReadReg(ExecContext& ctx,
                           NondetRegStruct arg0,
                           BoundLayout<NondetRegLayout> arg1,
                           InstInputStruct arg2,
                           Val arg3,
                           BoundLayout<ReadRegLayout> layout4) {
  // ReadReg(zirgen/circuit/rv32im/v2/dsl/inst.zir:33)
  Val x5 = arg2.mode;
  // ReadReg(zirgen/circuit/rv32im/v2/dsl/inst.zir:34)
  Val x6 = ((Val(1) - x5) * Val(1073725472));
  Val x7 = (((x5 * Val(1073725440)) + x6) + arg3);
  NondetRegStruct x8 = exec_Reg(ctx, x7, LAYOUT_LOOKUP(layout4, addr));
  // ReadReg(zirgen/circuit/rv32im/v2/dsl/inst.zir:35)
  GetDataStruct x9 = exec_MemoryRead(ctx, arg0, arg1, x8._super, LAYOUT_LOOKUP(layout4, _super));
  return x9;
}
WriteRdStruct exec_WriteRd(ExecContext& ctx,
                           NondetRegStruct arg0,
                           BoundLayout<NondetRegLayout> arg1,
                           InstInputStruct arg2,
                           DecoderStruct arg3,
                           Val arg4,
                           ValU32Struct arg5,
                           BoundLayout<WriteRdLayout> layout6) {
  // WriteRd(zirgen/circuit/rv32im/v2/dsl/inst.zir:38)
  Val x7 = arg3.rd;
  // WriteRd(zirgen/circuit/rv32im/v2/dsl/inst.zir:39)
  NondetRegStruct x8 = exec_IsZero(ctx, x7, LAYOUT_LOOKUP(layout6, isRd0));
  // WriteRd(zirgen/circuit/rv32im/v2/dsl/inst.zir:40)
  Val x9 = ((Val(1) - x8._super) * arg4);
  // WriteRd(zirgen/circuit/rv32im/v2/dsl/inst.zir:38)
  Val x10 = arg2.mode;
  // WriteRd(zirgen/circuit/rv32im/v2/dsl/inst.zir:42)
  Val x11 = ((Val(1) - x10) * Val(1073725472));
  Val x12 = ((Val(1) - x9) * Val(64));
  Val x13 = (((x10 * Val(1073725440)) + x11) + x12);
  NondetRegStruct x14 = exec_Reg(ctx, (x13 + (x9 * x7)), LAYOUT_LOOKUP(layout6, writeAddr));
  // WriteRd(zirgen/circuit/rv32im/v2/dsl/inst.zir:43)
  MemoryWriteStruct x15 =
      exec_MemoryWrite(ctx, arg0, arg1, x14._super, arg5, LAYOUT_LOOKUP(layout6, _0));
  return WriteRdStruct{};
}
ExpandU32Struct exec_ExpandU32(ExecContext& ctx,
                               ValU32Struct arg0,
                               Val arg1,
                               BoundLayout<ExpandU32Layout> layout2) {
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:49)
  Val x3 = arg0.low;
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:50)
  NondetRegStruct x4 = exec_NondetU8Reg(ctx, bitAnd(x3, Val(255)), LAYOUT_LOOKUP(layout2, b0));
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:51)
  NondetRegStruct x5 =
      exec_NondetU8Reg(ctx, (bitAnd(x3, Val(65280)) * Val(2005401601)), LAYOUT_LOOKUP(layout2, b1));
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:49)
  Val x6 = arg0.high;
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:52)
  NondetRegStruct x7 = exec_NondetU8Reg(ctx, bitAnd(x6, Val(255)), LAYOUT_LOOKUP(layout2, b2));
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:53)
  NondetRegStruct x8 =
      exec_NondetU8Reg(ctx, (bitAnd(x6, Val(65280)) * Val(2005401601)), LAYOUT_LOOKUP(layout2, b3));
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:59)
  NondetRegStruct x9 = exec_NondetU8Reg(
      ctx, (bitAnd(x6, Val(32512)) * Val(1997537281)), LAYOUT_LOOKUP(layout2, b3Top7times2));
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:60)
  NondetRegStruct x10 = exec_NondetBitReg(
      ctx, (bitAnd(x6, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout2, topBit));
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:62)
  Val x11 = (x4._super + (x5._super * Val(256)));
  EQZ((x3 - x11), "ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:62)");
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:59)
  Val x12 = x9._super;
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:60)
  Val x13 = x10._super;
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:63)
  Val x14 = ((x7._super + (x12 * Val(128))) + (x13 * Val(32768)));
  EQZ((x6 - x14), "ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:63)");
  // ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:67)
  Val x15 = (x8._super - ((x12 * Val(1006632961)) + (x13 * Val(128))));
  EQZ(x15, "ExpandU32(zirgen/circuit/rv32im/v2/dsl/mult.zir:67)");
  return ExpandU32Struct{.b0 = x4, .b1 = x5, .b2 = x7, .b3 = x8, .neg = (x13 * arg1)};
}
SplitTotalStruct
exec_SplitTotal(ExecContext& ctx, Val arg0, BoundLayout<SplitTotalLayout> layout1) {
  // SplitTotal(zirgen/circuit/rv32im/v2/dsl/mult.zir:97)
  NondetRegStruct x2 =
      exec_NondetU16Reg(ctx, bitAnd(arg0, Val(65535)), LAYOUT_LOOKUP(layout1, out));
  // SplitTotal(zirgen/circuit/rv32im/v2/dsl/mult.zir:98)
  NondetRegStruct x3 = exec_NondetU8Reg(
      ctx, (bitAnd(arg0, Val(16711680)) * Val(2013235201)), LAYOUT_LOOKUP(layout1, carryByte));
  // SplitTotal(zirgen/circuit/rv32im/v2/dsl/mult.zir:99)
  NondetFakeTwitRegStruct x4 = exec_NondetFakeTwitReg(
      ctx, (bitAnd(arg0, Val(251658240)) * Val(2013265801)), LAYOUT_LOOKUP(layout1, carryExtra));
  Val x5 = x4._super;
  // SplitTotal(zirgen/circuit/rv32im/v2/dsl/mult.zir:98)
  Val x6 = x3._super;
  // SplitTotal(zirgen/circuit/rv32im/v2/dsl/mult.zir:100)
  Val x7 = (((x5 * Val(16777216)) + (x6 * Val(65536))) + x2._super);
  EQZ((arg0 - x7), "SplitTotal(zirgen/circuit/rv32im/v2/dsl/mult.zir:100)");
  return SplitTotalStruct{.out = x2, .carry = ((x5 * Val(256)) + x6)};
}
MultiplyAccumulateStruct exec_MultiplyAccumulate(ExecContext& ctx,
                                                 ValU32Struct arg0,
                                                 ValU32Struct arg1,
                                                 ValU32Struct arg2,
                                                 MultiplySettingsStruct arg3,
                                                 BoundLayout<MultiplyAccumulateLayout> layout4) {
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:115)
  ExpandU32Struct x5 = exec_ExpandU32(ctx, arg0, arg3.aSigned, LAYOUT_LOOKUP(layout4, ax));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:116)
  ExpandU32Struct x6 = exec_ExpandU32(ctx, arg1, arg3.bSigned, LAYOUT_LOOKUP(layout4, bx));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:113)
  Val x7 = arg2.high;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:118)
  NondetRegStruct x8 = exec_NondetBitReg(
      ctx, (bitAnd(x7, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout4, cSign));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:119)
  NondetRegStruct x9 = exec_NondetU16Reg(
      ctx, (bitAnd(x7, Val(32767)) * Val(2)), LAYOUT_LOOKUP(layout4, cRestTimes2));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:118)
  Val x10 = x8._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:120)
  Val x11 = ((x10 * Val(32768)) + (x9._super * Val(1006632961)));
  EQZ((x7 - x11), "MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:120)");
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:115)
  Val x12 = x5.b0._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:116)
  Val x13 = x6.b0._super;
  Val x14 = x6.b1._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:115)
  Val x15 = x5.b1._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:125)
  Val x16 = (((x12 * x14) + (x15 * x13)) * Val(256));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:124)
  Val x17 = ((arg2.low + (x12 * x13)) + x16);
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:122)
  SplitTotalStruct x18 = exec_SplitTotal(ctx, x17, LAYOUT_LOOKUP(layout4, s0));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:116)
  Val x19 = x6.b2._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:131)
  Val x20 = (((x7 + x18.carry) + (x12 * x19)) + (x15 * x14));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:115)
  Val x21 = x5.b2._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:116)
  Val x22 = x6.b3._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:132)
  Val x23 = (((x12 * x22) + (x15 * x19)) + (x21 * x14));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:115)
  Val x24 = x5.b3._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:131)
  Val x25 = ((x20 + (x21 * x13)) + ((x23 + (x24 * x13)) * Val(256)));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:128)
  SplitTotalStruct x26 = exec_SplitTotal(ctx, x25, LAYOUT_LOOKUP(layout4, s1));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:138)
  Val x27 = ((x10 * Val(65535)) * arg3.cSigned);
  Val x28 = ((x26.carry + x27) + Val(131072));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:116)
  Val x29 = x6.neg;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:115)
  Val x30 = x5.neg;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:139)
  Val x31 = ((x28 - ((x12 + (x15 * Val(256))) * x29)) - ((x13 + (x14 * Val(256))) * x30));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:140)
  Val x32 = (((x31 + (x15 * x22)) + (x21 * x19)) + (x24 * x14));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:141)
  Val x33 = (((x21 * x22) + (x24 * x19)) * Val(256));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:136)
  SplitTotalStruct x34 = exec_SplitTotal(ctx, (x32 + x33), LAYOUT_LOOKUP(layout4, s2));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:147)
  Val x35 = ((x34.carry + x27) + Val(131070));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:148)
  Val x36 = ((x35 - ((x21 + (x24 * Val(256))) * x29)) - ((x19 + (x22 * Val(256))) * x30));
  Val x37 = (x36 + (x24 * x22));
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:150)
  NondetRegStruct x38 =
      exec_NondetU16Reg(ctx, bitAnd(x37, Val(65535)), LAYOUT_LOOKUP(layout4, s3Out));
  Val x39 = x38._super;
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:151)
  FakeTwitRegStruct x40 =
      exec_FakeTwitReg(ctx, ((x37 - x39) * Val(2013235201)), LAYOUT_LOOKUP(layout4, s3Carry));
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:152)
  ValU32Struct x41 = ValU32Struct{.low = x18.out._super, .high = x26.out._super};
  // MultiplyAccumulate(zirgen/circuit/rv32im/v2/dsl/mult.zir:153)
  ValU32Struct x42 = ValU32Struct{.low = x34.out._super, .high = x39};
  return MultiplyAccumulateStruct{.outLow = x41, .outHigh = x42};
}
DivInputStruct exec_DivInput(ExecContext& ctx,
                             NondetRegStruct arg0,
                             BoundLayout<NondetRegLayout> arg1,
                             InstInputStruct arg2,
                             BoundLayout<DivInputLayout> layout3) {
  // DivInput(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:8)
  EQZ((arg2.state - Val(32)), "DivInput(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:8)");
  // DivInput(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:10)
  DecoderStruct x4 = exec_DecodeInst(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, decoded));
  // DivInput(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:11)
  GetDataStruct x5 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs1, LAYOUT_LOOKUP(layout3, rs1));
  // DivInput(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:12)
  GetDataStruct x6 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs2, LAYOUT_LOOKUP(layout3, rs2));
  return DivInputStruct{._super = arg2, .ii = arg2, .decoded = x4, .rs1 = x5, .rs2 = x6};
}
DoDivStruct exec_DoDiv(ExecContext& ctx,
                       ValU32Struct arg0,
                       ValU32Struct arg1,
                       Val arg2,
                       Val arg3,
                       BoundLayout<DoDivLayout> layout4) {
  // Divide(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:43)
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:47)
  Val x5 = arg0.low;
  Val x6 = arg0.high;
  auto [x7, x8, x9, x10] =
      INVOKE_EXTERN(ctx, divide, x5, x6, arg1.low, arg1.high, (arg2 + (arg3 * Val(2))));
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:50)
  NondetRegStruct x11 = exec_NondetReg(ctx, x7, LAYOUT_LOOKUP(layout4, quotLow));
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:51)
  NondetRegStruct x12 = exec_NondetReg(ctx, x8, LAYOUT_LOOKUP(layout4, quotHigh));
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:52)
  ValU32Struct x13 = ValU32Struct{.low = x11._super, .high = x12._super};
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:54)
  NondetRegStruct x14 = exec_NondetU16Reg(ctx, x9, LAYOUT_LOOKUP(layout4, remLow));
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:55)
  NondetRegStruct x15 = exec_NondetU16Reg(ctx, x10, LAYOUT_LOOKUP(layout4, remHigh));
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:56)
  ValU32Struct x16 = ValU32Struct{.low = x14._super, .high = x15._super};
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:60)
  MultiplyAccumulateStruct x17 = exec_MultiplyAccumulate(
      ctx,
      x13,
      arg1,
      x16,
      MultiplySettingsStruct{.aSigned = arg2, .bSigned = arg2, .cSigned = arg2},
      LAYOUT_LOOKUP(layout4, mul));
  ValU32Struct x18 = x17.outLow;
  // AssertEqU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:106)
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:62)
  EQZ((x18.low - x5),
      "loc(callsite( AssertEqU32 ( zirgen/circuit/rv32im/v2/dsl/u32.zir :106:10) at  DoDiv ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :62:15)))");
  // AssertEqU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:107)
  EQZ((x18.high - x6),
      "loc(callsite( AssertEqU32 ( zirgen/circuit/rv32im/v2/dsl/u32.zir :107:11) at  DoDiv ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :62:15)))");
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:60)
  ValU32Struct x19 = x17.outHigh;
  Val x20 = x19.low;
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:64)
  NondetRegStruct x21 =
      exec_NondetBitReg(ctx, (Val(1) - isz(x20)), LAYOUT_LOOKUP(layout4, topBitType));
  // DoDiv(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:65)
  Val x22 = (x21._super * Val(65535));
  // AssertEqU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:106)
  EQZ((x20 - x22),
      "loc(callsite( AssertEqU32 ( zirgen/circuit/rv32im/v2/dsl/u32.zir :106:10) at  DoDiv ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :65:15)))");
  // AssertEqU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:107)
  EQZ((x19.high - x22),
      "loc(callsite( AssertEqU32 ( zirgen/circuit/rv32im/v2/dsl/u32.zir :107:11) at  DoDiv ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :65:15)))");
  return DoDivStruct{.quot = x13, .rem = x16};
}
ValU32Struct exec_OpSRL(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpSRLLayout> layout1) {
  // OpSRL(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:84)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSRL(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:85)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSRL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :85:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(5)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSRL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :85:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSRL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :85:20)))");
  // OpSRL(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:84)
  Val x4 = arg0.rs2._super.low;
  // OpSRL(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:86)
  ValU32Struct x5 = exec_DynPo2(ctx, x4, LAYOUT_LOOKUP(layout1, shiftMul));
  // OpSRL(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:87)
  DoDivStruct x6 = exec_DoDiv(ctx, arg0.rs1._super, x5, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x6.quot;
}
NondetRegStruct
exec_TopBit(ExecContext& ctx, ValU32Struct arg0, BoundLayout<TopBitLayout> layout1) {
  // TopBit(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:69)
  Val x2 = arg0.high;
  // TopBit(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:70)
  NondetRegStruct x3 = exec_NondetBitReg(
      ctx, (bitAnd(x2, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout1, _super));
  // TopBit(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:71)
  Val x4 = (x3._super * Val(32768));
  NondetRegStruct x5 = exec_NondetU16Reg(ctx, ((x2 - x4) * Val(2)), LAYOUT_LOOKUP(layout1, rest));
  // TopBit(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:72)
  Val x6 = ((x5._super * Val(1006632961)) + x4);
  EQZ((x2 - x6), "TopBit(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:72)");
  return x3;
}
ValU32Struct exec_OpSRA(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpSRALayout> layout1) {
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:90)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:91)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSRA ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :91:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(5)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSRA ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :91:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(32)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSRA ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :91:20)))");
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:90)
  Val x4 = arg0.rs2._super.low;
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:92)
  ValU32Struct x5 = exec_DynPo2(ctx, x4, LAYOUT_LOOKUP(layout1, shiftMul));
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:90)
  ValU32Struct x6 = arg0.rs1._super;
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:93)
  NondetRegStruct x7 = exec_TopBit(ctx, x6, LAYOUT_LOOKUP(layout1, flip));
  Val x8 = x7._super;
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:80)
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:94)
  Val x9 = x6.low;
  // FlipU16(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:77)
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:81)
  Val x10 = (Val(1) - x8);
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:80)
  Val x11 = x6.high;
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:81)
  ValU32Struct x12 = ValU32Struct{.low = ((x8 * (Val(65535) - x9)) + (x10 * x9)),
                                  .high = ((x8 * (Val(65535) - x11)) + (x10 * x11))};
  DoDivStruct x13 = exec_DoDiv(ctx, x12, x5, Val(0), Val(1), LAYOUT_LOOKUP(layout1, _0));
  ValU32Struct x14 = x13.quot;
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:80)
  // OpSRA(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:95)
  Val x15 = x14.low;
  Val x16 = x14.high;
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:81)
  ValU32Struct x17 = ValU32Struct{.low = ((x8 * (Val(65535) - x15)) + (x10 * x15)),
                                  .high = ((x8 * (Val(65535) - x16)) + (x10 * x16))};
  return x17;
}
ValU32Struct exec_OpSRLI(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpSRLILayout> layout1) {
  // OpSRLI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:98)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSRLI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:99)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSRLI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :99:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(5)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSRLI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :99:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSRLI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :99:20)))");
  // OpSRLI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:100)
  ValU32Struct x4 = exec_DynPo2(ctx, x2.rs2, LAYOUT_LOOKUP(layout1, shiftMul));
  // OpSRLI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:101)
  DoDivStruct x5 = exec_DoDiv(ctx, arg0.rs1._super, x4, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x5.quot;
}
ValU32Struct exec_OpSRAI(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpSRAILayout> layout1) {
  // OpSRAI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:104)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSRAI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:105)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSRAI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :105:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(5)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSRAI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :105:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(32)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSRAI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :105:20)))");
  // OpSRAI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:106)
  ValU32Struct x4 = exec_DynPo2(ctx, x2.rs2, LAYOUT_LOOKUP(layout1, shiftMul));
  // OpSRAI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:104)
  ValU32Struct x5 = arg0.rs1._super;
  // OpSRAI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:107)
  NondetRegStruct x6 = exec_TopBit(ctx, x5, LAYOUT_LOOKUP(layout1, flip));
  Val x7 = x6._super;
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:80)
  // OpSRAI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:108)
  Val x8 = x5.low;
  // FlipU16(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:77)
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:81)
  Val x9 = (Val(1) - x7);
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:80)
  Val x10 = x5.high;
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:81)
  ValU32Struct x11 = ValU32Struct{.low = ((x7 * (Val(65535) - x8)) + (x9 * x8)),
                                  .high = ((x7 * (Val(65535) - x10)) + (x9 * x10))};
  DoDivStruct x12 = exec_DoDiv(ctx, x11, x4, Val(0), Val(1), LAYOUT_LOOKUP(layout1, _0));
  ValU32Struct x13 = x12.quot;
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:80)
  // OpSRAI(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:109)
  Val x14 = x13.low;
  Val x15 = x13.high;
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // FlipU32(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:81)
  ValU32Struct x16 = ValU32Struct{.low = ((x7 * (Val(65535) - x14)) + (x9 * x14)),
                                  .high = ((x7 * (Val(65535) - x15)) + (x9 * x15))};
  return x16;
}
ValU32Struct exec_OpDIV(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpDIVLayout> layout1) {
  // OpDIV(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:112)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpDIV(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:113)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpDIV ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :113:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(4)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpDIV ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :113:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpDIV ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :113:20)))");
  // OpDIV(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:114)
  DoDivStruct x4 =
      exec_DoDiv(ctx, arg0.rs1._super, arg0.rs2._super, Val(1), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x4.quot;
}
ValU32Struct exec_OpDIVU(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpDIVULayout> layout1) {
  // OpDIVU(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:117)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpDIVU(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:118)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpDIVU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :118:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(5)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpDIVU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :118:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpDIVU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :118:20)))");
  // OpDIVU(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:119)
  DoDivStruct x4 =
      exec_DoDiv(ctx, arg0.rs1._super, arg0.rs2._super, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x4.quot;
}
ValU32Struct exec_OpREM(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpREMLayout> layout1) {
  // OpREM(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:122)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpREM(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:123)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpREM ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :123:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(6)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpREM ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :123:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpREM ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :123:20)))");
  // OpREM(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:124)
  DoDivStruct x4 =
      exec_DoDiv(ctx, arg0.rs1._super, arg0.rs2._super, Val(1), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x4.rem;
}
ValU32Struct exec_OpREMU(ExecContext& ctx, DivInputStruct arg0, BoundLayout<OpREMULayout> layout1) {
  // OpREMU(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:127)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpREMU(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:128)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpREMU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :128:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(7)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpREMU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :128:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpREMU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_div.zir :128:20)))");
  // OpREMU(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:129)
  DoDivStruct x4 =
      exec_DoDiv(ctx, arg0.rs1._super, arg0.rs2._super, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x4.rem;
}
InstOutputStruct exec_Div0(ExecContext& ctx,
                           NondetRegStruct arg0,
                           BoundLayout<NondetRegLayout> arg1,
                           InstInputStruct arg2,
                           BoundLayout<Div0Layout> layout3) {
  // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
  BoundLayout<Div0MulOutputLayout> x4 = LAYOUT_LOOKUP(layout3, mulOutput);
  // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:22)
  DivInputStruct x5 = exec_DivInput(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, input));
  NondetRegStruct8Array x6 = x5._super.minorOnehot._super;
  // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
  BoundLayout<Div0Arm0Layout> x7 = LAYOUT_LOOKUP(x4, arm0);
  BoundLayout<Div0Arm2Layout> x8 = LAYOUT_LOOKUP(x4, arm2);
  BoundLayout<Div0Arm4Layout> x9 = LAYOUT_LOOKUP(x4, arm4);
  BoundLayout<Div0Arm5Layout> x10 = LAYOUT_LOOKUP(x4, arm5);
  BoundLayout<Div0Arm6Layout> x11 = LAYOUT_LOOKUP(x4, arm6);
  BoundLayout<Div0Arm7Layout> x12 = LAYOUT_LOOKUP(x4, arm7);
  BoundLayout<Reg> x13 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra0), count), _super);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra1), count), _super);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  ValU32Struct x23;
  if (to_size_t(x6[0]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:24)
    ValU32Struct x24 = exec_OpSRL(ctx, x5, LAYOUT_LOOKUP(x7, _super));
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
    STORE(x13, Val(0));
    EQZ(LOAD(x13, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    x23 = x24;
  } else if (to_size_t(x6[1]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:25)
    ValU32Struct x25 = exec_OpSRA(ctx, x5, LAYOUT_LOOKUP(x4, arm1));
    x23 = x25;
  } else if (to_size_t(x6[2]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:26)
    ValU32Struct x26 = exec_OpSRLI(ctx, x5, LAYOUT_LOOKUP(x8, _super));
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
    STORE(x14, Val(0));
    EQZ(LOAD(x14, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    x23 = x26;
  } else if (to_size_t(x6[3]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:27)
    ValU32Struct x27 = exec_OpSRAI(ctx, x5, LAYOUT_LOOKUP(x4, arm3));
    x23 = x27;
  } else if (to_size_t(x6[4]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:28)
    ValU32Struct x28 = exec_OpDIV(ctx, x5, LAYOUT_LOOKUP(x9, _super));
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    x23 = x28;
  } else if (to_size_t(x6[5]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:29)
    ValU32Struct x29 = exec_OpDIVU(ctx, x5, LAYOUT_LOOKUP(x10, _super));
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    x23 = x29;
  } else if (to_size_t(x6[6]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:30)
    ValU32Struct x30 = exec_OpREM(ctx, x5, LAYOUT_LOOKUP(x11, _super));
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    x23 = x30;
  } else if (to_size_t(x6[7]._super)) {
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:31)
    ValU32Struct x31 = exec_OpREMU(ctx, x5, LAYOUT_LOOKUP(x12, _super));
    // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:23)");
    x23 = x31;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:33)
  WriteRdStruct x32 =
      exec_WriteRd(ctx, arg0, arg1, x5.ii, x5.decoded, Val(1), x23, LAYOUT_LOOKUP(layout3, _0));
  // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:21)
  ValU32Struct x33 = arg2.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // Div0(zirgen/circuit/rv32im/v2/dsl/inst_div.zir:34)
  DenormedValU32Struct x34 = DenormedValU32Struct{.low = (x33.low + Val(4)), .high = x33.high};
  NormalizeU32Struct x35 = exec_NormalizeU32(ctx, x34, LAYOUT_LOOKUP(layout3, pcAdd));
  return InstOutputStruct{.newPc = x35._super, .newState = Val(32), .newMode = arg2.mode};
}
MiscInputStruct exec_MiscInput(ExecContext& ctx,
                               NondetRegStruct arg0,
                               BoundLayout<NondetRegLayout> arg1,
                               InstInputStruct arg2,
                               BoundLayout<MiscInputLayout> layout3) {
  // MiscInput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:7)
  EQZ((arg2.state - Val(32)), "MiscInput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:7)");
  // MiscInput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:9)
  DecoderStruct x4 = exec_DecodeInst(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, decoded));
  // MiscInput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:10)
  GetDataStruct x5 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs1, LAYOUT_LOOKUP(layout3, rs1));
  // MiscInput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:11)
  GetDataStruct x6 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs2, LAYOUT_LOOKUP(layout3, rs2));
  return MiscInputStruct{._super = arg2, .ii = arg2, .decoded = x4, .rs1 = x5, .rs2 = x6};
}
InstOutputStruct exec_FinalizeMisc(ExecContext& ctx,
                                   NondetRegStruct arg0,
                                   BoundLayout<NondetRegLayout> arg1,
                                   MiscInputStruct arg2,
                                   MiscOutputStruct arg3,
                                   BoundLayout<FinalizeMiscLayout> layout4) {
  // FinalizeMisc(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:22)
  NormalizeU32Struct x5 = exec_NormalizeU32(ctx, arg3.toWrite, LAYOUT_LOOKUP(layout4, writeData));
  // FinalizeMisc(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:23)
  NormalizeU32Struct x6 = exec_NormalizeU32(ctx, arg3.newPc, LAYOUT_LOOKUP(layout4, pcNorm));
  // FinalizeMisc(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:21)
  InstInputStruct x7 = arg2.ii;
  // FinalizeMisc(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:24)
  WriteRdStruct x8 = exec_WriteRd(
      ctx, arg0, arg1, x7, arg2.decoded, arg3.doWrite, x5._super, LAYOUT_LOOKUP(layout4, _0));
  return InstOutputStruct{.newPc = x6._super, .newState = Val(32), .newMode = x7.mode};
}
MiscOutputStruct
exec_OpXOR(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpXORLayout> layout1) {
  // OpXOR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:95)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpXOR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:96)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpXOR ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :96:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(4)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpXOR ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :96:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpXOR ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :96:20)))");
  // OpXOR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:97)
  ValU32Struct x4 =
      exec_BitwiseXor(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, _0));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.low, .high = x4.high},
                       .newPc = x6};
  return x7;
}
MiscOutputStruct
exec_OpOR(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpORLayout> layout1) {
  // OpOR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:100)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpOR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:101)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpOR ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :101:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(6)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpOR ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :101:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpOR ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :101:20)))");
  // OpOR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:102)
  ValU32Struct x4 =
      exec_BitwiseOr(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, _0));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.low, .high = x4.high},
                       .newPc = x6};
  return x7;
}
MiscOutputStruct
exec_OpAND(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpANDLayout> layout1) {
  // OpAND(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:105)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpAND(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:106)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpAND ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :106:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(7)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpAND ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :106:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpAND ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :106:20)))");
  // OpAND(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:107)
  ValU32Struct x4 =
      exec_BitwiseAnd(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, _0));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.low, .high = x4.high},
                       .newPc = x6};
  return x7;
}
MiscOutputStruct
exec_OpSLT(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpSLTLayout> layout1) {
  // OpSLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:110)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:111)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSLT ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :111:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(2)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSLT ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :111:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSLT ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :111:20)))");
  // OpSLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:112)
  CmpLessThanStruct x4 =
      exec_CmpLessThan(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // OpSLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:113)
  DenormedValU32Struct x5 = DenormedValU32Struct{.low = x4.isLessThan._super, .high = Val(0)};
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x6 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x7 = DenormedValU32Struct{.low = (x6.low + Val(4)), .high = x6.high};
  return MiscOutputStruct{.doWrite = Val(1), .toWrite = x5, .newPc = x7};
}
MiscOutputStruct
exec_OpSLTU(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpSLTULayout> layout1) {
  // OpSLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:116)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:117)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSLTU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :117:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(3)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSLTU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :117:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSLTU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :117:20)))");
  // OpSLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:118)
  CmpLessThanUnsignedStruct x4 =
      exec_CmpLessThanUnsigned(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  // OpSLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:119)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.isLessThan, .high = Val(0)},
                       .newPc = x6};
  return x7;
}
InstOutputStruct exec_Misc0(ExecContext& ctx,
                            NondetRegStruct arg0,
                            BoundLayout<NondetRegLayout> arg1,
                            InstInputStruct arg2,
                            BoundLayout<Misc0Layout> layout3) {
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
  BoundLayout<Misc0MiscOutputLayout> x4 = LAYOUT_LOOKUP(layout3, miscOutput);
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:29)
  MiscInputStruct x5 = exec_MiscInput(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, input));
  InstInputStruct x6 = x5._super;
  NondetRegStruct8Array x7 = x6.minorOnehot._super;
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
  BoundLayout<Misc0Arm0Layout> x8 = LAYOUT_LOOKUP(x4, arm0);
  BoundLayout<Misc0Arm1Layout> x9 = LAYOUT_LOOKUP(x4, arm1);
  BoundLayout<Misc0Arm2Layout> x10 = LAYOUT_LOOKUP(x4, arm2);
  BoundLayout<Misc0Arm3Layout> x11 = LAYOUT_LOOKUP(x4, arm3);
  BoundLayout<Misc0Arm4Layout> x12 = LAYOUT_LOOKUP(x4, arm4);
  BoundLayout<Misc0Arm6Layout> x13 = LAYOUT_LOOKUP(x4, arm6);
  BoundLayout<Misc0Arm7Layout> x14 = LAYOUT_LOOKUP(x4, arm7);
  // OpADD(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:85)
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:31)
  DecoderStruct x15 = x5.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:61)
  // OpADD(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:86)
  Val x16 = x15.opcode._super;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  Val x17 = (x16 - Val(51));
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:61)
  Val x18 = x15.func3;
  Val x19 = x15.func7;
  // OpADD(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:85)
  ValU32Struct x20 = x5.rs1._super;
  ValU32Struct x21 = x5.rs2._super;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // OpADD(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:87)
  Val x22 = x20.low;
  Val x23 = x21.low;
  Val x24 = x20.high;
  Val x25 = x21.high;
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x26 = x6.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x27 = DenormedValU32Struct{.low = (x26.low + Val(4)), .high = x26.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x28 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = (x22 + x23), .high = (x24 + x25)},
                       .newPc = x27};
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra3), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra4), count), _super);
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // SubU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:33)
  // OpSUB(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:92)
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:32)
  DenormedValU32Struct x34 =
      DenormedValU32Struct{.low = ((x22 + Val(65536)) - x23), .high = ((x24 + Val(65535)) - x25)};
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra1), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra2), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra3), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra4), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x42 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra2), count), _super);
  BoundLayout<Reg> x43 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra3), count), _super);
  BoundLayout<Reg> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra4), count), _super);
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x48 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x49 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  BoundLayout<Reg> x50 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x51 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x52 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x53 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x54 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra4), count), _super);
  BoundLayout<Reg> x55 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra0), count), _super);
  BoundLayout<Reg> x56 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra1), count), _super);
  BoundLayout<Reg> x57 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra2), count), _super);
  // OpADDI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:122)
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:38)
  ValU32Struct x58 = x15.immI;
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  // OpADDI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:124)
  MiscOutputStruct x59 = MiscOutputStruct{
      .doWrite = Val(1),
      .toWrite = DenormedValU32Struct{.low = (x22 + x58.low), .high = (x24 + x58.high)},
      .newPc = x27};
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
  BoundLayout<Reg> x60 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra0), count), _super);
  BoundLayout<Reg> x61 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra1), count), _super);
  BoundLayout<Reg> x62 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra2), count), _super);
  BoundLayout<Reg> x63 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra3), count), _super);
  BoundLayout<Reg> x64 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra4), count), _super);
  MiscOutputStruct x65;
  if (to_size_t(x7[0]._super)) {
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
    // OpADD(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:86)
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:31)
    EQZ(x17,
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at "
        "callsite( OpADD ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :86:20) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :31:11))))");
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
    EQZ(x18,
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at "
        "callsite( OpADD ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :86:20) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :31:11))))");
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
    EQZ(x19,
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at "
        "callsite( OpADD ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :86:20) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :31:11))))");
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    x65 = x28;
  } else if (to_size_t(x7[1]._super)) {
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
    // OpSUB(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:91)
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:32)
    EQZ(x17,
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at "
        "callsite( OpSUB ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :91:20) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :32:11))))");
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
    EQZ(x18,
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at "
        "callsite( OpSUB ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :91:20) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :32:11))))");
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
    EQZ((x19 - Val(32)),
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at "
        "callsite( OpSUB ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :91:20) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :32:11))))");
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    x65 = MiscOutputStruct{.doWrite = Val(1), .toWrite = x34, .newPc = x27};
  } else if (to_size_t(x7[2]._super)) {
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:33)
    MiscOutputStruct x66 = exec_OpXOR(ctx, x5, LAYOUT_LOOKUP(x10, _super));
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x42, Val(0));
    EQZ(LOAD(x42, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x43, Val(0));
    EQZ(LOAD(x43, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x44, Val(0));
    EQZ(LOAD(x44, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    x65 = x66;
  } else if (to_size_t(x7[3]._super)) {
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:34)
    MiscOutputStruct x67 = exec_OpOR(ctx, x5, LAYOUT_LOOKUP(x11, _super));
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x48, Val(0));
    EQZ(LOAD(x48, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x49, Val(0));
    EQZ(LOAD(x49, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    x65 = x67;
  } else if (to_size_t(x7[4]._super)) {
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:35)
    MiscOutputStruct x68 = exec_OpAND(ctx, x5, LAYOUT_LOOKUP(x12, _super));
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
    STORE(x50, Val(0));
    EQZ(LOAD(x50, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x51, Val(0));
    EQZ(LOAD(x51, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x52, Val(0));
    EQZ(LOAD(x52, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x53, Val(0));
    EQZ(LOAD(x53, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x54, Val(0));
    EQZ(LOAD(x54, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    x65 = x68;
  } else if (to_size_t(x7[5]._super)) {
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:36)
    MiscOutputStruct x69 = exec_OpSLT(ctx, x5, LAYOUT_LOOKUP(x4, arm5));
    x65 = x69;
  } else if (to_size_t(x7[6]._super)) {
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:37)
    MiscOutputStruct x70 = exec_OpSLTU(ctx, x5, LAYOUT_LOOKUP(x13, _super));
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
    STORE(x55, Val(0));
    EQZ(LOAD(x55, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x56, Val(0));
    EQZ(LOAD(x56, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x57, Val(0));
    EQZ(LOAD(x57, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    x65 = x70;
  } else if (to_size_t(x7[7]._super)) {
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
    // OpADDI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:123)
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:38)
    EQZ((x16 - Val(19)),
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at callsite( "
        "OpADDI ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :123:18) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :38:12))))");
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
    EQZ(x18,
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at callsite( "
        "OpADDI ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :123:18) at  Misc0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :38:12))))");
    // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)
    STORE(x60, Val(0));
    EQZ(LOAD(x60, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x61, Val(0));
    EQZ(LOAD(x61, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x62, Val(0));
    EQZ(LOAD(x62, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x63, Val(0));
    EQZ(LOAD(x63, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    STORE(x64, Val(0));
    EQZ(LOAD(x64, 0), "Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:30)");
    x65 = x59;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Misc0(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:40)
  InstOutputStruct x71 =
      exec_FinalizeMisc(ctx, arg0, arg1, x5, x65, LAYOUT_LOOKUP(layout3, _super));
  return x71;
}
MiscOutputStruct
exec_OpXORI(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpXORILayout> layout1) {
  // OpXORI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:127)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpXORI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:128)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpXORI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :128:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(4)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpXORI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :128:18)))");
  // OpXORI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:129)
  ValU32Struct x4 = exec_BitwiseXor(ctx, arg0.rs1._super, x2.immI, LAYOUT_LOOKUP(layout1, _0));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.low, .high = x4.high},
                       .newPc = x6};
  return x7;
}
MiscOutputStruct
exec_OpORI(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpORILayout> layout1) {
  // OpORI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:132)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpORI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:133)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpORI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :133:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(6)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpORI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :133:18)))");
  // OpORI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:134)
  ValU32Struct x4 = exec_BitwiseOr(ctx, arg0.rs1._super, x2.immI, LAYOUT_LOOKUP(layout1, _0));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.low, .high = x4.high},
                       .newPc = x6};
  return x7;
}
MiscOutputStruct
exec_OpANDI(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpANDILayout> layout1) {
  // OpANDI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:137)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpANDI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:138)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpANDI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :138:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(7)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpANDI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :138:18)))");
  // OpANDI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:139)
  ValU32Struct x4 = exec_BitwiseAnd(ctx, arg0.rs1._super, x2.immI, LAYOUT_LOOKUP(layout1, _0));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.low, .high = x4.high},
                       .newPc = x6};
  return x7;
}
MiscOutputStruct
exec_OpSLTI(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpSLTILayout> layout1) {
  // OpSLTI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:142)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpSLTI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:143)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpSLTI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :143:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(2)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpSLTI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :143:18)))");
  // OpSLTI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:144)
  CmpLessThanStruct x4 =
      exec_CmpLessThan(ctx, arg0.rs1._super, x2.immI, LAYOUT_LOOKUP(layout1, cmp));
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // OpSLTI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:145)
  DenormedValU32Struct x5 = DenormedValU32Struct{.low = x4.isLessThan._super, .high = Val(0)};
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  ValU32Struct x6 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x7 = DenormedValU32Struct{.low = (x6.low + Val(4)), .high = x6.high};
  return MiscOutputStruct{.doWrite = Val(1), .toWrite = x5, .newPc = x7};
}
MiscOutputStruct
exec_OpSLTIU(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpSLTIULayout> layout1) {
  // OpSLTIU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:148)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpSLTIU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:149)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpSLTIU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :149:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(3)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpSLTIU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :149:18)))");
  // OpSLTIU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:150)
  CmpLessThanUnsignedStruct x4 =
      exec_CmpLessThanUnsigned(ctx, arg0.rs1._super, x2.immI, LAYOUT_LOOKUP(layout1, cmp));
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:73)
  // OpSLTIU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:151)
  ValU32Struct x5 = arg0._super.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:74)
  DenormedValU32Struct x6 = DenormedValU32Struct{.low = (x5.low + Val(4)), .high = x5.high};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  MiscOutputStruct x7 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = x4.isLessThan, .high = Val(0)},
                       .newPc = x6};
  return x7;
}
MiscOutputStruct
exec_OpBEQ(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpBEQLayout> layout1) {
  // OpBEQ(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:154)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpBEQ(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:155)
  Val x3 = (x2.opcode._super - Val(99));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpBEQ ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :155:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ(x2.func3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpBEQ ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :155:18)))");
  // OpBEQ(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:156)
  CmpEqualStruct x4 =
      exec_CmpEqual(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  Val x5 = x4.isEqual._super;
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:78)
  // OpBEQ(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:157)
  ValU32Struct x6 = arg0._super.pcU32;
  ValU32Struct x7 = x2.immB;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:80)
  Val x8 = x6.low;
  Val x9 = x6.high;
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:101)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:79)
  Val x10 = (Val(1) - x5);
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:100)
  DenormedValU32Struct x11 =
      DenormedValU32Struct{.low = ((x5 * (x8 + x7.low)) + (x10 * (x8 + Val(4)))),
                           .high = ((x5 * (x9 + x7.high)) + (x10 * x9))};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:82)
  MiscOutputStruct x12 =
      MiscOutputStruct{.doWrite = Val(0),
                       .toWrite = DenormedValU32Struct{.low = Val(0), .high = Val(0)},
                       .newPc = x11};
  return x12;
}
MiscOutputStruct
exec_OpBNE(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpBNELayout> layout1) {
  // OpBNE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:160)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpBNE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:161)
  Val x3 = (x2.opcode._super - Val(99));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpBNE ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :161:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(1)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpBNE ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :161:18)))");
  // OpBNE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:162)
  CmpEqualStruct x4 =
      exec_CmpEqual(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  // OpBNE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:163)
  Val x5 = (Val(1) - x4.isEqual._super);
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:78)
  ValU32Struct x6 = arg0._super.pcU32;
  ValU32Struct x7 = x2.immB;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:80)
  Val x8 = x6.low;
  Val x9 = x6.high;
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:101)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:79)
  Val x10 = (Val(1) - x5);
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:100)
  DenormedValU32Struct x11 =
      DenormedValU32Struct{.low = ((x5 * (x8 + x7.low)) + (x10 * (x8 + Val(4)))),
                           .high = ((x5 * (x9 + x7.high)) + (x10 * x9))};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:82)
  MiscOutputStruct x12 =
      MiscOutputStruct{.doWrite = Val(0),
                       .toWrite = DenormedValU32Struct{.low = Val(0), .high = Val(0)},
                       .newPc = x11};
  return x12;
}
MiscOutputStruct
exec_OpBLT(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpBLTLayout> layout1) {
  // OpBLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:166)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpBLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:167)
  Val x3 = (x2.opcode._super - Val(99));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpBLT ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :167:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(4)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpBLT ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :167:18)))");
  // OpBLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:168)
  CmpLessThanStruct x4 =
      exec_CmpLessThan(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  Val x5 = x4.isLessThan._super;
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:78)
  // OpBLT(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:169)
  ValU32Struct x6 = arg0._super.pcU32;
  ValU32Struct x7 = x2.immB;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:80)
  Val x8 = x6.low;
  Val x9 = x6.high;
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:101)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:79)
  Val x10 = (Val(1) - x5);
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:100)
  DenormedValU32Struct x11 =
      DenormedValU32Struct{.low = ((x5 * (x8 + x7.low)) + (x10 * (x8 + Val(4)))),
                           .high = ((x5 * (x9 + x7.high)) + (x10 * x9))};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:82)
  MiscOutputStruct x12 =
      MiscOutputStruct{.doWrite = Val(0),
                       .toWrite = DenormedValU32Struct{.low = Val(0), .high = Val(0)},
                       .newPc = x11};
  return x12;
}
InstOutputStruct exec_Misc1(ExecContext& ctx,
                            NondetRegStruct arg0,
                            BoundLayout<NondetRegLayout> arg1,
                            InstInputStruct arg2,
                            BoundLayout<Misc1Layout> layout3) {
  // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
  BoundLayout<Misc1MiscOutputLayout> x4 = LAYOUT_LOOKUP(layout3, miscOutput);
  // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:44)
  MiscInputStruct x5 = exec_MiscInput(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, input));
  NondetRegStruct8Array x6 = x5._super.minorOnehot._super;
  // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
  BoundLayout<Misc1Arm0Layout> x7 = LAYOUT_LOOKUP(x4, arm0);
  BoundLayout<Misc1Arm1Layout> x8 = LAYOUT_LOOKUP(x4, arm1);
  BoundLayout<Misc1Arm2Layout> x9 = LAYOUT_LOOKUP(x4, arm2);
  BoundLayout<Misc1Arm4Layout> x10 = LAYOUT_LOOKUP(x4, arm4);
  BoundLayout<Misc1Arm5Layout> x11 = LAYOUT_LOOKUP(x4, arm5);
  BoundLayout<Misc1Arm6Layout> x12 = LAYOUT_LOOKUP(x4, arm6);
  BoundLayout<Reg> x13 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra0), count), _super);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra1), count), _super);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra2), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra3), count), _super);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra4), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra3), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra4), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra1), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra2), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra3), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra4), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra2), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra4), count), _super);
  MiscOutputStruct x41;
  if (to_size_t(x6[0]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:46)
    MiscOutputStruct x42 = exec_OpXORI(ctx, x5, LAYOUT_LOOKUP(x7, _super));
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
    STORE(x13, Val(0));
    EQZ(LOAD(x13, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x14, Val(0));
    EQZ(LOAD(x14, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    x41 = x42;
  } else if (to_size_t(x6[1]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:47)
    MiscOutputStruct x43 = exec_OpORI(ctx, x5, LAYOUT_LOOKUP(x8, _super));
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    x41 = x43;
  } else if (to_size_t(x6[2]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:48)
    MiscOutputStruct x44 = exec_OpANDI(ctx, x5, LAYOUT_LOOKUP(x9, _super));
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    x41 = x44;
  } else if (to_size_t(x6[3]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:49)
    MiscOutputStruct x45 = exec_OpSLTI(ctx, x5, LAYOUT_LOOKUP(x4, arm3));
    x41 = x45;
  } else if (to_size_t(x6[4]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:50)
    MiscOutputStruct x46 = exec_OpSLTIU(ctx, x5, LAYOUT_LOOKUP(x10, _super));
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    x41 = x46;
  } else if (to_size_t(x6[5]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:51)
    MiscOutputStruct x47 = exec_OpBEQ(ctx, x5, LAYOUT_LOOKUP(x11, _super));
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    x41 = x47;
  } else if (to_size_t(x6[6]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:52)
    MiscOutputStruct x48 = exec_OpBNE(ctx, x5, LAYOUT_LOOKUP(x12, _super));
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:45)");
    x41 = x48;
  } else if (to_size_t(x6[7]._super)) {
    // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:53)
    MiscOutputStruct x49 = exec_OpBLT(ctx, x5, LAYOUT_LOOKUP(x4, arm7));
    x41 = x49;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Misc1(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:55)
  InstOutputStruct x50 =
      exec_FinalizeMisc(ctx, arg0, arg1, x5, x41, LAYOUT_LOOKUP(layout3, _super));
  return x50;
}
MiscOutputStruct
exec_OpBGE(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpBGELayout> layout1) {
  // OpBGE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:172)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpBGE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:173)
  Val x3 = (x2.opcode._super - Val(99));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpBGE ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :173:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(5)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpBGE ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :173:18)))");
  // OpBGE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:174)
  CmpLessThanStruct x4 =
      exec_CmpLessThan(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  // OpBGE(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:175)
  Val x5 = (Val(1) - x4.isLessThan._super);
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:78)
  ValU32Struct x6 = arg0._super.pcU32;
  ValU32Struct x7 = x2.immB;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:80)
  Val x8 = x6.low;
  Val x9 = x6.high;
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:101)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:79)
  Val x10 = (Val(1) - x5);
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:100)
  DenormedValU32Struct x11 =
      DenormedValU32Struct{.low = ((x5 * (x8 + x7.low)) + (x10 * (x8 + Val(4)))),
                           .high = ((x5 * (x9 + x7.high)) + (x10 * x9))};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:82)
  MiscOutputStruct x12 =
      MiscOutputStruct{.doWrite = Val(0),
                       .toWrite = DenormedValU32Struct{.low = Val(0), .high = Val(0)},
                       .newPc = x11};
  return x12;
}
MiscOutputStruct
exec_OpBLTU(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpBLTULayout> layout1) {
  // OpBLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:178)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpBLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:179)
  Val x3 = (x2.opcode._super - Val(99));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpBLTU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :179:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(6)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpBLTU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :179:18)))");
  // OpBLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:180)
  CmpLessThanUnsignedStruct x4 =
      exec_CmpLessThanUnsigned(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  Val x5 = x4.isLessThan;
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:78)
  // OpBLTU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:181)
  ValU32Struct x6 = arg0._super.pcU32;
  ValU32Struct x7 = x2.immB;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:80)
  Val x8 = x6.low;
  Val x9 = x6.high;
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:101)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:79)
  Val x10 = (Val(1) - x5);
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:100)
  DenormedValU32Struct x11 =
      DenormedValU32Struct{.low = ((x5 * (x8 + x7.low)) + (x10 * (x8 + Val(4)))),
                           .high = ((x5 * (x9 + x7.high)) + (x10 * x9))};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:82)
  MiscOutputStruct x12 =
      MiscOutputStruct{.doWrite = Val(0),
                       .toWrite = DenormedValU32Struct{.low = Val(0), .high = Val(0)},
                       .newPc = x11};
  return x12;
}
MiscOutputStruct
exec_OpBGEU(ExecContext& ctx, MiscInputStruct arg0, BoundLayout<OpBGEULayout> layout1) {
  // OpBGEU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:184)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpBGEU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:185)
  Val x3 = (x2.opcode._super - Val(99));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpBGEU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :185:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(7)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpBGEU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :185:18)))");
  // OpBGEU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:186)
  CmpLessThanUnsignedStruct x4 =
      exec_CmpLessThanUnsigned(ctx, arg0.rs1._super, arg0.rs2._super, LAYOUT_LOOKUP(layout1, cmp));
  // OpBGEU(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:187)
  Val x5 = (Val(1) - x4.isLessThan);
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:78)
  ValU32Struct x6 = arg0._super.pcU32;
  ValU32Struct x7 = x2.immB;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:80)
  Val x8 = x6.low;
  Val x9 = x6.high;
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:101)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:79)
  Val x10 = (Val(1) - x5);
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // CondDenormed(zirgen/circuit/rv32im/v2/dsl/u32.zir:100)
  DenormedValU32Struct x11 =
      DenormedValU32Struct{.low = ((x5 * (x8 + x7.low)) + (x10 * (x8 + Val(4)))),
                           .high = ((x5 * (x9 + x7.high)) + (x10 * x9))};
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // CmpOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:82)
  MiscOutputStruct x12 =
      MiscOutputStruct{.doWrite = Val(0),
                       .toWrite = DenormedValU32Struct{.low = Val(0), .high = Val(0)},
                       .newPc = x11};
  return x12;
}
InstOutputStruct exec_Misc2(ExecContext& ctx,
                            NondetRegStruct arg0,
                            BoundLayout<NondetRegLayout> arg1,
                            InstInputStruct arg2,
                            BoundLayout<Misc2Layout> layout3) {
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
  BoundLayout<Misc2MiscOutputLayout> x4 = LAYOUT_LOOKUP(layout3, miscOutput);
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:59)
  MiscInputStruct x5 = exec_MiscInput(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, input));
  InstInputStruct x6 = x5._super;
  NondetRegStruct8Array x7 = x6.minorOnehot._super;
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
  BoundLayout<Misc2Arm1Layout> x8 = LAYOUT_LOOKUP(x4, arm1);
  BoundLayout<Misc2Arm2Layout> x9 = LAYOUT_LOOKUP(x4, arm2);
  BoundLayout<Misc2Arm3Layout> x10 = LAYOUT_LOOKUP(x4, arm3);
  BoundLayout<Misc2Arm4Layout> x11 = LAYOUT_LOOKUP(x4, arm4);
  BoundLayout<Misc2Arm5Layout> x12 = LAYOUT_LOOKUP(x4, arm5);
  BoundLayout<Misc2Arm6Layout> x13 = LAYOUT_LOOKUP(x4, arm6);
  BoundLayout<Misc2Arm7Layout> x14 = LAYOUT_LOOKUP(x4, arm7);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra1), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra2), count), _super);
  // OpJAL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:190)
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:64)
  DecoderStruct x21 = x5.decoded;
  // VerifyOpcode(zirgen/circuit/rv32im/v2/dsl/inst.zir:52)
  // OpJAL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:191)
  Val x22 = x21.opcode._super;
  // OpJAL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:190)
  ValU32Struct x23 = x6.pcU32;
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:26)
  // OpJAL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:193)
  Val x24 = x23.low;
  Val x25 = x23.high;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  DenormedValU32Struct x26 = DenormedValU32Struct{.low = (x24 + Val(4)), .high = x25};
  // OpJAL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:190)
  ValU32Struct x27 = x21.immJ;
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra2), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra3), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra4), count), _super);
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:56)
  // OpJALR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:198)
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:65)
  Val x33 = x21.func3;
  // OpJALR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:197)
  ValU32Struct x34 = x21.immI;
  ValU32Struct x35 = x5.rs1._super;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // OpJALR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:201)
  DenormedValU32Struct x36 =
      DenormedValU32Struct{.low = (x35.low + x34.low), .high = (x35.high + x34.high)};
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  // OpLUI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:204)
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:66)
  ValU32Struct x42 = x21.immU;
  // Denorm(zirgen/circuit/rv32im/v2/dsl/u32.zir:37)
  // OpLUI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:206)
  Val x43 = x42.low;
  Val x44 = x42.high;
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x48 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x49 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra4), count), _super);
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // SimpleOp(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:75)
  // OpAUIPC(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:211)
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:67)
  MiscOutputStruct x50 =
      MiscOutputStruct{.doWrite = Val(1),
                       .toWrite = DenormedValU32Struct{.low = (x24 + x43), .high = (x25 + x44)},
                       .newPc = x26};
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
  BoundLayout<Reg> x51 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra0), count), _super);
  BoundLayout<Reg> x52 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra1), count), _super);
  BoundLayout<Reg> x53 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra2), count), _super);
  BoundLayout<Reg> x54 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra3), count), _super);
  BoundLayout<Reg> x55 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra4), count), _super);
  // MiscOutput(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:15)
  // OpECALL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:217)
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:68)
  MiscOutputStruct x56 =
      MiscOutputStruct{.doWrite = Val(0),
                       .toWrite = DenormedValU32Struct{.low = Val(0), .high = Val(0)},
                       .newPc = DenormedValU32Struct{.low = x24, .high = x25}};
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
  BoundLayout<Reg> x57 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra0), count), _super);
  BoundLayout<Reg> x58 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra1), count), _super);
  BoundLayout<Reg> x59 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra2), count), _super);
  BoundLayout<Reg> x60 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra3), count), _super);
  BoundLayout<Reg> x61 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra4), count), _super);
  MiscOutputStruct x62;
  if (to_size_t(x7[0]._super)) {
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:61)
    MiscOutputStruct x63 = exec_OpBGE(ctx, x5, LAYOUT_LOOKUP(x4, arm0));
    x62 = x63;
  } else if (to_size_t(x7[1]._super)) {
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:62)
    MiscOutputStruct x64 = exec_OpBLTU(ctx, x5, LAYOUT_LOOKUP(x8, _super));
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    x62 = x64;
  } else if (to_size_t(x7[2]._super)) {
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:63)
    MiscOutputStruct x65 = exec_OpBGEU(ctx, x5, LAYOUT_LOOKUP(x9, _super));
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    x62 = x65;
  } else if (to_size_t(x7[3]._super)) {
    // VerifyOpcode(zirgen/circuit/rv32im/v2/dsl/inst.zir:53)
    // OpJAL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:191)
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:64)
    EQZ((x22 - Val(111)),
        "loc(callsite( VerifyOpcode ( zirgen/circuit/rv32im/v2/dsl/inst.zir :53:19) at callsite( "
        "OpJAL ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :191:16) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :64:11))))");
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    x62 = MiscOutputStruct{
        .doWrite = Val(1),
        .toWrite = x26,
        .newPc = DenormedValU32Struct{.low = (x24 + x27.low), .high = (x25 + x27.high)}};
  } else if (to_size_t(x7[4]._super)) {
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
    // OpJALR(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:198)
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:65)
    EQZ((x22 - Val(103)),
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at callsite( "
        "OpJALR ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :198:18) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :65:12))))");
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
    EQZ(x33,
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at callsite( "
        "OpJALR ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :198:18) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :65:12))))");
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    x62 = MiscOutputStruct{.doWrite = Val(1), .toWrite = x26, .newPc = x36};
  } else if (to_size_t(x7[5]._super)) {
    // VerifyOpcode(zirgen/circuit/rv32im/v2/dsl/inst.zir:53)
    // OpLUI(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:205)
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:66)
    EQZ((x22 - Val(55)),
        "loc(callsite( VerifyOpcode ( zirgen/circuit/rv32im/v2/dsl/inst.zir :53:19) at callsite( "
        "OpLUI ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :205:16) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :66:11))))");
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x48, Val(0));
    EQZ(LOAD(x48, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x49, Val(0));
    EQZ(LOAD(x49, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    x62 = MiscOutputStruct{
        .doWrite = Val(1), .toWrite = DenormedValU32Struct{.low = x43, .high = x44}, .newPc = x26};
  } else if (to_size_t(x7[6]._super)) {
    // VerifyOpcode(zirgen/circuit/rv32im/v2/dsl/inst.zir:53)
    // OpAUIPC(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:210)
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:67)
    EQZ((x22 - Val(23)),
        "loc(callsite( VerifyOpcode ( zirgen/circuit/rv32im/v2/dsl/inst.zir :53:19) at callsite( "
        "OpAUIPC ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :210:16) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :67:13))))");
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
    STORE(x51, Val(0));
    EQZ(LOAD(x51, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x52, Val(0));
    EQZ(LOAD(x52, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x53, Val(0));
    EQZ(LOAD(x53, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x54, Val(0));
    EQZ(LOAD(x54, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x55, Val(0));
    EQZ(LOAD(x55, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    x62 = x50;
  } else if (to_size_t(x7[7]._super)) {
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
    // OpECALL(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:216)
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:68)
    EQZ((x22 - Val(115)),
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at "
        "callsite( OpECALL ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :216:20) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :68:13))))");
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
    EQZ(x33,
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at "
        "callsite( OpECALL ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :216:20) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :68:13))))");
    // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
    EQZ(x21.func7,
        "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at "
        "callsite( OpECALL ( zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :216:20) at  Misc2 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_misc.zir :68:13))))");
    // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)
    STORE(x57, Val(0));
    EQZ(LOAD(x57, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x58, Val(0));
    EQZ(LOAD(x58, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x59, Val(0));
    EQZ(LOAD(x59, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x60, Val(0));
    EQZ(LOAD(x60, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    STORE(x61, Val(0));
    EQZ(LOAD(x61, 0), "Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:60)");
    x62 = x56;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Misc2(zirgen/circuit/rv32im/v2/dsl/inst_misc.zir:70)
  InstOutputStruct x66 =
      exec_FinalizeMisc(ctx, arg0, arg1, x5, x62, LAYOUT_LOOKUP(layout3, _super));
  return x66;
}
MulInputStruct exec_MulInput(ExecContext& ctx,
                             NondetRegStruct arg0,
                             BoundLayout<NondetRegLayout> arg1,
                             InstInputStruct arg2,
                             BoundLayout<MulInputLayout> layout3) {
  // MulInput(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:8)
  EQZ((arg2.state - Val(32)), "MulInput(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:8)");
  // MulInput(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:10)
  DecoderStruct x4 = exec_DecodeInst(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, decoded));
  // MulInput(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:11)
  GetDataStruct x5 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs1, LAYOUT_LOOKUP(layout3, rs1));
  // MulInput(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:12)
  GetDataStruct x6 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs2, LAYOUT_LOOKUP(layout3, rs2));
  return MulInputStruct{._super = arg2, .ii = arg2, .decoded = x4, .rs1 = x5, .rs2 = x6};
}
DoMulStruct exec_DoMul(ExecContext& ctx,
                       ValU32Struct arg0,
                       ValU32Struct arg1,
                       Val arg2,
                       Val arg3,
                       BoundLayout<DoMulLayout> layout4) {
  // DoMul(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:40)
  MultiplyAccumulateStruct x5 = exec_MultiplyAccumulate(
      ctx,
      arg0,
      arg1,
      ValU32Struct{.low = Val(0), .high = Val(0)},
      MultiplySettingsStruct{.aSigned = arg2, .bSigned = arg3, .cSigned = Val(0)},
      LAYOUT_LOOKUP(layout4, mul));
  return DoMulStruct{.low = x5.outLow, .high = x5.outHigh};
}
ValU32Struct exec_OpSLL(ExecContext& ctx, MulInputStruct arg0, BoundLayout<OpSLLLayout> layout1) {
  // OpSLL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:45)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSLL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:46)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSLL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :46:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSLL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :46:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSLL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :46:20)))");
  // OpSLL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:45)
  Val x4 = arg0.rs2._super.low;
  // OpSLL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:47)
  ValU32Struct x5 = exec_DynPo2(ctx, x4, LAYOUT_LOOKUP(layout1, shiftMul));
  // OpSLL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:48)
  DoMulStruct x6 = exec_DoMul(ctx, arg0.rs1._super, x5, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x6.low;
}
ValU32Struct exec_OpSLLI(ExecContext& ctx, MulInputStruct arg0, BoundLayout<OpSLLILayout> layout1) {
  // OpSLLI(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:51)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpSLLI(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:52)
  Val x3 = (x2.opcode._super - Val(19));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpSLLI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :52:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpSLLI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :52:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ(x2.func7,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpSLLI ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :52:20)))");
  // OpSLLI(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:53)
  ValU32Struct x4 = exec_DynPo2(ctx, x2.rs2, LAYOUT_LOOKUP(layout1, shiftMul));
  // OpSLLI(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:54)
  DoMulStruct x5 = exec_DoMul(ctx, arg0.rs1._super, x4, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x5.low;
}
ValU32Struct exec_OpMUL(ExecContext& ctx, MulInputStruct arg0, BoundLayout<OpMULLayout> layout1) {
  // OpMUL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:57)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpMUL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:58)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpMUL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :58:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ(x2.func3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpMUL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :58:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpMUL ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :58:20)))");
  // OpMUL(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:59)
  DoMulStruct x4 =
      exec_DoMul(ctx, arg0.rs1._super, arg0.rs2._super, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x4.low;
}
ValU32Struct exec_OpMULH(ExecContext& ctx, MulInputStruct arg0, BoundLayout<OpMULHLayout> layout1) {
  // OpMULH(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:62)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpMULH(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:63)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpMULH ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :63:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpMULH ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :63:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpMULH ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :63:20)))");
  // OpMULH(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:64)
  DoMulStruct x4 =
      exec_DoMul(ctx, arg0.rs1._super, arg0.rs2._super, Val(1), Val(1), LAYOUT_LOOKUP(layout1, _0));
  return x4.high;
}
ValU32Struct
exec_OpMULHSU(ExecContext& ctx, MulInputStruct arg0, BoundLayout<OpMULHSULayout> layout1) {
  // OpMULHSU(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:67)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpMULHSU(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:68)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpMULHSU "
      "( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :68:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(2)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpMULHSU "
      "( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :68:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpMULHSU "
      "( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :68:20)))");
  // OpMULHSU(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:69)
  DoMulStruct x4 =
      exec_DoMul(ctx, arg0.rs1._super, arg0.rs2._super, Val(1), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x4.high;
}
ValU32Struct
exec_OpMULHU(ExecContext& ctx, MulInputStruct arg0, BoundLayout<OpMULHULayout> layout1) {
  // OpMULHU(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:72)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:62)
  // OpMULHU(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:73)
  Val x3 = (x2.opcode._super - Val(51));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :62:19) at  OpMULHU "
      "( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :73:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:63)
  EQZ((x2.func3 - Val(3)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :63:18) at  OpMULHU "
      "( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :73:20)))");
  // VerifyOpcodeF3F7(zirgen/circuit/rv32im/v2/dsl/inst.zir:64)
  EQZ((x2.func7 - Val(1)),
      "loc(callsite( VerifyOpcodeF3F7 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :64:18) at  OpMULHU "
      "( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :73:20)))");
  // OpMULHU(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:74)
  DoMulStruct x4 =
      exec_DoMul(ctx, arg0.rs1._super, arg0.rs2._super, Val(0), Val(0), LAYOUT_LOOKUP(layout1, _0));
  return x4.high;
}
InstOutputStruct exec_Mul0(ExecContext& ctx,
                           NondetRegStruct arg0,
                           BoundLayout<NondetRegLayout> arg1,
                           InstInputStruct arg2,
                           BoundLayout<Mul0Layout> layout3) {
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
  BoundLayout<Mul0MulOutputLayout> x4 = LAYOUT_LOOKUP(layout3, mulOutput);
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:22)
  MulInputStruct x5 = exec_MulInput(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, input));
  NondetRegStruct8Array x6 = x5._super.minorOnehot._super;
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
  BoundLayout<Mul0Arm2Layout> x7 = LAYOUT_LOOKUP(x4, arm2);
  BoundLayout<Mul0Arm3Layout> x8 = LAYOUT_LOOKUP(x4, arm3);
  BoundLayout<Mul0Arm4Layout> x9 = LAYOUT_LOOKUP(x4, arm4);
  BoundLayout<Mul0Arm5Layout> x10 = LAYOUT_LOOKUP(x4, arm5);
  BoundLayout<Mul0Arm6Layout> x11 = LAYOUT_LOOKUP(x4, arm6);
  BoundLayout<Mul0Arm7Layout> x12 = LAYOUT_LOOKUP(x4, arm7);
  BoundLayout<Reg> x13 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra0), count), _super);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // IllegalMulOp(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:18)
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:30)
  ValU32Struct x17 = ValU32Struct{.low = Val(0), .high = Val(0)};
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra5), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra6), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra7), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra8), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra9), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra10), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra11), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra12), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra13), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra14), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra15), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra16), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra17), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra18), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra4), count), _super);
  BoundLayout<Reg> x42 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra5), count), _super);
  BoundLayout<Reg> x43 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra6), count), _super);
  BoundLayout<Reg> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra7), count), _super);
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra8), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra9), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra10), count), _super);
  BoundLayout<Reg> x48 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra11), count), _super);
  BoundLayout<Reg> x49 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra12), count), _super);
  BoundLayout<Reg> x50 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra13), count), _super);
  BoundLayout<Reg> x51 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra14), count), _super);
  BoundLayout<Reg> x52 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra15), count), _super);
  BoundLayout<Reg> x53 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra16), count), _super);
  BoundLayout<Reg> x54 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra17), count), _super);
  BoundLayout<Reg> x55 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra18), count), _super);
  ValU32Struct x56;
  if (to_size_t(x6[0]._super)) {
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:24)
    ValU32Struct x57 = exec_OpSLL(ctx, x5, LAYOUT_LOOKUP(x4, arm0));
    x56 = x57;
  } else if (to_size_t(x6[1]._super)) {
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:25)
    ValU32Struct x58 = exec_OpSLLI(ctx, x5, LAYOUT_LOOKUP(x4, arm1));
    x56 = x58;
  } else if (to_size_t(x6[2]._super)) {
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:26)
    ValU32Struct x59 = exec_OpMUL(ctx, x5, LAYOUT_LOOKUP(x7, _super));
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
    STORE(x13, Val(0));
    EQZ(LOAD(x13, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    x56 = x59;
  } else if (to_size_t(x6[3]._super)) {
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:27)
    ValU32Struct x60 = exec_OpMULH(ctx, x5, LAYOUT_LOOKUP(x8, _super));
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
    STORE(x14, Val(0));
    EQZ(LOAD(x14, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    x56 = x60;
  } else if (to_size_t(x6[4]._super)) {
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:28)
    ValU32Struct x61 = exec_OpMULHSU(ctx, x5, LAYOUT_LOOKUP(x9, _super));
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    x56 = x61;
  } else if (to_size_t(x6[5]._super)) {
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:29)
    ValU32Struct x62 = exec_OpMULHU(ctx, x5, LAYOUT_LOOKUP(x10, _super));
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    x56 = x62;
  } else if (to_size_t(x6[6]._super)) {
    // IllegalMulOp(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:17)
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:30)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalMulOp ( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :17:6) at  Mul0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :30:18)))");
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    x56 = x17;
  } else if (to_size_t(x6[7]._super)) {
    // IllegalMulOp(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:17)
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:31)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalMulOp ( zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :17:6) at  Mul0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mul.zir :31:18)))");
    // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x42, Val(0));
    EQZ(LOAD(x42, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x43, Val(0));
    EQZ(LOAD(x43, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x44, Val(0));
    EQZ(LOAD(x44, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x48, Val(0));
    EQZ(LOAD(x48, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x49, Val(0));
    EQZ(LOAD(x49, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x50, Val(0));
    EQZ(LOAD(x50, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x51, Val(0));
    EQZ(LOAD(x51, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x52, Val(0));
    EQZ(LOAD(x52, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x53, Val(0));
    EQZ(LOAD(x53, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x54, Val(0));
    EQZ(LOAD(x54, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    STORE(x55, Val(0));
    EQZ(LOAD(x55, 0), "Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:23)");
    x56 = x17;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:33)
  WriteRdStruct x63 =
      exec_WriteRd(ctx, arg0, arg1, x5.ii, x5.decoded, Val(1), x56, LAYOUT_LOOKUP(layout3, _0));
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:21)
  ValU32Struct x64 = arg2.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // Mul0(zirgen/circuit/rv32im/v2/dsl/inst_mul.zir:34)
  DenormedValU32Struct x65 = DenormedValU32Struct{.low = (x64.low + Val(4)), .high = x64.high};
  NormalizeU32Struct x66 = exec_NormalizeU32(ctx, x65, LAYOUT_LOOKUP(layout3, pcAdd));
  return InstOutputStruct{.newPc = x66._super, .newState = Val(32), .newMode = arg2.mode};
}
MemLoadInputStruct exec_MemLoadInput(ExecContext& ctx,
                                     NondetRegStruct arg0,
                                     BoundLayout<NondetRegLayout> arg1,
                                     InstInputStruct arg2,
                                     BoundLayout<MemLoadInputLayout> layout3) {
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:8)
  EQZ((arg2.state - Val(32)), "MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:8)");
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:10)
  DecoderStruct x4 = exec_DecodeInst(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, decoded));
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:11)
  GetDataStruct x5 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs1, LAYOUT_LOOKUP(layout3, rs1));
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:10)
  ValU32Struct x6 = x4.immI;
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:11)
  ValU32Struct x7 = x5._super;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:12)
  DenormedValU32Struct x8 =
      DenormedValU32Struct{.low = (x7.low + x6.low), .high = (x7.high + x6.high)};
  NormalizeU32Struct x9 = exec_NormalizeU32(ctx, x8, LAYOUT_LOOKUP(layout3, addrU32));
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:13)
  AddrDecomposeBitsStruct x10 =
      exec_AddrDecomposeBits(ctx, x9._super, arg2.mode, LAYOUT_LOOKUP(layout3, addr));
  // MemLoadInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:14)
  GetDataStruct x11 = exec_MemoryRead(ctx, arg0, arg1, x10.addr, LAYOUT_LOOKUP(layout3, data));
  return MemLoadInputStruct{.ii = arg2, .decoded = x4, .addr = x10, .data = x11};
}
MemStoreInputStruct exec_MemStoreInput(ExecContext& ctx,
                                       NondetRegStruct arg0,
                                       BoundLayout<NondetRegLayout> arg1,
                                       InstInputStruct arg2,
                                       BoundLayout<MemStoreInputLayout> layout3) {
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:18)
  EQZ((arg2.state - Val(32)), "MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:18)");
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:20)
  DecoderStruct x4 = exec_DecodeInst(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, decoded));
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:21)
  GetDataStruct x5 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs1, LAYOUT_LOOKUP(layout3, rs1));
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:22)
  GetDataStruct x6 = exec_ReadReg(ctx, arg0, arg1, arg2, x4.rs2, LAYOUT_LOOKUP(layout3, rs2));
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:20)
  ValU32Struct x7 = x4.immS;
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:21)
  ValU32Struct x8 = x5._super;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:23)
  DenormedValU32Struct x9 =
      DenormedValU32Struct{.low = (x8.low + x7.low), .high = (x8.high + x7.high)};
  NormalizeU32Struct x10 = exec_NormalizeU32(ctx, x9, LAYOUT_LOOKUP(layout3, addrU32));
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:24)
  AddrDecomposeBitsStruct x11 =
      exec_AddrDecomposeBits(ctx, x10._super, arg2.mode, LAYOUT_LOOKUP(layout3, addr));
  // MemStoreInput(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:25)
  GetDataStruct x12 = exec_MemoryRead(ctx, arg0, arg1, x11.addr, LAYOUT_LOOKUP(layout3, data));
  return MemStoreInputStruct{.decoded = x4, .rs2 = x6, .addr = x11, .data = x12};
}
MemStoreFinalizeStruct exec_MemStoreFinalize(ExecContext& ctx,
                                             NondetRegStruct arg0,
                                             BoundLayout<NondetRegLayout> arg1,
                                             MemStoreInputStruct arg2,
                                             ValU32Struct arg3,
                                             BoundLayout<MemStoreFinalizeLayout> layout4) {
  // MemStoreFinalize(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:29)
  MemoryWriteStruct x5 =
      exec_MemoryWrite(ctx, arg0, arg1, arg2.addr.addr, arg3, LAYOUT_LOOKUP(layout4, _0));
  return MemStoreFinalizeStruct{};
}
SplitWordStruct exec_SplitWord(ExecContext& ctx, Val arg0, BoundLayout<SplitWordLayout> layout1) {
  // SplitWord(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:33)
  NondetRegStruct x2 = exec_NondetU8Reg(ctx, bitAnd(arg0, Val(255)), LAYOUT_LOOKUP(layout1, byte0));
  // SplitWord(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:34)
  NondetRegStruct x3 = exec_NondetU8Reg(
      ctx, (bitAnd(arg0, Val(65280)) * Val(2005401601)), LAYOUT_LOOKUP(layout1, byte1));
  // SplitWord(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:35)
  Val x4 = ((x3._super * Val(256)) + x2._super);
  EQZ((arg0 - x4), "SplitWord(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:35)");
  return SplitWordStruct{.byte0 = x2, .byte1 = x3};
}
ValU32Struct exec_OpLB(ExecContext& ctx, MemLoadInputStruct arg0, BoundLayout<OpLBLayout> layout1) {
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:82)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:83)
  Val x3 = (x2.opcode._super - Val(3));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpLB ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :83:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ(x2.func3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpLB ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :83:18)))");
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:82)
  AddrDecomposeBitsStruct x4 = arg0.addr;
  ValU32Struct x5 = arg0.data._super;
  Val x6 = x4.low1._super;
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:84)
  Val x7 = ((Val(1) - x6) * x5.low);
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:85)
  SplitWordStruct x8 = exec_SplitWord(ctx, ((x6 * x5.high) + x7), LAYOUT_LOOKUP(layout1, bytes));
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:82)
  Val x9 = x4.low0._super;
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:86)
  Val x10 = ((Val(1) - x9) * x8.byte0._super);
  Val x11 = ((x9 * x8.byte1._super) + x10);
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:87)
  NondetRegStruct x12 = exec_NondetBitReg(
      ctx, (bitAnd(x11, Val(128)) * Val(1997537281)), LAYOUT_LOOKUP(layout1, highBit));
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:88)
  NondetRegStruct x13 =
      exec_NondetU8Reg(ctx, (bitAnd(x11, Val(127)) * Val(2)), LAYOUT_LOOKUP(layout1, low7x2));
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:87)
  Val x14 = x12._super;
  // OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:89)
  Val x15 = ((x14 * Val(128)) + (x13._super * Val(1006632961)));
  EQZ((x11 - x15), "OpLB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:89)");
  return ValU32Struct{.low = (x11 + (x14 * Val(65280))), .high = (x14 * Val(65535))};
}
ValU32Struct exec_OpLH(ExecContext& ctx, MemLoadInputStruct arg0, BoundLayout<OpLHLayout> layout1) {
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:93)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:94)
  Val x3 = (x2.opcode._super - Val(3));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpLH ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :94:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(1)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpLH ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :94:18)))");
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:93)
  AddrDecomposeBitsStruct x4 = arg0.addr;
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:95)
  EQZ(x4.low0._super, "OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:95)");
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:93)
  ValU32Struct x5 = arg0.data._super;
  Val x6 = x4.low1._super;
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:96)
  Val x7 = ((Val(1) - x6) * x5.low);
  Val x8 = ((x6 * x5.high) + x7);
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:97)
  NondetRegStruct x9 = exec_NondetBitReg(
      ctx, (bitAnd(x8, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout1, highBit));
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:98)
  NondetRegStruct x10 =
      exec_NondetU8Reg(ctx, (bitAnd(x8, Val(32767)) * Val(2)), LAYOUT_LOOKUP(layout1, low15x2));
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:97)
  Val x11 = x9._super;
  // OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:99)
  Val x12 = ((x11 * Val(32768)) + (x10._super * Val(1006632961)));
  EQZ((x8 - x12), "OpLH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:99)");
  return ValU32Struct{.low = x8, .high = (x11 * Val(65535))};
}
ValU32Struct
exec_OpLBU(ExecContext& ctx, MemLoadInputStruct arg0, BoundLayout<OpLBULayout> layout1) {
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:110)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:111)
  Val x3 = (x2.opcode._super - Val(3));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpLBU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :111:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ((x2.func3 - Val(4)),
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpLBU ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :111:18)))");
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:110)
  AddrDecomposeBitsStruct x4 = arg0.addr;
  ValU32Struct x5 = arg0.data._super;
  Val x6 = x4.low1._super;
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:112)
  Val x7 = ((Val(1) - x6) * x5.low);
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:113)
  SplitWordStruct x8 = exec_SplitWord(ctx, ((x6 * x5.high) + x7), LAYOUT_LOOKUP(layout1, bytes));
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:110)
  Val x9 = x4.low0._super;
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:114)
  Val x10 = ((Val(1) - x9) * x8.byte0._super);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // OpLBU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:115)
  ValU32Struct x11 = ValU32Struct{.low = ((x9 * x8.byte1._super) + x10), .high = Val(0)};
  return x11;
}
InstOutputStruct exec_Mem0(ExecContext& ctx,
                           NondetRegStruct arg0,
                           BoundLayout<NondetRegLayout> arg1,
                           InstInputStruct arg2,
                           BoundLayout<Mem0Layout> layout3) {
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
  BoundLayout<Mem0OutputLayout> x4 = LAYOUT_LOOKUP(layout3, output);
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:49)
  MemLoadInputStruct x5 = exec_MemLoadInput(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, input));
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:48)
  NondetRegStruct8Array x6 = arg2.minorOnehot._super;
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
  BoundLayout<Mem0Arm1Layout> x7 = LAYOUT_LOOKUP(x4, arm1);
  BoundLayout<Mem0Arm2Layout> x8 = LAYOUT_LOOKUP(x4, arm2);
  BoundLayout<Mem0Arm3Layout> x9 = LAYOUT_LOOKUP(x4, arm3);
  BoundLayout<Mem0Arm4Layout> x10 = LAYOUT_LOOKUP(x4, arm4);
  BoundLayout<Mem0Arm5Layout> x11 = LAYOUT_LOOKUP(x4, arm5);
  BoundLayout<Mem0Arm6Layout> x12 = LAYOUT_LOOKUP(x4, arm6);
  BoundLayout<Mem0Arm7Layout> x13 = LAYOUT_LOOKUP(x4, arm7);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra0), count), _super);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra1), count), _super);
  // OpLW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:103)
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:53)
  DecoderStruct x16 = x5.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpLW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:104)
  Val x17 = (x16.opcode._super - Val(3));
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:56)
  Val x18 = x16.func3;
  // OpLW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:103)
  AddrDecomposeBitsStruct x19 = x5.addr;
  Val x20 = x19.low0._super;
  Val x21 = x19.low1._super;
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  ValU32Struct x25 = x5.data._super;
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  // OpLHU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:121)
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:55)
  Val x27 = ((Val(1) - x21) * x25.low);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // OpLHU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:122)
  ValU32Struct x28 = ValU32Struct{.low = ((x21 * x25.high) + x27), .high = Val(0)};
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra2), count), _super);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // IllegalLoadOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:40)
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:56)
  ValU32Struct x32 = ValU32Struct{.low = Val(0), .high = Val(0)};
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra0), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra1), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra2), count), _super);
  ValU32Struct x42;
  if (to_size_t(x6[0]._super)) {
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:51)
    ValU32Struct x43 = exec_OpLB(ctx, x5, LAYOUT_LOOKUP(x4, arm0));
    x42 = x43;
  } else if (to_size_t(x6[1]._super)) {
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:52)
    ValU32Struct x44 = exec_OpLH(ctx, x5, LAYOUT_LOOKUP(x7, _super));
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
    STORE(x14, Val(0));
    EQZ(LOAD(x14, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    x42 = x44;
  } else if (to_size_t(x6[2]._super)) {
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
    // OpLW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:104)
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:53)
    EQZ(x17,
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at callsite( "
        "OpLW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :104:18) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :53:10))))");
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
    EQZ((x18 - Val(2)),
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at callsite( "
        "OpLW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :104:18) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :53:10))))");
    // OpLW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:105)
    EQZ(x20,
        "loc(callsite( OpLW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :105:20) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :53:10)))");
    // OpLW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:106)
    EQZ(x21,
        "loc(callsite( OpLW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :106:20) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :53:10)))");
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    x42 = x25;
  } else if (to_size_t(x6[3]._super)) {
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:54)
    ValU32Struct x45 = exec_OpLBU(ctx, x5, LAYOUT_LOOKUP(x9, _super));
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    x42 = x45;
  } else if (to_size_t(x6[4]._super)) {
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
    // OpLHU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:119)
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:55)
    EQZ(x17,
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at callsite( "
        "OpLHU ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :119:18) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :55:11))))");
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
    EQZ((x18 - Val(5)),
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at callsite( "
        "OpLHU ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :119:18) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :55:11))))");
    // OpLHU(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:120)
    EQZ(x20,
        "loc(callsite( OpLHU ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :120:20) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :55:11)))");
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    x42 = x28;
  } else if (to_size_t(x6[5]._super)) {
    // IllegalLoadOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:39)
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:56)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalLoadOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :39:6) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :56:19)))");
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    x42 = x32;
  } else if (to_size_t(x6[6]._super)) {
    // IllegalLoadOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:39)
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:57)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalLoadOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :39:6) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :57:19)))");
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    x42 = x32;
  } else if (to_size_t(x6[7]._super)) {
    // IllegalLoadOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:39)
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:58)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalLoadOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :39:6) at  Mem0 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :58:19)))");
    // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:50)");
    x42 = x32;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:60)
  WriteRdStruct x46 =
      exec_WriteRd(ctx, arg0, arg1, x5.ii, x16, Val(1), x42, LAYOUT_LOOKUP(layout3, _0));
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:48)
  ValU32Struct x47 = arg2.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // Mem0(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:61)
  DenormedValU32Struct x48 = DenormedValU32Struct{.low = (x47.low + Val(4)), .high = x47.high};
  NormalizeU32Struct x49 = exec_NormalizeU32(ctx, x48, LAYOUT_LOOKUP(layout3, pcAdd));
  return InstOutputStruct{.newPc = x49._super, .newState = Val(32), .newMode = arg2.mode};
}
ValU32Struct
exec_OpSB(ExecContext& ctx, MemStoreInputStruct arg0, BoundLayout<OpSBLayout> layout1) {
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:125)
  DecoderStruct x2 = arg0.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:126)
  Val x3 = (x2.opcode._super - Val(35));
  EQZ(x3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at  OpSB ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :126:18)))");
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
  EQZ(x2.func3,
      "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at  OpSB ( "
      "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :126:18)))");
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:125)
  AddrDecomposeBitsStruct x4 = arg0.addr;
  ValU32Struct x5 = arg0.data._super;
  Val x6 = x5.high;
  Val x7 = x4.low1._super;
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:127)
  Val x8 = (Val(1) - x7);
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:125)
  Val x9 = x5.low;
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:128)
  SplitWordStruct x10 =
      exec_SplitWord(ctx, ((x7 * x6) + (x8 * x9)), LAYOUT_LOOKUP(layout1, origBytes));
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:125)
  Val x11 = arg0.rs2._super.low;
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:129)
  SplitWordStruct x12 = exec_SplitWord(ctx, x11, LAYOUT_LOOKUP(layout1, newBytes));
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:125)
  Val x13 = x4.low0._super;
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:134)
  Val x14 = (Val(1) - x13);
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:129)
  Val x15 = x12.byte0._super;
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:135)
  Val x16 = (((x14 * x10.byte1._super) + (x13 * x15)) * Val(256));
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:134)
  Val x17 = (((x13 * x10.byte0._super) + (x14 * x15)) + x16);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // OpSB(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:137)
  ValU32Struct x18 =
      ValU32Struct{.low = ((x7 * x9) + (x8 * x17)), .high = ((x8 * x6) + (x7 * x17))};
  return x18;
}
InstOutputStruct exec_Mem1(ExecContext& ctx,
                           NondetRegStruct arg0,
                           BoundLayout<NondetRegLayout> arg1,
                           InstInputStruct arg2,
                           BoundLayout<Mem1Layout> layout3) {
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
  BoundLayout<Mem1OutputLayout> x4 = LAYOUT_LOOKUP(layout3, output);
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:66)
  MemStoreInputStruct x5 = exec_MemStoreInput(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, input));
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:65)
  NondetRegStruct8Array x6 = arg2.minorOnehot._super;
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
  BoundLayout<Mem1Arm1Layout> x7 = LAYOUT_LOOKUP(x4, arm1);
  BoundLayout<Mem1Arm2Layout> x8 = LAYOUT_LOOKUP(x4, arm2);
  BoundLayout<Mem1Arm3Layout> x9 = LAYOUT_LOOKUP(x4, arm3);
  BoundLayout<Mem1Arm4Layout> x10 = LAYOUT_LOOKUP(x4, arm4);
  BoundLayout<Mem1Arm5Layout> x11 = LAYOUT_LOOKUP(x4, arm5);
  BoundLayout<Mem1Arm6Layout> x12 = LAYOUT_LOOKUP(x4, arm6);
  BoundLayout<Mem1Arm7Layout> x13 = LAYOUT_LOOKUP(x4, arm7);
  // OpSH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:143)
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:69)
  DecoderStruct x14 = x5.decoded;
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
  // OpSH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:144)
  Val x15 = (x14.opcode._super - Val(35));
  // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:56)
  Val x16 = x14.func3;
  // OpSH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:143)
  AddrDecomposeBitsStruct x17 = x5.addr;
  Val x18 = x17.low0._super;
  ValU32Struct x19 = x5.rs2._super;
  Val x20 = x19.low;
  ValU32Struct x21 = x5.data._super;
  Val x22 = x17.low1._super;
  // OpSH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:150)
  Val x23 = (Val(1) - x22);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // OpSH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:149)
  ValU32Struct x24 = ValU32Struct{.low = ((x22 * x21.low) + (x23 * x20)),
                                  .high = ((x23 * x21.high) + (x22 * x20))};
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra0), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra1), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra2), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _extra3), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra3), count), _super);
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // IllegalStoreOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:45)
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:71)
  ValU32Struct x33 = ValU32Struct{.low = Val(0), .high = Val(0)};
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra1), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra2), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra3), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra2), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra3), count), _super);
  BoundLayout<Reg> x42 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x43 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x48 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x49 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x50 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra0), count), _super);
  BoundLayout<Reg> x51 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra1), count), _super);
  BoundLayout<Reg> x52 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra2), count), _super);
  BoundLayout<Reg> x53 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra3), count), _super);
  ValU32Struct x54;
  if (to_size_t(x6[0]._super)) {
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:68)
    ValU32Struct x55 = exec_OpSB(ctx, x5, LAYOUT_LOOKUP(x4, arm0));
    x54 = x55;
  } else if (to_size_t(x6[1]._super)) {
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
    // OpSH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:144)
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:69)
    EQZ(x15,
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at callsite( "
        "OpSH ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :144:18) at  Mem1 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :69:10))))");
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
    EQZ((x16 - Val(1)),
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at callsite( "
        "OpSH ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :144:18) at  Mem1 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :69:10))))");
    // OpSH(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:145)
    EQZ(x18,
        "loc(callsite( OpSH ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :145:20) at  Mem1 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :69:10)))");
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    x54 = x24;
  } else if (to_size_t(x6[2]._super)) {
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:57)
    // OpSW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:156)
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:70)
    EQZ(x15,
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :57:19) at callsite( "
        "OpSW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :156:18) at  Mem1 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :70:10))))");
    // VerifyOpcodeF3(zirgen/circuit/rv32im/v2/dsl/inst.zir:58)
    EQZ((x16 - Val(2)),
        "loc(callsite( VerifyOpcodeF3 ( zirgen/circuit/rv32im/v2/dsl/inst.zir :58:18) at callsite( "
        "OpSW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :156:18) at  Mem1 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :70:10))))");
    // OpSW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:157)
    EQZ(x18,
        "loc(callsite( OpSW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :157:20) at  Mem1 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :70:10)))");
    // OpSW(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:158)
    EQZ(x22,
        "loc(callsite( OpSW ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :158:20) at  Mem1 ( "
        "zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :70:10)))");
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    x54 = x19;
  } else if (to_size_t(x6[3]._super)) {
    // IllegalStoreOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:44)
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:71)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalStoreOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :44:6) at  Mem1 "
        "( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :71:20)))");
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    x54 = x33;
  } else if (to_size_t(x6[4]._super)) {
    // IllegalStoreOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:44)
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:72)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalStoreOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :44:6) at  Mem1 "
        "( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :72:20)))");
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    x54 = x33;
  } else if (to_size_t(x6[5]._super)) {
    // IllegalStoreOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:44)
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:73)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalStoreOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :44:6) at  Mem1 "
        "( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :73:20)))");
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
    STORE(x42, Val(0));
    EQZ(LOAD(x42, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x43, Val(0));
    EQZ(LOAD(x43, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x44, Val(0));
    EQZ(LOAD(x44, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    x54 = x33;
  } else if (to_size_t(x6[6]._super)) {
    // IllegalStoreOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:44)
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:74)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalStoreOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :44:6) at  Mem1 "
        "( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :74:20)))");
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x48, Val(0));
    EQZ(LOAD(x48, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x49, Val(0));
    EQZ(LOAD(x49, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    x54 = x33;
  } else if (to_size_t(x6[7]._super)) {
    // IllegalStoreOp(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:44)
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:75)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalStoreOp ( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :44:6) at  Mem1 "
        "( zirgen/circuit/rv32im/v2/dsl/inst_mem.zir :75:20)))");
    // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)
    STORE(x50, Val(0));
    EQZ(LOAD(x50, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x51, Val(0));
    EQZ(LOAD(x51, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x52, Val(0));
    EQZ(LOAD(x52, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    STORE(x53, Val(0));
    EQZ(LOAD(x53, 0), "Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:67)");
    x54 = x33;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:77)
  MemStoreFinalizeStruct x56 =
      exec_MemStoreFinalize(ctx, arg0, arg1, x5, x54, LAYOUT_LOOKUP(layout3, _0));
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:65)
  ValU32Struct x57 = arg2.pcU32;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // Mem1(zirgen/circuit/rv32im/v2/dsl/inst_mem.zir:78)
  DenormedValU32Struct x58 = DenormedValU32Struct{.low = (x57.low + Val(4)), .high = x57.high};
  NormalizeU32Struct x59 = exec_NormalizeU32(ctx, x58, LAYOUT_LOOKUP(layout3, pcAdd));
  return InstOutputStruct{.newPc = x59._super, .newState = Val(32), .newMode = arg2.mode};
}
DigestRegStruct
back_DigestReg(ExecContext& ctx, Index distance0, BoundLayout<DigestRegLayout> layout1) {
  // DigestReg(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:7)
  DigestRegValues_SuperStruct8Array x2 =
      map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
          LAYOUT_LOOKUP(layout1, values),
          std::function([&](Val8Array::value_type x3,
                            BoundLayout<DigestRegValues_SuperLayout8LayoutArray::value_type> x4) {
            // DigestReg(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:8)
            NondetRegStruct x5 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(x4, low));
            // DigestReg(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:9)
            NondetRegStruct x6 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(x4, high));
            return DigestRegValues_SuperStruct{.low = x5, .high = x6};
          }));
  return DigestRegStruct{.values = x2};
}
DigestRegStruct
exec_DigestReg(ExecContext& ctx, ValU32Struct8Array arg0, BoundLayout<DigestRegLayout> layout1) {
  // DigestReg(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:7)
  DigestRegValues_SuperStruct8Array x2 =
      map(arg0,
          LAYOUT_LOOKUP(layout1, values),
          std::function([&](ValU32Struct8Array::value_type x3,
                            BoundLayout<DigestRegValues_SuperLayout8LayoutArray::value_type> x4) {
            // DigestReg(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:8)
            NondetRegStruct x5 = exec_Reg(ctx, x3.low, LAYOUT_LOOKUP(x4, low));
            // DigestReg(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:9)
            NondetRegStruct x6 = exec_Reg(ctx, x3.high, LAYOUT_LOOKUP(x4, high));
            return DigestRegValues_SuperStruct{.low = x5, .high = x6};
          }));
  return DigestRegStruct{.values = x2};
}
InstOutputStruct exec_ControlLoadRoot(ExecContext& ctx,
                                      NondetRegStruct arg0,
                                      BoundLayout<NondetRegLayout> arg1,
                                      InstInputStruct arg2,
                                      BoundLayout<ControlLoadRootLayout> layout3,
                                      GlobalBuf global4) {
  // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:18)
  BoundLayout<_globalLayout> x5 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:19)
  BoundLayout<DigestRegLayout> x6 = LAYOUT_LOOKUP(x5, stateIn);
  // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:20)
  EQZ(arg2.state, "ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:20)");
  // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:22)
  ControlLoadRoot__0Struct8Array x7 = map(
      Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
      LAYOUT_LOOKUP(layout3, _0),
      std::function([&](Val8Array::value_type x8,
                        BoundLayout<ControlLoadRoot__0_SuperLayout8LayoutArray::value_type> x9) {
        // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:23)
        GetDataStruct x10 =
            exec_MemoryPageIn(ctx, arg0, arg1, (x8 + Val(1140850680)), LAYOUT_LOOKUP(x9, mem));
        // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:19)
        DigestRegStruct x11 = back_DigestReg(ctx, 0, x6);
        NondetRegStruct x12 = x11.values[to_size_t(x8)].low;
        // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:23)
        ValU32Struct x13 = x10._super;
        // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:24)
        EQZ((x12._super - x13.low),
            "ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:24)");
        // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:19)
        DigestRegStruct x14 = back_DigestReg(ctx, 0, x6);
        NondetRegStruct x15 = x14.values[to_size_t(x8)].high;
        // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:25)
        EQZ((x15._super - x13.high),
            "ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:25)");
        return ControlLoadRoot__0Struct{};
      }));
  // InstOutput(zirgen/circuit/rv32im/v2/dsl/inst.zir:46)
  // ControlLoadRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:27)
  InstOutputStruct x16 = InstOutputStruct{
      .newPc = ValU32Struct{.low = Val(0), .high = Val(0)}, .newState = Val(16), .newMode = Val(0)};
  return x16;
}
InstOutputStruct exec_ControlResume(ExecContext& ctx,
                                    NondetRegStruct arg0,
                                    BoundLayout<NondetRegLayout> arg1,
                                    InstInputStruct arg2,
                                    BoundLayout<ControlResumeLayout> layout3,
                                    GlobalBuf global4) {
  // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:30)
  BoundLayout<_globalLayout> x5 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)
  BoundLayout<ControlResume_SuperLayout> x6 = LAYOUT_LOOKUP(layout3, _super);
  // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:31)
  EQZ((arg2.state - Val(1)), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:31)");
  // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:30)
  ValU32Struct x7 = arg2.pcU32;
  // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:33)
  NondetRegStruct x8 = exec_IsZero(ctx, (x7.low + x7.high), LAYOUT_LOOKUP(layout3, pcZero));
  Val x9 = x8._super;
  // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)
  BoundLayout<ControlResumeArm0Layout> x10 = LAYOUT_LOOKUP(x6, arm0);
  BoundLayout<ControlResumeArm0_SuperLayout> x11 = LAYOUT_LOOKUP(x10, _super);
  BoundLayout<Reg> x12 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x13 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra2), count), _super);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra3), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra4), count), _super);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra5), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra6), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra7), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra8), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra9), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra10), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra11), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra12), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra13), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra14), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra15), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra16), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra17), count), _super);
  InstOutputStruct x30;
  if (to_size_t(x9)) {
    // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:36)
    GetDataStruct x31 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725572), LAYOUT_LOOKUP(x11, pc));
    // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:37)
    GetDataStruct x32 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725573), LAYOUT_LOOKUP(x11, mode));
    // InstOutput(zirgen/circuit/rv32im/v2/dsl/inst.zir:46)
    // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:38)
    InstOutputStruct x33 =
        InstOutputStruct{.newPc = x31._super, .newState = Val(1), .newMode = x32._super.low};
    // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)
    STORE(x12, Val(0));
    EQZ(LOAD(x12, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x13, Val(0));
    EQZ(LOAD(x13, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x14, Val(0));
    EQZ(LOAD(x14, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:34)");
    x30 = x33;
  } else if (to_size_t((Val(1) - x9))) {
    // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:41)
    ControlResumeArm1_Super__0Struct8Array x34 = map(
        Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, arm1), _0),
        std::function(
            [&](Val8Array::value_type x35,
                BoundLayout<ControlResumeArm1_Super__0_SuperLayout8LayoutArray::value_type> x36) {
              // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:40)
              DigestRegStruct x37 = back_DigestReg(ctx, 0, LAYOUT_LOOKUP(x5, input));
              NondetRegStruct x38 = x37.values[to_size_t(x35)].low;
              NondetRegStruct x39 = x37.values[to_size_t(x35)].high;
              // ControlResume(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:42)
              MemoryWriteStruct x40 =
                  exec_MemoryWrite(ctx,
                                   arg0,
                                   arg1,
                                   (x35 + Val(1073725592)),
                                   ValU32Struct{.low = x38._super, .high = x39._super},
                                   LAYOUT_LOOKUP(x36, _0));
              return ControlResumeArm1_Super__0Struct{};
            }));
    x30 = InstOutputStruct{.newPc = x7, .newState = Val(32), .newMode = arg2.mode};
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  return x30;
}
InstOutputStruct exec_ControlUserECALL(ExecContext& ctx,
                                       NondetRegStruct arg0,
                                       BoundLayout<NondetRegLayout> arg1,
                                       InstInputStruct arg2,
                                       BoundLayout<ControlUserECALLLayout> layout3) {
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:49)
  Val x4 = arg2.mode;
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:50)
  NondetRegStruct x5 = exec_Reg(ctx, x4, LAYOUT_LOOKUP(layout3, safeMode));
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:49)
  ValU32Struct x6 = arg2.pcU32;
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:51)
  AddrDecomposeBitsStruct x7 =
      exec_AddrDecomposeBits(ctx, x6, x5._super, LAYOUT_LOOKUP(layout3, pcAddr));
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:52)
  EQZ(x7.low2, "ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:52)");
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:53)
  GetDataStruct x8 = exec_MemoryRead(ctx, arg0, arg1, x7._super, LAYOUT_LOOKUP(layout3, loadInst));
  ValU32Struct x9 = x8._super;
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:54)
  EQZ(x9.high, "ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:54)");
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:55)
  EQZ((x9.low - Val(115)), "ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:55)");
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:56)
  EQZ((arg2.state - Val(32)), "ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:56)");
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:57)
  EQZ(x4, "ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:57)");
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:58)
  GetDataStruct x10 =
      exec_MemoryRead(ctx, arg0, arg1, Val(1073725489), LAYOUT_LOOKUP(layout3, dispatchIdx));
  ValU32Struct x11 = x10._super;
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:59)
  EQZ(x11.high, "ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:59)");
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:58)
  Val x12 = x11.low;
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:60)
  U16RegStruct x13 = exec_U16Reg(ctx, (x12 * Val(128)), LAYOUT_LOOKUP(layout3, _0));
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:61)
  GetDataStruct x14 =
      exec_MemoryRead(ctx, arg0, arg1, (x12 + Val(1073726464)), LAYOUT_LOOKUP(layout3, newPcAddr));
  // ControlUserECALL(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:62)
  MemoryWriteStruct x15 =
      exec_MemoryWrite(ctx, arg0, arg1, Val(1073725568), x6, LAYOUT_LOOKUP(layout3, _1));
  return InstOutputStruct{.newPc = x14._super, .newState = Val(32), .newMode = Val(1)};
}
InstOutputStruct exec_ControlMRET(ExecContext& ctx,
                                  NondetRegStruct arg0,
                                  BoundLayout<NondetRegLayout> arg1,
                                  InstInputStruct arg2,
                                  BoundLayout<ControlMRETLayout> layout3) {
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:66)
  Val x4 = arg2.mode;
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:67)
  NondetRegStruct x5 = exec_Reg(ctx, x4, LAYOUT_LOOKUP(layout3, safeMode));
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:68)
  AddrDecomposeBitsStruct x6 =
      exec_AddrDecomposeBits(ctx, arg2.pcU32, x5._super, LAYOUT_LOOKUP(layout3, pcAddr));
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:69)
  EQZ(x6.low2, "ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:69)");
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:70)
  GetDataStruct x7 = exec_MemoryRead(ctx, arg0, arg1, x6._super, LAYOUT_LOOKUP(layout3, loadInst));
  ValU32Struct x8 = x7._super;
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:71)
  EQZ((x8.high - Val(12320)), "ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:71)");
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:72)
  EQZ((x8.low - Val(115)), "ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:72)");
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:73)
  EQZ((arg2.state - Val(32)), "ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:73)");
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:74)
  EQZ((x4 - Val(1)), "ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:74)");
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:75)
  GetDataStruct x9 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725568), LAYOUT_LOOKUP(layout3, pc));
  ValU32Struct x10 = x9._super;
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  // ControlMRET(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:76)
  DenormedValU32Struct x11 = DenormedValU32Struct{.low = (x10.low + Val(4)), .high = x10.high};
  NormalizeU32Struct x12 = exec_NormalizeU32(ctx, x11, LAYOUT_LOOKUP(layout3, pcAdd));
  return InstOutputStruct{.newPc = x12._super, .newState = Val(32), .newMode = Val(0)};
}
InstOutputStruct exec_ControlSuspend(ExecContext& ctx,
                                     NondetRegStruct arg0,
                                     BoundLayout<NondetRegLayout> arg1,
                                     InstInputStruct arg2,
                                     BoundLayout<ControlSuspendLayout> layout3,
                                     GlobalBuf global4) {
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:80)
  BoundLayout<_globalLayout> x5 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)
  BoundLayout<ControlSuspend_SuperLayout> x6 = LAYOUT_LOOKUP(layout3, _super);
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:80)
  Val x7 = arg2.state;
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:81)
  EQZ((x7 - Val(4)), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:81)");
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:80)
  ValU32Struct x8 = arg2.pcU32;
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:83)
  NondetRegStruct x9 = exec_IsZero(ctx, (x8.low + x8.high), LAYOUT_LOOKUP(layout3, pcZero));
  Val x10 = x9._super;
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)
  BoundLayout<ControlSuspendArm1Layout> x11 = LAYOUT_LOOKUP(x6, arm1);
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:86)
  BoundLayout<NondetRegLayout> x12 = LAYOUT_LOOKUP(x5, isTerminate);
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:91)
  ComponentStruct x13 = ComponentStruct{};
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:103)
  ValU32Struct x14 = ValU32Struct{.low = Val(0), .high = Val(0)};
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:104)
  BoundLayout<ControlSuspendArm1_SuperLayout> x15 = LAYOUT_LOOKUP(x11, _super);
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:80)
  Val x16 = arg2.mode;
  // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra5), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra6), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra7), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra8), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra9), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra10), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra11), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra12), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra13), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra14), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra15), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra16), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra17), count), _super);
  InstOutputStruct x35;
  if (to_size_t(x10)) {
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:89)
    GetDataStruct8Array x36 =
        map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
            LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, arm0), _0),
            std::function([&](Val8Array::value_type x37,
                              BoundLayout<MemoryReadLayout8LayoutArray::value_type> x38) {
              GetDataStruct x39 = exec_MemoryRead(ctx, arg0, arg1, (x37 + Val(1073725584)), x38);
              return x39;
            }));
    ValU32Struct8Array x40 = ValU32Struct8Array{x36[0]._super,
                                                x36[1]._super,
                                                x36[2]._super,
                                                x36[3]._super,
                                                x36[4]._super,
                                                x36[5]._super,
                                                x36[6]._super,
                                                x36[7]._super};
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:88)
    DigestRegStruct x41 = exec_DigestReg(ctx, x40, LAYOUT_LOOKUP(x5, output));
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:86)
    NondetRegStruct x42 = back_Reg(ctx, 0, x12);
    NondetRegStruct x43 = back_Reg(ctx, 0, x12);
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:91)
    Val x44 = (Val(1) - (Val(1) - x43._super));
    ComponentStruct x45;
    if (to_size_t((Val(1) - x42._super))) {
      // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:92)
      NondetRegStruct x46 = exec_Reg(ctx, Val(0), LAYOUT_LOOKUP(x5, termA0low));
      // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:93)
      NondetRegStruct x47 = exec_Reg(ctx, Val(0), LAYOUT_LOOKUP(x5, termA0high));
      // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:94)
      NondetRegStruct x48 = exec_Reg(ctx, Val(0), LAYOUT_LOOKUP(x5, termA1low));
      // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:95)
      NondetRegStruct x49 = exec_Reg(ctx, Val(0), LAYOUT_LOOKUP(x5, termA1high));
      x45 = x13;
    } else if (to_size_t(x44)) {
      x45 = x13;
    } else {
      assert(0 && "Reached unreachable mux arm");
    }
    x35 = InstOutputStruct{.newPc = x14, .newState = Val(16), .newMode = Val(3)};
  } else if (to_size_t((Val(1) - x10))) {
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:107)
    NondetRegStruct x50 = exec_Reg(ctx, x7, LAYOUT_LOOKUP(x15, state));
    Val x51 = x50._super;
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:108)
    Val x52 = (x51 - Val(32));
    EQZ((x52 * (x51 - Val(4))),
        "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:108)");
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:111)
    NondetRegStruct x53 = exec_Reg(ctx, (x52 * Val(1797558858)), x12);
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:113)
    MemoryWriteStruct x54 =
        exec_MemoryWrite(ctx, arg0, arg1, Val(1073725572), x8, LAYOUT_LOOKUP(x15, _0));
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:114)
    MemoryWriteStruct x55 = exec_MemoryWrite(ctx,
                                             arg0,
                                             arg1,
                                             Val(1073725573),
                                             ValU32Struct{.low = x16, .high = Val(0)},
                                             LAYOUT_LOOKUP(x15, _1));
    // ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "ControlSuspend(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:84)");
    x35 = InstOutputStruct{.newPc = x14, .newState = Val(4), .newMode = x16};
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  return x35;
}
InstOutputStruct exec_ControlStoreRoot(ExecContext& ctx,
                                       NondetRegStruct arg0,
                                       BoundLayout<NondetRegLayout> arg1,
                                       InstInputStruct arg2,
                                       BoundLayout<ControlStoreRootLayout> layout3,
                                       GlobalBuf global4) {
  // ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:120)
  BoundLayout<_globalLayout> x5 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:121)
  EQZ((arg2.state - Val(5)), "ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:121)");
  // ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:123)
  GetDataStruct8Array x6 =
      map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
          LAYOUT_LOOKUP(layout3, _0),
          std::function([&](Val8Array::value_type x7,
                            BoundLayout<MemoryPageOutLayout8LayoutArray::value_type> x8) {
            // ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:124)
            GetDataStruct x9 = exec_MemoryPageOut(ctx, arg0, arg1, (x7 + Val(1140850680)), x8);
            return x9;
          }));
  // ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:123)
  ValU32Struct8Array x10 = ValU32Struct8Array{x6[0]._super,
                                              x6[1]._super,
                                              x6[2]._super,
                                              x6[3]._super,
                                              x6[4]._super,
                                              x6[5]._super,
                                              x6[6]._super,
                                              x6[7]._super};
  // ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:122)
  DigestRegStruct x11 = exec_DigestReg(ctx, x10, LAYOUT_LOOKUP(x5, stateOut));
  // InstOutput(zirgen/circuit/rv32im/v2/dsl/inst.zir:46)
  // ControlStoreRoot(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:127)
  InstOutputStruct x12 = InstOutputStruct{
      .newPc = ValU32Struct{.low = Val(0), .high = Val(0)}, .newState = Val(6), .newMode = Val(0)};
  return x12;
}
InstOutputStruct exec_ControlTable(ExecContext& ctx,
                                   NondetRegStruct arg0,
                                   InstInputStruct arg1,
                                   BoundLayout<ControlTableLayout> layout2) {
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)
  BoundLayout<ControlTable_SuperLayout> x3 = LAYOUT_LOOKUP(layout2, _super);
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:131)
  EQZ((arg1.state - Val(6)), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:131)");
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:132)
  NondetRegStruct x4 = exec_Reg(ctx, arg1.pcU32.low, LAYOUT_LOOKUP(layout2, entry));
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:133)
  NondetRegStruct x5 = exec_Reg(ctx, arg1.mode, LAYOUT_LOOKUP(layout2, mode));
  Val x6 = x5._super;
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:132)
  Val x7 = x4._super;
  // Log(<preamble>:22)
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:134)
  INVOKE_EXTERN(ctx, log, "mode/entry = ", std::initializer_list<Val>{x6, x7});
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)
  BoundLayout<ControlTableArm0Layout> x8 = LAYOUT_LOOKUP(x3, arm0);
  BoundLayout<ControlTableArm1Layout> x9 = LAYOUT_LOOKUP(x3, arm1);
  BoundLayout<ControlTableArm0_SuperLayout> x10 = LAYOUT_LOOKUP(x8, _super);
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:137)
  Val16Array x11 = Val16Array{Val(0),
                              Val(1),
                              Val(2),
                              Val(3),
                              Val(4),
                              Val(5),
                              Val(6),
                              Val(7),
                              Val(8),
                              Val(9),
                              Val(10),
                              Val(11),
                              Val(12),
                              Val(13),
                              Val(14),
                              Val(15)};
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:142)
  Val x12 = (x7 + Val(16));
  // ValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:10)
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:145)
  ValU32Struct x13 = ValU32Struct{.low = Val(0), .high = Val(0)};
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:147)
  ValU32Struct x14 = ValU32Struct{.low = x12, .high = Val(0)};
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra3), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra4), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra5), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra6), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra7), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra8), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra9), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra10), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra11), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra12), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra13), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra14), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra15), count), _super);
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:149)
  BoundLayout<ControlTableArm1_SuperLayout> x31 = LAYOUT_LOOKUP(x9, _super);
  // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra1), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra2), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra3), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra4), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra5), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra6), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra7), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra8), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra9), count), _super);
  BoundLayout<Reg> x42 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra10), count), _super);
  BoundLayout<Reg> x43 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra11), count), _super);
  BoundLayout<Reg> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra12), count), _super);
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra13), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra14), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra15), count), _super);
  InstOutputStruct x48;
  if (to_size_t(x6)) {
    // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:137)
    ControlTableArm0_Super__0Struct16Array x49 = map(
        x11,
        LAYOUT_LOOKUP(x10, _0),
        std::function(
            [&](Val16Array::value_type x50,
                BoundLayout<ControlTableArm0_Super__0_SuperLayout16LayoutArray::value_type> x51) {
              // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:138)
              Val x52 = (x7 + x50);
              // LookupCurrent(zirgen/circuit/rv32im/v2/dsl/lookups.zir:5)
              // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:139)
              Val x53 = INVOKE_EXTERN(ctx, lookupCurrent, Val(16), x52);
              ArgU16Struct x54 = exec_ArgU16(ctx, neg_0(x53), x52, LAYOUT_LOOKUP(x51, arg));
              // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:140)
              Val x55 = (x54.val._super - x52);
              EQZ(x55, "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:140)");
              return ControlTableArm0_Super__0Struct{};
            }));
    // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:143)
    NondetRegStruct x56 = exec_IsZero(ctx, (x12 - Val(65536)), LAYOUT_LOOKUP(x10, done));
    Val x57 = x56._super;
    InstOutputStruct x58;
    if (to_size_t(x57)) {
      x58 = InstOutputStruct{.newPc = x13, .newState = Val(7), .newMode = Val(0)};
    } else if (to_size_t((Val(1) - x57))) {
      x58 = InstOutputStruct{.newPc = x14, .newState = Val(6), .newMode = Val(1)};
    } else {
      assert(0 && "Reached unreachable mux arm");
    }
    // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    x48 = x58;
  } else if (to_size_t((Val(1) - x6))) {
    // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:151)
    ControlTableArm1_Super__0Struct16Array x59 = map(
        x11,
        LAYOUT_LOOKUP(x31, _0),
        std::function(
            [&](Val16Array::value_type x60,
                BoundLayout<ControlTableArm1_Super__0_SuperLayout16LayoutArray::value_type> x61) {
              // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:152)
              Val x62 = (x7 + x60);
              // LookupCurrent(zirgen/circuit/rv32im/v2/dsl/lookups.zir:5)
              // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:153)
              Val x63 = INVOKE_EXTERN(ctx, lookupCurrent, Val(8), x62);
              ArgU8Struct x64 = exec_ArgU8(ctx, neg_0(x63), x62, LAYOUT_LOOKUP(x61, arg));
              // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:154)
              Val x65 = (x64.val._super - x62);
              EQZ(x65, "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:154)");
              return ControlTableArm1_Super__0Struct{};
            }));
    // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:157)
    NondetRegStruct x66 = exec_IsZero(ctx, (x12 - Val(256)), LAYOUT_LOOKUP(x31, done));
    Val x67 = x66._super;
    InstOutputStruct x68;
    if (to_size_t(x67)) {
      x68 = InstOutputStruct{.newPc = x13, .newState = Val(6), .newMode = Val(1)};
    } else if (to_size_t((Val(1) - x67))) {
      x68 = InstOutputStruct{.newPc = x14, .newState = Val(6), .newMode = Val(0)};
    } else {
      assert(0 && "Reached unreachable mux arm");
    }
    // ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x42, Val(0));
    EQZ(LOAD(x42, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x43, Val(0));
    EQZ(LOAD(x43, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x44, Val(0));
    EQZ(LOAD(x44, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "ControlTable(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:135)");
    x48 = x68;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  return x48;
}
InstOutputStruct exec_Control0(ExecContext& ctx,
                               NondetRegStruct arg0,
                               BoundLayout<NondetRegLayout> arg1,
                               InstInputStruct arg2,
                               BoundLayout<Control0Layout> layout3,
                               GlobalBuf global4) {
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
  BoundLayout<Control0_SuperLayout> x5 = LAYOUT_LOOKUP(layout3, _super);
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:171)
  Val x6 = arg0._super;
  // GetDiffCount(zirgen/circuit/rv32im/v2/dsl/mem.zir:22)
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:173)
  Val x7 = INVOKE_EXTERN(ctx, getDiffCount, x6);
  CycleArgStruct x8 = exec_CycleArg(ctx, neg_0(x7), x6, LAYOUT_LOOKUP(layout3, arg));
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:175)
  Val x9 = (x8.cycle._super - x6);
  EQZ(x9, "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:175)");
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:171)
  NondetRegStruct8Array x10 = arg2.minorOnehot._super;
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
  BoundLayout<Control0Arm0Layout> x11 = LAYOUT_LOOKUP(x5, arm0);
  BoundLayout<Control0Arm1Layout> x12 = LAYOUT_LOOKUP(x5, arm1);
  BoundLayout<Control0Arm2Layout> x13 = LAYOUT_LOOKUP(x5, arm2);
  BoundLayout<Control0Arm3Layout> x14 = LAYOUT_LOOKUP(x5, arm3);
  BoundLayout<Control0Arm4Layout> x15 = LAYOUT_LOOKUP(x5, arm4);
  BoundLayout<Control0Arm5Layout> x16 = LAYOUT_LOOKUP(x5, arm5);
  BoundLayout<Control0Arm6Layout> x17 = LAYOUT_LOOKUP(x5, arm6);
  BoundLayout<Control0Arm7Layout> x18 = LAYOUT_LOOKUP(x5, arm7);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra5), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra6), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra7), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra8), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra9), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra10), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra11), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra12), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra13), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra14), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra15), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra16), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra17), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra18), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra19), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra20), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra21), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra22), count), _super);
  BoundLayout<Reg> x42 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra23), count), _super);
  BoundLayout<Reg> x43 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra24), count), _super);
  BoundLayout<Reg> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra25), count), _super);
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra26), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra27), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra28), count), _super);
  BoundLayout<Reg> x48 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra29), count), _super);
  BoundLayout<Reg> x49 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra30), count), _super);
  BoundLayout<Reg> x50 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra31), count), _super);
  BoundLayout<Reg> x51 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra32), count), _super);
  BoundLayout<Reg> x52 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra33), count), _super);
  BoundLayout<Reg> x53 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra34), count), _super);
  BoundLayout<Reg> x54 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra35), count), _super);
  BoundLayout<Reg> x55 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra36), count), _super);
  BoundLayout<Reg> x56 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra37), count), _super);
  BoundLayout<Reg> x57 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra38), count), _super);
  BoundLayout<Reg> x58 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra39), count), _super);
  BoundLayout<Reg> x59 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x60 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x61 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x62 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x63 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra4), count), _super);
  BoundLayout<Reg> x64 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra5), count), _super);
  BoundLayout<Reg> x65 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra6), count), _super);
  BoundLayout<Reg> x66 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra7), count), _super);
  BoundLayout<Reg> x67 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra8), count), _super);
  BoundLayout<Reg> x68 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra9), count), _super);
  BoundLayout<Reg> x69 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra10), count), _super);
  BoundLayout<Reg> x70 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra11), count), _super);
  BoundLayout<Reg> x71 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra12), count), _super);
  BoundLayout<Reg> x72 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra13), count), _super);
  BoundLayout<Reg> x73 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra14), count), _super);
  BoundLayout<Reg> x74 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra15), count), _super);
  BoundLayout<Reg> x75 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra16), count), _super);
  BoundLayout<Reg> x76 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra17), count), _super);
  BoundLayout<Reg> x77 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra18), count), _super);
  BoundLayout<Reg> x78 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra19), count), _super);
  BoundLayout<Reg> x79 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra20), count), _super);
  BoundLayout<Reg> x80 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra21), count), _super);
  BoundLayout<Reg> x81 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra22), count), _super);
  BoundLayout<Reg> x82 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra23), count), _super);
  BoundLayout<Reg> x83 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra24), count), _super);
  BoundLayout<Reg> x84 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra25), count), _super);
  BoundLayout<Reg> x85 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra26), count), _super);
  BoundLayout<Reg> x86 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra27), count), _super);
  BoundLayout<Reg> x87 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra28), count), _super);
  BoundLayout<Reg> x88 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra29), count), _super);
  BoundLayout<Reg> x89 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra30), count), _super);
  BoundLayout<Reg> x90 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra31), count), _super);
  BoundLayout<Reg> x91 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra0), count), _super);
  BoundLayout<Reg> x92 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra1), count), _super);
  BoundLayout<Reg> x93 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra2), count), _super);
  BoundLayout<Reg> x94 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra3), count), _super);
  BoundLayout<Reg> x95 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra4), count), _super);
  BoundLayout<Reg> x96 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra5), count), _super);
  BoundLayout<Reg> x97 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra6), count), _super);
  BoundLayout<Reg> x98 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra7), count), _super);
  BoundLayout<Reg> x99 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra8), count), _super);
  BoundLayout<Reg> x100 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra9), count), _super);
  BoundLayout<Reg> x101 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra10), count), _super);
  BoundLayout<Reg> x102 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra11), count), _super);
  BoundLayout<Reg> x103 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra12), count), _super);
  BoundLayout<Reg> x104 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra13), count), _super);
  BoundLayout<Reg> x105 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra14), count), _super);
  BoundLayout<Reg> x106 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra15), count), _super);
  BoundLayout<Reg> x107 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra16), count), _super);
  BoundLayout<Reg> x108 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra17), count), _super);
  BoundLayout<Reg> x109 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra18), count), _super);
  BoundLayout<Reg> x110 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra19), count), _super);
  BoundLayout<Reg> x111 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra20), count), _super);
  BoundLayout<Reg> x112 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra21), count), _super);
  BoundLayout<Reg> x113 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra22), count), _super);
  BoundLayout<Reg> x114 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra23), count), _super);
  BoundLayout<Reg> x115 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra24), count), _super);
  BoundLayout<Reg> x116 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra25), count), _super);
  BoundLayout<Reg> x117 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra26), count), _super);
  BoundLayout<Reg> x118 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra27), count), _super);
  BoundLayout<Reg> x119 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra28), count), _super);
  BoundLayout<Reg> x120 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra29), count), _super);
  BoundLayout<Reg> x121 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra30), count), _super);
  BoundLayout<Reg> x122 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra31), count), _super);
  BoundLayout<Reg> x123 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra32), count), _super);
  BoundLayout<Reg> x124 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra33), count), _super);
  BoundLayout<Reg> x125 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra34), count), _super);
  BoundLayout<Reg> x126 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra35), count), _super);
  BoundLayout<Reg> x127 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra36), count), _super);
  BoundLayout<Reg> x128 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra37), count), _super);
  BoundLayout<Reg> x129 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra38), count), _super);
  BoundLayout<Reg> x130 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra39), count), _super);
  BoundLayout<Reg> x131 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra40), count), _super);
  BoundLayout<Reg> x132 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra0), count), _super);
  BoundLayout<Reg> x133 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra1), count), _super);
  BoundLayout<Reg> x134 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra2), count), _super);
  BoundLayout<Reg> x135 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra3), count), _super);
  BoundLayout<Reg> x136 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra4), count), _super);
  BoundLayout<Reg> x137 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra5), count), _super);
  BoundLayout<Reg> x138 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra6), count), _super);
  BoundLayout<Reg> x139 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra7), count), _super);
  BoundLayout<Reg> x140 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra8), count), _super);
  BoundLayout<Reg> x141 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra9), count), _super);
  BoundLayout<Reg> x142 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra10), count), _super);
  BoundLayout<Reg> x143 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra11), count), _super);
  BoundLayout<Reg> x144 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra12), count), _super);
  BoundLayout<Reg> x145 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra13), count), _super);
  BoundLayout<Reg> x146 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra14), count), _super);
  BoundLayout<Reg> x147 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra15), count), _super);
  BoundLayout<Reg> x148 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra16), count), _super);
  BoundLayout<Reg> x149 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra17), count), _super);
  BoundLayout<Reg> x150 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra18), count), _super);
  BoundLayout<Reg> x151 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra19), count), _super);
  BoundLayout<Reg> x152 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra20), count), _super);
  BoundLayout<Reg> x153 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra21), count), _super);
  BoundLayout<Reg> x154 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra22), count), _super);
  BoundLayout<Reg> x155 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra23), count), _super);
  BoundLayout<Reg> x156 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra24), count), _super);
  BoundLayout<Reg> x157 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra25), count), _super);
  BoundLayout<Reg> x158 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra26), count), _super);
  BoundLayout<Reg> x159 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra27), count), _super);
  BoundLayout<Reg> x160 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra28), count), _super);
  BoundLayout<Reg> x161 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra29), count), _super);
  BoundLayout<Reg> x162 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra30), count), _super);
  BoundLayout<Reg> x163 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra31), count), _super);
  BoundLayout<Reg> x164 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra32), count), _super);
  BoundLayout<Reg> x165 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra33), count), _super);
  BoundLayout<Reg> x166 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra34), count), _super);
  BoundLayout<Reg> x167 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra35), count), _super);
  BoundLayout<Reg> x168 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra36), count), _super);
  BoundLayout<Reg> x169 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra37), count), _super);
  BoundLayout<Reg> x170 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra38), count), _super);
  BoundLayout<Reg> x171 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra39), count), _super);
  BoundLayout<Reg> x172 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra40), count), _super);
  BoundLayout<Reg> x173 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra41), count), _super);
  BoundLayout<Reg> x174 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra42), count), _super);
  BoundLayout<Reg> x175 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra43), count), _super);
  BoundLayout<Reg> x176 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra44), count), _super);
  BoundLayout<Reg> x177 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra45), count), _super);
  BoundLayout<Reg> x178 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra0), count), _super);
  BoundLayout<Reg> x179 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra1), count), _super);
  BoundLayout<Reg> x180 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra2), count), _super);
  BoundLayout<Reg> x181 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra3), count), _super);
  BoundLayout<Reg> x182 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra4), count), _super);
  BoundLayout<Reg> x183 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra5), count), _super);
  BoundLayout<Reg> x184 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra6), count), _super);
  BoundLayout<Reg> x185 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra7), count), _super);
  BoundLayout<Reg> x186 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra8), count), _super);
  BoundLayout<Reg> x187 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra9), count), _super);
  BoundLayout<Reg> x188 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra10), count), _super);
  BoundLayout<Reg> x189 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra11), count), _super);
  BoundLayout<Reg> x190 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra12), count), _super);
  BoundLayout<Reg> x191 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra13), count), _super);
  BoundLayout<Reg> x192 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra14), count), _super);
  BoundLayout<Reg> x193 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra15), count), _super);
  BoundLayout<Reg> x194 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra16), count), _super);
  BoundLayout<Reg> x195 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra17), count), _super);
  BoundLayout<Reg> x196 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra18), count), _super);
  BoundLayout<Reg> x197 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra19), count), _super);
  BoundLayout<Reg> x198 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra20), count), _super);
  BoundLayout<Reg> x199 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra21), count), _super);
  BoundLayout<Reg> x200 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra22), count), _super);
  BoundLayout<Reg> x201 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra23), count), _super);
  BoundLayout<Reg> x202 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra24), count), _super);
  BoundLayout<Reg> x203 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra25), count), _super);
  BoundLayout<Reg> x204 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra26), count), _super);
  BoundLayout<Reg> x205 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra27), count), _super);
  BoundLayout<Reg> x206 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra28), count), _super);
  BoundLayout<Reg> x207 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra29), count), _super);
  BoundLayout<Reg> x208 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra30), count), _super);
  BoundLayout<Reg> x209 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra31), count), _super);
  BoundLayout<Reg> x210 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra0), count), _super);
  BoundLayout<Reg> x211 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra1), count), _super);
  BoundLayout<Reg> x212 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra2), count), _super);
  BoundLayout<Reg> x213 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra3), count), _super);
  BoundLayout<Reg> x214 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra4), count), _super);
  BoundLayout<Reg> x215 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra5), count), _super);
  BoundLayout<Reg> x216 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra6), count), _super);
  BoundLayout<Reg> x217 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra7), count), _super);
  BoundLayout<Reg> x218 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra8), count), _super);
  BoundLayout<Reg> x219 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra9), count), _super);
  BoundLayout<Reg> x220 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra10), count), _super);
  BoundLayout<Reg> x221 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra11), count), _super);
  BoundLayout<Reg> x222 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra12), count), _super);
  BoundLayout<Reg> x223 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra13), count), _super);
  BoundLayout<Reg> x224 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra14), count), _super);
  BoundLayout<Reg> x225 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra15), count), _super);
  BoundLayout<Reg> x226 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra16), count), _super);
  BoundLayout<Reg> x227 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra17), count), _super);
  BoundLayout<Reg> x228 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra18), count), _super);
  BoundLayout<Reg> x229 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra19), count), _super);
  BoundLayout<Reg> x230 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra20), count), _super);
  BoundLayout<Reg> x231 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra21), count), _super);
  BoundLayout<Reg> x232 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra22), count), _super);
  BoundLayout<Reg> x233 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra23), count), _super);
  BoundLayout<Reg> x234 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra24), count), _super);
  BoundLayout<Reg> x235 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra25), count), _super);
  BoundLayout<Reg> x236 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra26), count), _super);
  BoundLayout<Reg> x237 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra27), count), _super);
  BoundLayout<Reg> x238 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra28), count), _super);
  BoundLayout<Reg> x239 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra29), count), _super);
  BoundLayout<Reg> x240 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra30), count), _super);
  BoundLayout<Reg> x241 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra31), count), _super);
  BoundLayout<Reg> x242 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra0), count), _super);
  BoundLayout<Reg> x243 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra1), count), _super);
  BoundLayout<Reg> x244 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra2), count), _super);
  BoundLayout<Reg> x245 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra3), count), _super);
  BoundLayout<Reg> x246 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra4), count), _super);
  BoundLayout<Reg> x247 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra5), count), _super);
  BoundLayout<Reg> x248 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra6), count), _super);
  BoundLayout<Reg> x249 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra7), count), _super);
  BoundLayout<Reg> x250 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra8), count), _super);
  BoundLayout<Reg> x251 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra9), count), _super);
  BoundLayout<Reg> x252 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra10), count), _super);
  BoundLayout<Reg> x253 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra11), count), _super);
  BoundLayout<Reg> x254 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra12), count), _super);
  BoundLayout<Reg> x255 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra13), count), _super);
  BoundLayout<Reg> x256 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra14), count), _super);
  BoundLayout<Reg> x257 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra15), count), _super);
  BoundLayout<Reg> x258 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra16), count), _super);
  BoundLayout<Reg> x259 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra17), count), _super);
  BoundLayout<Reg> x260 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra18), count), _super);
  BoundLayout<Reg> x261 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra19), count), _super);
  BoundLayout<Reg> x262 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra20), count), _super);
  BoundLayout<Reg> x263 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra21), count), _super);
  BoundLayout<Reg> x264 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra22), count), _super);
  BoundLayout<Reg> x265 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _extra23), count), _super);
  // InstOutput(zirgen/circuit/rv32im/v2/dsl/inst.zir:46)
  // ControlDone(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:168)
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:184)
  InstOutputStruct x266 = InstOutputStruct{
      .newPc = ValU32Struct{.low = Val(0), .high = Val(0)}, .newState = Val(7), .newMode = Val(0)};
  // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
  BoundLayout<Reg> x267 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra0), count), _super);
  BoundLayout<Reg> x268 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra1), count), _super);
  BoundLayout<Reg> x269 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra2), count), _super);
  BoundLayout<Reg> x270 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra3), count), _super);
  BoundLayout<Reg> x271 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra4), count), _super);
  BoundLayout<Reg> x272 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra5), count), _super);
  BoundLayout<Reg> x273 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra6), count), _super);
  BoundLayout<Reg> x274 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra7), count), _super);
  BoundLayout<Reg> x275 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra8), count), _super);
  BoundLayout<Reg> x276 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra9), count), _super);
  BoundLayout<Reg> x277 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra10), count), _super);
  BoundLayout<Reg> x278 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra11), count), _super);
  BoundLayout<Reg> x279 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra12), count), _super);
  BoundLayout<Reg> x280 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra13), count), _super);
  BoundLayout<Reg> x281 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra14), count), _super);
  BoundLayout<Reg> x282 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra15), count), _super);
  BoundLayout<Reg> x283 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra16), count), _super);
  BoundLayout<Reg> x284 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra17), count), _super);
  BoundLayout<Reg> x285 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra18), count), _super);
  BoundLayout<Reg> x286 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra19), count), _super);
  BoundLayout<Reg> x287 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra20), count), _super);
  BoundLayout<Reg> x288 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra21), count), _super);
  BoundLayout<Reg> x289 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra22), count), _super);
  BoundLayout<Reg> x290 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra23), count), _super);
  BoundLayout<Reg> x291 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra24), count), _super);
  BoundLayout<Reg> x292 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra25), count), _super);
  BoundLayout<Reg> x293 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra26), count), _super);
  BoundLayout<Reg> x294 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra27), count), _super);
  BoundLayout<Reg> x295 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra28), count), _super);
  BoundLayout<Reg> x296 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra29), count), _super);
  BoundLayout<Reg> x297 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra30), count), _super);
  BoundLayout<Reg> x298 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra31), count), _super);
  BoundLayout<Reg> x299 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra32), count), _super);
  BoundLayout<Reg> x300 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra33), count), _super);
  BoundLayout<Reg> x301 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra34), count), _super);
  BoundLayout<Reg> x302 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra35), count), _super);
  BoundLayout<Reg> x303 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra36), count), _super);
  BoundLayout<Reg> x304 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra37), count), _super);
  BoundLayout<Reg> x305 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra38), count), _super);
  BoundLayout<Reg> x306 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra39), count), _super);
  BoundLayout<Reg> x307 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra40), count), _super);
  BoundLayout<Reg> x308 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra41), count), _super);
  BoundLayout<Reg> x309 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra42), count), _super);
  BoundLayout<Reg> x310 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra43), count), _super);
  BoundLayout<Reg> x311 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra44), count), _super);
  BoundLayout<Reg> x312 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra45), count), _super);
  BoundLayout<Reg> x313 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra46), count), _super);
  BoundLayout<Reg> x314 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra47), count), _super);
  BoundLayout<Reg> x315 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra48), count), _super);
  BoundLayout<Reg> x316 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra49), count), _super);
  BoundLayout<Reg> x317 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra50), count), _super);
  BoundLayout<Reg> x318 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra51), count), _super);
  BoundLayout<Reg> x319 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra52), count), _super);
  BoundLayout<Reg> x320 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra53), count), _super);
  BoundLayout<Reg> x321 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra54), count), _super);
  BoundLayout<Reg> x322 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, _extra55), count), _super);
  InstOutputStruct x323;
  if (to_size_t(x10[0]._super)) {
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:177)
    InstOutputStruct x324 =
        exec_ControlLoadRoot(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x11, _super), global4);
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x42, Val(0));
    EQZ(LOAD(x42, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x43, Val(0));
    EQZ(LOAD(x43, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x44, Val(0));
    EQZ(LOAD(x44, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x48, Val(0));
    EQZ(LOAD(x48, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x49, Val(0));
    EQZ(LOAD(x49, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x50, Val(0));
    EQZ(LOAD(x50, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x51, Val(0));
    EQZ(LOAD(x51, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x52, Val(0));
    EQZ(LOAD(x52, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x53, Val(0));
    EQZ(LOAD(x53, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x54, Val(0));
    EQZ(LOAD(x54, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x55, Val(0));
    EQZ(LOAD(x55, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x56, Val(0));
    EQZ(LOAD(x56, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x57, Val(0));
    EQZ(LOAD(x57, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x58, Val(0));
    EQZ(LOAD(x58, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x324;
  } else if (to_size_t(x10[1]._super)) {
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:178)
    InstOutputStruct x325 =
        exec_ControlResume(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x12, _super), global4);
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x59, Val(0));
    EQZ(LOAD(x59, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x60, Val(0));
    EQZ(LOAD(x60, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x61, Val(0));
    EQZ(LOAD(x61, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x62, Val(0));
    EQZ(LOAD(x62, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x63, Val(0));
    EQZ(LOAD(x63, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x64, Val(0));
    EQZ(LOAD(x64, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x65, Val(0));
    EQZ(LOAD(x65, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x66, Val(0));
    EQZ(LOAD(x66, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x67, Val(0));
    EQZ(LOAD(x67, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x68, Val(0));
    EQZ(LOAD(x68, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x69, Val(0));
    EQZ(LOAD(x69, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x70, Val(0));
    EQZ(LOAD(x70, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x71, Val(0));
    EQZ(LOAD(x71, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x72, Val(0));
    EQZ(LOAD(x72, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x73, Val(0));
    EQZ(LOAD(x73, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x74, Val(0));
    EQZ(LOAD(x74, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x75, Val(0));
    EQZ(LOAD(x75, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x76, Val(0));
    EQZ(LOAD(x76, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x77, Val(0));
    EQZ(LOAD(x77, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x78, Val(0));
    EQZ(LOAD(x78, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x79, Val(0));
    EQZ(LOAD(x79, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x80, Val(0));
    EQZ(LOAD(x80, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x81, Val(0));
    EQZ(LOAD(x81, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x82, Val(0));
    EQZ(LOAD(x82, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x83, Val(0));
    EQZ(LOAD(x83, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x84, Val(0));
    EQZ(LOAD(x84, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x85, Val(0));
    EQZ(LOAD(x85, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x86, Val(0));
    EQZ(LOAD(x86, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x87, Val(0));
    EQZ(LOAD(x87, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x88, Val(0));
    EQZ(LOAD(x88, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x89, Val(0));
    EQZ(LOAD(x89, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x90, Val(0));
    EQZ(LOAD(x90, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x325;
  } else if (to_size_t(x10[2]._super)) {
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:179)
    InstOutputStruct x326 =
        exec_ControlUserECALL(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x13, _super));
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x91, Val(0));
    EQZ(LOAD(x91, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x92, Val(0));
    EQZ(LOAD(x92, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x93, Val(0));
    EQZ(LOAD(x93, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x94, Val(0));
    EQZ(LOAD(x94, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x95, Val(0));
    EQZ(LOAD(x95, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x96, Val(0));
    EQZ(LOAD(x96, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x97, Val(0));
    EQZ(LOAD(x97, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x98, Val(0));
    EQZ(LOAD(x98, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x99, Val(0));
    EQZ(LOAD(x99, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x100, Val(0));
    EQZ(LOAD(x100, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x101, Val(0));
    EQZ(LOAD(x101, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x102, Val(0));
    EQZ(LOAD(x102, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x103, Val(0));
    EQZ(LOAD(x103, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x104, Val(0));
    EQZ(LOAD(x104, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x105, Val(0));
    EQZ(LOAD(x105, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x106, Val(0));
    EQZ(LOAD(x106, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x107, Val(0));
    EQZ(LOAD(x107, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x108, Val(0));
    EQZ(LOAD(x108, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x109, Val(0));
    EQZ(LOAD(x109, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x110, Val(0));
    EQZ(LOAD(x110, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x111, Val(0));
    EQZ(LOAD(x111, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x112, Val(0));
    EQZ(LOAD(x112, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x113, Val(0));
    EQZ(LOAD(x113, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x114, Val(0));
    EQZ(LOAD(x114, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x115, Val(0));
    EQZ(LOAD(x115, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x116, Val(0));
    EQZ(LOAD(x116, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x117, Val(0));
    EQZ(LOAD(x117, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x118, Val(0));
    EQZ(LOAD(x118, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x119, Val(0));
    EQZ(LOAD(x119, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x120, Val(0));
    EQZ(LOAD(x120, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x121, Val(0));
    EQZ(LOAD(x121, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x122, Val(0));
    EQZ(LOAD(x122, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x123, Val(0));
    EQZ(LOAD(x123, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x124, Val(0));
    EQZ(LOAD(x124, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x125, Val(0));
    EQZ(LOAD(x125, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x126, Val(0));
    EQZ(LOAD(x126, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x127, Val(0));
    EQZ(LOAD(x127, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x128, Val(0));
    EQZ(LOAD(x128, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x129, Val(0));
    EQZ(LOAD(x129, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x130, Val(0));
    EQZ(LOAD(x130, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x131, Val(0));
    EQZ(LOAD(x131, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x326;
  } else if (to_size_t(x10[3]._super)) {
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:180)
    InstOutputStruct x327 = exec_ControlMRET(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x14, _super));
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x132, Val(0));
    EQZ(LOAD(x132, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x133, Val(0));
    EQZ(LOAD(x133, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x134, Val(0));
    EQZ(LOAD(x134, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x135, Val(0));
    EQZ(LOAD(x135, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x136, Val(0));
    EQZ(LOAD(x136, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x137, Val(0));
    EQZ(LOAD(x137, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x138, Val(0));
    EQZ(LOAD(x138, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x139, Val(0));
    EQZ(LOAD(x139, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x140, Val(0));
    EQZ(LOAD(x140, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x141, Val(0));
    EQZ(LOAD(x141, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x142, Val(0));
    EQZ(LOAD(x142, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x143, Val(0));
    EQZ(LOAD(x143, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x144, Val(0));
    EQZ(LOAD(x144, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x145, Val(0));
    EQZ(LOAD(x145, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x146, Val(0));
    EQZ(LOAD(x146, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x147, Val(0));
    EQZ(LOAD(x147, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x148, Val(0));
    EQZ(LOAD(x148, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x149, Val(0));
    EQZ(LOAD(x149, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x150, Val(0));
    EQZ(LOAD(x150, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x151, Val(0));
    EQZ(LOAD(x151, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x152, Val(0));
    EQZ(LOAD(x152, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x153, Val(0));
    EQZ(LOAD(x153, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x154, Val(0));
    EQZ(LOAD(x154, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x155, Val(0));
    EQZ(LOAD(x155, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x156, Val(0));
    EQZ(LOAD(x156, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x157, Val(0));
    EQZ(LOAD(x157, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x158, Val(0));
    EQZ(LOAD(x158, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x159, Val(0));
    EQZ(LOAD(x159, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x160, Val(0));
    EQZ(LOAD(x160, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x161, Val(0));
    EQZ(LOAD(x161, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x162, Val(0));
    EQZ(LOAD(x162, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x163, Val(0));
    EQZ(LOAD(x163, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x164, Val(0));
    EQZ(LOAD(x164, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x165, Val(0));
    EQZ(LOAD(x165, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x166, Val(0));
    EQZ(LOAD(x166, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x167, Val(0));
    EQZ(LOAD(x167, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x168, Val(0));
    EQZ(LOAD(x168, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x169, Val(0));
    EQZ(LOAD(x169, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x170, Val(0));
    EQZ(LOAD(x170, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x171, Val(0));
    EQZ(LOAD(x171, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x172, Val(0));
    EQZ(LOAD(x172, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x173, Val(0));
    EQZ(LOAD(x173, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x174, Val(0));
    EQZ(LOAD(x174, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x175, Val(0));
    EQZ(LOAD(x175, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x176, Val(0));
    EQZ(LOAD(x176, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x177, Val(0));
    EQZ(LOAD(x177, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x327;
  } else if (to_size_t(x10[4]._super)) {
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:181)
    InstOutputStruct x328 =
        exec_ControlSuspend(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x15, _super), global4);
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x178, Val(0));
    EQZ(LOAD(x178, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x179, Val(0));
    EQZ(LOAD(x179, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x180, Val(0));
    EQZ(LOAD(x180, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x181, Val(0));
    EQZ(LOAD(x181, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x182, Val(0));
    EQZ(LOAD(x182, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x183, Val(0));
    EQZ(LOAD(x183, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x184, Val(0));
    EQZ(LOAD(x184, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x185, Val(0));
    EQZ(LOAD(x185, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x186, Val(0));
    EQZ(LOAD(x186, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x187, Val(0));
    EQZ(LOAD(x187, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x188, Val(0));
    EQZ(LOAD(x188, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x189, Val(0));
    EQZ(LOAD(x189, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x190, Val(0));
    EQZ(LOAD(x190, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x191, Val(0));
    EQZ(LOAD(x191, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x192, Val(0));
    EQZ(LOAD(x192, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x193, Val(0));
    EQZ(LOAD(x193, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x194, Val(0));
    EQZ(LOAD(x194, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x195, Val(0));
    EQZ(LOAD(x195, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x196, Val(0));
    EQZ(LOAD(x196, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x197, Val(0));
    EQZ(LOAD(x197, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x198, Val(0));
    EQZ(LOAD(x198, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x199, Val(0));
    EQZ(LOAD(x199, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x200, Val(0));
    EQZ(LOAD(x200, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x201, Val(0));
    EQZ(LOAD(x201, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x202, Val(0));
    EQZ(LOAD(x202, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x203, Val(0));
    EQZ(LOAD(x203, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x204, Val(0));
    EQZ(LOAD(x204, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x205, Val(0));
    EQZ(LOAD(x205, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x206, Val(0));
    EQZ(LOAD(x206, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x207, Val(0));
    EQZ(LOAD(x207, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x208, Val(0));
    EQZ(LOAD(x208, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x209, Val(0));
    EQZ(LOAD(x209, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x328;
  } else if (to_size_t(x10[5]._super)) {
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:182)
    InstOutputStruct x329 =
        exec_ControlStoreRoot(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x16, _super), global4);
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x210, Val(0));
    EQZ(LOAD(x210, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x211, Val(0));
    EQZ(LOAD(x211, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x212, Val(0));
    EQZ(LOAD(x212, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x213, Val(0));
    EQZ(LOAD(x213, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x214, Val(0));
    EQZ(LOAD(x214, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x215, Val(0));
    EQZ(LOAD(x215, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x216, Val(0));
    EQZ(LOAD(x216, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x217, Val(0));
    EQZ(LOAD(x217, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x218, Val(0));
    EQZ(LOAD(x218, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x219, Val(0));
    EQZ(LOAD(x219, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x220, Val(0));
    EQZ(LOAD(x220, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x221, Val(0));
    EQZ(LOAD(x221, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x222, Val(0));
    EQZ(LOAD(x222, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x223, Val(0));
    EQZ(LOAD(x223, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x224, Val(0));
    EQZ(LOAD(x224, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x225, Val(0));
    EQZ(LOAD(x225, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x226, Val(0));
    EQZ(LOAD(x226, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x227, Val(0));
    EQZ(LOAD(x227, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x228, Val(0));
    EQZ(LOAD(x228, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x229, Val(0));
    EQZ(LOAD(x229, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x230, Val(0));
    EQZ(LOAD(x230, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x231, Val(0));
    EQZ(LOAD(x231, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x232, Val(0));
    EQZ(LOAD(x232, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x233, Val(0));
    EQZ(LOAD(x233, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x234, Val(0));
    EQZ(LOAD(x234, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x235, Val(0));
    EQZ(LOAD(x235, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x236, Val(0));
    EQZ(LOAD(x236, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x237, Val(0));
    EQZ(LOAD(x237, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x238, Val(0));
    EQZ(LOAD(x238, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x239, Val(0));
    EQZ(LOAD(x239, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x240, Val(0));
    EQZ(LOAD(x240, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x241, Val(0));
    EQZ(LOAD(x241, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x329;
  } else if (to_size_t(x10[6]._super)) {
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:183)
    InstOutputStruct x330 = exec_ControlTable(ctx, arg0, arg2, LAYOUT_LOOKUP(x17, _super));
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x242, Val(0));
    EQZ(LOAD(x242, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x243, Val(0));
    EQZ(LOAD(x243, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x244, Val(0));
    EQZ(LOAD(x244, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x245, Val(0));
    EQZ(LOAD(x245, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x246, Val(0));
    EQZ(LOAD(x246, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x247, Val(0));
    EQZ(LOAD(x247, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x248, Val(0));
    EQZ(LOAD(x248, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x249, Val(0));
    EQZ(LOAD(x249, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x250, Val(0));
    EQZ(LOAD(x250, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x251, Val(0));
    EQZ(LOAD(x251, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x252, Val(0));
    EQZ(LOAD(x252, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x253, Val(0));
    EQZ(LOAD(x253, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x254, Val(0));
    EQZ(LOAD(x254, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x255, Val(0));
    EQZ(LOAD(x255, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x256, Val(0));
    EQZ(LOAD(x256, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x257, Val(0));
    EQZ(LOAD(x257, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x258, Val(0));
    EQZ(LOAD(x258, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x259, Val(0));
    EQZ(LOAD(x259, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x260, Val(0));
    EQZ(LOAD(x260, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x261, Val(0));
    EQZ(LOAD(x261, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x262, Val(0));
    EQZ(LOAD(x262, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x263, Val(0));
    EQZ(LOAD(x263, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x264, Val(0));
    EQZ(LOAD(x264, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x265, Val(0));
    EQZ(LOAD(x265, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x330;
  } else if (to_size_t(x10[7]._super)) {
    // ControlDone(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:167)
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:184)
    EQZ((arg2.state - Val(7)),
        "loc(callsite( ControlDone ( zirgen/circuit/rv32im/v2/dsl/inst_control.zir :167:16) at  "
        "Control0 ( zirgen/circuit/rv32im/v2/dsl/inst_control.zir :184:17)))");
    // Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)
    STORE(x267, Val(0));
    EQZ(LOAD(x267, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x268, Val(0));
    EQZ(LOAD(x268, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x269, Val(0));
    EQZ(LOAD(x269, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x270, Val(0));
    EQZ(LOAD(x270, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x271, Val(0));
    EQZ(LOAD(x271, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x272, Val(0));
    EQZ(LOAD(x272, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x273, Val(0));
    EQZ(LOAD(x273, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x274, Val(0));
    EQZ(LOAD(x274, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x275, Val(0));
    EQZ(LOAD(x275, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x276, Val(0));
    EQZ(LOAD(x276, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x277, Val(0));
    EQZ(LOAD(x277, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x278, Val(0));
    EQZ(LOAD(x278, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x279, Val(0));
    EQZ(LOAD(x279, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x280, Val(0));
    EQZ(LOAD(x280, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x281, Val(0));
    EQZ(LOAD(x281, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x282, Val(0));
    EQZ(LOAD(x282, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x283, Val(0));
    EQZ(LOAD(x283, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x284, Val(0));
    EQZ(LOAD(x284, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x285, Val(0));
    EQZ(LOAD(x285, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x286, Val(0));
    EQZ(LOAD(x286, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x287, Val(0));
    EQZ(LOAD(x287, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x288, Val(0));
    EQZ(LOAD(x288, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x289, Val(0));
    EQZ(LOAD(x289, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x290, Val(0));
    EQZ(LOAD(x290, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x291, Val(0));
    EQZ(LOAD(x291, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x292, Val(0));
    EQZ(LOAD(x292, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x293, Val(0));
    EQZ(LOAD(x293, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x294, Val(0));
    EQZ(LOAD(x294, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x295, Val(0));
    EQZ(LOAD(x295, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x296, Val(0));
    EQZ(LOAD(x296, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x297, Val(0));
    EQZ(LOAD(x297, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x298, Val(0));
    EQZ(LOAD(x298, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x299, Val(0));
    EQZ(LOAD(x299, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x300, Val(0));
    EQZ(LOAD(x300, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x301, Val(0));
    EQZ(LOAD(x301, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x302, Val(0));
    EQZ(LOAD(x302, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x303, Val(0));
    EQZ(LOAD(x303, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x304, Val(0));
    EQZ(LOAD(x304, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x305, Val(0));
    EQZ(LOAD(x305, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x306, Val(0));
    EQZ(LOAD(x306, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x307, Val(0));
    EQZ(LOAD(x307, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x308, Val(0));
    EQZ(LOAD(x308, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x309, Val(0));
    EQZ(LOAD(x309, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x310, Val(0));
    EQZ(LOAD(x310, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x311, Val(0));
    EQZ(LOAD(x311, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x312, Val(0));
    EQZ(LOAD(x312, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x313, Val(0));
    EQZ(LOAD(x313, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x314, Val(0));
    EQZ(LOAD(x314, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x315, Val(0));
    EQZ(LOAD(x315, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x316, Val(0));
    EQZ(LOAD(x316, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x317, Val(0));
    EQZ(LOAD(x317, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x318, Val(0));
    EQZ(LOAD(x318, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x319, Val(0));
    EQZ(LOAD(x319, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x320, Val(0));
    EQZ(LOAD(x320, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x321, Val(0));
    EQZ(LOAD(x321, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    STORE(x322, Val(0));
    EQZ(LOAD(x322, 0), "Control0(zirgen/circuit/rv32im/v2/dsl/inst_control.zir:176)");
    x323 = x266;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  return x323;
}
OneHot_4_Struct exec_OneHot_4_(ExecContext& ctx, Val arg0, BoundLayout<OneHot_4_Layout> layout1) {
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:7)
  NondetRegStruct4Array x2 =
      map(Val4Array{Val(0), Val(1), Val(2), Val(3)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val4Array::value_type x3,
                            BoundLayout<NondetRegLayout4LayoutArray::value_type> x4) {
            NondetRegStruct x5 = exec_NondetBitReg(ctx, isz((x3 - arg0)), x4);
            return x5;
          }));
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)
  Val x6 = x2[1]._super;
  Val x7 = (x2[0]._super + x6);
  Val x8 = x2[2]._super;
  Val x9 = x2[3]._super;
  Val x10 = (((x7 + x8) + x9) - Val(1));
  EQZ(x10, "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)");
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)
  Val x11 = (((x6 + (x8 * Val(2))) + (x9 * Val(3))) - arg0);
  EQZ(x11, "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)");
  return OneHot_4_Struct{._super = x2};
}
ECallOutputStruct exec_MachineECall(ExecContext& ctx,
                                    NondetRegStruct arg0,
                                    BoundLayout<NondetRegLayout> arg1,
                                    InstInputStruct arg2,
                                    Val arg3,
                                    BoundLayout<MachineECallLayout> layout4) {
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:26)
  GetDataStruct x5 = exec_MemoryRead(ctx, arg0, arg1, arg3, LAYOUT_LOOKUP(layout4, loadInst));
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:27)
  EQZ((arg2.state - Val(32)), "MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:27)");
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:26)
  ValU32Struct x6 = x5._super;
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:28)
  EQZ(x6.high, "MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:28)");
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:29)
  EQZ((x6.low - Val(115)), "MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:29)");
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:30)
  EQZ((arg2.mode - Val(1)), "MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:30)");
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:31)
  GetDataStruct x7 =
      exec_MemoryRead(ctx, arg0, arg1, Val(1073725457), LAYOUT_LOOKUP(layout4, dispatchIdx));
  ValU32Struct x8 = x7._super;
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:32)
  EQZ(x8.high, "MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:32)");
  // MachineECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:33)
  OneHot_4_Struct x9 = exec_OneHot_4_(ctx, x8.low, LAYOUT_LOOKUP(layout4, dispatch));
  NondetRegStruct4Array x10 = x9._super;
  Val x11;
  if (to_size_t(x10[0]._super)) {
    x11 = Val(9);
  } else if (to_size_t(x10[1]._super)) {
    x11 = Val(10);
  } else if (to_size_t(x10[2]._super)) {
    x11 = Val(11);
  } else if (to_size_t(x10[3]._super)) {
    x11 = Val(16);
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  return ECallOutputStruct{.state = x11, .s0 = Val(0), .s1 = Val(0), .s2 = Val(0)};
}
ECallOutputStruct exec_ECallTerminate(ExecContext& ctx,
                                      NondetRegStruct arg0,
                                      BoundLayout<NondetRegLayout> arg1,
                                      InstInputStruct arg2,
                                      BoundLayout<ECallTerminateLayout> layout3,
                                      GlobalBuf global4) {
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:43)
  BoundLayout<_globalLayout> x5 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:44)
  EQZ((arg2.state - Val(9)), "ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:44)");
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:45)
  GetDataStruct x6 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725482), LAYOUT_LOOKUP(layout3, a0));
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:46)
  GetDataStruct x7 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725483), LAYOUT_LOOKUP(layout3, a1));
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:45)
  ValU32Struct x8 = x6._super;
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:47)
  NondetRegStruct x9 = exec_Reg(ctx, x8.low, LAYOUT_LOOKUP(x5, termA0low));
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:48)
  NondetRegStruct x10 = exec_Reg(ctx, x8.high, LAYOUT_LOOKUP(x5, termA0high));
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:46)
  ValU32Struct x11 = x7._super;
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:49)
  NondetRegStruct x12 = exec_Reg(ctx, x11.low, LAYOUT_LOOKUP(x5, termA1low));
  // ECallTerminate(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:50)
  NondetRegStruct x13 = exec_Reg(ctx, x11.high, LAYOUT_LOOKUP(x5, termA1high));
  return ECallOutputStruct{.state = Val(4), .s0 = Val(0), .s1 = Val(0), .s2 = Val(0)};
}
DecomposeLow2Struct
exec_DecomposeLow2(ExecContext& ctx, Val arg0, BoundLayout<DecomposeLow2Layout> layout1) {
  // DecomposeLow2(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:57)
  NondetRegStruct x2 = exec_NondetReg(
      ctx, (bitAnd(arg0, Val(65532)) * Val(1509949441)), LAYOUT_LOOKUP(layout1, high));
  // DecomposeLow2(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:58)
  NondetRegStruct x3 = exec_NondetReg(ctx, bitAnd(arg0, Val(3)), LAYOUT_LOOKUP(layout1, low2));
  // DecomposeLow2(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:59)
  OneHot_4_Struct x4 = exec_OneHot_4_(ctx, x3._super, LAYOUT_LOOKUP(layout1, low2Hot));
  // DecomposeLow2(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:60)
  NondetRegStruct x5 = exec_IsZero(ctx, x2._super, LAYOUT_LOOKUP(layout1, highZero));
  // DecomposeLow2(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:59)
  NondetRegStruct4Array x6 = x4._super;
  // DecomposeLow2(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:61)
  Val x7 = (x5._super * x6[0]._super);
  NondetRegStruct x8 = exec_Reg(ctx, x7, LAYOUT_LOOKUP(layout1, isZero));
  // DecomposeLow2(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:63)
  Val x9 = (x6[1]._super + x6[2]._super);
  return DecomposeLow2Struct{.high = x2,
                             .low2 = x3,
                             .low2Hot = x4,
                             .highZero = x5,
                             .isZero = x8,
                             .low2Nonzero = (x9 + x6[3]._super)};
}
ECallOutputStruct exec_ECallHostReadSetup(ExecContext& ctx,
                                          NondetRegStruct arg0,
                                          BoundLayout<NondetRegLayout> arg1,
                                          InstInputStruct arg2,
                                          BoundLayout<ECallHostReadSetupLayout> layout3) {
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:67)
  EQZ((arg2.state - Val(10)), "ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:67)");
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:69)
  GetDataStruct x4 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725450), LAYOUT_LOOKUP(layout3, fd));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:70)
  GetDataStruct x5 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725451), LAYOUT_LOOKUP(layout3, ptr));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:71)
  GetDataStruct x6 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725452), LAYOUT_LOOKUP(layout3, len));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:69)
  ValU32Struct x7 = x4._super;
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:73)
  EQZ(x7.high, "ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:73)");
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:71)
  ValU32Struct x8 = x6._super;
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:75)
  EQZ(x8.high, "ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:75)");
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:71)
  Val x9 = x8.low;
  // HostReadPrepare(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:8)
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:77)
  Val x10 = INVOKE_EXTERN(ctx, hostReadPrepare, x7.low, x9);
  NondetRegStruct x11 = exec_NondetU16Reg(ctx, x10, LAYOUT_LOOKUP(layout3, newLen));
  Val x12 = x11._super;
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:79)
  U16RegStruct x13 = exec_U16Reg(ctx, (x9 - x12), LAYOUT_LOOKUP(layout3, diff));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:81)
  MemoryWriteStruct x14 = exec_MemoryWrite(ctx,
                                           arg0,
                                           arg1,
                                           Val(1073725450),
                                           ValU32Struct{.low = x12, .high = Val(0)},
                                           LAYOUT_LOOKUP(layout3, _0));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:70)
  ValU32Struct x15 = x5._super;
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:83)
  DecomposeLow2Struct x16 = exec_DecomposeLow2(ctx, x15.low, LAYOUT_LOOKUP(layout3, ptrDecomp));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:84)
  Val x17 = ((x15.high * Val(16384)) + x16.high._super);
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:85)
  DecomposeLow2Struct x18 = exec_DecomposeLow2(ctx, x12, LAYOUT_LOOKUP(layout3, lenDecomp));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:87)
  Val x19 = (x18.highZero._super * x18.low2Nonzero);
  NondetRegStruct x20 = exec_Reg(ctx, x19, LAYOUT_LOOKUP(layout3, len123));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:89)
  NondetRegStruct x21 =
      exec_Reg(ctx, (x20._super * x16.low2Nonzero), LAYOUT_LOOKUP(layout3, uneven));
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:85)
  Val x22 = x18.isZero._super;
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:95)
  Val x23 = (Val(1) - x22);
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:89)
  Val x24 = x21._super;
  // ECallHostReadSetup(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:93)
  Val x25 = ((x22 * Val(32)) + ((x23 * x24) * Val(12)));
  return ECallOutputStruct{.state = (x25 + ((x23 * (Val(1) - x24)) * Val(13))),
                           .s0 = x17,
                           .s1 = x16.low2._super,
                           .s2 = x12};
}
ECallOutputStruct exec_ECallHostWrite(ExecContext& ctx,
                                      NondetRegStruct arg0,
                                      BoundLayout<NondetRegLayout> arg1,
                                      InstInputStruct arg2,
                                      BoundLayout<ECallHostWriteLayout> layout3) {
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:102)
  EQZ((arg2.state - Val(11)), "ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:102)");
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:104)
  GetDataStruct x4 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725450), LAYOUT_LOOKUP(layout3, fd));
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:105)
  GetDataStruct x5 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725451), LAYOUT_LOOKUP(layout3, ptr));
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:106)
  GetDataStruct x6 = exec_MemoryRead(ctx, arg0, arg1, Val(1073725452), LAYOUT_LOOKUP(layout3, len));
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:104)
  ValU32Struct x7 = x4._super;
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:107)
  EQZ(x7.high, "ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:107)");
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:106)
  ValU32Struct x8 = x6._super;
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:108)
  EQZ(x8.high, "ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:108)");
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:106)
  Val x9 = x8.low;
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:105)
  ValU32Struct x10 = x5._super;
  // HostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:11)
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:110)
  Val x11 = INVOKE_EXTERN(ctx, hostWrite, x7.low, x10.low, x10.high, x9);
  NondetRegStruct x12 = exec_NondetU16Reg(ctx, x11, LAYOUT_LOOKUP(layout3, newLen));
  Val x13 = x12._super;
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:112)
  U16RegStruct x14 = exec_U16Reg(ctx, (x9 - x13), LAYOUT_LOOKUP(layout3, diff));
  // ECallHostWrite(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:114)
  MemoryWriteStruct x15 = exec_MemoryWrite(ctx,
                                           arg0,
                                           arg1,
                                           Val(1073725450),
                                           ValU32Struct{.low = x13, .high = Val(0)},
                                           LAYOUT_LOOKUP(layout3, _0));
  return ECallOutputStruct{.state = Val(32), .s0 = Val(0), .s1 = Val(0), .s2 = Val(0)};
}
ECallOutputStruct exec_ECallHostReadWords(ExecContext& ctx,
                                          NondetRegStruct arg0,
                                          BoundLayout<NondetRegLayout> arg1,
                                          InstInputStruct arg2,
                                          Val arg3,
                                          Val arg4,
                                          BoundLayout<ECallHostReadWordsLayout> layout5) {
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:127)
  EQZ((arg2.state - Val(13)),
      "ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:127)");
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:128)
  DecomposeLow2Struct x6 = exec_DecomposeLow2(ctx, arg4, LAYOUT_LOOKUP(layout5, lenDecomp));
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:129)
  DecomposeLow2Struct x7 =
      exec_DecomposeLow2(ctx, x6.high._super, LAYOUT_LOOKUP(layout5, wordsDecomp));
  NondetRegStruct4Array x8 = x7.low2Hot._super;
  Val x9 = x7.highZero._super;
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:131)
  Val x10 = (x8[1]._super * x9);
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:132)
  Val x11 = (x8[2]._super * x9);
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:133)
  Val x12 = (x8[3]._super * x9);
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:134)
  Val x13 = (Val(1) - x9);
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:136)
  Val x14 = (((x10 + x11) + x12) + x13);
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:137)
  ECallHostReadWords__0Struct4Array x15 =
      map(Val4Array{Val(0), Val(1), Val(2), Val(3)},
          LAYOUT_LOOKUP(layout5, _0),
          std::function(
              [&](Val4Array::value_type x16,
                  BoundLayout<ECallHostReadWords__0_SuperLayout4LayoutArray::value_type> x17) {
                // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:138)
                Val x18 = (Val4Array{x10, x11, x12, x13}[to_size_t(x16)] * (arg3 + x16));
                Val x19 = (Val(1) - Val4Array{x10, x11, x12, x13}[to_size_t(x16)]);
                NondetRegStruct x20 =
                    exec_Reg(ctx, (x18 + (x19 * Val(1073725504))), LAYOUT_LOOKUP(x17, addr));
                // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:139)
                MemoryWriteUnconstrainedStruct x21 = exec_MemoryWriteUnconstrained(
                    ctx, arg0, arg1, x20._super, LAYOUT_LOOKUP(x17, _0));
                return ECallHostReadWords__0Struct{};
              }));
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:141)
  Val x22 = (arg4 - (x14 * Val(4)));
  NondetRegStruct x23 = exec_IsZero(ctx, x22, LAYOUT_LOOKUP(layout5, lenZero));
  Val x24 = x23._super;
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:146)
  Val x25 = (Val(1) - x24);
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:128)
  Val x26 = x6.low2Nonzero;
  // ECallHostReadWords(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:144)
  Val x27 = ((x24 * Val(32)) + ((x25 * x26) * Val(12)));
  return ECallOutputStruct{.state = (x27 + ((x25 * (Val(1) - x26)) * Val(13))),
                           .s0 = (arg3 + x14),
                           .s1 = Val(0),
                           .s2 = x22};
}
InstOutputStruct exec_ECall0(ExecContext& ctx,
                             NondetRegStruct arg0,
                             BoundLayout<NondetRegLayout> arg1,
                             InstInputStruct arg2,
                             BoundLayout<ECall0Layout> layout3,
                             GlobalBuf global4) {
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:153)
  BoundLayout<NondetRegLayout> x5 = LAYOUT_LOOKUP(layout3, s0);
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:155)
  BoundLayout<NondetRegLayout> x6 = LAYOUT_LOOKUP(layout3, s2);
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
  BoundLayout<ECall0OutputLayout> x7 = LAYOUT_LOOKUP(layout3, output);
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:152)
  ValU32Struct x8 = arg2.pcU32;
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:156)
  AddrDecomposeBitsStruct x9 =
      exec_AddrDecomposeBits(ctx, x8, arg2.mode, LAYOUT_LOOKUP(layout3, pcAddr));
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:157)
  EQZ(x9.low2, "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:157)");
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:152)
  NondetRegStruct8Array x10 = arg2.minorOnehot._super;
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
  BoundLayout<ECall0Arm0Layout> x11 = LAYOUT_LOOKUP(x7, arm0);
  BoundLayout<ECall0Arm1Layout> x12 = LAYOUT_LOOKUP(x7, arm1);
  BoundLayout<ECall0Arm4Layout> x13 = LAYOUT_LOOKUP(x7, arm4);
  BoundLayout<ECall0Arm5Layout> x14 = LAYOUT_LOOKUP(x7, arm5);
  BoundLayout<ECall0Arm6Layout> x15 = LAYOUT_LOOKUP(x7, arm6);
  BoundLayout<ECall0Arm7Layout> x16 = LAYOUT_LOOKUP(x7, arm7);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra5), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra6), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra7), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra4), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra5), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra6), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra7), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra0), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra1), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra2), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra3), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra4), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra5), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra6), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra7), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra8), count), _super);
  BoundLayout<Reg> x42 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra9), count), _super);
  BoundLayout<Reg> x43 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra10), count), _super);
  BoundLayout<Reg> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra11), count), _super);
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra12), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra13), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra0), count), _super);
  BoundLayout<Reg> x48 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra1), count), _super);
  // ECallOutput(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:13)
  // IllegalECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:22)
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:165)
  ECallOutputStruct x49 =
      ECallOutputStruct{.state = Val(0), .s0 = Val(0), .s1 = Val(0), .s2 = Val(0)};
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
  BoundLayout<Reg> x50 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra0), count), _super);
  BoundLayout<Reg> x51 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra1), count), _super);
  BoundLayout<Reg> x52 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra2), count), _super);
  BoundLayout<Reg> x53 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra3), count), _super);
  BoundLayout<Reg> x54 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra4), count), _super);
  BoundLayout<Reg> x55 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra5), count), _super);
  BoundLayout<Reg> x56 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra6), count), _super);
  BoundLayout<Reg> x57 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra7), count), _super);
  BoundLayout<Reg> x58 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra8), count), _super);
  BoundLayout<Reg> x59 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra9), count), _super);
  BoundLayout<Reg> x60 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra10), count), _super);
  BoundLayout<Reg> x61 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra11), count), _super);
  BoundLayout<Reg> x62 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra12), count), _super);
  BoundLayout<Reg> x63 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra13), count), _super);
  BoundLayout<Reg> x64 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra0), count), _super);
  BoundLayout<Reg> x65 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra1), count), _super);
  BoundLayout<Reg> x66 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra2), count), _super);
  BoundLayout<Reg> x67 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra3), count), _super);
  BoundLayout<Reg> x68 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra4), count), _super);
  BoundLayout<Reg> x69 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra5), count), _super);
  BoundLayout<Reg> x70 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra6), count), _super);
  BoundLayout<Reg> x71 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra7), count), _super);
  BoundLayout<Reg> x72 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra8), count), _super);
  BoundLayout<Reg> x73 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra9), count), _super);
  BoundLayout<Reg> x74 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra10), count), _super);
  BoundLayout<Reg> x75 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra11), count), _super);
  BoundLayout<Reg> x76 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra12), count), _super);
  BoundLayout<Reg> x77 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x16, _extra13), count), _super);
  ECallOutputStruct x78;
  if (to_size_t(x10[0]._super)) {
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:159)
    ECallOutputStruct x79 =
        exec_MachineECall(ctx, arg0, arg1, arg2, x9._super, LAYOUT_LOOKUP(x11, _super));
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    x78 = x79;
  } else if (to_size_t(x10[1]._super)) {
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:160)
    ECallOutputStruct x80 =
        exec_ECallTerminate(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x12, _super), global4);
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    x78 = x80;
  } else if (to_size_t(x10[2]._super)) {
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:161)
    ECallOutputStruct x81 = exec_ECallHostReadSetup(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x7, arm2));
    x78 = x81;
  } else if (to_size_t(x10[3]._super)) {
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:162)
    ECallOutputStruct x82 = exec_ECallHostWrite(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x7, arm3));
    x78 = x82;
  } else if (to_size_t(x10[4]._super)) {
    // ECallHostReadBytes(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:121)
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:163)
    EQZ((arg2.state - Val(12)),
        "loc(callsite( ECallHostReadBytes ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :121:16) "
        "at  ECall0 ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :163:24)))");
    // ECallHostReadBytes(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:122)
    EQZ(Val(2013265920),
        "loc(callsite( ECallHostReadBytes ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :122:6) at "
        " ECall0 ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :163:24)))");
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x42, Val(0));
    EQZ(LOAD(x42, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x43, Val(0));
    EQZ(LOAD(x43, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x44, Val(0));
    EQZ(LOAD(x44, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    x78 = ECallOutputStruct{.state = Val(16), .s0 = Val(0), .s1 = Val(0), .s2 = Val(0)};
  } else if (to_size_t(x10[5]._super)) {
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:164)
    NondetRegStruct x83 = back_Reg(ctx, 1, x5);
    NondetRegStruct x84 = back_Reg(ctx, 1, x6);
    ECallOutputStruct x85 = exec_ECallHostReadWords(
        ctx, arg0, arg1, arg2, x83._super, x84._super, LAYOUT_LOOKUP(x14, _super));
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x48, Val(0));
    EQZ(LOAD(x48, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    x78 = x85;
  } else if (to_size_t(x10[6]._super)) {
    // IllegalECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:21)
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:165)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalECall ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :21:6) at  "
        "ECall0 ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :165:18)))");
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
    STORE(x50, Val(0));
    EQZ(LOAD(x50, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x51, Val(0));
    EQZ(LOAD(x51, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x52, Val(0));
    EQZ(LOAD(x52, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x53, Val(0));
    EQZ(LOAD(x53, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x54, Val(0));
    EQZ(LOAD(x54, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x55, Val(0));
    EQZ(LOAD(x55, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x56, Val(0));
    EQZ(LOAD(x56, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x57, Val(0));
    EQZ(LOAD(x57, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x58, Val(0));
    EQZ(LOAD(x58, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x59, Val(0));
    EQZ(LOAD(x59, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x60, Val(0));
    EQZ(LOAD(x60, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x61, Val(0));
    EQZ(LOAD(x61, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x62, Val(0));
    EQZ(LOAD(x62, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x63, Val(0));
    EQZ(LOAD(x63, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    x78 = x49;
  } else if (to_size_t(x10[7]._super)) {
    // IllegalECall(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:21)
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:166)
    EQZ(Val(2013265920),
        "loc(callsite( IllegalECall ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :21:6) at  "
        "ECall0 ( zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir :166:18)))");
    // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
    STORE(x64, Val(0));
    EQZ(LOAD(x64, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x65, Val(0));
    EQZ(LOAD(x65, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x66, Val(0));
    EQZ(LOAD(x66, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x67, Val(0));
    EQZ(LOAD(x67, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x68, Val(0));
    EQZ(LOAD(x68, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x69, Val(0));
    EQZ(LOAD(x69, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x70, Val(0));
    EQZ(LOAD(x70, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x71, Val(0));
    EQZ(LOAD(x71, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x72, Val(0));
    EQZ(LOAD(x72, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x73, Val(0));
    EQZ(LOAD(x73, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x74, Val(0));
    EQZ(LOAD(x74, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x75, Val(0));
    EQZ(LOAD(x75, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x76, Val(0));
    EQZ(LOAD(x76, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    STORE(x77, Val(0));
    EQZ(LOAD(x77, 0), "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)");
    x78 = x49;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:168)
  NondetRegStruct x86 = exec_Reg(ctx, x78.s0, x5);
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:169)
  NondetRegStruct x87 = exec_Reg(ctx, x78.s1, LAYOUT_LOOKUP(layout3, s1));
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:170)
  NondetRegStruct x88 = exec_Reg(ctx, x78.s2, x6);
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:158)
  Val x89 = x78.state;
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:171)
  NondetRegStruct x90 = exec_IsZero(ctx, (x89 - Val(32)), LAYOUT_LOOKUP(layout3, isDecode));
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:172)
  NondetRegStruct x91 = exec_IsZero(ctx, (x89 - Val(16)), LAYOUT_LOOKUP(layout3, isP2Entry));
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:173)
  Val x92 = ((x90._super + x91._super) * Val(4));
  // DenormedValU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:20)
  // AddU32(zirgen/circuit/rv32im/v2/dsl/u32.zir:27)
  DenormedValU32Struct x93 = DenormedValU32Struct{.low = (x8.low + x92), .high = x8.high};
  NormalizeU32Struct x94 = exec_NormalizeU32(ctx, x93, LAYOUT_LOOKUP(layout3, addPC));
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:152)
  Val x95 = arg0._super;
  // GetDiffCount(zirgen/circuit/rv32im/v2/dsl/mem.zir:22)
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:175)
  Val x96 = INVOKE_EXTERN(ctx, getDiffCount, x95);
  CycleArgStruct x97 = exec_CycleArg(ctx, neg_0(x96), x95, LAYOUT_LOOKUP(layout3, arg));
  // ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:177)
  Val x98 = (x97.cycle._super - x95);
  EQZ(x98, "ECall0(zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir:177)");
  return InstOutputStruct{.newPc = x94._super, .newState = x89, .newMode = Val(1)};
}
NondetRegStruct exec_SBox(ExecContext& ctx, Val arg0, BoundLayout<SBoxLayout> layout1) {
  // SBox(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:25)
  NondetRegStruct x2 = exec_Reg(ctx, ((arg0 * arg0) * arg0), LAYOUT_LOOKUP(layout1, cubed));
  Val x3 = x2._super;
  // SBox(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:26)
  NondetRegStruct x4 = exec_Reg(ctx, ((x3 * x3) * arg0), LAYOUT_LOOKUP(layout1, _super));
  return x4;
}
MultiplyByMIntStruct exec_DoIntRound(ExecContext& ctx,
                                     Val24Array arg0,
                                     Val arg1,
                                     BoundLayout<DoIntRoundLayout> layout2) {
  // DoIntRound(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:32)
  NondetRegStruct x3 = exec_SBox(ctx, (arg0[0] + arg1), LAYOUT_LOOKUP(layout2, sbox));
  Val x4 = x3._super;
  // MultiplyByMInt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:13)
  // DoIntRound(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:34)
  Val x5 = (((x4 + arg0[1]) + arg0[2]) + arg0[3]);
  Val x6 = (((x5 + arg0[4]) + arg0[5]) + arg0[6]);
  Val x7 = (((x6 + arg0[7]) + arg0[8]) + arg0[9]);
  Val x8 = (((x7 + arg0[10]) + arg0[11]) + arg0[12]);
  Val x9 = (((x8 + arg0[13]) + arg0[14]) + arg0[15]);
  Val x10 = (((x9 + arg0[16]) + arg0[17]) + arg0[18]);
  Val x11 = (((x10 + arg0[19]) + arg0[20]) + arg0[21]);
  Val x12 = ((x11 + arg0[22]) + arg0[23]);
  // MultiplyByMInt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:14)
  MultiplyByMInt_Super_SuperStruct24Array x13 = MultiplyByMInt_Super_SuperStruct24Array{
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (x4 * Val(1083257840)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[1] * Val(375892129)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[2] * Val(111593398)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[3] * Val(1867716110)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[4] * Val(658182609)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[5] * Val(51866717)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[6] * Val(1928969209)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[7] * Val(1942928017)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[8] * Val(1558116381)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[9] * Val(20525701)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[10] * Val(1188752902)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[11] * Val(106789798)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[12] * Val(1389833583)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[13] * Val(98371040)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[14] * Val(1001081699)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[15] * Val(1792686146)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[16] * Val(801504236)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[17] * Val(1997365680)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[18] * Val(1461037801)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[19] * Val(65998480)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[20] * Val(1974912880)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[21] * Val(606789471)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[22] * Val(13683276)))},
      MultiplyByMInt_Super_SuperStruct{._super = (x12 + (arg0[23] * Val(918610824)))}};
  return MultiplyByMIntStruct{._super = x13};
}
DoIntRoundsStruct
exec_DoIntRounds(ExecContext& ctx, Val24Array arg0, BoundLayout<DoIntRoundsLayout> layout1) {
  // DoIntRounds(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:38)
  DoIntRounds__0_SuperStruct21Array x2 =
      DoIntRounds__0_SuperStruct21Array{DoIntRounds__0_SuperStruct{._super = Val(497520322)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1930103076)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1052077299)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1540960371)},
                                        DoIntRounds__0_SuperStruct{._super = Val(924863639)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1365519753)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1726563304)},
                                        DoIntRounds__0_SuperStruct{._super = Val(440300254)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1891545577)},
                                        DoIntRounds__0_SuperStruct{._super = Val(822033215)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1111544260)},
                                        DoIntRounds__0_SuperStruct{._super = Val(308575117)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1708681573)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1240419708)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1199068823)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1186174623)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1551596046)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1886977120)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1327682690)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1210751726)},
                                        DoIntRounds__0_SuperStruct{._super = Val(1810596765)}};
  Val24Array x3 =
      reduce(x2,
             arg0,
             LAYOUT_LOOKUP(layout1, _super),
             std::function([&](Val24Array x4,
                               DoIntRounds__0_SuperStruct21Array::value_type x5,
                               BoundLayout<DoIntRoundLayout21LayoutArray::value_type> x6) {
               MultiplyByMIntStruct x7 = exec_DoIntRound(ctx, x4, x5._super, x6);
               MultiplyByMInt_Super_SuperStruct24Array x8 = x7._super;
               Val24Array x9 = Val24Array{
                   x8[0]._super,  x8[1]._super,  x8[2]._super,  x8[3]._super,  x8[4]._super,
                   x8[5]._super,  x8[6]._super,  x8[7]._super,  x8[8]._super,  x8[9]._super,
                   x8[10]._super, x8[11]._super, x8[12]._super, x8[13]._super, x8[14]._super,
                   x8[15]._super, x8[16]._super, x8[17]._super, x8[18]._super, x8[19]._super,
                   x8[20]._super, x8[21]._super, x8[22]._super, x8[23]._super};
               return x9;
             }));
  return DoIntRoundsStruct{._super = x3};
}
MultiplyByMExtStruct exec_DoExtRound(ExecContext& ctx,
                                     Val24Array arg0,
                                     Val24Array arg1,
                                     BoundLayout<DoExtRoundLayout> layout2) {
  // DoExtRound(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:103)
  NondetRegStruct24Array x3 =
      map(Val24Array{Val(0),  Val(1),  Val(2),  Val(3),  Val(4),  Val(5),  Val(6),  Val(7),
                     Val(8),  Val(9),  Val(10), Val(11), Val(12), Val(13), Val(14), Val(15),
                     Val(16), Val(17), Val(18), Val(19), Val(20), Val(21), Val(22), Val(23)},
          LAYOUT_LOOKUP(layout2, _0),
          std::function([&](Val24Array::value_type x4,
                            BoundLayout<SBoxLayout24LayoutArray::value_type> x5) {
            NondetRegStruct x6 = exec_SBox(ctx, (arg0[to_size_t(x4)] + arg1[to_size_t(x4)]), x5);
            return x6;
          }));
  Val x7 = x3[1]._super;
  Val x8 = x3[3]._super;
  Val x9 = x3[5]._super;
  Val x10 = x3[7]._super;
  Val x11 = x3[9]._super;
  Val x12 = x3[11]._super;
  Val x13 = x3[13]._super;
  Val x14 = x3[15]._super;
  Val x15 = x3[17]._super;
  Val x16 = x3[19]._super;
  Val x17 = x3[21]._super;
  Val x18 = x3[23]._super;
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  // MultiplyByMExt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:61)
  Val x19 = (x3[0]._super + x7);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x20 = (x3[2]._super + x8);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x21 = ((x7 * Val(2)) + x20);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x22 = ((x8 * Val(2)) + x19);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x23 = ((x20 * Val(4)) + x22);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x24 = ((x19 * Val(4)) + x21);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x25 = (x22 + x24);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x26 = (x21 + x23);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x27 = (x3[4]._super + x9);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x28 = (x3[6]._super + x10);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x29 = ((x9 * Val(2)) + x28);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x30 = ((x10 * Val(2)) + x27);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x31 = ((x28 * Val(4)) + x30);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x32 = ((x27 * Val(4)) + x29);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x33 = (x30 + x32);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x34 = (x29 + x31);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x35 = (x3[8]._super + x11);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x36 = (x3[10]._super + x12);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x37 = ((x11 * Val(2)) + x36);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x38 = ((x12 * Val(2)) + x35);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x39 = ((x36 * Val(4)) + x38);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x40 = ((x35 * Val(4)) + x37);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x41 = (x38 + x40);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x42 = (x37 + x39);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x43 = (x3[12]._super + x13);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x44 = (x3[14]._super + x14);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x45 = ((x13 * Val(2)) + x44);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x46 = ((x14 * Val(2)) + x43);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x47 = ((x44 * Val(4)) + x46);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x48 = ((x43 * Val(4)) + x45);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x49 = (x46 + x48);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x50 = (x45 + x47);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x51 = (x3[16]._super + x15);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x52 = (x3[18]._super + x16);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x53 = ((x15 * Val(2)) + x52);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x54 = ((x16 * Val(2)) + x51);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x55 = ((x52 * Val(4)) + x54);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x56 = ((x51 * Val(4)) + x53);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x57 = (x54 + x56);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x58 = (x53 + x55);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x59 = (x3[20]._super + x17);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x60 = (x3[22]._super + x18);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x61 = ((x17 * Val(2)) + x60);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x62 = ((x18 * Val(2)) + x59);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x63 = ((x60 * Val(4)) + x62);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x64 = ((x59 * Val(4)) + x61);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x65 = (x62 + x64);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x66 = (x61 + x63);
  // ReduceVec4(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:54)
  // MultiplyByMExt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:64)
  Val x67 = (((x25 + x33) + x41) + x49);
  Val x68 = (((x24 + x32) + x40) + x48);
  Val x69 = (((x26 + x34) + x42) + x50);
  Val x70 = (((x23 + x31) + x39) + x47);
  Val x71 = ((x67 + x57) + x65);
  Val x72 = ((x68 + x56) + x64);
  Val x73 = ((x69 + x58) + x66);
  Val x74 = ((x70 + x55) + x63);
  // MultiplyByMExt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:65)
  MultiplyByMExt_Super_SuperStruct24Array x75 = MultiplyByMExt_Super_SuperStruct24Array{
      MultiplyByMExt_Super_SuperStruct{._super = (x25 + x71)},
      MultiplyByMExt_Super_SuperStruct{._super = (x24 + x72)},
      MultiplyByMExt_Super_SuperStruct{._super = (x26 + x73)},
      MultiplyByMExt_Super_SuperStruct{._super = (x23 + x74)},
      MultiplyByMExt_Super_SuperStruct{._super = (x33 + x71)},
      MultiplyByMExt_Super_SuperStruct{._super = (x32 + x72)},
      MultiplyByMExt_Super_SuperStruct{._super = (x34 + x73)},
      MultiplyByMExt_Super_SuperStruct{._super = (x31 + x74)},
      MultiplyByMExt_Super_SuperStruct{._super = (x41 + x71)},
      MultiplyByMExt_Super_SuperStruct{._super = (x40 + x72)},
      MultiplyByMExt_Super_SuperStruct{._super = (x42 + x73)},
      MultiplyByMExt_Super_SuperStruct{._super = (x39 + x74)},
      MultiplyByMExt_Super_SuperStruct{._super = (x49 + x71)},
      MultiplyByMExt_Super_SuperStruct{._super = (x48 + x72)},
      MultiplyByMExt_Super_SuperStruct{._super = (x50 + x73)},
      MultiplyByMExt_Super_SuperStruct{._super = (x47 + x74)},
      MultiplyByMExt_Super_SuperStruct{._super = (x57 + x71)},
      MultiplyByMExt_Super_SuperStruct{._super = (x56 + x72)},
      MultiplyByMExt_Super_SuperStruct{._super = (x58 + x73)},
      MultiplyByMExt_Super_SuperStruct{._super = (x55 + x74)},
      MultiplyByMExt_Super_SuperStruct{._super = (x65 + x71)},
      MultiplyByMExt_Super_SuperStruct{._super = (x64 + x72)},
      MultiplyByMExt_Super_SuperStruct{._super = (x66 + x73)},
      MultiplyByMExt_Super_SuperStruct{._super = (x63 + x74)}};
  return MultiplyByMExtStruct{._super = x75};
}
MultiplyByMExtStruct exec_DoExtRoundByIdx(ExecContext& ctx,
                                          Val24Array arg0,
                                          Val arg1,
                                          BoundLayout<DoExtRoundByIdxLayout> layout2) {
  // DoExtRoundByIdx(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:115)
  OneHot_8_Struct x3 = exec_OneHot_8_(ctx, arg1, LAYOUT_LOOKUP(layout2, idxHot));
  NondetRegStruct8Array x4 = x3.bits;
  Val x5 = x4[0]._super;
  Val x6 = x4[1]._super;
  Val x7 = x4[2]._super;
  Val x8 = x4[3]._super;
  Val x9 = x4[4]._super;
  Val x10 = x4[5]._super;
  Val x11 = x4[6]._super;
  Val x12 = x4[7]._super;
  // AddConsts(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:107)
  // DoExtRoundByIdx(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:118)
  Val x13 = (((x5 * Val(262278199)) + (x6 * Val(695835963))) + (x7 * Val(1147522062)));
  Val x14 = (((x5 * Val(127253399)) + (x6 * Val(1845603984))) + (x7 * Val(27129487)));
  Val x15 = (((x5 * Val(314968988)) + (x6 * Val(540703332))) + (x7 * Val(1257820264)));
  Val x16 = (((x5 * Val(246143118)) + (x6 * Val(1333667262))) + (x7 * Val(142102402)));
  Val x17 = (((x5 * Val(157582794)) + (x6 * Val(1917861751))) + (x7 * Val(217046702)));
  Val x18 = (((x5 * Val(118043943)) + (x6 * Val(1170029417))) + (x7 * Val(1664590951)));
  Val x19 = (((x5 * Val(454905424)) + (x6 * Val(1989924532))) + (x7 * Val(855276054)));
  Val x20 = (((x5 * Val(815798990)) + (x6 * Val(1518763784))) + (x7 * Val(1215259350)));
  Val x21 = (((x5 * Val(1004040026)) + (x6 * Val(1339793538))) + (x7 * Val(946500736)));
  Val x22 = (((x5 * Val(1773108264)) + (x6 * Val(622609176))) + (x7 * Val(552696906)));
  Val x23 = (((x5 * Val(1066694495)) + (x6 * Val(686842369))) + (x7 * Val(1424297384)));
  Val x24 = (((x5 * Val(1930780904)) + (x6 * Val(1737016378))) + (x7 * Val(538103555)));
  Val x25 = (((x5 * Val(1180307149)) + (x6 * Val(1282239129))) + (x7 * Val(1608853840)));
  Val x26 = (((x5 * Val(1464793095)) + (x6 * Val(897025192))) + (x7 * Val(162510541)));
  Val x27 = (((x5 * Val(1660766320)) + (x6 * Val(716894289))) + (x7 * Val(623051854)));
  Val x28 = (((x5 * Val(1389166148)) + (x6 * Val(1997503974))) + (x7 * Val(1549062383)));
  Val x29 = (((x5 * Val(343354132)) + (x6 * Val(395622276))) + (x7 * Val(1908416316)));
  Val x30 = (((x5 * Val(1307439985)) + (x6 * Val(1201063290))) + (x7 * Val(1622328571)));
  Val x31 = (((x5 * Val(638242172)) + (x6 * Val(1917549072))) + (x7 * Val(1079030649)));
  Val x32 = (((x5 * Val(525458520)) + (x6 * Val(1150912935))) + (x7 * Val(1584033957)));
  Val x33 = (((x5 * Val(1964135730)) + (x6 * Val(1687379185))) + (x7 * Val(1099252725)));
  Val x34 = (((x5 * Val(1751797115)) + (x6 * Val(1507936940))) + (x7 * Val(1910423126)));
  Val x35 = (((x5 * Val(1421525369)) + (x6 * Val(241306552))) + (x7 * Val(447555988)));
  Val x36 = (((x5 * Val(831813382)) + (x6 * Val(989176635))) + (x7 * Val(862495875)));
  Val x37 = (((x13 + (x8 * Val(128479034))) + (x9 * Val(53041581))) + (x10 * Val(1209164052)));
  Val x38 = (((x14 + (x8 * Val(1587822577))) + (x9 * Val(723038058))) + (x10 * Val(714957516)));
  Val x39 = (((x15 + (x8 * Val(608401422))) + (x9 * Val(1439947916))) + (x10 * Val(390340387)));
  Val x40 = (((x16 + (x8 * Val(1290028279))) + (x9 * Val(1136469704))) + (x10 * Val(1213686459)));
  Val x41 = (((x17 + (x8 * Val(342857858))) + (x9 * Val(205609311))) + (x10 * Val(790726260)));
  Val x42 = (((x18 + (x8 * Val(825405577))) + (x9 * Val(1883820770))) + (x10 * Val(117294666)));
  Val x43 = (((x19 + (x8 * Val(427731030))) + (x9 * Val(14387587))) + (x10 * Val(140621810)));
  Val x44 = (((x20 + (x8 * Val(1718628547))) + (x9 * Val(720724951))) + (x10 * Val(993455846)));
  Val x45 = (((x21 + (x8 * Val(588764636))) + (x9 * Val(1854174607))) + (x10 * Val(1889603648)));
  Val x46 = (((x22 + (x8 * Val(204228775))) + (x9 * Val(1629316321))) + (x10 * Val(78845751)));
  Val x47 = (((x23 + (x8 * Val(1454563174))) + (x9 * Val(530151394))) + (x10 * Val(925018226)));
  Val x48 = (((x24 + (x8 * Val(1740472809))) + (x9 * Val(1679178250))) + (x10 * Val(708123747)));
  Val x49 = (((x25 + (x8 * Val(1338899225))) + (x9 * Val(1549779579))) + (x10 * Val(1647665372)));
  Val x50 = (((x26 + (x8 * Val(1269493554))) + (x9 * Val(48375137))) + (x10 * Val(1649953458)));
  Val x51 = (((x27 + (x8 * Val(53007114))) + (x9 * Val(976057819))) + (x10 * Val(942439428)));
  Val x52 = (((x28 + (x8 * Val(1647670797))) + (x9 * Val(463976218))) + (x10 * Val(1006235079)));
  Val x53 = (((x29 + (x8 * Val(306391314))) + (x9 * Val(875839332))) + (x10 * Val(238616145)));
  Val x54 = (((x30 + (x8 * Val(172614232))) + (x9 * Val(1946596189))) + (x10 * Val(930036496)));
  Val x55 = (((x31 + (x8 * Val(51256176))) + (x9 * Val(434078361))) + (x10 * Val(1401020792)));
  Val x56 = (((x32 + (x8 * Val(1221257987))) + (x9 * Val(1878280202))) + (x10 * Val(989618631)));
  Val x57 = (((x33 + (x8 * Val(1239734761))) + (x9 * Val(1363837384))) + (x10 * Val(1545325389)));
  Val x58 = (((x34 + (x8 * Val(273790406))) + (x9 * Val(1470845646))) + (x10 * Val(1715719711)));
  Val x59 = (((x35 + (x8 * Val(1781980094))) + (x9 * Val(1792450386))) + (x10 * Val(755691969)));
  Val x60 = (((x36 + (x8 * Val(1291790245))) + (x9 * Val(1040977421))) + (x10 * Val(150307788)));
  Val24Array x61 = Val24Array{((x37 + (x11 * Val(1567618575))) + (x12 * Val(1206940496))),
                              ((x38 + (x11 * Val(1663353317))) + (x12 * Val(1896271507))),
                              ((x39 + (x11 * Val(1950429111))) + (x12 * Val(1003792297))),
                              ((x40 + (x11 * Val(1891637550))) + (x12 * Val(738091882))),
                              ((x41 + (x11 * Val(192082241))) + (x12 * Val(1124078057))),
                              ((x42 + (x11 * Val(1080533265))) + (x12 * Val(1889898))),
                              ((x43 + (x11 * Val(1463323727))) + (x12 * Val(813674331))),
                              ((x44 + (x11 * Val(890243564))) + (x12 * Val(228520958))),
                              ((x45 + (x11 * Val(158646617))) + (x12 * Val(1832911930))),
                              ((x46 + (x11 * Val(1402624179))) + (x12 * Val(781141772))),
                              ((x47 + (x11 * Val(59510015))) + (x12 * Val(459826664))),
                              ((x48 + (x11 * Val(1198261138))) + (x12 * Val(202271745))),
                              ((x49 + (x11 * Val(1065075039))) + (x12 * Val(1296144415))),
                              ((x50 + (x11 * Val(1150410028))) + (x12 * Val(1111203133))),
                              ((x51 + (x11 * Val(1293938517))) + (x12 * Val(1090783436))),
                              ((x52 + (x11 * Val(76770019))) + (x12 * Val(641665156))),
                              ((x53 + (x11 * Val(1478577620))) + (x12 * Val(1393671120))),
                              ((x54 + (x11 * Val(1748789933))) + (x12 * Val(1303271640))),
                              ((x55 + (x11 * Val(457372011))) + (x12 * Val(809508074))),
                              ((x56 + (x11 * Val(1841795381))) + (x12 * Val(162506101))),
                              ((x57 + (x11 * Val(760115692))) + (x12 * Val(1262312258))),
                              ((x58 + (x11 * Val(1042892522))) + (x12 * Val(1672219447))),
                              ((x59 + (x11 * Val(1507649755))) + (x12 * Val(1608891156))),
                              ((x60 + (x11 * Val(1827572010))) + (x12 * Val(1380248020)))};
  // DoExtRoundByIdx(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:122)
  MultiplyByMExtStruct x62 = exec_DoExtRound(ctx, arg0, x61, LAYOUT_LOOKUP(layout2, _super));
  return x62;
}
PoseidonStateStruct
back_PoseidonState(ExecContext& ctx, Index distance0, BoundLayout<PoseidonStateLayout> layout1) {
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:34)
  NondetRegStruct x2 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, hasState));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:35)
  NondetRegStruct x3 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, stateAddr));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:36)
  NondetRegStruct x4 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, bufOutAddr));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:37)
  NondetRegStruct x5 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, isElem));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:38)
  NondetRegStruct x6 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, checkOut));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:39)
  NondetRegStruct x7 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, loadTxType));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:41)
  NondetRegStruct x8 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, nextState));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:42)
  NondetRegStruct x9 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, subState));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:43)
  NondetRegStruct x10 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, bufInAddr));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:44)
  NondetRegStruct x11 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, count));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:45)
  NondetRegStruct x12 = back_Reg(ctx, distance0, LAYOUT_LOOKUP(layout1, mode));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:47)
  NondetRegStruct24Array x13 =
      map(Val24Array{Val(0),  Val(1),  Val(2),  Val(3),  Val(4),  Val(5),  Val(6),  Val(7),
                     Val(8),  Val(9),  Val(10), Val(11), Val(12), Val(13), Val(14), Val(15),
                     Val(16), Val(17), Val(18), Val(19), Val(20), Val(21), Val(22), Val(23)},
          LAYOUT_LOOKUP(layout1, inner),
          std::function([&](Val24Array::value_type x14,
                            BoundLayout<NondetRegLayout24LayoutArray::value_type> x15) {
            NondetRegStruct x16 = back_Reg(ctx, distance0, x15);
            return x16;
          }));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:48)
  NondetExtRegStruct x17 = back_ExtReg(ctx, distance0, LAYOUT_LOOKUP(layout1, zcheck));
  return PoseidonStateStruct{.hasState = x2,
                             .stateAddr = x3,
                             .bufOutAddr = x4,
                             .isElem = x5,
                             .checkOut = x6,
                             .loadTxType = x7,
                             .nextState = x8,
                             .subState = x9,
                             .bufInAddr = x10,
                             .count = x11,
                             .mode = x12,
                             .inner = x13,
                             .zcheck = x17};
}
PoseidonStateStruct exec_PoseidonState(ExecContext& ctx,
                                       PoseidonOpDefStruct arg0,
                                       Val arg1,
                                       Val arg2,
                                       Val arg3,
                                       Val arg4,
                                       Val arg5,
                                       Val24Array arg6,
                                       ExtVal arg7,
                                       BoundLayout<PoseidonStateLayout> layout8) {
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:34)
  NondetRegStruct x9 = exec_Reg(ctx, arg0.hasState, LAYOUT_LOOKUP(layout8, hasState));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:35)
  NondetRegStruct x10 = exec_Reg(ctx, arg0.stateAddr, LAYOUT_LOOKUP(layout8, stateAddr));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:36)
  NondetRegStruct x11 = exec_Reg(ctx, arg0.bufOutAddr, LAYOUT_LOOKUP(layout8, bufOutAddr));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:37)
  NondetRegStruct x12 = exec_Reg(ctx, arg0.isElem, LAYOUT_LOOKUP(layout8, isElem));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:38)
  NondetRegStruct x13 = exec_Reg(ctx, arg0.checkOut, LAYOUT_LOOKUP(layout8, checkOut));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:39)
  NondetRegStruct x14 = exec_Reg(ctx, arg0.loadTxType, LAYOUT_LOOKUP(layout8, loadTxType));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:41)
  NondetRegStruct x15 = exec_Reg(ctx, arg1, LAYOUT_LOOKUP(layout8, nextState));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:42)
  NondetRegStruct x16 = exec_Reg(ctx, arg2, LAYOUT_LOOKUP(layout8, subState));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:43)
  NondetRegStruct x17 = exec_Reg(ctx, arg3, LAYOUT_LOOKUP(layout8, bufInAddr));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:44)
  NondetRegStruct x18 = exec_Reg(ctx, arg4, LAYOUT_LOOKUP(layout8, count));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:45)
  NondetRegStruct x19 = exec_Reg(ctx, arg5, LAYOUT_LOOKUP(layout8, mode));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:47)
  NondetRegStruct24Array x20 =
      map(arg6,
          LAYOUT_LOOKUP(layout8, inner),
          std::function([&](Val24Array::value_type x21,
                            BoundLayout<NondetRegLayout24LayoutArray::value_type> x22) {
            NondetRegStruct x23 = exec_Reg(ctx, x21, x22);
            return x23;
          }));
  // PoseidonState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:48)
  NondetExtRegStruct x24 = exec_ExtReg(ctx, arg7, LAYOUT_LOOKUP(layout8, zcheck));
  return PoseidonStateStruct{.hasState = x9,
                             .stateAddr = x10,
                             .bufOutAddr = x11,
                             .isElem = x12,
                             .checkOut = x13,
                             .loadTxType = x14,
                             .nextState = x15,
                             .subState = x16,
                             .bufInAddr = x17,
                             .count = x18,
                             .mode = x19,
                             .inner = x20,
                             .zcheck = x24};
}
PoseidonStateStruct exec_PoseidonInvalid(ExecContext& ctx,
                                         BoundLayout<PoseidonStateLayout> layout0) {
  // PoseidonInvalid(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:52)
  EQZ(Val(2013265920), "PoseidonInvalid(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:52)");
  // PoseidonInvalid(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:61)
  PoseidonStateStruct x1 =
      exec_PoseidonState(ctx,
                         PoseidonOpDefStruct{.hasState = Val(0),
                                             .stateAddr = Val(0),
                                             .bufOutAddr = Val(0),
                                             .isElem = Val(0),
                                             .checkOut = Val(0),
                                             .loadTxType = Val(0)},
                         Val(0),
                         Val(0),
                         Val(0),
                         Val(0),
                         Val(0),
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout0);
  return x1;
}
ReadAddrStruct exec_ReadAddr(ExecContext& ctx,
                             NondetRegStruct arg0,
                             BoundLayout<NondetRegLayout> arg1,
                             Val arg2,
                             BoundLayout<ReadAddrLayout> layout3) {
  // ReadAddr(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:78)
  GetDataStruct x4 =
      exec_MemoryRead(ctx, arg0, arg1, (arg2 + Val(1073725440)), LAYOUT_LOOKUP(layout3, addr32));
  ValU32Struct x5 = x4._super;
  // ReadAddr(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:79)
  Val x6 = ((x5.high * Val(16384)) + (x5.low * Val(1509949441)));
  return ReadAddrStruct{._super = x6};
}
PoseidonStateStruct exec_PoseidonEcall(ExecContext& ctx,
                                       NondetRegStruct arg0,
                                       BoundLayout<NondetRegLayout> arg1,
                                       Val arg2,
                                       BoundLayout<PoseidonEcallLayout> layout3) {
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:84)
  ReadAddrStruct x4 = exec_ReadAddr(ctx, arg0, arg1, Val(10), LAYOUT_LOOKUP(layout3, stateAddr));
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:85)
  ReadAddrStruct x5 = exec_ReadAddr(ctx, arg0, arg1, Val(11), LAYOUT_LOOKUP(layout3, bufInAddr));
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:86)
  ReadAddrStruct x6 = exec_ReadAddr(ctx, arg0, arg1, Val(12), LAYOUT_LOOKUP(layout3, bufOutAddr));
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:87)
  GetDataStruct x7 =
      exec_MemoryRead(ctx, arg0, arg1, Val(1073725453), LAYOUT_LOOKUP(layout3, bitsAndCount));
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:84)
  Val x8 = x4._super;
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:90)
  NondetRegStruct x9 = exec_IsZero(ctx, x8, LAYOUT_LOOKUP(layout3, _0));
  Val x10 = (Val(1) - x9._super);
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:87)
  ValU32Struct x11 = x7._super;
  Val x12 = x11.low;
  Val x13 = x11.high;
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:94)
  NondetRegStruct x14 = exec_NondetBitReg(
      ctx, (bitAnd(x13, Val(32768)) * Val(2013204481)), LAYOUT_LOOKUP(layout3, isElem));
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:95)
  NondetRegStruct x15 = exec_NondetBitReg(
      ctx, (bitAnd(x13, Val(16384)) * Val(2013143041)), LAYOUT_LOOKUP(layout3, checkOut));
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:94)
  Val x16 = x14._super;
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:95)
  Val x17 = x15._super;
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:96)
  EQZ((x13 - ((x16 * Val(32768)) + (x17 * Val(16384)))),
      "PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:96)");
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:99)
  NondetRegStruct x18 = exec_IsZero(ctx, x12, LAYOUT_LOOKUP(layout3, countZero));
  Val x19 = x18._super;
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:102)
  Val x20 = (Val(1) - x19);
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:101)
  Val x21 = ((x19 * Val(32)) + ((x20 * x10) * Val(17)));
  // PoseidonEcall(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:105)
  PoseidonStateStruct x22 =
      exec_PoseidonState(ctx,
                         PoseidonOpDefStruct{.hasState = x10,
                                             .stateAddr = x8,
                                             .bufOutAddr = x6._super,
                                             .isElem = x16,
                                             .checkOut = x17,
                                             .loadTxType = Val(0)},
                         (x21 + ((x20 * (Val(1) - x10)) * Val(18))),
                         Val(0),
                         x5._super,
                         x12,
                         arg2,
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         LAYOUT_LOOKUP(layout3, _super));
  return x22;
}
PoseidonStateStruct exec_PoseidonPagingEntry(ExecContext& ctx,
                                             NondetRegStruct arg0,
                                             Val arg1,
                                             BoundLayout<PoseidonStateLayout> layout2) {
  // Div(<preamble>:19)
  // PoseidonPagingEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:110)
  Val x3 = (arg1 * Val(1342177281));
  // PoseidonPagingEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:114)
  Val x4 = ((Val(1) - x3) * Val(1140850688));
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // PoseidonPagingEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:111)
  PoseidonOpDefStruct x5 = PoseidonOpDefStruct{.hasState = Val(0),
                                               .stateAddr = Val(0),
                                               .bufOutAddr = ((x3 * Val(1073741824)) + x4),
                                               .isElem = Val(1),
                                               .checkOut = Val(1),
                                               .loadTxType = Val(1)};
  // PoseidonPagingEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:119)
  PoseidonStateStruct x6 =
      exec_PoseidonState(ctx,
                         x5,
                         Val(22),
                         Val(0),
                         Val(0),
                         Val(0),
                         arg1,
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout2);
  return x6;
}
PoseidonStateStruct exec_PoseidonEntry(ExecContext& ctx,
                                       NondetRegStruct arg0,
                                       BoundLayout<NondetRegLayout> arg1,
                                       ValU32Struct arg2,
                                       Val arg3,
                                       BoundLayout<PoseidonEntryLayout> layout4) {
  // PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)
  BoundLayout<PoseidonEntry_SuperLayout> x5 = LAYOUT_LOOKUP(layout4, _super);
  // PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:131)
  NondetRegStruct x6 = exec_IsZero(ctx, (arg2.low + arg2.high), LAYOUT_LOOKUP(layout4, pcZero));
  Val x7 = x6._super;
  // PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)
  BoundLayout<PoseidonEntryArm0Layout> x8 = LAYOUT_LOOKUP(x5, arm0);
  BoundLayout<Reg> x9 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x10 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x11 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  BoundLayout<Reg> x12 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra3), count), _super);
  BoundLayout<Reg> x13 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra4), count), _super);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra5), count), _super);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra6), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra7), count), _super);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra8), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra9), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra10), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra11), count), _super);
  PoseidonStateStruct x21;
  if (to_size_t(x7)) {
    // PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:133)
    PoseidonStateStruct x22 = exec_PoseidonPagingEntry(ctx, arg0, arg3, LAYOUT_LOOKUP(x8, _super));
    // PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)
    STORE(x9, Val(0));
    EQZ(LOAD(x9, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x10, Val(0));
    EQZ(LOAD(x10, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x11, Val(0));
    EQZ(LOAD(x11, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x12, Val(0));
    EQZ(LOAD(x12, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x13, Val(0));
    EQZ(LOAD(x13, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x14, Val(0));
    EQZ(LOAD(x14, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)");
    x21 = x22;
  } else if (to_size_t((Val(1) - x7))) {
    // PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:135)
    PoseidonStateStruct x23 = exec_PoseidonEcall(ctx, arg0, arg1, arg3, LAYOUT_LOOKUP(x5, arm1));
    x21 = x23;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // PoseidonEntry(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:132)
  PoseidonStateStruct x24 = back_PoseidonState(ctx, 0, LAYOUT_LOOKUP(x5, _super));
  return x24;
}
ReadElemStruct exec_ReadElem(ExecContext& ctx,
                             NondetRegStruct arg0,
                             BoundLayout<NondetRegLayout> arg1,
                             Val arg2,
                             BoundLayout<ReadElemLayout> layout3) {
  // ReadElem(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:140)
  GetDataStruct x4 = exec_MemoryRead(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(layout3, elem32));
  ValU32Struct x5 = x4._super;
  // ReadElem(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:141)
  Val x6 = ((x5.high * Val(65536)) + x5.low);
  return ReadElemStruct{._super = x6};
}
PoseidonStateStruct exec_PoseidonLoadState(ExecContext& ctx,
                                           NondetRegStruct arg0,
                                           BoundLayout<NondetRegLayout> arg1,
                                           PoseidonStateStruct arg2,
                                           BoundLayout<PoseidonLoadStateLayout> layout3) {
  // PoseidonLoadState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:144)
  Val x4 = arg2.stateAddr._super;
  // PoseidonLoadState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:146)
  ReadElemStruct8Array x5 =
      map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
          LAYOUT_LOOKUP(layout3, loadList),
          std::function([&](Val8Array::value_type x6,
                            BoundLayout<ReadElemLayout8LayoutArray::value_type> x7) {
            // PoseidonLoadState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:147)
            ReadElemStruct x8 = exec_ReadElem(ctx, arg0, arg1, (x4 + x6), x7);
            return x8;
          }));
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // GetDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:72)
  // PoseidonLoadState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:156)
  PoseidonOpDefStruct x9 = PoseidonOpDefStruct{.hasState = arg2.hasState._super,
                                               .stateAddr = x4,
                                               .bufOutAddr = arg2.bufOutAddr._super,
                                               .isElem = arg2.isElem._super,
                                               .checkOut = arg2.checkOut._super,
                                               .loadTxType = arg2.loadTxType._super};
  // PoseidonLoadState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:149)
  Val24Array x10 = Val24Array{Val(0),       Val(0),       Val(0),       Val(0),       Val(0),
                              Val(0),       Val(0),       Val(0),       Val(0),       Val(0),
                              Val(0),       Val(0),       Val(0),       Val(0),       Val(0),
                              Val(0),       x5[0]._super, x5[1]._super, x5[2]._super, x5[3]._super,
                              x5[4]._super, x5[5]._super, x5[6]._super, x5[7]._super};
  // PoseidonLoadState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:156)
  PoseidonStateStruct x11 = exec_PoseidonState(ctx,
                                               x9,
                                               Val(18),
                                               Val(0),
                                               arg2.bufInAddr._super,
                                               arg2.count._super,
                                               arg2.mode._super,
                                               x10,
                                               ExtVal(0, 0, 0, 0),
                                               LAYOUT_LOOKUP(layout3, _super));
  return x11;
}
PoseidonStateStruct exec_PoseidonLoadInShort(ExecContext& ctx,
                                             NondetRegStruct arg0,
                                             BoundLayout<NondetRegLayout> arg1,
                                             PoseidonStateStruct arg2,
                                             BoundLayout<PoseidonLoadInShortLayout> layout3,
                                             GlobalBuf global4) {
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:174)
  Val x5 = arg2.loadTxType._super;
  // Log(<preamble>:22)
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:175)
  INVOKE_EXTERN(ctx, log, "txnType", std::initializer_list<Val>{x5});
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:176)
  OneHot_3_Struct x6 = exec_OneHot_3_(ctx, x5, LAYOUT_LOOKUP(layout3, txType));
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:174)
  Val x7 = arg2.bufInAddr._super;
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:177)
  GetDataStruct8Array x8 =
      map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
          LAYOUT_LOOKUP(layout3, loadList),
          std::function([&](Val8Array::value_type x9,
                            BoundLayout<MemoryGetLayout8LayoutArray::value_type> x10) {
            // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:178)
            GetDataStruct x11 = exec_MemoryGet(ctx, arg0, arg1, (x7 + x9), x6, x10);
            return x11;
          }));
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:159)
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:180)
  BoundLayout<_globalLayout> x12 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:160)
  BoundLayout<NondetExtRegLayout> x13 = LAYOUT_LOOKUP(x12, rng);
  NondetExtRegStruct x14 = back_ExtReg(ctx, 0, x13);
  ExtVal x15 = x14._super;
  // PolyEvalStateReduce(zirgen/circuit/rv32im/v2/dsl/poly.zir:14)
  // PolyEval(zirgen/circuit/rv32im/v2/dsl/poly.zir:18)
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:170)
  ExtVal x16 = (x15 * ExtVal(1, 0, 0, 0));
  ExtVal x17 = (x8[0].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x18 = (x16 * x15);
  ExtVal x19 = (x8[0].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x20 = (((x17 * ExtVal(1, 0, 0, 0)) + ExtVal(0, 0, 0, 0)) + (x19 * x16));
  ExtVal x21 = (x18 * x15);
  ExtVal x22 = (x8[1].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x23 = (x21 * x15);
  ExtVal x24 = (x8[1].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x25 = (x23 * x15);
  ExtVal x26 = (x8[2].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x27 = (((x20 + (x22 * x18)) + (x24 * x21)) + (x26 * x23));
  ExtVal x28 = (x25 * x15);
  ExtVal x29 = (x8[2].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x30 = (x28 * x15);
  ExtVal x31 = (x8[3].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x32 = (x30 * x15);
  ExtVal x33 = (x8[3].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x34 = (((x27 + (x29 * x25)) + (x31 * x28)) + (x33 * x30));
  ExtVal x35 = (x32 * x15);
  ExtVal x36 = (x8[4].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x37 = (x35 * x15);
  ExtVal x38 = (x8[4].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x39 = (x37 * x15);
  ExtVal x40 = (x8[5].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x41 = (((x34 + (x36 * x32)) + (x38 * x35)) + (x40 * x37));
  ExtVal x42 = (x39 * x15);
  ExtVal x43 = (x8[5].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x44 = (x42 * x15);
  ExtVal x45 = (x8[6].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x46 = (x44 * x15);
  ExtVal x47 = (x8[6].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x48 = (((x41 + (x43 * x39)) + (x45 * x42)) + (x47 * x44));
  ExtVal x49 = (x8[7].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x50 = (x8[7].diffHigh + ExtVal(0, 0, 0, 0));
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:160)
  NondetExtRegStruct x51 = back_ExtReg(ctx, 0, x13);
  ExtVal x52 = x51._super;
  // Pow(zirgen/circuit/rv32im/v2/dsl/poly.zir:10)
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:171)
  ExtVal x53 = (((x52 * ExtVal(1, 0, 0, 0)) * x52) * x52);
  ExtVal x54 = (((x53 * x52) * x52) * x52);
  ExtVal x55 = (((x54 * x52) * x52) * x52);
  ExtVal x56 = (((x55 * x52) * x52) * x52);
  ExtVal x57 = (((x56 * x52) * x52) * x52);
  ExtVal x58 = (arg2.zcheck._super * (x57 * x52));
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:177)
  ValU32Struct x59 = x8[0]._super;
  Val x60 = x59.high;
  ValU32Struct x61 = x8[1]._super;
  Val x62 = x61.high;
  ValU32Struct x63 = x8[2]._super;
  Val x64 = x63.high;
  ValU32Struct x65 = x8[3]._super;
  Val x66 = x65.high;
  ValU32Struct x67 = x8[4]._super;
  Val x68 = x67.high;
  ValU32Struct x69 = x8[5]._super;
  Val x70 = x69.high;
  ValU32Struct x71 = x8[6]._super;
  Val x72 = x71.high;
  ValU32Struct x73 = x8[7]._super;
  Val x74 = x73.high;
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:174)
  NondetRegStruct24Array x75 = arg2.inner;
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:182)
  Val x76 = x75[17]._super;
  Val x77 = x75[19]._super;
  Val x78 = x75[21]._super;
  Val x79 = x75[23]._super;
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // GetDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:72)
  // PoseidonLoadInShort(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:194)
  PoseidonOpDefStruct x80 = PoseidonOpDefStruct{.hasState = arg2.hasState._super,
                                                .stateAddr = arg2.stateAddr._super,
                                                .bufOutAddr = arg2.bufOutAddr._super,
                                                .isElem = arg2.isElem._super,
                                                .checkOut = arg2.checkOut._super,
                                                .loadTxType = x5};
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  // MultiplyByMExt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:61)
  Val x81 = (x59.low + x60);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x82 = (x61.low + x62);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x83 = ((x60 * Val(2)) + x82);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x84 = ((x62 * Val(2)) + x81);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x85 = ((x82 * Val(4)) + x84);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x86 = ((x81 * Val(4)) + x83);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x87 = (x84 + x86);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x88 = (x83 + x85);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x89 = (x63.low + x64);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x90 = (x65.low + x66);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x91 = ((x64 * Val(2)) + x90);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x92 = ((x66 * Val(2)) + x89);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x93 = ((x90 * Val(4)) + x92);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x94 = ((x89 * Val(4)) + x91);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x95 = (x92 + x94);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x96 = (x91 + x93);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x97 = (x67.low + x68);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x98 = (x69.low + x70);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x99 = ((x68 * Val(2)) + x98);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x100 = ((x70 * Val(2)) + x97);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x101 = ((x98 * Val(4)) + x100);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x102 = ((x97 * Val(4)) + x99);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x103 = (x100 + x102);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x104 = (x99 + x101);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x105 = (x71.low + x72);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x106 = (x73.low + x74);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x107 = ((x72 * Val(2)) + x106);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x108 = ((x74 * Val(2)) + x105);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x109 = ((x106 * Val(4)) + x108);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x110 = ((x105 * Val(4)) + x107);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x111 = (x108 + x110);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x112 = (x107 + x109);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x113 = (x75[16]._super + x76);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x114 = (x75[18]._super + x77);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x115 = ((x76 * Val(2)) + x114);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x116 = ((x77 * Val(2)) + x113);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x117 = ((x114 * Val(4)) + x116);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x118 = ((x113 * Val(4)) + x115);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x119 = (x116 + x118);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x120 = (x115 + x117);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x121 = (x75[20]._super + x78);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x122 = (x75[22]._super + x79);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x123 = ((x78 * Val(2)) + x122);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x124 = ((x79 * Val(2)) + x121);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x125 = ((x122 * Val(4)) + x124);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x126 = ((x121 * Val(4)) + x123);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x127 = (x124 + x126);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x128 = (x123 + x125);
  // ReduceVec4(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:54)
  // MultiplyByMExt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:64)
  Val x129 = (((x87 + x95) + x103) + x111);
  Val x130 = (((x86 + x94) + x102) + x110);
  Val x131 = (((x88 + x96) + x104) + x112);
  Val x132 = (((x85 + x93) + x101) + x109);
  Val x133 = ((x129 + x119) + x127);
  Val x134 = ((x130 + x118) + x126);
  Val x135 = ((x131 + x120) + x128);
  Val x136 = ((x132 + x117) + x125);
  PoseidonStateStruct x137 = exec_PoseidonState(
      ctx,
      x80,
      Val(24),
      Val(0),
      (x7 + Val(8)),
      arg2.count._super,
      arg2.mode._super,
      Val24Array{(x87 + x133),  (x86 + x134),  (x88 + x135),  (x85 + x136),  (x95 + x133),
                 (x94 + x134),  (x96 + x135),  (x93 + x136),  (x103 + x133), (x102 + x134),
                 (x104 + x135), (x101 + x136), (x111 + x133), (x110 + x134), (x112 + x135),
                 (x109 + x136), (x119 + x133), (x118 + x134), (x120 + x135), (x117 + x136),
                 (x127 + x133), (x126 + x134), (x128 + x135), (x125 + x136)},
      (x58 + ((x48 + (x49 * x46)) + (x50 * (x46 * x15)))),
      LAYOUT_LOOKUP(layout3, _super));
  return x137;
}
PoseidonStateStruct exec_PoseidonLoadInLow(ExecContext& ctx,
                                           NondetRegStruct arg0,
                                           BoundLayout<NondetRegLayout> arg1,
                                           PoseidonStateStruct arg2,
                                           BoundLayout<PoseidonLoadInLowLayout> layout3,
                                           GlobalBuf global4) {
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:197)
  Val x5 = arg2.loadTxType._super;
  // Log(<preamble>:22)
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:198)
  INVOKE_EXTERN(ctx, log, "txnType", std::initializer_list<Val>{x5});
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:199)
  OneHot_3_Struct x6 = exec_OneHot_3_(ctx, x5, LAYOUT_LOOKUP(layout3, txType));
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:197)
  Val x7 = arg2.bufInAddr._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  GetDataStruct8Array x8 =
      map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
          LAYOUT_LOOKUP(layout3, loadList),
          std::function([&](Val8Array::value_type x9,
                            BoundLayout<MemoryGetLayout8LayoutArray::value_type> x10) {
            // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:201)
            GetDataStruct x11 = exec_MemoryGet(ctx, arg0, arg1, (x7 + x9), x6, x10);
            return x11;
          }));
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:159)
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:203)
  BoundLayout<_globalLayout> x12 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:160)
  BoundLayout<NondetExtRegLayout> x13 = LAYOUT_LOOKUP(x12, rng);
  NondetExtRegStruct x14 = back_ExtReg(ctx, 0, x13);
  ExtVal x15 = x14._super;
  // PolyEvalStateReduce(zirgen/circuit/rv32im/v2/dsl/poly.zir:14)
  // PolyEval(zirgen/circuit/rv32im/v2/dsl/poly.zir:18)
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:170)
  ExtVal x16 = (x15 * ExtVal(1, 0, 0, 0));
  ExtVal x17 = (x8[0].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x18 = (x16 * x15);
  ExtVal x19 = (x8[0].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x20 = (((x17 * ExtVal(1, 0, 0, 0)) + ExtVal(0, 0, 0, 0)) + (x19 * x16));
  ExtVal x21 = (x18 * x15);
  ExtVal x22 = (x8[1].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x23 = (x21 * x15);
  ExtVal x24 = (x8[1].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x25 = (x23 * x15);
  ExtVal x26 = (x8[2].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x27 = (((x20 + (x22 * x18)) + (x24 * x21)) + (x26 * x23));
  ExtVal x28 = (x25 * x15);
  ExtVal x29 = (x8[2].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x30 = (x28 * x15);
  ExtVal x31 = (x8[3].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x32 = (x30 * x15);
  ExtVal x33 = (x8[3].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x34 = (((x27 + (x29 * x25)) + (x31 * x28)) + (x33 * x30));
  ExtVal x35 = (x32 * x15);
  ExtVal x36 = (x8[4].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x37 = (x35 * x15);
  ExtVal x38 = (x8[4].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x39 = (x37 * x15);
  ExtVal x40 = (x8[5].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x41 = (((x34 + (x36 * x32)) + (x38 * x35)) + (x40 * x37));
  ExtVal x42 = (x39 * x15);
  ExtVal x43 = (x8[5].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x44 = (x42 * x15);
  ExtVal x45 = (x8[6].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x46 = (x44 * x15);
  ExtVal x47 = (x8[6].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x48 = (((x41 + (x43 * x39)) + (x45 * x42)) + (x47 * x44));
  ExtVal x49 = (x8[7].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x50 = (x8[7].diffHigh + ExtVal(0, 0, 0, 0));
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:160)
  NondetExtRegStruct x51 = back_ExtReg(ctx, 0, x13);
  ExtVal x52 = x51._super;
  // Pow(zirgen/circuit/rv32im/v2/dsl/poly.zir:10)
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:171)
  ExtVal x53 = (((x52 * ExtVal(1, 0, 0, 0)) * x52) * x52);
  ExtVal x54 = (((x53 * x52) * x52) * x52);
  ExtVal x55 = (((x54 * x52) * x52) * x52);
  ExtVal x56 = (((x55 * x52) * x52) * x52);
  ExtVal x57 = (((x56 * x52) * x52) * x52);
  ExtVal x58 = (arg2.zcheck._super * (x57 * x52));
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x59 = x8[0]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x60 = ((x59.high * Val(65536)) + x59.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x61 = x8[1]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x62 = ((x61.high * Val(65536)) + x61.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x63 = x8[2]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x64 = ((x63.high * Val(65536)) + x63.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x65 = x8[3]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x66 = ((x65.high * Val(65536)) + x65.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x67 = x8[4]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x68 = ((x67.high * Val(65536)) + x67.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x69 = x8[5]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x70 = ((x69.high * Val(65536)) + x69.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x71 = x8[6]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x72 = ((x71.high * Val(65536)) + x71.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:200)
  ValU32Struct x73 = x8[7]._super;
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:206)
  Val x74 = ((x73.high * Val(65536)) + x73.low);
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:197)
  NondetRegStruct24Array x75 = arg2.inner;
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // GetDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:72)
  // PoseidonLoadInLow(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:211)
  PoseidonOpDefStruct x76 = PoseidonOpDefStruct{.hasState = arg2.hasState._super,
                                                .stateAddr = arg2.stateAddr._super,
                                                .bufOutAddr = arg2.bufOutAddr._super,
                                                .isElem = arg2.isElem._super,
                                                .checkOut = arg2.checkOut._super,
                                                .loadTxType = x5};
  PoseidonStateStruct x77 = exec_PoseidonState(ctx,
                                               x76,
                                               Val(18),
                                               Val(1),
                                               (x7 + Val(8)),
                                               arg2.count._super,
                                               arg2.mode._super,
                                               Val24Array{x60,
                                                          x62,
                                                          x64,
                                                          x66,
                                                          x68,
                                                          x70,
                                                          x72,
                                                          x74,
                                                          x75[8]._super,
                                                          x75[9]._super,
                                                          x75[10]._super,
                                                          x75[11]._super,
                                                          x75[12]._super,
                                                          x75[13]._super,
                                                          x75[14]._super,
                                                          x75[15]._super,
                                                          x75[16]._super,
                                                          x75[17]._super,
                                                          x75[18]._super,
                                                          x75[19]._super,
                                                          x75[20]._super,
                                                          x75[21]._super,
                                                          x75[22]._super,
                                                          x75[23]._super},
                                               (x58 + ((x48 + (x49 * x46)) + (x50 * (x46 * x15)))),
                                               LAYOUT_LOOKUP(layout3, _super));
  return x77;
}
PoseidonStateStruct exec_PoseidonLoadInHigh(ExecContext& ctx,
                                            NondetRegStruct arg0,
                                            BoundLayout<NondetRegLayout> arg1,
                                            PoseidonStateStruct arg2,
                                            BoundLayout<PoseidonLoadInHighLayout> layout3,
                                            GlobalBuf global4) {
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:214)
  Val x5 = arg2.loadTxType._super;
  // Log(<preamble>:22)
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:215)
  INVOKE_EXTERN(ctx, log, "txnType", std::initializer_list<Val>{x5});
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:216)
  OneHot_3_Struct x6 = exec_OneHot_3_(ctx, x5, LAYOUT_LOOKUP(layout3, txType));
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:214)
  Val x7 = arg2.bufInAddr._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  GetDataStruct8Array x8 =
      map(Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
          LAYOUT_LOOKUP(layout3, loadList),
          std::function([&](Val8Array::value_type x9,
                            BoundLayout<MemoryGetLayout8LayoutArray::value_type> x10) {
            // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:218)
            GetDataStruct x11 = exec_MemoryGet(ctx, arg0, arg1, (x7 + x9), x6, x10);
            return x11;
          }));
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:214)
  NondetRegStruct24Array x12 = arg2.inner;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:221)
  Val x13 = x12[1]._super;
  Val x14 = x12[3]._super;
  Val x15 = x12[5]._super;
  Val x16 = x12[7]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x17 = x8[0]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x18 = ((x17.high * Val(65536)) + x17.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x19 = x8[1]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x20 = ((x19.high * Val(65536)) + x19.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x21 = x8[2]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x22 = ((x21.high * Val(65536)) + x21.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x23 = x8[3]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x24 = ((x23.high * Val(65536)) + x23.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x25 = x8[4]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x26 = ((x25.high * Val(65536)) + x25.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x27 = x8[5]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x28 = ((x27.high * Val(65536)) + x27.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x29 = x8[6]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x30 = ((x29.high * Val(65536)) + x29.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:217)
  ValU32Struct x31 = x8[7]._super;
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:222)
  Val x32 = ((x31.high * Val(65536)) + x31.low);
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:221)
  Val x33 = x12[17]._super;
  Val x34 = x12[19]._super;
  Val x35 = x12[21]._super;
  Val x36 = x12[23]._super;
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:159)
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:227)
  BoundLayout<_globalLayout> x37 = BIND_LAYOUT(kLayoutGlobal, global4);
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:160)
  BoundLayout<NondetExtRegLayout> x38 = LAYOUT_LOOKUP(x37, rng);
  NondetExtRegStruct x39 = back_ExtReg(ctx, 0, x38);
  ExtVal x40 = x39._super;
  // PolyEvalStateReduce(zirgen/circuit/rv32im/v2/dsl/poly.zir:14)
  // PolyEval(zirgen/circuit/rv32im/v2/dsl/poly.zir:18)
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:170)
  ExtVal x41 = (x40 * ExtVal(1, 0, 0, 0));
  ExtVal x42 = (x8[0].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x43 = (x41 * x40);
  ExtVal x44 = (x8[0].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x45 = (((x42 * ExtVal(1, 0, 0, 0)) + ExtVal(0, 0, 0, 0)) + (x44 * x41));
  ExtVal x46 = (x43 * x40);
  ExtVal x47 = (x8[1].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x48 = (x46 * x40);
  ExtVal x49 = (x8[1].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x50 = (x48 * x40);
  ExtVal x51 = (x8[2].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x52 = (((x45 + (x47 * x43)) + (x49 * x46)) + (x51 * x48));
  ExtVal x53 = (x50 * x40);
  ExtVal x54 = (x8[2].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x55 = (x53 * x40);
  ExtVal x56 = (x8[3].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x57 = (x55 * x40);
  ExtVal x58 = (x8[3].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x59 = (((x52 + (x54 * x50)) + (x56 * x53)) + (x58 * x55));
  ExtVal x60 = (x57 * x40);
  ExtVal x61 = (x8[4].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x62 = (x60 * x40);
  ExtVal x63 = (x8[4].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x64 = (x62 * x40);
  ExtVal x65 = (x8[5].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x66 = (((x59 + (x61 * x57)) + (x63 * x60)) + (x65 * x62));
  ExtVal x67 = (x64 * x40);
  ExtVal x68 = (x8[5].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x69 = (x67 * x40);
  ExtVal x70 = (x8[6].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x71 = (x69 * x40);
  ExtVal x72 = (x8[6].diffHigh + ExtVal(0, 0, 0, 0));
  ExtVal x73 = (((x66 + (x68 * x64)) + (x70 * x67)) + (x72 * x69));
  ExtVal x74 = (x8[7].diffLow + ExtVal(0, 0, 0, 0));
  ExtVal x75 = (x8[7].diffHigh + ExtVal(0, 0, 0, 0));
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:160)
  NondetExtRegStruct x76 = back_ExtReg(ctx, 0, x38);
  ExtVal x77 = x76._super;
  // Pow(zirgen/circuit/rv32im/v2/dsl/poly.zir:10)
  // ShiftPoly(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:171)
  ExtVal x78 = (((x77 * ExtVal(1, 0, 0, 0)) * x77) * x77);
  ExtVal x79 = (((x78 * x77) * x77) * x77);
  ExtVal x80 = (((x79 * x77) * x77) * x77);
  ExtVal x81 = (((x80 * x77) * x77) * x77);
  ExtVal x82 = (((x81 * x77) * x77) * x77);
  ExtVal x83 = (arg2.zcheck._super * (x82 * x77));
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // GetDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:72)
  // PoseidonLoadInHigh(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:228)
  PoseidonOpDefStruct x84 = PoseidonOpDefStruct{.hasState = arg2.hasState._super,
                                                .stateAddr = arg2.stateAddr._super,
                                                .bufOutAddr = arg2.bufOutAddr._super,
                                                .isElem = arg2.isElem._super,
                                                .checkOut = arg2.checkOut._super,
                                                .loadTxType = x5};
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  // MultiplyByMExt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:61)
  Val x85 = (x12[0]._super + x13);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x86 = (x12[2]._super + x14);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x87 = ((x13 * Val(2)) + x86);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x88 = ((x14 * Val(2)) + x85);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x89 = ((x86 * Val(4)) + x88);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x90 = ((x85 * Val(4)) + x87);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x91 = (x88 + x90);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x92 = (x87 + x89);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x93 = (x12[4]._super + x15);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x94 = (x12[6]._super + x16);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x95 = ((x15 * Val(2)) + x94);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x96 = ((x16 * Val(2)) + x93);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x97 = ((x94 * Val(4)) + x96);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x98 = ((x93 * Val(4)) + x95);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x99 = (x96 + x98);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x100 = (x95 + x97);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x101 = (x18 + x20);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x102 = (x22 + x24);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x103 = ((x20 * Val(2)) + x102);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x104 = ((x24 * Val(2)) + x101);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x105 = ((x102 * Val(4)) + x104);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x106 = ((x101 * Val(4)) + x103);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x107 = (x104 + x106);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x108 = (x103 + x105);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x109 = (x26 + x28);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x110 = (x30 + x32);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x111 = ((x28 * Val(2)) + x110);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x112 = ((x32 * Val(2)) + x109);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x113 = ((x110 * Val(4)) + x112);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x114 = ((x109 * Val(4)) + x111);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x115 = (x112 + x114);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x116 = (x111 + x113);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x117 = (x12[16]._super + x33);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x118 = (x12[18]._super + x34);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x119 = ((x33 * Val(2)) + x118);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x120 = ((x34 * Val(2)) + x117);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x121 = ((x118 * Val(4)) + x120);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x122 = ((x117 * Val(4)) + x119);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x123 = (x120 + x122);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x124 = (x119 + x121);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:42)
  Val x125 = (x12[20]._super + x35);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:43)
  Val x126 = (x12[22]._super + x36);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:44)
  Val x127 = ((x35 * Val(2)) + x126);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:45)
  Val x128 = ((x36 * Val(2)) + x125);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:46)
  Val x129 = ((x126 * Val(4)) + x128);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:47)
  Val x130 = ((x125 * Val(4)) + x127);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:48)
  Val x131 = (x128 + x130);
  // MultiplyByCirculant(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:49)
  Val x132 = (x127 + x129);
  // ReduceVec4(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:54)
  // MultiplyByMExt(zirgen/circuit/rv32im/v2/dsl/poseidon2.zir:64)
  Val x133 = (((x91 + x99) + x107) + x115);
  Val x134 = (((x90 + x98) + x106) + x114);
  Val x135 = (((x92 + x100) + x108) + x116);
  Val x136 = (((x89 + x97) + x105) + x113);
  Val x137 = ((x133 + x123) + x131);
  Val x138 = ((x134 + x122) + x130);
  Val x139 = ((x135 + x124) + x132);
  Val x140 = ((x136 + x121) + x129);
  PoseidonStateStruct x141 = exec_PoseidonState(
      ctx,
      x84,
      Val(24),
      Val(0),
      (x7 + Val(8)),
      arg2.count._super,
      arg2.mode._super,
      Val24Array{(x91 + x137),  (x90 + x138),  (x92 + x139),  (x89 + x140),  (x99 + x137),
                 (x98 + x138),  (x100 + x139), (x97 + x140),  (x107 + x137), (x106 + x138),
                 (x108 + x139), (x105 + x140), (x115 + x137), (x114 + x138), (x116 + x139),
                 (x113 + x140), (x123 + x137), (x122 + x138), (x124 + x139), (x121 + x140),
                 (x131 + x137), (x130 + x138), (x132 + x139), (x129 + x140)},
      (x83 + ((x73 + (x74 * x71)) + (x75 * (x71 * x40)))),
      LAYOUT_LOOKUP(layout3, _super));
  return x141;
}
PoseidonStateStruct exec_PoseidonLoadIn(ExecContext& ctx,
                                        NondetRegStruct arg0,
                                        BoundLayout<NondetRegLayout> arg1,
                                        PoseidonStateStruct arg2,
                                        BoundLayout<PoseidonLoadInLayout> layout3,
                                        GlobalBuf global4) {
  // PoseidonLoadIn(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:233)
  BoundLayout<PoseidonLoadIn_SuperLayout> x5 = LAYOUT_LOOKUP(layout3, _super);
  // PoseidonLoadIn(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:232)
  Val x6 = (arg2.isElem._super + arg2.subState._super);
  // PoseidonLoadIn(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:233)
  OneHot_3_Struct x7 = exec_OneHot_3_(ctx, x6, LAYOUT_LOOKUP(layout3, _0));
  NondetRegStruct3Array x8 = x7._super;
  PoseidonStateStruct x9;
  if (to_size_t(x8[0]._super)) {
    // PoseidonLoadIn(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:234)
    PoseidonStateStruct x10 =
        exec_PoseidonLoadInShort(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x5, arm0), global4);
    x9 = x10;
  } else if (to_size_t(x8[1]._super)) {
    // PoseidonLoadIn(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:235)
    PoseidonStateStruct x11 =
        exec_PoseidonLoadInLow(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x5, arm1), global4);
    x9 = x11;
  } else if (to_size_t(x8[2]._super)) {
    // PoseidonLoadIn(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:236)
    PoseidonStateStruct x12 =
        exec_PoseidonLoadInHigh(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x5, arm2), global4);
    x9 = x12;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // PoseidonLoadIn(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:233)
  PoseidonStateStruct x13 = back_PoseidonState(ctx, 0, LAYOUT_LOOKUP(x5, _super));
  return x13;
}
PoseidonStateStruct exec_PoseidonExtRound(ExecContext& ctx,
                                          PoseidonStateStruct arg0,
                                          BoundLayout<PoseidonExtRoundLayout> layout1) {
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:240)
  Val x2 = arg0.subState._super;
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:241)
  NondetRegStruct x3 = exec_IsZero(ctx, (x2 - Val(3)), LAYOUT_LOOKUP(layout1, isRound3));
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:242)
  NondetRegStruct x4 = exec_IsZero(ctx, (x2 - Val(7)), LAYOUT_LOOKUP(layout1, isRound7));
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:240)
  Val x5 = arg0.count._super;
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:244)
  NondetRegStruct x6 = exec_IsZero(ctx, (x5 - Val(1)), LAYOUT_LOOKUP(layout1, lastBlock));
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:242)
  Val x7 = x4._super;
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:241)
  Val x8 = x3._super;
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:248)
  Val x9 = ((Val(1) - x8) - x7);
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:244)
  Val x10 = x6._super;
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:248)
  Val x11 = (((x8 * Val(25)) + (x9 * Val(24))) + ((x7 * (Val(1) - x10)) * Val(18)));
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:240)
  NondetRegStruct24Array x12 = arg0.inner;
  Val24Array x13 =
      Val24Array{x12[0]._super,  x12[1]._super,  x12[2]._super,  x12[3]._super,  x12[4]._super,
                 x12[5]._super,  x12[6]._super,  x12[7]._super,  x12[8]._super,  x12[9]._super,
                 x12[10]._super, x12[11]._super, x12[12]._super, x12[13]._super, x12[14]._super,
                 x12[15]._super, x12[16]._super, x12[17]._super, x12[18]._super, x12[19]._super,
                 x12[20]._super, x12[21]._super, x12[22]._super, x12[23]._super};
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:252)
  MultiplyByMExtStruct x14 = exec_DoExtRoundByIdx(ctx, x13, x2, LAYOUT_LOOKUP(layout1, nextInner));
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // GetDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:72)
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:253)
  PoseidonOpDefStruct x15 = PoseidonOpDefStruct{.hasState = arg0.hasState._super,
                                                .stateAddr = arg0.stateAddr._super,
                                                .bufOutAddr = arg0.bufOutAddr._super,
                                                .isElem = arg0.isElem._super,
                                                .checkOut = arg0.checkOut._super,
                                                .loadTxType = arg0.loadTxType._super};
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:252)
  MultiplyByMExt_Super_SuperStruct24Array x16 = x14._super;
  Val24Array x17 =
      Val24Array{x16[0]._super,  x16[1]._super,  x16[2]._super,  x16[3]._super,  x16[4]._super,
                 x16[5]._super,  x16[6]._super,  x16[7]._super,  x16[8]._super,  x16[9]._super,
                 x16[10]._super, x16[11]._super, x16[12]._super, x16[13]._super, x16[14]._super,
                 x16[15]._super, x16[16]._super, x16[17]._super, x16[18]._super, x16[19]._super,
                 x16[20]._super, x16[21]._super, x16[22]._super, x16[23]._super};
  // PoseidonExtRound(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:253)
  PoseidonStateStruct x18 = exec_PoseidonState(ctx,
                                               x15,
                                               (x11 + ((x7 * x10) * Val(21))),
                                               (x9 * (x2 + Val(1))),
                                               arg0.bufInAddr._super,
                                               (x5 - x7),
                                               arg0.mode._super,
                                               x17,
                                               arg0.zcheck._super,
                                               LAYOUT_LOOKUP(layout1, _super));
  return x18;
}
PoseidonStateStruct exec_PoseidonIntRounds(ExecContext& ctx,
                                           PoseidonStateStruct arg0,
                                           BoundLayout<PoseidonIntRoundsLayout> layout1) {
  // PoseidonIntRounds(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:256)
  NondetRegStruct24Array x2 = arg0.inner;
  Val24Array x3 = Val24Array{
      x2[0]._super,  x2[1]._super,  x2[2]._super,  x2[3]._super,  x2[4]._super,  x2[5]._super,
      x2[6]._super,  x2[7]._super,  x2[8]._super,  x2[9]._super,  x2[10]._super, x2[11]._super,
      x2[12]._super, x2[13]._super, x2[14]._super, x2[15]._super, x2[16]._super, x2[17]._super,
      x2[18]._super, x2[19]._super, x2[20]._super, x2[21]._super, x2[22]._super, x2[23]._super};
  // PoseidonIntRounds(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:257)
  DoIntRoundsStruct x4 = exec_DoIntRounds(ctx, x3, LAYOUT_LOOKUP(layout1, nextInner));
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // GetDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:72)
  // PoseidonIntRounds(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:258)
  PoseidonOpDefStruct x5 = PoseidonOpDefStruct{.hasState = arg0.hasState._super,
                                               .stateAddr = arg0.stateAddr._super,
                                               .bufOutAddr = arg0.bufOutAddr._super,
                                               .isElem = arg0.isElem._super,
                                               .checkOut = arg0.checkOut._super,
                                               .loadTxType = arg0.loadTxType._super};
  PoseidonStateStruct x6 = exec_PoseidonState(ctx,
                                              x5,
                                              Val(24),
                                              Val(4),
                                              arg0.bufInAddr._super,
                                              arg0.count._super,
                                              arg0.mode._super,
                                              x4._super,
                                              arg0.zcheck._super,
                                              LAYOUT_LOOKUP(layout1, _super));
  return x6;
}
PoseidonStateStruct exec_PoseidonCheckOut(ExecContext& ctx,
                                          NondetRegStruct arg0,
                                          BoundLayout<NondetRegLayout> arg1,
                                          PoseidonStateStruct arg2,
                                          BoundLayout<PoseidonCheckOutLayout> layout3) {
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:261)
  NondetRegStruct24Array x4 = arg2.inner;
  Val x5 = arg2.bufOutAddr._super;
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:262)
  PoseidonCheckOut__0Struct8Array x6 = map(
      Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
      LAYOUT_LOOKUP(layout3, _0),
      std::function([&](Val8Array::value_type x7,
                        BoundLayout<PoseidonCheckOut__0_SuperLayout8LayoutArray::value_type> x8) {
        // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:264)
        ReadElemStruct x9 = exec_ReadElem(ctx, arg0, arg1, (x5 + x7), LAYOUT_LOOKUP(x8, goal));
        // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:265)
        Val x10 = (x9._super - x4[to_size_t(x7)]._super);
        EQZ(x10, "PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:265)");
        return PoseidonCheckOut__0Struct{};
      }));
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:261)
  Val x11 = arg2.loadTxType._super;
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:267)
  NondetRegStruct x12 = exec_IsZero(ctx, x11, LAYOUT_LOOKUP(layout3, isNormal));
  Val x13 = x12._super;
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:268)
  Val x14 = ((Val(1) - x13) * Val(22));
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:261)
  Val x15 = arg2.hasState._super;
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:269)
  Val x16 = ((Val(1) - x15) * ((x13 * Val(32)) + x14));
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:261)
  ExtVal x17 = arg2.zcheck._super;
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:274)
  NondetExtRegStruct x18 = exec_NondetExtReg(ctx, inv_0(x17), LAYOUT_LOOKUP(layout3, extInv));
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:275)
  ExtVal x19 = ((x18._super * x17) - ExtVal(1, 0, 0, 0));
  EQZ(x19,
      "loc(callsite(unknown at  PoseidonCheckOut ( zirgen/circuit/rv32im/v2/dsl/inst_p2.zir "
      ":275:10)))");
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:261)
  Val24Array x20 = Val24Array{
      x4[0]._super,  x4[1]._super,  x4[2]._super,  x4[3]._super,  x4[4]._super,  x4[5]._super,
      x4[6]._super,  x4[7]._super,  x4[8]._super,  x4[9]._super,  x4[10]._super, x4[11]._super,
      x4[12]._super, x4[13]._super, x4[14]._super, x4[15]._super, x4[16]._super, x4[17]._super,
      x4[18]._super, x4[19]._super, x4[20]._super, x4[21]._super, x4[22]._super, x4[23]._super};
  // PoseidonCheckOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:276)
  PoseidonStateStruct x21 =
      exec_PoseidonState(ctx,
                         PoseidonOpDefStruct{.hasState = x15,
                                             .stateAddr = arg2.stateAddr._super,
                                             .bufOutAddr = x5,
                                             .isElem = arg2.isElem._super,
                                             .checkOut = arg2.checkOut._super,
                                             .loadTxType = x11},
                         ((x15 * Val(23)) + x16),
                         Val(0),
                         Val(0),
                         Val(0),
                         arg2.mode._super,
                         x20,
                         ExtVal(0, 0, 0, 0),
                         LAYOUT_LOOKUP(layout3, _super));
  return x21;
}
PoseidonStateStruct exec_PoseidonStoreOut(ExecContext& ctx,
                                          NondetRegStruct arg0,
                                          BoundLayout<NondetRegLayout> arg1,
                                          PoseidonStateStruct arg2,
                                          BoundLayout<PoseidonStoreOutLayout> layout3) {
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:279)
  NondetRegStruct24Array x4 = arg2.inner;
  Val x5 = arg2.bufOutAddr._super;
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:280)
  PoseidonStoreOut__0Struct8Array x6 = map(
      Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
      LAYOUT_LOOKUP(layout3, _0),
      std::function([&](Val8Array::value_type x7,
                        BoundLayout<PoseidonStoreOut__0_SuperLayout8LayoutArray::value_type> x8) {
        // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:279)
        Val x9 = x4[to_size_t(x7)]._super;
        // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:282)
        NondetRegStruct x10 =
            exec_NondetU16Reg(ctx, bitAnd(x9, Val(65535)), LAYOUT_LOOKUP(x8, low));
        Val x11 = x10._super;
        // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:283)
        U16RegStruct x12 =
            exec_U16Reg(ctx, ((x9 - x11) * Val(2013235201)), LAYOUT_LOOKUP(x8, high));
        // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:284)
        MemoryWriteStruct x13 = exec_MemoryWrite(ctx,
                                                 arg0,
                                                 arg1,
                                                 (x5 + x7),
                                                 ValU32Struct{.low = x11, .high = x12._super},
                                                 LAYOUT_LOOKUP(x8, _0));
        return PoseidonStoreOut__0Struct{};
      }));
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:279)
  Val x14 = arg2.loadTxType._super;
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:286)
  NondetRegStruct x15 = exec_IsZero(ctx, x14, LAYOUT_LOOKUP(layout3, isNormal));
  Val x16 = x15._super;
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:287)
  Val x17 = ((Val(1) - x16) * Val(22));
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:279)
  Val x18 = arg2.hasState._super;
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:290)
  Val x19 = ((Val(1) - x18) * ((x16 * Val(32)) + x17));
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:291)
  ExtVal x20 = inv_0(arg2.zcheck._super);
  NondetExtRegStruct x21 = exec_NondetExtReg(ctx, x20, LAYOUT_LOOKUP(layout3, extInv));
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:279)
  Val24Array x22 = Val24Array{
      x4[0]._super,  x4[1]._super,  x4[2]._super,  x4[3]._super,  x4[4]._super,  x4[5]._super,
      x4[6]._super,  x4[7]._super,  x4[8]._super,  x4[9]._super,  x4[10]._super, x4[11]._super,
      x4[12]._super, x4[13]._super, x4[14]._super, x4[15]._super, x4[16]._super, x4[17]._super,
      x4[18]._super, x4[19]._super, x4[20]._super, x4[21]._super, x4[22]._super, x4[23]._super};
  // PoseidonStoreOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:292)
  PoseidonStateStruct x23 =
      exec_PoseidonState(ctx,
                         PoseidonOpDefStruct{.hasState = x18,
                                             .stateAddr = arg2.stateAddr._super,
                                             .bufOutAddr = x5,
                                             .isElem = arg2.isElem._super,
                                             .checkOut = arg2.checkOut._super,
                                             .loadTxType = x14},
                         ((x18 * Val(23)) + x19),
                         Val(0),
                         Val(0),
                         Val(0),
                         arg2.mode._super,
                         x22,
                         ExtVal(0, 0, 0, 0),
                         LAYOUT_LOOKUP(layout3, _super));
  return x23;
}
PoseidonStateStruct exec_PoseidonDoOut(ExecContext& ctx,
                                       NondetRegStruct arg0,
                                       BoundLayout<NondetRegLayout> arg1,
                                       PoseidonStateStruct arg2,
                                       BoundLayout<PoseidonDoOutLayout> layout3) {
  // PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)
  BoundLayout<PoseidonDoOut_SuperLayout> x4 = LAYOUT_LOOKUP(layout3, _super);
  // PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:295)
  Val x5 = arg2.checkOut._super;
  // PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)
  BoundLayout<PoseidonDoOutArm0Layout> x6 = LAYOUT_LOOKUP(x4, arm0);
  BoundLayout<Reg> x7 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra0), count), _super);
  BoundLayout<Reg> x8 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra1), count), _super);
  BoundLayout<Reg> x9 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra2), count), _super);
  BoundLayout<Reg> x10 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra3), count), _super);
  BoundLayout<Reg> x11 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra4), count), _super);
  BoundLayout<Reg> x12 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra5), count), _super);
  BoundLayout<Reg> x13 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra6), count), _super);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra7), count), _super);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra8), count), _super);
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra9), count), _super);
  BoundLayout<Reg> x17 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra10), count), _super);
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra11), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra12), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra13), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra14), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x6, _extra15), count), _super);
  PoseidonStateStruct x23;
  if (to_size_t(x5)) {
    PoseidonStateStruct x24 =
        exec_PoseidonCheckOut(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x6, _super));
    STORE(x7, Val(0));
    EQZ(LOAD(x7, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x8, Val(0));
    EQZ(LOAD(x8, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x9, Val(0));
    EQZ(LOAD(x9, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x10, Val(0));
    EQZ(LOAD(x10, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x11, Val(0));
    EQZ(LOAD(x11, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x12, Val(0));
    EQZ(LOAD(x12, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x13, Val(0));
    EQZ(LOAD(x13, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x14, Val(0));
    EQZ(LOAD(x14, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x15, Val(0));
    EQZ(LOAD(x15, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x16, Val(0));
    EQZ(LOAD(x16, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x17, Val(0));
    EQZ(LOAD(x17, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "PoseidonDoOut(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:296)");
    x23 = x24;
  } else if (to_size_t((Val(1) - x5))) {
    PoseidonStateStruct x25 = exec_PoseidonStoreOut(ctx, arg0, arg1, arg2, LAYOUT_LOOKUP(x4, arm1));
    x23 = x25;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  PoseidonStateStruct x26 = back_PoseidonState(ctx, 0, LAYOUT_LOOKUP(x4, _super));
  return x26;
}
PoseidonStateStruct exec_PoseidonStoreState(ExecContext& ctx,
                                            NondetRegStruct arg0,
                                            BoundLayout<NondetRegLayout> arg1,
                                            PoseidonStateStruct arg2,
                                            BoundLayout<PoseidonStoreStateLayout> layout3) {
  // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:299)
  NondetRegStruct24Array x4 = arg2.inner;
  Val x5 = arg2.stateAddr._super;
  // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:300)
  PoseidonStoreState__0Struct8Array x6 = map(
      Val8Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7)},
      LAYOUT_LOOKUP(layout3, _0),
      std::function([&](Val8Array::value_type x7,
                        BoundLayout<PoseidonStoreState__0_SuperLayout8LayoutArray::value_type> x8) {
        // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:299)
        Val x9 = x4[to_size_t((x7 + Val(16)))]._super;
        // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:302)
        NondetRegStruct x10 =
            exec_NondetU16Reg(ctx, bitAnd(x9, Val(65535)), LAYOUT_LOOKUP(x8, low));
        Val x11 = x10._super;
        // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:303)
        U16RegStruct x12 =
            exec_U16Reg(ctx, ((x9 - x11) * Val(2013235201)), LAYOUT_LOOKUP(x8, high));
        // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:304)
        MemoryWriteStruct x13 = exec_MemoryWrite(ctx,
                                                 arg0,
                                                 arg1,
                                                 (x5 + x7),
                                                 ValU32Struct{.low = x11, .high = x12._super},
                                                 LAYOUT_LOOKUP(x8, _0));
        return PoseidonStoreState__0Struct{};
      }));
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // GetDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:72)
  // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:306)
  PoseidonOpDefStruct x14 = PoseidonOpDefStruct{.hasState = arg2.hasState._super,
                                                .stateAddr = x5,
                                                .bufOutAddr = arg2.bufOutAddr._super,
                                                .isElem = arg2.isElem._super,
                                                .checkOut = arg2.checkOut._super,
                                                .loadTxType = arg2.loadTxType._super};
  // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:299)
  Val24Array x15 = Val24Array{
      x4[0]._super,  x4[1]._super,  x4[2]._super,  x4[3]._super,  x4[4]._super,  x4[5]._super,
      x4[6]._super,  x4[7]._super,  x4[8]._super,  x4[9]._super,  x4[10]._super, x4[11]._super,
      x4[12]._super, x4[13]._super, x4[14]._super, x4[15]._super, x4[16]._super, x4[17]._super,
      x4[18]._super, x4[19]._super, x4[20]._super, x4[21]._super, x4[22]._super, x4[23]._super};
  // PoseidonStoreState(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:306)
  PoseidonStateStruct x16 = exec_PoseidonState(ctx,
                                               x14,
                                               Val(32),
                                               Val(0),
                                               Val(0),
                                               Val(0),
                                               arg2.mode._super,
                                               x15,
                                               ExtVal(0, 0, 0, 0),
                                               LAYOUT_LOOKUP(layout3, _super));
  return x16;
}
IsU24Struct exec_IsU24(ExecContext& ctx, Val arg0, BoundLayout<IsU24Layout> layout1) {
  // IsU24(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:320)
  NondetRegStruct x2 =
      exec_NondetU16Reg(ctx, bitAnd(arg0, Val(65535)), LAYOUT_LOOKUP(layout1, low16));
  // IsU24(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:321)
  U8RegStruct x3 =
      exec_U8Reg(ctx, ((arg0 - x2._super) * Val(2013235201)), LAYOUT_LOOKUP(layout1, _0));
  return IsU24Struct{};
}
PoseidonStateStruct exec_PoseidonPagingLoadNode(ExecContext& ctx,
                                                NondetRegStruct arg0,
                                                Val arg1,
                                                BoundLayout<PoseidonStateLayout> layout2) {
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // PoseidonPagingLoadNode(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:325)
  PoseidonOpDefStruct x3 = PoseidonOpDefStruct{.hasState = Val(0),
                                               .stateAddr = Val(0),
                                               .bufOutAddr = (Val(1140850688) - (arg1 * Val(8))),
                                               .isElem = Val(1),
                                               .checkOut = Val(1),
                                               .loadTxType = Val(1)};
  // NodeIdxToAddr(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:317)
  // PoseidonPagingLoadNode(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:337)
  Val x4 = (((arg1 * Val(2)) + Val(1)) * Val(8));
  // PoseidonPagingLoadNode(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:333)
  PoseidonStateStruct x5 =
      exec_PoseidonState(ctx,
                         x3,
                         Val(18),
                         Val(0),
                         (Val(1140850688) - x4),
                         Val(1),
                         Val(0),
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout2);
  return x5;
}
PoseidonStateStruct exec_PoseidonPagingLoadPage(ExecContext& ctx,
                                                NondetRegStruct arg0,
                                                Val arg1,
                                                BoundLayout<PoseidonStateLayout> layout2) {
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // PoseidonPagingLoadPage(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:347)
  PoseidonOpDefStruct x3 = PoseidonOpDefStruct{.hasState = Val(0),
                                               .stateAddr = Val(0),
                                               .bufOutAddr = (Val(1140850688) - (arg1 * Val(8))),
                                               .isElem = Val(0),
                                               .checkOut = Val(1),
                                               .loadTxType = Val(1)};
  // PoseidonPagingLoadPage(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:355)
  PoseidonStateStruct x4 =
      exec_PoseidonState(ctx,
                         x3,
                         Val(18),
                         Val(0),
                         ((arg1 - Val(4194304)) * Val(256)),
                         Val(32),
                         Val(1),
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout2);
  return x4;
}
PoseidonStateStruct exec_PoseidonPagingLoadDone(ExecContext& ctx,
                                                BoundLayout<PoseidonStateLayout> layout0) {
  // PoseidonPagingLoadDone(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:369)
  PoseidonStateStruct x1 =
      exec_PoseidonState(ctx,
                         PoseidonOpDefStruct{.hasState = Val(0),
                                             .stateAddr = Val(0),
                                             .bufOutAddr = Val(1073741824),
                                             .isElem = Val(0),
                                             .checkOut = Val(0),
                                             .loadTxType = Val(0)},
                         Val(1),
                         Val(0),
                         Val(0),
                         Val(0),
                         Val(2),
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout0);
  return x1;
}
PoseidonStateStruct exec_PoseidonPagingStoreNode(ExecContext& ctx,
                                                 NondetRegStruct arg0,
                                                 Val arg1,
                                                 BoundLayout<PoseidonStateLayout> layout2) {
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // PoseidonPagingStoreNode(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:373)
  PoseidonOpDefStruct x3 = PoseidonOpDefStruct{.hasState = Val(0),
                                               .stateAddr = Val(0),
                                               .bufOutAddr = (Val(1140850688) - (arg1 * Val(8))),
                                               .isElem = Val(1),
                                               .checkOut = Val(0),
                                               .loadTxType = Val(2)};
  // NodeIdxToAddr(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:317)
  // PoseidonPagingStoreNode(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:385)
  Val x4 = (((arg1 * Val(2)) + Val(1)) * Val(8));
  // PoseidonPagingStoreNode(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:381)
  PoseidonStateStruct x5 =
      exec_PoseidonState(ctx,
                         x3,
                         Val(18),
                         Val(0),
                         (Val(1140850688) - x4),
                         Val(1),
                         Val(4),
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout2);
  return x5;
}
PoseidonStateStruct exec_PoseidonPagingStorePage(ExecContext& ctx,
                                                 NondetRegStruct arg0,
                                                 Val arg1,
                                                 BoundLayout<PoseidonStateLayout> layout2) {
  // PoseidonOpDef(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:8)
  // PoseidonPagingStorePage(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:395)
  PoseidonOpDefStruct x3 = PoseidonOpDefStruct{.hasState = Val(0),
                                               .stateAddr = Val(0),
                                               .bufOutAddr = (Val(1140850688) - (arg1 * Val(8))),
                                               .isElem = Val(0),
                                               .checkOut = Val(0),
                                               .loadTxType = Val(2)};
  // PoseidonPagingStorePage(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:403)
  PoseidonStateStruct x4 =
      exec_PoseidonState(ctx,
                         x3,
                         Val(18),
                         Val(0),
                         ((arg1 - Val(4194304)) * Val(256)),
                         Val(32),
                         Val(3),
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout2);
  return x4;
}
PoseidonStateStruct exec_PoseidonPagingStoreDone(ExecContext& ctx,
                                                 BoundLayout<PoseidonStateLayout> layout0) {
  // PoseidonPagingStoreDone(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:417)
  PoseidonStateStruct x1 =
      exec_PoseidonState(ctx,
                         PoseidonOpDefStruct{.hasState = Val(0),
                                             .stateAddr = Val(0),
                                             .bufOutAddr = Val(1140850688),
                                             .isElem = Val(0),
                                             .checkOut = Val(0),
                                             .loadTxType = Val(0)},
                         Val(5),
                         Val(0),
                         Val(0),
                         Val(0),
                         Val(5),
                         Val24Array{Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0),
                                    Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0), Val(0)},
                         ExtVal(0, 0, 0, 0),
                         layout0);
  return x1;
}
OneHot_6_Struct exec_OneHot_6_(ExecContext& ctx, Val arg0, BoundLayout<OneHot_6_Layout> layout1) {
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:7)
  NondetRegStruct6Array x2 =
      map(Val6Array{Val(0), Val(1), Val(2), Val(3), Val(4), Val(5)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val6Array::value_type x3,
                            BoundLayout<NondetRegLayout6LayoutArray::value_type> x4) {
            NondetRegStruct x5 = exec_NondetBitReg(ctx, isz((x3 - arg0)), x4);
            return x5;
          }));
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)
  Val x6 = x2[1]._super;
  Val x7 = (x2[0]._super + x6);
  Val x8 = x2[2]._super;
  Val x9 = x2[3]._super;
  Val x10 = x2[4]._super;
  Val x11 = (((x7 + x8) + x9) + x10);
  Val x12 = x2[5]._super;
  EQZ(((x11 + x12) - Val(1)), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)");
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)
  Val x13 = (((x6 + (x8 * Val(2))) + (x9 * Val(3))) + (x10 * Val(4)));
  EQZ(((x13 + (x12 * Val(5))) - arg0), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)");
  return OneHot_6_Struct{._super = x2, .bits = x2};
}
PoseidonStateStruct exec_PoseidonPaging(ExecContext& ctx,
                                        NondetRegStruct arg0,
                                        Val arg1,
                                        PoseidonStateStruct arg2,
                                        BoundLayout<PoseidonPagingLayout> layout3) {
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:428)
  BoundLayout<PoseidonPaging__0Layout> x4 = LAYOUT_LOOKUP(layout3, _2);
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:435)
  BoundLayout<PoseidonPaging_SuperLayout> x5 = LAYOUT_LOOKUP(layout3, _super);
  // NodeAddrToIdx(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:316)
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:421)
  Val x6 = (Val(1140850688) - arg2.bufOutAddr._super);
  // Div(<preamble>:19)
  Val x7 = (x6 * Val(1761607681));
  // nextPagingIdx(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:314)
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:422)
  auto [x8, x9] = INVOKE_EXTERN(ctx, nextPagingIdx);
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:423)
  NondetRegStruct x10 = exec_NondetReg(ctx, x8, LAYOUT_LOOKUP(layout3, curIdx));
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:424)
  NondetRegStruct x11 = exec_NondetReg(ctx, x9, LAYOUT_LOOKUP(layout3, curMode));
  Val x12 = x11._super;
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:425)
  OneHot_6_Struct x13 = exec_OneHot_6_(ctx, x12, LAYOUT_LOOKUP(layout3, modeSplit));
  NondetRegStruct6Array x14 = x13.bits;
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:426)
  Val x15 = (x14[0]._super + x14[1]._super);
  Val x16 = (x15 + x14[2]._super);
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:423)
  Val x17 = x10._super;
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:427)
  IsU24Struct x18 = exec_IsU24(ctx, x17, LAYOUT_LOOKUP(layout3, _0));
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:428)
  ComponentStruct x19 = ComponentStruct{};
  ComponentStruct x20;
  if (to_size_t(x16)) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:429)
    IsU24Struct x21 =
        exec_IsU24(ctx, (x17 - (x7 + Val(1))), LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, arm0), _0));
    x20 = x19;
  } else if (to_size_t((Val(1) - x16))) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:431)
    IsU24Struct x22 =
        exec_IsU24(ctx, ((x7 - Val(1)) - x17), LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, arm1), _0));
    x20 = x19;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:434)
  BitRegStruct x23 = exec_BitReg(ctx, (x12 - arg1), LAYOUT_LOOKUP(layout3, _3));
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:425)
  NondetRegStruct6Array x24 = x13._super;
  PoseidonStateStruct x25;
  if (to_size_t(x24[0]._super)) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:436)
    PoseidonStateStruct x26 = exec_PoseidonPagingLoadNode(ctx, arg0, x17, LAYOUT_LOOKUP(x5, arm0));
    x25 = x26;
  } else if (to_size_t(x24[1]._super)) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:437)
    PoseidonStateStruct x27 = exec_PoseidonPagingLoadPage(ctx, arg0, x17, LAYOUT_LOOKUP(x5, arm1));
    x25 = x27;
  } else if (to_size_t(x24[2]._super)) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:438)
    PoseidonStateStruct x28 = exec_PoseidonPagingLoadDone(ctx, LAYOUT_LOOKUP(x5, arm2));
    x25 = x28;
  } else if (to_size_t(x24[3]._super)) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:439)
    PoseidonStateStruct x29 = exec_PoseidonPagingStorePage(ctx, arg0, x17, LAYOUT_LOOKUP(x5, arm3));
    x25 = x29;
  } else if (to_size_t(x24[4]._super)) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:440)
    PoseidonStateStruct x30 = exec_PoseidonPagingStoreNode(ctx, arg0, x17, LAYOUT_LOOKUP(x5, arm4));
    x25 = x30;
  } else if (to_size_t(x24[5]._super)) {
    // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:441)
    PoseidonStateStruct x31 = exec_PoseidonPagingStoreDone(ctx, LAYOUT_LOOKUP(x5, arm5));
    x25 = x31;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // PoseidonPaging(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:435)
  PoseidonStateStruct x32 = back_PoseidonState(ctx, 0, LAYOUT_LOOKUP(x5, _super));
  return x32;
}
InstOutputStruct exec_Poseidon0(ExecContext& ctx,
                                NondetRegStruct arg0,
                                BoundLayout<NondetRegLayout> arg1,
                                InstInputStruct arg2,
                                BoundLayout<Poseidon0Layout> layout3,
                                GlobalBuf global4) {
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
  BoundLayout<Poseidon0StateLayout> x5 = LAYOUT_LOOKUP(layout3, state);
  BoundLayout<PoseidonStateLayout> x6 = LAYOUT_LOOKUP(x5, _super);
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:445)
  NondetRegStruct8Array x7 = arg2.minorOnehot._super;
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
  BoundLayout<Poseidon0Arm0Layout> x8 = LAYOUT_LOOKUP(x5, arm0);
  BoundLayout<Poseidon0Arm1Layout> x9 = LAYOUT_LOOKUP(x5, arm1);
  BoundLayout<Poseidon0Arm2Layout> x10 = LAYOUT_LOOKUP(x5, arm2);
  BoundLayout<Poseidon0Arm3Layout> x11 = LAYOUT_LOOKUP(x5, arm3);
  BoundLayout<Poseidon0Arm4Layout> x12 = LAYOUT_LOOKUP(x5, arm4);
  BoundLayout<Poseidon0Arm5Layout> x13 = LAYOUT_LOOKUP(x5, arm5);
  BoundLayout<Poseidon0Arm6Layout> x14 = LAYOUT_LOOKUP(x5, arm6);
  BoundLayout<Poseidon0Arm7Layout> x15 = LAYOUT_LOOKUP(x5, arm7);
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:445)
  ValU32Struct x16 = arg2.pcU32;
  Val x17 = arg2.mode;
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
  BoundLayout<Reg> x18 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra0), count), _super);
  BoundLayout<Reg> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra1), count), _super);
  BoundLayout<Reg> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra2), count), _super);
  BoundLayout<Reg> x21 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra3), count), _super);
  BoundLayout<Reg> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra4), count), _super);
  BoundLayout<Reg> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra5), count), _super);
  BoundLayout<Reg> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra6), count), _super);
  BoundLayout<Reg> x25 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra7), count), _super);
  BoundLayout<Reg> x26 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra8), count), _super);
  BoundLayout<Reg> x27 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra9), count), _super);
  BoundLayout<Reg> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra10), count), _super);
  BoundLayout<Reg> x29 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra11), count), _super);
  BoundLayout<Reg> x30 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra12), count), _super);
  BoundLayout<Reg> x31 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra13), count), _super);
  BoundLayout<Reg> x32 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra14), count), _super);
  BoundLayout<Reg> x33 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra15), count), _super);
  BoundLayout<Reg> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra16), count), _super);
  BoundLayout<Reg> x35 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra17), count), _super);
  BoundLayout<Reg> x36 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra18), count), _super);
  BoundLayout<Reg> x37 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra19), count), _super);
  BoundLayout<Reg> x38 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra20), count), _super);
  BoundLayout<Reg> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra21), count), _super);
  BoundLayout<Reg> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra22), count), _super);
  BoundLayout<Reg> x41 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra23), count), _super);
  BoundLayout<Reg> x42 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra24), count), _super);
  BoundLayout<Reg> x43 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra25), count), _super);
  BoundLayout<Reg> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra26), count), _super);
  BoundLayout<Reg> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra27), count), _super);
  BoundLayout<Reg> x46 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra28), count), _super);
  BoundLayout<Reg> x47 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x8, _extra29), count), _super);
  BoundLayout<Reg> x48 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra0), count), _super);
  BoundLayout<Reg> x49 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra1), count), _super);
  BoundLayout<Reg> x50 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra2), count), _super);
  BoundLayout<Reg> x51 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra3), count), _super);
  BoundLayout<Reg> x52 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra4), count), _super);
  BoundLayout<Reg> x53 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra5), count), _super);
  BoundLayout<Reg> x54 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra6), count), _super);
  BoundLayout<Reg> x55 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra7), count), _super);
  BoundLayout<Reg> x56 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra8), count), _super);
  BoundLayout<Reg> x57 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra9), count), _super);
  BoundLayout<Reg> x58 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra10), count), _super);
  BoundLayout<Reg> x59 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra11), count), _super);
  BoundLayout<Reg> x60 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra12), count), _super);
  BoundLayout<Reg> x61 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra13), count), _super);
  BoundLayout<Reg> x62 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra14), count), _super);
  BoundLayout<Reg> x63 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra15), count), _super);
  BoundLayout<Reg> x64 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra16), count), _super);
  BoundLayout<Reg> x65 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x9, _extra17), count), _super);
  BoundLayout<Reg> x66 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra0), count), _super);
  BoundLayout<Reg> x67 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra1), count), _super);
  BoundLayout<Reg> x68 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra2), count), _super);
  BoundLayout<Reg> x69 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra3), count), _super);
  BoundLayout<Reg> x70 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra4), count), _super);
  BoundLayout<Reg> x71 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra5), count), _super);
  BoundLayout<Reg> x72 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra6), count), _super);
  BoundLayout<Reg> x73 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra7), count), _super);
  BoundLayout<Reg> x74 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra8), count), _super);
  BoundLayout<Reg> x75 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra9), count), _super);
  BoundLayout<Reg> x76 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra10), count), _super);
  BoundLayout<Reg> x77 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra11), count), _super);
  BoundLayout<Reg> x78 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra12), count), _super);
  BoundLayout<Reg> x79 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra13), count), _super);
  BoundLayout<Reg> x80 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra14), count), _super);
  BoundLayout<Reg> x81 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra15), count), _super);
  BoundLayout<Reg> x82 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra16), count), _super);
  BoundLayout<Reg> x83 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x10, _extra17), count), _super);
  BoundLayout<Reg> x84 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra0), count), _super);
  BoundLayout<Reg> x85 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra1), count), _super);
  BoundLayout<Reg> x86 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra2), count), _super);
  BoundLayout<Reg> x87 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra3), count), _super);
  BoundLayout<Reg> x88 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra4), count), _super);
  BoundLayout<Reg> x89 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra5), count), _super);
  BoundLayout<Reg> x90 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra6), count), _super);
  BoundLayout<Reg> x91 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra7), count), _super);
  BoundLayout<Reg> x92 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra8), count), _super);
  BoundLayout<Reg> x93 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra9), count), _super);
  BoundLayout<Reg> x94 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra10), count), _super);
  BoundLayout<Reg> x95 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra11), count), _super);
  BoundLayout<Reg> x96 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra12), count), _super);
  BoundLayout<Reg> x97 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra13), count), _super);
  BoundLayout<Reg> x98 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra14), count), _super);
  BoundLayout<Reg> x99 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra15), count), _super);
  BoundLayout<Reg> x100 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra16), count), _super);
  BoundLayout<Reg> x101 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra17), count), _super);
  BoundLayout<Reg> x102 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra18), count), _super);
  BoundLayout<Reg> x103 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra19), count), _super);
  BoundLayout<Reg> x104 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra20), count), _super);
  BoundLayout<Reg> x105 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra21), count), _super);
  BoundLayout<Reg> x106 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra22), count), _super);
  BoundLayout<Reg> x107 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra23), count), _super);
  BoundLayout<Reg> x108 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra24), count), _super);
  BoundLayout<Reg> x109 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra25), count), _super);
  BoundLayout<Reg> x110 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra26), count), _super);
  BoundLayout<Reg> x111 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra27), count), _super);
  BoundLayout<Reg> x112 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra28), count), _super);
  BoundLayout<Reg> x113 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra29), count), _super);
  BoundLayout<Reg> x114 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra30), count), _super);
  BoundLayout<Reg> x115 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra31), count), _super);
  BoundLayout<Reg> x116 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra32), count), _super);
  BoundLayout<Reg> x117 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra33), count), _super);
  BoundLayout<Reg> x118 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra34), count), _super);
  BoundLayout<Reg> x119 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra35), count), _super);
  BoundLayout<Reg> x120 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra36), count), _super);
  BoundLayout<Reg> x121 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra37), count), _super);
  BoundLayout<Reg> x122 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra38), count), _super);
  BoundLayout<Reg> x123 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra39), count), _super);
  BoundLayout<Reg> x124 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra40), count), _super);
  BoundLayout<Reg> x125 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, _extra41), count), _super);
  BoundLayout<Reg> x126 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra0), count), _super);
  BoundLayout<Reg> x127 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra1), count), _super);
  BoundLayout<Reg> x128 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra2), count), _super);
  BoundLayout<Reg> x129 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra3), count), _super);
  BoundLayout<Reg> x130 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra4), count), _super);
  BoundLayout<Reg> x131 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra5), count), _super);
  BoundLayout<Reg> x132 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra6), count), _super);
  BoundLayout<Reg> x133 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra7), count), _super);
  BoundLayout<Reg> x134 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra8), count), _super);
  BoundLayout<Reg> x135 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra9), count), _super);
  BoundLayout<Reg> x136 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra10), count), _super);
  BoundLayout<Reg> x137 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra11), count), _super);
  BoundLayout<Reg> x138 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra12), count), _super);
  BoundLayout<Reg> x139 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra13), count), _super);
  BoundLayout<Reg> x140 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra14), count), _super);
  BoundLayout<Reg> x141 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra15), count), _super);
  BoundLayout<Reg> x142 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra16), count), _super);
  BoundLayout<Reg> x143 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra17), count), _super);
  BoundLayout<Reg> x144 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra18), count), _super);
  BoundLayout<Reg> x145 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra19), count), _super);
  BoundLayout<Reg> x146 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra20), count), _super);
  BoundLayout<Reg> x147 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra21), count), _super);
  BoundLayout<Reg> x148 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra22), count), _super);
  BoundLayout<Reg> x149 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra23), count), _super);
  BoundLayout<Reg> x150 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra24), count), _super);
  BoundLayout<Reg> x151 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra25), count), _super);
  BoundLayout<Reg> x152 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra26), count), _super);
  BoundLayout<Reg> x153 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra27), count), _super);
  BoundLayout<Reg> x154 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra28), count), _super);
  BoundLayout<Reg> x155 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra29), count), _super);
  BoundLayout<Reg> x156 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra30), count), _super);
  BoundLayout<Reg> x157 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra31), count), _super);
  BoundLayout<Reg> x158 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra32), count), _super);
  BoundLayout<Reg> x159 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra33), count), _super);
  BoundLayout<Reg> x160 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra34), count), _super);
  BoundLayout<Reg> x161 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra35), count), _super);
  BoundLayout<Reg> x162 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra36), count), _super);
  BoundLayout<Reg> x163 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra37), count), _super);
  BoundLayout<Reg> x164 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra38), count), _super);
  BoundLayout<Reg> x165 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra39), count), _super);
  BoundLayout<Reg> x166 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra40), count), _super);
  BoundLayout<Reg> x167 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x12, _extra41), count), _super);
  BoundLayout<Reg> x168 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra0), count), _super);
  BoundLayout<Reg> x169 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x13, _extra1), count), _super);
  BoundLayout<Reg> x170 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra0), count), _super);
  BoundLayout<Reg> x171 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra1), count), _super);
  BoundLayout<Reg> x172 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra2), count), _super);
  BoundLayout<Reg> x173 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra3), count), _super);
  BoundLayout<Reg> x174 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra4), count), _super);
  BoundLayout<Reg> x175 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra5), count), _super);
  BoundLayout<Reg> x176 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra6), count), _super);
  BoundLayout<Reg> x177 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra7), count), _super);
  BoundLayout<Reg> x178 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra8), count), _super);
  BoundLayout<Reg> x179 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra9), count), _super);
  BoundLayout<Reg> x180 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra10), count), _super);
  BoundLayout<Reg> x181 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra11), count), _super);
  BoundLayout<Reg> x182 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra12), count), _super);
  BoundLayout<Reg> x183 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra13), count), _super);
  BoundLayout<Reg> x184 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra14), count), _super);
  BoundLayout<Reg> x185 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra15), count), _super);
  BoundLayout<Reg> x186 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra16), count), _super);
  BoundLayout<Reg> x187 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra17), count), _super);
  BoundLayout<Reg> x188 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra18), count), _super);
  BoundLayout<Reg> x189 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra19), count), _super);
  BoundLayout<Reg> x190 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra20), count), _super);
  BoundLayout<Reg> x191 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra21), count), _super);
  BoundLayout<Reg> x192 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra22), count), _super);
  BoundLayout<Reg> x193 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra23), count), _super);
  BoundLayout<Reg> x194 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra24), count), _super);
  BoundLayout<Reg> x195 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra25), count), _super);
  BoundLayout<Reg> x196 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra26), count), _super);
  BoundLayout<Reg> x197 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra27), count), _super);
  BoundLayout<Reg> x198 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra28), count), _super);
  BoundLayout<Reg> x199 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra29), count), _super);
  BoundLayout<Reg> x200 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra30), count), _super);
  BoundLayout<Reg> x201 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra31), count), _super);
  BoundLayout<Reg> x202 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra32), count), _super);
  BoundLayout<Reg> x203 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra33), count), _super);
  BoundLayout<Reg> x204 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra34), count), _super);
  BoundLayout<Reg> x205 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra35), count), _super);
  BoundLayout<Reg> x206 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra36), count), _super);
  BoundLayout<Reg> x207 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x14, _extra37), count), _super);
  BoundLayout<Reg> x208 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra0), count), _super);
  BoundLayout<Reg> x209 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x15, _extra1), count), _super);
  PoseidonStateStruct x210;
  if (to_size_t(x7[0]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:449)
    PoseidonStateStruct x211 =
        exec_PoseidonEntry(ctx, arg0, arg1, x16, x17, LAYOUT_LOOKUP(x8, _super));
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x18, Val(0));
    EQZ(LOAD(x18, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x19, Val(0));
    EQZ(LOAD(x19, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x20, Val(0));
    EQZ(LOAD(x20, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x21, Val(0));
    EQZ(LOAD(x21, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x22, Val(0));
    EQZ(LOAD(x22, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x23, Val(0));
    EQZ(LOAD(x23, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x24, Val(0));
    EQZ(LOAD(x24, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x25, Val(0));
    EQZ(LOAD(x25, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x26, Val(0));
    EQZ(LOAD(x26, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x27, Val(0));
    EQZ(LOAD(x27, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x28, Val(0));
    EQZ(LOAD(x28, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x29, Val(0));
    EQZ(LOAD(x29, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x30, Val(0));
    EQZ(LOAD(x30, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x31, Val(0));
    EQZ(LOAD(x31, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x32, Val(0));
    EQZ(LOAD(x32, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x33, Val(0));
    EQZ(LOAD(x33, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x34, Val(0));
    EQZ(LOAD(x34, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x35, Val(0));
    EQZ(LOAD(x35, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x36, Val(0));
    EQZ(LOAD(x36, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x37, Val(0));
    EQZ(LOAD(x37, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x38, Val(0));
    EQZ(LOAD(x38, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x39, Val(0));
    EQZ(LOAD(x39, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x40, Val(0));
    EQZ(LOAD(x40, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x41, Val(0));
    EQZ(LOAD(x41, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x42, Val(0));
    EQZ(LOAD(x42, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x43, Val(0));
    EQZ(LOAD(x43, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x44, Val(0));
    EQZ(LOAD(x44, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x45, Val(0));
    EQZ(LOAD(x45, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x46, Val(0));
    EQZ(LOAD(x46, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x47, Val(0));
    EQZ(LOAD(x47, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x211;
  } else if (to_size_t(x7[1]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:450)
    PoseidonStateStruct x212 = back_PoseidonState(ctx, 1, x6);
    PoseidonStateStruct x213 =
        exec_PoseidonLoadState(ctx, arg0, arg1, x212, LAYOUT_LOOKUP(x9, _super));
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x48, Val(0));
    EQZ(LOAD(x48, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x49, Val(0));
    EQZ(LOAD(x49, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x50, Val(0));
    EQZ(LOAD(x50, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x51, Val(0));
    EQZ(LOAD(x51, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x52, Val(0));
    EQZ(LOAD(x52, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x53, Val(0));
    EQZ(LOAD(x53, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x54, Val(0));
    EQZ(LOAD(x54, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x55, Val(0));
    EQZ(LOAD(x55, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x56, Val(0));
    EQZ(LOAD(x56, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x57, Val(0));
    EQZ(LOAD(x57, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x58, Val(0));
    EQZ(LOAD(x58, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x59, Val(0));
    EQZ(LOAD(x59, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x60, Val(0));
    EQZ(LOAD(x60, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x61, Val(0));
    EQZ(LOAD(x61, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x62, Val(0));
    EQZ(LOAD(x62, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x63, Val(0));
    EQZ(LOAD(x63, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x64, Val(0));
    EQZ(LOAD(x64, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x65, Val(0));
    EQZ(LOAD(x65, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x213;
  } else if (to_size_t(x7[2]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:451)
    PoseidonStateStruct x214 = back_PoseidonState(ctx, 1, x6);
    PoseidonStateStruct x215 =
        exec_PoseidonLoadIn(ctx, arg0, arg1, x214, LAYOUT_LOOKUP(x10, _super), global4);
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x66, Val(0));
    EQZ(LOAD(x66, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x67, Val(0));
    EQZ(LOAD(x67, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x68, Val(0));
    EQZ(LOAD(x68, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x69, Val(0));
    EQZ(LOAD(x69, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x70, Val(0));
    EQZ(LOAD(x70, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x71, Val(0));
    EQZ(LOAD(x71, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x72, Val(0));
    EQZ(LOAD(x72, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x73, Val(0));
    EQZ(LOAD(x73, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x74, Val(0));
    EQZ(LOAD(x74, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x75, Val(0));
    EQZ(LOAD(x75, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x76, Val(0));
    EQZ(LOAD(x76, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x77, Val(0));
    EQZ(LOAD(x77, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x78, Val(0));
    EQZ(LOAD(x78, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x79, Val(0));
    EQZ(LOAD(x79, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x80, Val(0));
    EQZ(LOAD(x80, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x81, Val(0));
    EQZ(LOAD(x81, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x82, Val(0));
    EQZ(LOAD(x82, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x83, Val(0));
    EQZ(LOAD(x83, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x215;
  } else if (to_size_t(x7[3]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:452)
    PoseidonStateStruct x216 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x11, _super));
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x84, Val(0));
    EQZ(LOAD(x84, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x85, Val(0));
    EQZ(LOAD(x85, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x86, Val(0));
    EQZ(LOAD(x86, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x87, Val(0));
    EQZ(LOAD(x87, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x88, Val(0));
    EQZ(LOAD(x88, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x89, Val(0));
    EQZ(LOAD(x89, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x90, Val(0));
    EQZ(LOAD(x90, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x91, Val(0));
    EQZ(LOAD(x91, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x92, Val(0));
    EQZ(LOAD(x92, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x93, Val(0));
    EQZ(LOAD(x93, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x94, Val(0));
    EQZ(LOAD(x94, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x95, Val(0));
    EQZ(LOAD(x95, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x96, Val(0));
    EQZ(LOAD(x96, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x97, Val(0));
    EQZ(LOAD(x97, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x98, Val(0));
    EQZ(LOAD(x98, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x99, Val(0));
    EQZ(LOAD(x99, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x100, Val(0));
    EQZ(LOAD(x100, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x101, Val(0));
    EQZ(LOAD(x101, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x102, Val(0));
    EQZ(LOAD(x102, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x103, Val(0));
    EQZ(LOAD(x103, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x104, Val(0));
    EQZ(LOAD(x104, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x105, Val(0));
    EQZ(LOAD(x105, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x106, Val(0));
    EQZ(LOAD(x106, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x107, Val(0));
    EQZ(LOAD(x107, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x108, Val(0));
    EQZ(LOAD(x108, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x109, Val(0));
    EQZ(LOAD(x109, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x110, Val(0));
    EQZ(LOAD(x110, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x111, Val(0));
    EQZ(LOAD(x111, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x112, Val(0));
    EQZ(LOAD(x112, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x113, Val(0));
    EQZ(LOAD(x113, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x114, Val(0));
    EQZ(LOAD(x114, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x115, Val(0));
    EQZ(LOAD(x115, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x116, Val(0));
    EQZ(LOAD(x116, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x117, Val(0));
    EQZ(LOAD(x117, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x118, Val(0));
    EQZ(LOAD(x118, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x119, Val(0));
    EQZ(LOAD(x119, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x120, Val(0));
    EQZ(LOAD(x120, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x121, Val(0));
    EQZ(LOAD(x121, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x122, Val(0));
    EQZ(LOAD(x122, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x123, Val(0));
    EQZ(LOAD(x123, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x124, Val(0));
    EQZ(LOAD(x124, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x125, Val(0));
    EQZ(LOAD(x125, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x216;
  } else if (to_size_t(x7[4]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:453)
    PoseidonStateStruct x217 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x12, _super));
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x126, Val(0));
    EQZ(LOAD(x126, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x127, Val(0));
    EQZ(LOAD(x127, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x128, Val(0));
    EQZ(LOAD(x128, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x129, Val(0));
    EQZ(LOAD(x129, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x130, Val(0));
    EQZ(LOAD(x130, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x131, Val(0));
    EQZ(LOAD(x131, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x132, Val(0));
    EQZ(LOAD(x132, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x133, Val(0));
    EQZ(LOAD(x133, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x134, Val(0));
    EQZ(LOAD(x134, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x135, Val(0));
    EQZ(LOAD(x135, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x136, Val(0));
    EQZ(LOAD(x136, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x137, Val(0));
    EQZ(LOAD(x137, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x138, Val(0));
    EQZ(LOAD(x138, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x139, Val(0));
    EQZ(LOAD(x139, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x140, Val(0));
    EQZ(LOAD(x140, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x141, Val(0));
    EQZ(LOAD(x141, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x142, Val(0));
    EQZ(LOAD(x142, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x143, Val(0));
    EQZ(LOAD(x143, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x144, Val(0));
    EQZ(LOAD(x144, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x145, Val(0));
    EQZ(LOAD(x145, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x146, Val(0));
    EQZ(LOAD(x146, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x147, Val(0));
    EQZ(LOAD(x147, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x148, Val(0));
    EQZ(LOAD(x148, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x149, Val(0));
    EQZ(LOAD(x149, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x150, Val(0));
    EQZ(LOAD(x150, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x151, Val(0));
    EQZ(LOAD(x151, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x152, Val(0));
    EQZ(LOAD(x152, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x153, Val(0));
    EQZ(LOAD(x153, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x154, Val(0));
    EQZ(LOAD(x154, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x155, Val(0));
    EQZ(LOAD(x155, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x156, Val(0));
    EQZ(LOAD(x156, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x157, Val(0));
    EQZ(LOAD(x157, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x158, Val(0));
    EQZ(LOAD(x158, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x159, Val(0));
    EQZ(LOAD(x159, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x160, Val(0));
    EQZ(LOAD(x160, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x161, Val(0));
    EQZ(LOAD(x161, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x162, Val(0));
    EQZ(LOAD(x162, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x163, Val(0));
    EQZ(LOAD(x163, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x164, Val(0));
    EQZ(LOAD(x164, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x165, Val(0));
    EQZ(LOAD(x165, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x166, Val(0));
    EQZ(LOAD(x166, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x167, Val(0));
    EQZ(LOAD(x167, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x217;
  } else if (to_size_t(x7[5]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:454)
    PoseidonStateStruct x218 = back_PoseidonState(ctx, 1, x6);
    PoseidonStateStruct x219 =
        exec_PoseidonDoOut(ctx, arg0, arg1, x218, LAYOUT_LOOKUP(x13, _super));
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x168, Val(0));
    EQZ(LOAD(x168, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x169, Val(0));
    EQZ(LOAD(x169, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x219;
  } else if (to_size_t(x7[6]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:455)
    PoseidonStateStruct x220 = back_PoseidonState(ctx, 1, x6);
    PoseidonStateStruct x221 =
        exec_PoseidonPaging(ctx, arg0, x17, x220, LAYOUT_LOOKUP(x14, _super));
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x170, Val(0));
    EQZ(LOAD(x170, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x171, Val(0));
    EQZ(LOAD(x171, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x172, Val(0));
    EQZ(LOAD(x172, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x173, Val(0));
    EQZ(LOAD(x173, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x174, Val(0));
    EQZ(LOAD(x174, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x175, Val(0));
    EQZ(LOAD(x175, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x176, Val(0));
    EQZ(LOAD(x176, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x177, Val(0));
    EQZ(LOAD(x177, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x178, Val(0));
    EQZ(LOAD(x178, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x179, Val(0));
    EQZ(LOAD(x179, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x180, Val(0));
    EQZ(LOAD(x180, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x181, Val(0));
    EQZ(LOAD(x181, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x182, Val(0));
    EQZ(LOAD(x182, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x183, Val(0));
    EQZ(LOAD(x183, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x184, Val(0));
    EQZ(LOAD(x184, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x185, Val(0));
    EQZ(LOAD(x185, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x186, Val(0));
    EQZ(LOAD(x186, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x187, Val(0));
    EQZ(LOAD(x187, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x188, Val(0));
    EQZ(LOAD(x188, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x189, Val(0));
    EQZ(LOAD(x189, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x190, Val(0));
    EQZ(LOAD(x190, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x191, Val(0));
    EQZ(LOAD(x191, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x192, Val(0));
    EQZ(LOAD(x192, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x193, Val(0));
    EQZ(LOAD(x193, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x194, Val(0));
    EQZ(LOAD(x194, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x195, Val(0));
    EQZ(LOAD(x195, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x196, Val(0));
    EQZ(LOAD(x196, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x197, Val(0));
    EQZ(LOAD(x197, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x198, Val(0));
    EQZ(LOAD(x198, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x199, Val(0));
    EQZ(LOAD(x199, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x200, Val(0));
    EQZ(LOAD(x200, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x201, Val(0));
    EQZ(LOAD(x201, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x202, Val(0));
    EQZ(LOAD(x202, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x203, Val(0));
    EQZ(LOAD(x203, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x204, Val(0));
    EQZ(LOAD(x204, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x205, Val(0));
    EQZ(LOAD(x205, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x206, Val(0));
    EQZ(LOAD(x206, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x207, Val(0));
    EQZ(LOAD(x207, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x221;
  } else if (to_size_t(x7[7]._super)) {
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:456)
    PoseidonStateStruct x222 = back_PoseidonState(ctx, 1, x6);
    PoseidonStateStruct x223 =
        exec_PoseidonStoreState(ctx, arg0, arg1, x222, LAYOUT_LOOKUP(x15, _super));
    // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
    STORE(x208, Val(0));
    EQZ(LOAD(x208, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    STORE(x209, Val(0));
    EQZ(LOAD(x209, 0), "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)");
    x210 = x223;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:445)
  Val x224 = arg0._super;
  // GetDiffCount(zirgen/circuit/rv32im/v2/dsl/mem.zir:22)
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:458)
  Val x225 = INVOKE_EXTERN(ctx, getDiffCount, x224);
  CycleArgStruct x226 = exec_CycleArg(ctx, neg_0(x225), x224, LAYOUT_LOOKUP(layout3, arg));
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:460)
  Val x227 = (x226.cycle._super - x224);
  EQZ(x227, "Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:460)");
  // Poseidon0(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:448)
  PoseidonStateStruct x228 = back_PoseidonState(ctx, 0, x6);
  return InstOutputStruct{
      .newPc = x16, .newState = x228.nextState._super, .newMode = x228.mode._super};
}
InstOutputStruct exec_Poseidon1(ExecContext& ctx,
                                NondetRegStruct arg0,
                                BoundLayout<NondetRegLayout> arg1,
                                InstInputStruct arg2,
                                BoundLayout<Poseidon1Layout> layout3) {
  // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:467)
  BoundLayout<Poseidon1StateLayout> x4 = LAYOUT_LOOKUP(layout3, state);
  BoundLayout<PoseidonStateLayout> x5 = LAYOUT_LOOKUP(x4, _super);
  // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:464)
  NondetRegStruct8Array x6 = arg2.minorOnehot._super;
  PoseidonStateStruct x7;
  if (to_size_t(x6[0]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:468)
    PoseidonStateStruct x8 = back_PoseidonState(ctx, 1, x5);
    PoseidonStateStruct x9 = exec_PoseidonExtRound(ctx, x8, LAYOUT_LOOKUP(x4, arm0));
    x7 = x9;
  } else if (to_size_t(x6[1]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:469)
    PoseidonStateStruct x10 = back_PoseidonState(ctx, 1, x5);
    PoseidonStateStruct x11 = exec_PoseidonIntRounds(ctx, x10, LAYOUT_LOOKUP(x4, arm1));
    x7 = x11;
  } else if (to_size_t(x6[2]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:470)
    PoseidonStateStruct x12 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x4, arm2));
    x7 = x12;
  } else if (to_size_t(x6[3]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:471)
    PoseidonStateStruct x13 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x4, arm3));
    x7 = x13;
  } else if (to_size_t(x6[4]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:472)
    PoseidonStateStruct x14 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x4, arm4));
    x7 = x14;
  } else if (to_size_t(x6[5]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:473)
    PoseidonStateStruct x15 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x4, arm5));
    x7 = x15;
  } else if (to_size_t(x6[6]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:474)
    PoseidonStateStruct x16 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x4, arm6));
    x7 = x16;
  } else if (to_size_t(x6[7]._super)) {
    // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:475)
    PoseidonStateStruct x17 = exec_PoseidonInvalid(ctx, LAYOUT_LOOKUP(x4, arm7));
    x7 = x17;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:464)
  Val x18 = arg0._super;
  // GetDiffCount(zirgen/circuit/rv32im/v2/dsl/mem.zir:22)
  // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:477)
  Val x19 = INVOKE_EXTERN(ctx, getDiffCount, x18);
  CycleArgStruct x20 = exec_CycleArg(ctx, neg_0(x19), x18, LAYOUT_LOOKUP(layout3, arg));
  // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:479)
  Val x21 = (x20.cycle._super - x18);
  EQZ(x21, "Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:479)");
  // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:467)
  PoseidonStateStruct x22 = back_PoseidonState(ctx, 0, x5);
  // InstOutput(zirgen/circuit/rv32im/v2/dsl/inst.zir:46)
  // Poseidon1(zirgen/circuit/rv32im/v2/dsl/inst_p2.zir:480)
  InstOutputStruct x23 = InstOutputStruct{
      .newPc = arg2.pcU32, .newState = x22.nextState._super, .newMode = x22.mode._super};
  return x23;
}
OneHot_11_Struct
exec_OneHot_11_(ExecContext& ctx, Val arg0, BoundLayout<OneHot_11_Layout> layout1) {
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:7)
  NondetRegStruct11Array x2 = map(
      Val11Array{
          Val(0), Val(1), Val(2), Val(3), Val(4), Val(5), Val(6), Val(7), Val(8), Val(9), Val(10)},
      LAYOUT_LOOKUP(layout1, _super),
      std::function(
          [&](Val11Array::value_type x3, BoundLayout<NondetRegLayout11LayoutArray::value_type> x4) {
            NondetRegStruct x5 = exec_NondetBitReg(ctx, isz((x3 - arg0)), x4);
            return x5;
          }));
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)
  Val x6 = x2[1]._super;
  Val x7 = (x2[0]._super + x6);
  Val x8 = x2[2]._super;
  Val x9 = x2[3]._super;
  Val x10 = x2[4]._super;
  Val x11 = (((x7 + x8) + x9) + x10);
  Val x12 = x2[5]._super;
  Val x13 = x2[6]._super;
  Val x14 = x2[7]._super;
  Val x15 = (((x11 + x12) + x13) + x14);
  Val x16 = x2[8]._super;
  Val x17 = x2[9]._super;
  Val x18 = x2[10]._super;
  Val x19 = (((x15 + x16) + x17) + x18);
  EQZ((x19 - Val(1)), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:9)");
  // OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)
  Val x20 = (((x6 + (x8 * Val(2))) + (x9 * Val(3))) + (x10 * Val(4)));
  Val x21 = (((x20 + (x12 * Val(5))) + (x13 * Val(6))) + (x14 * Val(7)));
  Val x22 = (((x21 + (x16 * Val(8))) + (x17 * Val(9))) + (x18 * Val(10)));
  EQZ((x22 - arg0), "OneHot(zirgen/circuit/rv32im/v2/dsl/one_hot.zir:11)");
  return OneHot_11_Struct{._super = x2};
}
TopStruct exec_Top(ExecContext& ctx, BoundLayout<TopLayout> layout0, GlobalBuf global1) {
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:41)
  BoundLayout<NondetRegLayout> x2 = LAYOUT_LOOKUP(layout0, nextPcLow);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:42)
  BoundLayout<NondetRegLayout> x3 = LAYOUT_LOOKUP(layout0, nextPcHigh);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:43)
  BoundLayout<NondetRegLayout> x4 = LAYOUT_LOOKUP(layout0, nextState_0);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:44)
  BoundLayout<NondetRegLayout> x5 = LAYOUT_LOOKUP(layout0, nextMachineMode);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:49)
  BoundLayout<NondetRegLayout> x6 = LAYOUT_LOOKUP(layout0, cycle);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:69)
  BoundLayout<TopInstResultLayout> x7 = LAYOUT_LOOKUP(layout0, instResult);
  // IsFirstCycle(zirgen/circuit/rv32im/v2/dsl/top.zir:17)
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:47)
  Val x8 = INVOKE_EXTERN(ctx, isFirstCycle_0);
  NondetRegStruct x9 = exec_NondetReg(ctx, x8, LAYOUT_LOOKUP(layout0, isFirstCycle));
  // GetCycle(zirgen/circuit/rv32im/v2/dsl/top.zir:18)
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:48)
  Val x10 = INVOKE_EXTERN(ctx, getCycle);
  NondetRegStruct x11 = exec_NondetReg(ctx, x10, LAYOUT_LOOKUP(layout0, cycleND));
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:49)
  NondetRegStruct x12 = exec_Reg(ctx, x11._super, x6);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:47)
  Val x13 = x9._super;
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:52)
  Val x14 = (Val(1) - x13);
  NondetRegStruct x15 = back_Reg(ctx, 1, x2);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:53)
  NondetRegStruct x16 = back_Reg(ctx, 1, x3);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:56)
  NondetRegStruct x17 = back_Reg(ctx, 1, x4);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:58)
  NondetRegStruct x18 = back_Reg(ctx, 1, x5);
  // GetMajorMinor(zirgen/circuit/rv32im/v2/dsl/top.zir:25)
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:60)
  auto [x19, x20] = INVOKE_EXTERN(ctx, getMajorMinor);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:62)
  NondetRegStruct x21 = exec_NondetReg(ctx, x19, LAYOUT_LOOKUP(layout0, major));
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:63)
  NondetRegStruct x22 = exec_NondetReg(ctx, x20, LAYOUT_LOOKUP(layout0, minor));
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:62)
  Val x23 = x21._super;
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:63)
  Val x24 = x22._super;
  // Log(<preamble>:22)
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:64)
  INVOKE_EXTERN(ctx, log, "Major/Minor = ", std::initializer_list<Val>{x23, x24});
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:49)
  Val x25 = x12._super;
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:66)
  InstInputStruct x26 =
      exec_InstInput(ctx,
                     x25,
                     x23,
                     x24,
                     ValU32Struct{.low = (x14 * x15._super), .high = (x14 * x16._super)},
                     (x14 * x17._super),
                     ((x14 * x18._super) + x13),
                     LAYOUT_LOOKUP(layout0, instInput));
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:68)
  OneHot_11_Struct x27 = exec_OneHot_11_(ctx, x23, LAYOUT_LOOKUP(layout0, majorOnehot));
  NondetRegStruct11Array x28 = x27._super;
  InstOutputStruct x29;
  if (to_size_t(x28[0]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:70)
    InstOutputStruct x30 = exec_Misc0(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm0));
    x29 = x30;
  } else if (to_size_t(x28[1]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:71)
    InstOutputStruct x31 = exec_Misc1(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm1));
    x29 = x31;
  } else if (to_size_t(x28[2]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:72)
    InstOutputStruct x32 = exec_Misc2(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm2));
    x29 = x32;
  } else if (to_size_t(x28[3]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:73)
    InstOutputStruct x33 = exec_Mul0(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm3));
    x29 = x33;
  } else if (to_size_t(x28[4]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:74)
    InstOutputStruct x34 = exec_Div0(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm4));
    x29 = x34;
  } else if (to_size_t(x28[5]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:75)
    InstOutputStruct x35 = exec_Mem0(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm5));
    x29 = x35;
  } else if (to_size_t(x28[6]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:76)
    InstOutputStruct x36 = exec_Mem1(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm6));
    x29 = x36;
  } else if (to_size_t(x28[7]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:77)
    InstOutputStruct x37 = exec_Control0(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm7), global1);
    x29 = x37;
  } else if (to_size_t(x28[8]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:78)
    InstOutputStruct x38 = exec_ECall0(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm8), global1);
    x29 = x38;
  } else if (to_size_t(x28[9]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:79)
    InstOutputStruct x39 = exec_Poseidon0(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm9), global1);
    x29 = x39;
  } else if (to_size_t(x28[10]._super)) {
    // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:80)
    InstOutputStruct x40 = exec_Poseidon1(ctx, x12, x6, x26, LAYOUT_LOOKUP(x7, arm10));
    x29 = x40;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:69)
  ValU32Struct x41 = x29.newPc;
  Val x42 = x41.low;
  Val x43 = x41.high;
  Val x44 = x29.newState;
  Val x45 = x29.newMode;
  // Log(<preamble>:22)
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:84)
  INVOKE_EXTERN(
      ctx,
      log,
      "Cycle, pc, state, mm",
      std::initializer_list<Val>{x25, ((x42 * Val(1509949441)) + (x43 * Val(16384))), x44, x45});
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:85)
  NondetRegStruct x46 = exec_Reg(ctx, x42, x2);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:86)
  NondetRegStruct x47 = exec_Reg(ctx, x43, x3);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:87)
  NondetRegStruct x48 = exec_Reg(ctx, x44, x4);
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:88)
  NondetRegStruct x49 = exec_Reg(ctx, x45, x5);
  return TopStruct{};
}
void step_Top(ExecContext& ctx, MutableBuf data0, GlobalBuf global1) {
  // Top(zirgen/circuit/rv32im/v2/dsl/top.zir:27)
  BoundLayout<TopLayout> x2 = BIND_LAYOUT(kLayout_Top, data0);
  TopStruct x3 = exec_Top(ctx, x2, global1);
  return;
}
ComponentStruct exec_TopAccum(ExecContext& ctx,
                              BoundLayout<TopLayout> arg0,
                              BoundLayout<LayoutAccumLayout> layout1,
                              GlobalBuf mix2) {
  // zirgen/dsl/passes/GenerateAccum.cpp:526
  BoundLayout<_mixLayout> x3 = BIND_LAYOUT(kLayoutMix, mix2);
  // zirgen/dsl/passes/GenerateAccum.cpp:533
  BoundLayout<_accumLayout> x4 = LAYOUT_LOOKUP(x3, randomness);
  // zirgen/dsl/passes/GenerateAccum.cpp:545
  BoundLayout<TopInstResultLayout> x5 = LAYOUT_LOOKUP(arg0, instResult);
  // zirgen/dsl/passes/GenerateAccum.cpp:574
  BoundLayout<NondetRegLayout11LayoutArray> x6 = LAYOUT_LOOKUP(x5, _selector);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Misc0Layout> x7 = LAYOUT_LOOKUP(x5, arm0);
  // zirgen/dsl/passes/GenerateAccum.cpp:604
  BoundLayout<Reg19LayoutArray> x8 = LAYOUT_LOOKUP(layout1, columns);
  // zirgen/dsl/passes/GenerateAccum.cpp:53
  BoundLayout<Reg> x9 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, argU16), val);
  BoundLayout<Arg_MemoryArgLayout> x10 = LAYOUT_LOOKUP(x4, memoryArg);
  BoundLayout<Reg> x11 = LAYOUT_LOOKUP(x10, addr);
  BoundLayout<Reg> x12 = LAYOUT_LOOKUP(x10, cycle);
  BoundLayout<Reg> x13 = LAYOUT_LOOKUP(x10, dataLow);
  BoundLayout<Reg> x14 = LAYOUT_LOOKUP(x10, dataHigh);
  BoundLayout<Reg> x15 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, cycleArg), cycle);
  // zirgen/dsl/passes/GenerateAccum.cpp:83
  BoundLayout<Reg> x16 = LAYOUT_LOOKUP(x4, _offset);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<FinalizeMiscLayout> x17 = LAYOUT_LOOKUP(x7, _super);
  BoundLayout<NormalizeU32Layout> x18 = LAYOUT_LOOKUP(x17, writeData);
  BoundLayout<ArgU16Layout> x19 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, low16), arg);
  BoundLayout<ArgU16Layout> x20 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x18, high16), arg);
  BoundLayout<NormalizeU32Layout> x21 = LAYOUT_LOOKUP(x17, pcNorm);
  BoundLayout<ArgU16Layout> x22 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x21, low16), arg);
  BoundLayout<ArgU16Layout> x23 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x21, high16), arg);
  BoundLayout<MemoryWriteLayout> x24 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x17, _0), _0);
  BoundLayout<MemoryIOLayout> x25 = LAYOUT_LOOKUP(x24, io);
  BoundLayout<MemoryArgLayout> x26 = LAYOUT_LOOKUP(x25, oldTxn);
  BoundLayout<MemoryArgLayout> x27 = LAYOUT_LOOKUP(x25, newTxn);
  BoundLayout<CycleArgLayout> x28 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x24, _0), _0), arg);
  BoundLayout<MiscInputLayout> x29 = LAYOUT_LOOKUP(x7, input);
  BoundLayout<DecodeInstLayout> x30 = LAYOUT_LOOKUP(x29, decoded);
  BoundLayout<CycleArgLayout> x31 = LAYOUT_LOOKUP(x30, arg);
  BoundLayout<AddrDecomposeLayout> x32 = LAYOUT_LOOKUP(x30, pcAddr);
  BoundLayout<ArgU16Layout> x33 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x32, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x34 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x32, med14), arg);
  BoundLayout<MemoryReadLayout> x35 = LAYOUT_LOOKUP(x30, loadInst);
  BoundLayout<MemoryIOLayout> x36 = LAYOUT_LOOKUP(x35, io);
  BoundLayout<MemoryArgLayout> x37 = LAYOUT_LOOKUP(x36, oldTxn);
  BoundLayout<MemoryArgLayout> x38 = LAYOUT_LOOKUP(x36, newTxn);
  BoundLayout<CycleArgLayout> x39 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x35, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x40 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x29, rs1), _super);
  BoundLayout<MemoryIOLayout> x41 = LAYOUT_LOOKUP(x40, io);
  BoundLayout<MemoryArgLayout> x42 = LAYOUT_LOOKUP(x41, oldTxn);
  BoundLayout<MemoryArgLayout> x43 = LAYOUT_LOOKUP(x41, newTxn);
  BoundLayout<CycleArgLayout> x44 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x40, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x45 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x29, rs2), _super);
  BoundLayout<MemoryIOLayout> x46 = LAYOUT_LOOKUP(x45, io);
  BoundLayout<MemoryArgLayout> x47 = LAYOUT_LOOKUP(x46, oldTxn);
  BoundLayout<MemoryArgLayout> x48 = LAYOUT_LOOKUP(x46, newTxn);
  BoundLayout<CycleArgLayout> x49 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x45, _0), _0), arg);
  BoundLayout<ArgU16Layout5LayoutArray> x50 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, _argumentsMiscOutput), argU16);
  // zirgen/dsl/passes/GenerateAccum.cpp:624
  ComponentStruct x51 = ComponentStruct{};
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Misc1Layout> x52 = LAYOUT_LOOKUP(x5, arm1);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<FinalizeMiscLayout> x53 = LAYOUT_LOOKUP(x52, _super);
  BoundLayout<NormalizeU32Layout> x54 = LAYOUT_LOOKUP(x53, writeData);
  BoundLayout<ArgU16Layout> x55 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x54, low16), arg);
  BoundLayout<ArgU16Layout> x56 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x54, high16), arg);
  BoundLayout<NormalizeU32Layout> x57 = LAYOUT_LOOKUP(x53, pcNorm);
  BoundLayout<ArgU16Layout> x58 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x57, low16), arg);
  BoundLayout<ArgU16Layout> x59 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x57, high16), arg);
  BoundLayout<MemoryWriteLayout> x60 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x53, _0), _0);
  BoundLayout<MemoryIOLayout> x61 = LAYOUT_LOOKUP(x60, io);
  BoundLayout<MemoryArgLayout> x62 = LAYOUT_LOOKUP(x61, oldTxn);
  BoundLayout<MemoryArgLayout> x63 = LAYOUT_LOOKUP(x61, newTxn);
  BoundLayout<CycleArgLayout> x64 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x60, _0), _0), arg);
  BoundLayout<MiscInputLayout> x65 = LAYOUT_LOOKUP(x52, input);
  BoundLayout<DecodeInstLayout> x66 = LAYOUT_LOOKUP(x65, decoded);
  BoundLayout<CycleArgLayout> x67 = LAYOUT_LOOKUP(x66, arg);
  BoundLayout<AddrDecomposeLayout> x68 = LAYOUT_LOOKUP(x66, pcAddr);
  BoundLayout<ArgU16Layout> x69 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x68, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x70 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x68, med14), arg);
  BoundLayout<MemoryReadLayout> x71 = LAYOUT_LOOKUP(x66, loadInst);
  BoundLayout<MemoryIOLayout> x72 = LAYOUT_LOOKUP(x71, io);
  BoundLayout<MemoryArgLayout> x73 = LAYOUT_LOOKUP(x72, oldTxn);
  BoundLayout<MemoryArgLayout> x74 = LAYOUT_LOOKUP(x72, newTxn);
  BoundLayout<CycleArgLayout> x75 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x71, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x76 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x65, rs1), _super);
  BoundLayout<MemoryIOLayout> x77 = LAYOUT_LOOKUP(x76, io);
  BoundLayout<MemoryArgLayout> x78 = LAYOUT_LOOKUP(x77, oldTxn);
  BoundLayout<MemoryArgLayout> x79 = LAYOUT_LOOKUP(x77, newTxn);
  BoundLayout<CycleArgLayout> x80 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x76, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x81 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x65, rs2), _super);
  BoundLayout<MemoryIOLayout> x82 = LAYOUT_LOOKUP(x81, io);
  BoundLayout<MemoryArgLayout> x83 = LAYOUT_LOOKUP(x82, oldTxn);
  BoundLayout<MemoryArgLayout> x84 = LAYOUT_LOOKUP(x82, newTxn);
  BoundLayout<CycleArgLayout> x85 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x81, _0), _0), arg);
  BoundLayout<ArgU16Layout5LayoutArray> x86 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(x52, _argumentsMiscOutput), argU16);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Misc2Layout> x87 = LAYOUT_LOOKUP(x5, arm2);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<FinalizeMiscLayout> x88 = LAYOUT_LOOKUP(x87, _super);
  BoundLayout<NormalizeU32Layout> x89 = LAYOUT_LOOKUP(x88, writeData);
  BoundLayout<ArgU16Layout> x90 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x89, low16), arg);
  BoundLayout<ArgU16Layout> x91 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x89, high16), arg);
  BoundLayout<NormalizeU32Layout> x92 = LAYOUT_LOOKUP(x88, pcNorm);
  BoundLayout<ArgU16Layout> x93 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x92, low16), arg);
  BoundLayout<ArgU16Layout> x94 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x92, high16), arg);
  BoundLayout<MemoryWriteLayout> x95 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x88, _0), _0);
  BoundLayout<MemoryIOLayout> x96 = LAYOUT_LOOKUP(x95, io);
  BoundLayout<MemoryArgLayout> x97 = LAYOUT_LOOKUP(x96, oldTxn);
  BoundLayout<MemoryArgLayout> x98 = LAYOUT_LOOKUP(x96, newTxn);
  BoundLayout<CycleArgLayout> x99 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x95, _0), _0), arg);
  BoundLayout<MiscInputLayout> x100 = LAYOUT_LOOKUP(x87, input);
  BoundLayout<DecodeInstLayout> x101 = LAYOUT_LOOKUP(x100, decoded);
  BoundLayout<CycleArgLayout> x102 = LAYOUT_LOOKUP(x101, arg);
  BoundLayout<AddrDecomposeLayout> x103 = LAYOUT_LOOKUP(x101, pcAddr);
  BoundLayout<ArgU16Layout> x104 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x103, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x105 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x103, med14), arg);
  BoundLayout<MemoryReadLayout> x106 = LAYOUT_LOOKUP(x101, loadInst);
  BoundLayout<MemoryIOLayout> x107 = LAYOUT_LOOKUP(x106, io);
  BoundLayout<MemoryArgLayout> x108 = LAYOUT_LOOKUP(x107, oldTxn);
  BoundLayout<MemoryArgLayout> x109 = LAYOUT_LOOKUP(x107, newTxn);
  BoundLayout<CycleArgLayout> x110 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x106, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x111 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x100, rs1), _super);
  BoundLayout<MemoryIOLayout> x112 = LAYOUT_LOOKUP(x111, io);
  BoundLayout<MemoryArgLayout> x113 = LAYOUT_LOOKUP(x112, oldTxn);
  BoundLayout<MemoryArgLayout> x114 = LAYOUT_LOOKUP(x112, newTxn);
  BoundLayout<CycleArgLayout> x115 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x111, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x116 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x100, rs2), _super);
  BoundLayout<MemoryIOLayout> x117 = LAYOUT_LOOKUP(x116, io);
  BoundLayout<MemoryArgLayout> x118 = LAYOUT_LOOKUP(x117, oldTxn);
  BoundLayout<MemoryArgLayout> x119 = LAYOUT_LOOKUP(x117, newTxn);
  BoundLayout<CycleArgLayout> x120 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x116, _0), _0), arg);
  BoundLayout<ArgU16Layout5LayoutArray> x121 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(x87, _argumentsMiscOutput), argU16);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Mul0Layout> x122 = LAYOUT_LOOKUP(x5, arm3);
  // zirgen/dsl/passes/GenerateAccum.cpp:53
  BoundLayout<Reg> x123 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, argU8), val);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<MulInputLayout> x124 = LAYOUT_LOOKUP(x122, input);
  BoundLayout<DecodeInstLayout> x125 = LAYOUT_LOOKUP(x124, decoded);
  BoundLayout<CycleArgLayout> x126 = LAYOUT_LOOKUP(x125, arg);
  BoundLayout<AddrDecomposeLayout> x127 = LAYOUT_LOOKUP(x125, pcAddr);
  BoundLayout<ArgU16Layout> x128 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x127, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x129 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x127, med14), arg);
  BoundLayout<MemoryReadLayout> x130 = LAYOUT_LOOKUP(x125, loadInst);
  BoundLayout<MemoryIOLayout> x131 = LAYOUT_LOOKUP(x130, io);
  BoundLayout<MemoryArgLayout> x132 = LAYOUT_LOOKUP(x131, oldTxn);
  BoundLayout<MemoryArgLayout> x133 = LAYOUT_LOOKUP(x131, newTxn);
  BoundLayout<CycleArgLayout> x134 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x130, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x135 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x124, rs1), _super);
  BoundLayout<MemoryIOLayout> x136 = LAYOUT_LOOKUP(x135, io);
  BoundLayout<MemoryArgLayout> x137 = LAYOUT_LOOKUP(x136, oldTxn);
  BoundLayout<MemoryArgLayout> x138 = LAYOUT_LOOKUP(x136, newTxn);
  BoundLayout<CycleArgLayout> x139 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x135, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x140 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x124, rs2), _super);
  BoundLayout<MemoryIOLayout> x141 = LAYOUT_LOOKUP(x140, io);
  BoundLayout<MemoryArgLayout> x142 = LAYOUT_LOOKUP(x141, oldTxn);
  BoundLayout<MemoryArgLayout> x143 = LAYOUT_LOOKUP(x141, newTxn);
  BoundLayout<CycleArgLayout> x144 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x140, _0), _0), arg);
  BoundLayout<_Arguments_Mul0MulOutputLayout> x145 = LAYOUT_LOOKUP(x122, _argumentsMulOutput);
  BoundLayout<ArgU16Layout6LayoutArray> x146 = LAYOUT_LOOKUP(x145, argU16);
  BoundLayout<ArgU8Layout13LayoutArray> x147 = LAYOUT_LOOKUP(x145, argU8);
  BoundLayout<MemoryWriteLayout> x148 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x122, _0), _0);
  BoundLayout<MemoryIOLayout> x149 = LAYOUT_LOOKUP(x148, io);
  BoundLayout<MemoryArgLayout> x150 = LAYOUT_LOOKUP(x149, oldTxn);
  BoundLayout<MemoryArgLayout> x151 = LAYOUT_LOOKUP(x149, newTxn);
  BoundLayout<CycleArgLayout> x152 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x148, _0), _0), arg);
  BoundLayout<NormalizeU32Layout> x153 = LAYOUT_LOOKUP(x122, pcAdd);
  BoundLayout<ArgU16Layout> x154 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x153, low16), arg);
  BoundLayout<ArgU16Layout> x155 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x153, high16), arg);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Div0Layout> x156 = LAYOUT_LOOKUP(x5, arm4);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<DivInputLayout> x157 = LAYOUT_LOOKUP(x156, input);
  BoundLayout<DecodeInstLayout> x158 = LAYOUT_LOOKUP(x157, decoded);
  BoundLayout<CycleArgLayout> x159 = LAYOUT_LOOKUP(x158, arg);
  BoundLayout<AddrDecomposeLayout> x160 = LAYOUT_LOOKUP(x158, pcAddr);
  BoundLayout<ArgU16Layout> x161 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x160, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x162 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x160, med14), arg);
  BoundLayout<MemoryReadLayout> x163 = LAYOUT_LOOKUP(x158, loadInst);
  BoundLayout<MemoryIOLayout> x164 = LAYOUT_LOOKUP(x163, io);
  BoundLayout<MemoryArgLayout> x165 = LAYOUT_LOOKUP(x164, oldTxn);
  BoundLayout<MemoryArgLayout> x166 = LAYOUT_LOOKUP(x164, newTxn);
  BoundLayout<CycleArgLayout> x167 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x163, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x168 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x157, rs1), _super);
  BoundLayout<MemoryIOLayout> x169 = LAYOUT_LOOKUP(x168, io);
  BoundLayout<MemoryArgLayout> x170 = LAYOUT_LOOKUP(x169, oldTxn);
  BoundLayout<MemoryArgLayout> x171 = LAYOUT_LOOKUP(x169, newTxn);
  BoundLayout<CycleArgLayout> x172 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x168, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x173 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x157, rs2), _super);
  BoundLayout<MemoryIOLayout> x174 = LAYOUT_LOOKUP(x173, io);
  BoundLayout<MemoryArgLayout> x175 = LAYOUT_LOOKUP(x174, oldTxn);
  BoundLayout<MemoryArgLayout> x176 = LAYOUT_LOOKUP(x174, newTxn);
  BoundLayout<CycleArgLayout> x177 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x173, _0), _0), arg);
  BoundLayout<_Arguments_Div0MulOutputLayout> x178 = LAYOUT_LOOKUP(x156, _argumentsMulOutput);
  BoundLayout<ArgU16Layout9LayoutArray> x179 = LAYOUT_LOOKUP(x178, argU16);
  BoundLayout<ArgU8Layout13LayoutArray> x180 = LAYOUT_LOOKUP(x178, argU8);
  BoundLayout<MemoryWriteLayout> x181 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x156, _0), _0);
  BoundLayout<MemoryIOLayout> x182 = LAYOUT_LOOKUP(x181, io);
  BoundLayout<MemoryArgLayout> x183 = LAYOUT_LOOKUP(x182, oldTxn);
  BoundLayout<MemoryArgLayout> x184 = LAYOUT_LOOKUP(x182, newTxn);
  BoundLayout<CycleArgLayout> x185 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x181, _0), _0), arg);
  BoundLayout<NormalizeU32Layout> x186 = LAYOUT_LOOKUP(x156, pcAdd);
  BoundLayout<ArgU16Layout> x187 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x186, low16), arg);
  BoundLayout<ArgU16Layout> x188 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x186, high16), arg);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Mem0Layout> x189 = LAYOUT_LOOKUP(x5, arm5);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<MemLoadInputLayout> x190 = LAYOUT_LOOKUP(x189, input);
  BoundLayout<DecodeInstLayout> x191 = LAYOUT_LOOKUP(x190, decoded);
  BoundLayout<CycleArgLayout> x192 = LAYOUT_LOOKUP(x191, arg);
  BoundLayout<AddrDecomposeLayout> x193 = LAYOUT_LOOKUP(x191, pcAddr);
  BoundLayout<ArgU16Layout> x194 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x193, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x195 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x193, med14), arg);
  BoundLayout<MemoryReadLayout> x196 = LAYOUT_LOOKUP(x191, loadInst);
  BoundLayout<MemoryIOLayout> x197 = LAYOUT_LOOKUP(x196, io);
  BoundLayout<MemoryArgLayout> x198 = LAYOUT_LOOKUP(x197, oldTxn);
  BoundLayout<MemoryArgLayout> x199 = LAYOUT_LOOKUP(x197, newTxn);
  BoundLayout<CycleArgLayout> x200 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x196, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x201 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x190, rs1), _super);
  BoundLayout<MemoryIOLayout> x202 = LAYOUT_LOOKUP(x201, io);
  BoundLayout<MemoryArgLayout> x203 = LAYOUT_LOOKUP(x202, oldTxn);
  BoundLayout<MemoryArgLayout> x204 = LAYOUT_LOOKUP(x202, newTxn);
  BoundLayout<CycleArgLayout> x205 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x201, _0), _0), arg);
  BoundLayout<NormalizeU32Layout> x206 = LAYOUT_LOOKUP(x190, addrU32);
  BoundLayout<ArgU16Layout> x207 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x206, low16), arg);
  BoundLayout<ArgU16Layout> x208 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x206, high16), arg);
  BoundLayout<AddrDecomposeBitsLayout> x209 = LAYOUT_LOOKUP(x190, addr);
  BoundLayout<ArgU16Layout> x210 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x209, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x211 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x209, med14), arg);
  BoundLayout<MemoryReadLayout> x212 = LAYOUT_LOOKUP(x190, data);
  BoundLayout<MemoryIOLayout> x213 = LAYOUT_LOOKUP(x212, io);
  BoundLayout<MemoryArgLayout> x214 = LAYOUT_LOOKUP(x213, oldTxn);
  BoundLayout<MemoryArgLayout> x215 = LAYOUT_LOOKUP(x213, newTxn);
  BoundLayout<CycleArgLayout> x216 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x212, _0), _0), arg);
  BoundLayout<ArgU8Layout3LayoutArray> x217 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(x189, _argumentsOutput), argU8);
  BoundLayout<MemoryWriteLayout> x218 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x189, _0), _0);
  BoundLayout<MemoryIOLayout> x219 = LAYOUT_LOOKUP(x218, io);
  BoundLayout<MemoryArgLayout> x220 = LAYOUT_LOOKUP(x219, oldTxn);
  BoundLayout<MemoryArgLayout> x221 = LAYOUT_LOOKUP(x219, newTxn);
  BoundLayout<CycleArgLayout> x222 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x218, _0), _0), arg);
  BoundLayout<NormalizeU32Layout> x223 = LAYOUT_LOOKUP(x189, pcAdd);
  BoundLayout<ArgU16Layout> x224 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x223, low16), arg);
  BoundLayout<ArgU16Layout> x225 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x223, high16), arg);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Mem1Layout> x226 = LAYOUT_LOOKUP(x5, arm6);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<MemStoreInputLayout> x227 = LAYOUT_LOOKUP(x226, input);
  BoundLayout<DecodeInstLayout> x228 = LAYOUT_LOOKUP(x227, decoded);
  BoundLayout<CycleArgLayout> x229 = LAYOUT_LOOKUP(x228, arg);
  BoundLayout<AddrDecomposeLayout> x230 = LAYOUT_LOOKUP(x228, pcAddr);
  BoundLayout<ArgU16Layout> x231 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x230, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x232 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x230, med14), arg);
  BoundLayout<MemoryReadLayout> x233 = LAYOUT_LOOKUP(x228, loadInst);
  BoundLayout<MemoryIOLayout> x234 = LAYOUT_LOOKUP(x233, io);
  BoundLayout<MemoryArgLayout> x235 = LAYOUT_LOOKUP(x234, oldTxn);
  BoundLayout<MemoryArgLayout> x236 = LAYOUT_LOOKUP(x234, newTxn);
  BoundLayout<CycleArgLayout> x237 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x233, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x238 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x227, rs1), _super);
  BoundLayout<MemoryIOLayout> x239 = LAYOUT_LOOKUP(x238, io);
  BoundLayout<MemoryArgLayout> x240 = LAYOUT_LOOKUP(x239, oldTxn);
  BoundLayout<MemoryArgLayout> x241 = LAYOUT_LOOKUP(x239, newTxn);
  BoundLayout<CycleArgLayout> x242 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x238, _0), _0), arg);
  BoundLayout<MemoryReadLayout> x243 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x227, rs2), _super);
  BoundLayout<MemoryIOLayout> x244 = LAYOUT_LOOKUP(x243, io);
  BoundLayout<MemoryArgLayout> x245 = LAYOUT_LOOKUP(x244, oldTxn);
  BoundLayout<MemoryArgLayout> x246 = LAYOUT_LOOKUP(x244, newTxn);
  BoundLayout<CycleArgLayout> x247 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x243, _0), _0), arg);
  BoundLayout<NormalizeU32Layout> x248 = LAYOUT_LOOKUP(x227, addrU32);
  BoundLayout<ArgU16Layout> x249 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x248, low16), arg);
  BoundLayout<ArgU16Layout> x250 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x248, high16), arg);
  BoundLayout<AddrDecomposeBitsLayout> x251 = LAYOUT_LOOKUP(x227, addr);
  BoundLayout<ArgU16Layout> x252 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x251, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x253 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x251, med14), arg);
  BoundLayout<MemoryReadLayout> x254 = LAYOUT_LOOKUP(x227, data);
  BoundLayout<MemoryIOLayout> x255 = LAYOUT_LOOKUP(x254, io);
  BoundLayout<MemoryArgLayout> x256 = LAYOUT_LOOKUP(x255, oldTxn);
  BoundLayout<MemoryArgLayout> x257 = LAYOUT_LOOKUP(x255, newTxn);
  BoundLayout<CycleArgLayout> x258 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x254, _0), _0), arg);
  BoundLayout<ArgU8Layout4LayoutArray> x259 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(x226, _argumentsOutput), argU8);
  BoundLayout<MemoryWriteLayout> x260 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x226, _0), _0);
  BoundLayout<MemoryIOLayout> x261 = LAYOUT_LOOKUP(x260, io);
  BoundLayout<MemoryArgLayout> x262 = LAYOUT_LOOKUP(x261, oldTxn);
  BoundLayout<MemoryArgLayout> x263 = LAYOUT_LOOKUP(x261, newTxn);
  BoundLayout<CycleArgLayout> x264 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x260, _0), _0), arg);
  BoundLayout<NormalizeU32Layout> x265 = LAYOUT_LOOKUP(x226, pcAdd);
  BoundLayout<ArgU16Layout> x266 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x265, low16), arg);
  BoundLayout<ArgU16Layout> x267 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x265, high16), arg);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Control0Layout> x268 = LAYOUT_LOOKUP(x5, arm7);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<CycleArgLayout> x269 = LAYOUT_LOOKUP(x268, arg);
  BoundLayout<_Arguments_Control0_SuperLayout> x270 = LAYOUT_LOOKUP(x268, _arguments_Super);
  BoundLayout<MemoryArgLayout16LayoutArray> x271 = LAYOUT_LOOKUP(x270, memoryArg);
  BoundLayout<CycleArgLayout8LayoutArray> x272 = LAYOUT_LOOKUP(x270, cycleArg);
  BoundLayout<ArgU16Layout16LayoutArray> x273 = LAYOUT_LOOKUP(x270, argU16);
  BoundLayout<ArgU8Layout16LayoutArray> x274 = LAYOUT_LOOKUP(x270, argU8);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<ECall0Layout> x275 = LAYOUT_LOOKUP(x5, arm8);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<AddrDecomposeBitsLayout> x276 = LAYOUT_LOOKUP(x275, pcAddr);
  BoundLayout<ArgU16Layout> x277 =
      LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x276, upperDiff), ret), arg);
  BoundLayout<ArgU16Layout> x278 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x276, med14), arg);
  BoundLayout<_Arguments_ECall0OutputLayout> x279 = LAYOUT_LOOKUP(x275, _argumentsOutput);
  BoundLayout<MemoryArgLayout8LayoutArray> x280 = LAYOUT_LOOKUP(x279, memoryArg);
  BoundLayout<CycleArgLayout4LayoutArray> x281 = LAYOUT_LOOKUP(x279, cycleArg);
  BoundLayout<ArgU16Layout2LayoutArray> x282 = LAYOUT_LOOKUP(x279, argU16);
  BoundLayout<NormalizeU32Layout> x283 = LAYOUT_LOOKUP(x275, addPC);
  BoundLayout<ArgU16Layout> x284 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x283, low16), arg);
  BoundLayout<ArgU16Layout> x285 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x283, high16), arg);
  BoundLayout<CycleArgLayout> x286 = LAYOUT_LOOKUP(x275, arg);
  // zirgen/dsl/passes/GenerateAccum.cpp:602
  BoundLayout<Poseidon0Layout> x287 = LAYOUT_LOOKUP(x5, arm9);
  // zirgen/dsl/passes/GenerateAccum.cpp:276
  BoundLayout<_Arguments_Poseidon0StateLayout> x288 = LAYOUT_LOOKUP(x287, _argumentsState);
  BoundLayout<MemoryArgLayout16LayoutArray> x289 = LAYOUT_LOOKUP(x288, memoryArg);
  BoundLayout<CycleArgLayout8LayoutArray> x290 = LAYOUT_LOOKUP(x288, cycleArg);
  BoundLayout<ArgU16Layout16LayoutArray> x291 = LAYOUT_LOOKUP(x288, argU16);
  BoundLayout<ArgU8Layout2LayoutArray> x292 = LAYOUT_LOOKUP(x288, argU8);
  BoundLayout<CycleArgLayout> x293 = LAYOUT_LOOKUP(x287, arg);
  BoundLayout<CycleArgLayout> x294 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x5, arm10), arg);
  ComponentStruct x295;
  if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 0), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x296 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x19, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x297 = (x296 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x298 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x19, count), _super), 0) * inv_0(x297));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x299 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x298);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x300 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x20, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x301 = (x300 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x302 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x20, count), _super), 0) * inv_0(x301));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x303 = (x297 * x301);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x304 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x19, count), _super), 0) * x301);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x305 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x22, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x306 = (x305 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x307 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x22, count), _super), 0) * inv_0(x306));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x308 = ((x299 + x302) + x307);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x308);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x309 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x310 = (((x309 * (x303 * x306)) - (x304 * x306)) -
                   ((x297 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x20, count), _super), 0)) * x306));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x310 - (x303 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x22, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x311 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x23, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x312 = (x311 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x313 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x23, count), _super), 0) * inv_0(x312));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x314 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x26, addr), _super), 0));
    ExtVal x315 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x26, cycle), _super), 0));
    ExtVal x316 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x26, dataLow), _super), 0));
    ExtVal x317 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x26, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x318 = (((x314 + x315) + x316) + x317);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x319 = (x318 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x320 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x26, count), _super), 0) * inv_0(x319));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x321 = (x312 * x319);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x322 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x23, count), _super), 0) * x319);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x323 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x27, addr), _super), 0));
    ExtVal x324 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x27, cycle), _super), 0));
    ExtVal x325 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x27, dataLow), _super), 0));
    ExtVal x326 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x27, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x327 = (((x323 + x324) + x325) + x326);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x328 = (x327 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x329 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x27, count), _super), 0) * inv_0(x328));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x330 = (((x308 + x313) + x320) + x329);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x330);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x331 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x332 = (((x331 * (x321 * x328)) - (x322 * x328)) -
                   ((x312 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x26, count), _super), 0)) * x328));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x332 - (x321 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x27, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x333 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x28, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x334 = (x333 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x335 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x28, count), _super), 0) * inv_0(x334));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x336 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x31, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x337 = (x336 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x338 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x31, count), _super), 0) * inv_0(x337));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x339 = (x334 * x337);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x340 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x28, count), _super), 0) * x337);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x341 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x33, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x342 = (x341 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x343 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x33, count), _super), 0) * inv_0(x342));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x344 = (((x330 + x335) + x338) + x343);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x344);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x345 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x346 = (((x345 * (x339 * x342)) - (x340 * x342)) -
                   ((x334 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x31, count), _super), 0)) * x342));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x346 - (x339 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x33, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x347 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x34, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x348 = (x347 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x349 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x34, count), _super), 0) * inv_0(x348));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x350 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x37, addr), _super), 0));
    ExtVal x351 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x37, cycle), _super), 0));
    ExtVal x352 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x37, dataLow), _super), 0));
    ExtVal x353 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x37, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x354 = (((x350 + x351) + x352) + x353);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x355 = (x354 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x356 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x37, count), _super), 0) * inv_0(x355));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x357 = (x348 * x355);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x358 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x34, count), _super), 0) * x355);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x359 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x38, addr), _super), 0));
    ExtVal x360 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x38, cycle), _super), 0));
    ExtVal x361 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x38, dataLow), _super), 0));
    ExtVal x362 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x38, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x363 = (((x359 + x360) + x361) + x362);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x364 = (x363 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x365 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x38, count), _super), 0) * inv_0(x364));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x366 = (((x344 + x349) + x356) + x365);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x366);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x367 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x368 = (((x367 * (x357 * x364)) - (x358 * x364)) -
                   ((x348 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x37, count), _super), 0)) * x364));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x368 - (x357 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x38, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x369 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x39, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x370 = (x369 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x371 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x39, count), _super), 0) * inv_0(x370));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x372 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x42, addr), _super), 0));
    ExtVal x373 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x42, cycle), _super), 0));
    ExtVal x374 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x42, dataLow), _super), 0));
    ExtVal x375 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x42, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x376 = (((x372 + x373) + x374) + x375);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x377 = (x376 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x378 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x42, count), _super), 0) * inv_0(x377));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x379 = (x370 * x377);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x380 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x39, count), _super), 0) * x377);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x381 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x43, addr), _super), 0));
    ExtVal x382 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x43, cycle), _super), 0));
    ExtVal x383 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x43, dataLow), _super), 0));
    ExtVal x384 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x43, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x385 = (((x381 + x382) + x383) + x384);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x386 = (x385 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x387 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x43, count), _super), 0) * inv_0(x386));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x388 = (((x366 + x371) + x378) + x387);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x388);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x389 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x390 = (((x389 * (x379 * x386)) - (x380 * x386)) -
                   ((x370 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x42, count), _super), 0)) * x386));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x390 - (x379 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x43, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x391 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x44, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x392 = (x391 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x393 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x44, count), _super), 0) * inv_0(x392));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x394 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x47, addr), _super), 0));
    ExtVal x395 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x47, cycle), _super), 0));
    ExtVal x396 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x47, dataLow), _super), 0));
    ExtVal x397 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x47, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x398 = (((x394 + x395) + x396) + x397);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x399 = (x398 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x400 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x47, count), _super), 0) * inv_0(x399));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x401 = (x392 * x399);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x402 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x44, count), _super), 0) * x399);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x403 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x48, addr), _super), 0));
    ExtVal x404 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x48, cycle), _super), 0));
    ExtVal x405 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x48, dataLow), _super), 0));
    ExtVal x406 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x48, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x407 = (((x403 + x404) + x405) + x406);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x408 = (x407 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x409 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x48, count), _super), 0) * inv_0(x408));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x410 = (((x388 + x393) + x400) + x409);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x410);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x411 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x412 = (((x411 * (x401 * x408)) - (x402 * x408)) -
                   ((x392 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x47, count), _super), 0)) * x408));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x412 - (x401 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x48, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x413 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x49, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x414 = (x413 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x415 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x49, count), _super), 0) * inv_0(x414));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x416 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x417 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x418 = ((LOAD_EXT(x9, 0) * LOAD(x417, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x419 = ((x410 + x415) + (LOAD(x416, 0) * inv_0(x418)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x420 = (x414 * x418);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x421 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x49, count), _super), 0) * x418);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x422 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x423 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x424 = ((LOAD_EXT(x9, 0) * LOAD(x423, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x425 = (x419 + (LOAD(x422, 0) * inv_0(x424)));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x425);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x426 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x427 = (((x426 * (x420 * x424)) - (x421 * x424)) - ((x414 * LOAD(x416, 0)) * x424));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x427 - (x420 * LOAD(x422, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x428 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x429 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x430 = ((LOAD_EXT(x9, 0) * LOAD(x429, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x431 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x432 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x433 = ((LOAD_EXT(x9, 0) * LOAD(x432, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x434 = ((x425 + (LOAD(x428, 0) * inv_0(x430))) + (LOAD(x431, 0) * inv_0(x433)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x435 = (x430 * x433);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x436 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x437 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x50, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x438 = ((LOAD_EXT(x9, 0) * LOAD(x437, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x439 = (x434 + (LOAD(x436, 0) * inv_0(x438)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x440 = ((LOAD(x428, 0) * x433) * x438);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x439);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x441 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x442 = (((x441 * (x435 * x438)) - x440) - ((x430 * LOAD(x431, 0)) * x438));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x442 - (x435 * LOAD(x436, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), x439);
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x443 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x443, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 1), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x444 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x55, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x445 = (x444 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x446 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x55, count), _super), 0) * inv_0(x445));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x447 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x446);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x448 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x56, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x449 = (x448 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x450 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x56, count), _super), 0) * inv_0(x449));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x451 = (x445 * x449);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x452 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x55, count), _super), 0) * x449);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x453 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x58, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x454 = (x453 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x455 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x58, count), _super), 0) * inv_0(x454));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x456 = ((x447 + x450) + x455);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x456);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x457 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x458 = (((x457 * (x451 * x454)) - (x452 * x454)) -
                   ((x445 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x56, count), _super), 0)) * x454));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x458 - (x451 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x58, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x459 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x59, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x460 = (x459 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x461 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x59, count), _super), 0) * inv_0(x460));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x462 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x62, addr), _super), 0));
    ExtVal x463 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x62, cycle), _super), 0));
    ExtVal x464 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x62, dataLow), _super), 0));
    ExtVal x465 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x62, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x466 = (((x462 + x463) + x464) + x465);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x467 = (x466 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x468 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x62, count), _super), 0) * inv_0(x467));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x469 = (x460 * x467);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x470 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x59, count), _super), 0) * x467);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x471 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x63, addr), _super), 0));
    ExtVal x472 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x63, cycle), _super), 0));
    ExtVal x473 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x63, dataLow), _super), 0));
    ExtVal x474 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x63, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x475 = (((x471 + x472) + x473) + x474);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x476 = (x475 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x477 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x63, count), _super), 0) * inv_0(x476));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x478 = (((x456 + x461) + x468) + x477);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x478);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x479 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x480 = (((x479 * (x469 * x476)) - (x470 * x476)) -
                   ((x460 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x62, count), _super), 0)) * x476));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x480 - (x469 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x63, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x481 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x64, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x482 = (x481 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x483 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x64, count), _super), 0) * inv_0(x482));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x484 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x67, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x485 = (x484 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x486 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x67, count), _super), 0) * inv_0(x485));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x487 = (x482 * x485);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x488 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x64, count), _super), 0) * x485);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x489 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x69, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x490 = (x489 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x491 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x69, count), _super), 0) * inv_0(x490));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x492 = (((x478 + x483) + x486) + x491);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x492);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x493 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x494 = (((x493 * (x487 * x490)) - (x488 * x490)) -
                   ((x482 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x67, count), _super), 0)) * x490));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x494 - (x487 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x69, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x495 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x70, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x496 = (x495 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x497 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x70, count), _super), 0) * inv_0(x496));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x498 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x73, addr), _super), 0));
    ExtVal x499 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x73, cycle), _super), 0));
    ExtVal x500 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x73, dataLow), _super), 0));
    ExtVal x501 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x73, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x502 = (((x498 + x499) + x500) + x501);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x503 = (x502 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x504 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x73, count), _super), 0) * inv_0(x503));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x505 = (x496 * x503);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x506 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x70, count), _super), 0) * x503);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x507 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x74, addr), _super), 0));
    ExtVal x508 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x74, cycle), _super), 0));
    ExtVal x509 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x74, dataLow), _super), 0));
    ExtVal x510 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x74, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x511 = (((x507 + x508) + x509) + x510);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x512 = (x511 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x513 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x74, count), _super), 0) * inv_0(x512));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x514 = (((x492 + x497) + x504) + x513);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x514);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x515 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x516 = (((x515 * (x505 * x512)) - (x506 * x512)) -
                   ((x496 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x73, count), _super), 0)) * x512));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x516 - (x505 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x74, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x517 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x75, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x518 = (x517 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x519 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x75, count), _super), 0) * inv_0(x518));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x520 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x78, addr), _super), 0));
    ExtVal x521 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x78, cycle), _super), 0));
    ExtVal x522 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x78, dataLow), _super), 0));
    ExtVal x523 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x78, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x524 = (((x520 + x521) + x522) + x523);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x525 = (x524 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x526 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x78, count), _super), 0) * inv_0(x525));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x527 = (x518 * x525);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x528 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x75, count), _super), 0) * x525);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x529 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x79, addr), _super), 0));
    ExtVal x530 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x79, cycle), _super), 0));
    ExtVal x531 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x79, dataLow), _super), 0));
    ExtVal x532 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x79, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x533 = (((x529 + x530) + x531) + x532);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x534 = (x533 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x535 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x79, count), _super), 0) * inv_0(x534));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x536 = (((x514 + x519) + x526) + x535);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x536);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x537 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x538 = (((x537 * (x527 * x534)) - (x528 * x534)) -
                   ((x518 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x78, count), _super), 0)) * x534));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x538 - (x527 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x79, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x539 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x80, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x540 = (x539 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x541 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x80, count), _super), 0) * inv_0(x540));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x542 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x83, addr), _super), 0));
    ExtVal x543 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x83, cycle), _super), 0));
    ExtVal x544 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x83, dataLow), _super), 0));
    ExtVal x545 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x83, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x546 = (((x542 + x543) + x544) + x545);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x547 = (x546 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x548 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x83, count), _super), 0) * inv_0(x547));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x549 = (x540 * x547);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x550 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x80, count), _super), 0) * x547);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x551 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x84, addr), _super), 0));
    ExtVal x552 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x84, cycle), _super), 0));
    ExtVal x553 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x84, dataLow), _super), 0));
    ExtVal x554 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x84, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x555 = (((x551 + x552) + x553) + x554);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x556 = (x555 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x557 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x84, count), _super), 0) * inv_0(x556));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x558 = (((x536 + x541) + x548) + x557);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x558);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x559 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x560 = (((x559 * (x549 * x556)) - (x550 * x556)) -
                   ((x540 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x83, count), _super), 0)) * x556));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x560 - (x549 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x84, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x561 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x85, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x562 = (x561 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x563 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x85, count), _super), 0) * inv_0(x562));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x564 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x565 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x566 = ((LOAD_EXT(x9, 0) * LOAD(x565, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x567 = ((x558 + x563) + (LOAD(x564, 0) * inv_0(x566)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x568 = (x562 * x566);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x569 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x85, count), _super), 0) * x566);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x570 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x571 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x572 = ((LOAD_EXT(x9, 0) * LOAD(x571, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x573 = (x567 + (LOAD(x570, 0) * inv_0(x572)));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x573);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x574 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x575 = (((x574 * (x568 * x572)) - (x569 * x572)) - ((x562 * LOAD(x564, 0)) * x572));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x575 - (x568 * LOAD(x570, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x576 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x577 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x578 = ((LOAD_EXT(x9, 0) * LOAD(x577, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x579 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x580 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x581 = ((LOAD_EXT(x9, 0) * LOAD(x580, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x582 = ((x573 + (LOAD(x576, 0) * inv_0(x578))) + (LOAD(x579, 0) * inv_0(x581)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x583 = (x578 * x581);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x584 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x585 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x86, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x586 = ((LOAD_EXT(x9, 0) * LOAD(x585, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x587 = (x582 + (LOAD(x584, 0) * inv_0(x586)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x588 = ((LOAD(x576, 0) * x581) * x586);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x587);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x589 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x590 = (((x589 * (x583 * x586)) - x588) - ((x578 * LOAD(x579, 0)) * x586));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x590 - (x583 * LOAD(x584, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), x587);
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x591 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x591, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 2), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x592 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x90, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x593 = (x592 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x594 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x90, count), _super), 0) * inv_0(x593));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x595 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x594);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x596 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x91, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x597 = (x596 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x598 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x91, count), _super), 0) * inv_0(x597));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x599 = (x593 * x597);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x600 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x90, count), _super), 0) * x597);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x601 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x93, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x602 = (x601 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x603 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x93, count), _super), 0) * inv_0(x602));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x604 = ((x595 + x598) + x603);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x604);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x605 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x606 = (((x605 * (x599 * x602)) - (x600 * x602)) -
                   ((x593 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x91, count), _super), 0)) * x602));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x606 - (x599 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x93, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x607 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x94, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x608 = (x607 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x609 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x94, count), _super), 0) * inv_0(x608));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x610 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x97, addr), _super), 0));
    ExtVal x611 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x97, cycle), _super), 0));
    ExtVal x612 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x97, dataLow), _super), 0));
    ExtVal x613 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x97, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x614 = (((x610 + x611) + x612) + x613);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x615 = (x614 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x616 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x97, count), _super), 0) * inv_0(x615));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x617 = (x608 * x615);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x618 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x94, count), _super), 0) * x615);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x619 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x98, addr), _super), 0));
    ExtVal x620 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x98, cycle), _super), 0));
    ExtVal x621 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x98, dataLow), _super), 0));
    ExtVal x622 = (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x98, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x623 = (((x619 + x620) + x621) + x622);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x624 = (x623 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x625 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x98, count), _super), 0) * inv_0(x624));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x626 = (((x604 + x609) + x616) + x625);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x626);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x627 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x628 = (((x627 * (x617 * x624)) - (x618 * x624)) -
                   ((x608 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x97, count), _super), 0)) * x624));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x628 - (x617 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x98, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x629 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x99, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x630 = (x629 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x631 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x99, count), _super), 0) * inv_0(x630));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x632 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x102, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x633 = (x632 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x634 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x102, count), _super), 0) * inv_0(x633));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x635 = (x630 * x633);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x636 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x99, count), _super), 0) * x633);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x637 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x104, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x638 = (x637 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x639 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x104, count), _super), 0) * inv_0(x638));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x640 = (((x626 + x631) + x634) + x639);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x640);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x641 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x642 = (((x641 * (x635 * x638)) - (x636 * x638)) -
                   ((x630 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x102, count), _super), 0)) * x638));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x642 - (x635 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x104, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x643 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x105, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x644 = (x643 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x645 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x105, count), _super), 0) * inv_0(x644));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x646 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x108, addr), _super), 0));
    ExtVal x647 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x108, cycle), _super), 0));
    ExtVal x648 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x108, dataLow), _super), 0));
    ExtVal x649 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x108, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x650 = (((x646 + x647) + x648) + x649);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x651 = (x650 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x652 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x108, count), _super), 0) * inv_0(x651));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x653 = (x644 * x651);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x654 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x105, count), _super), 0) * x651);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x655 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x109, addr), _super), 0));
    ExtVal x656 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x109, cycle), _super), 0));
    ExtVal x657 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x109, dataLow), _super), 0));
    ExtVal x658 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x109, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x659 = (((x655 + x656) + x657) + x658);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x660 = (x659 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x661 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x109, count), _super), 0) * inv_0(x660));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x662 = (((x640 + x645) + x652) + x661);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x662);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x663 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x664 = (((x663 * (x653 * x660)) - (x654 * x660)) -
                   ((x644 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x108, count), _super), 0)) * x660));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x664 - (x653 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x109, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x665 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x110, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x666 = (x665 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x667 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x110, count), _super), 0) * inv_0(x666));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x668 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x113, addr), _super), 0));
    ExtVal x669 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x113, cycle), _super), 0));
    ExtVal x670 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x113, dataLow), _super), 0));
    ExtVal x671 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x113, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x672 = (((x668 + x669) + x670) + x671);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x673 = (x672 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x674 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x113, count), _super), 0) * inv_0(x673));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x675 = (x666 * x673);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x676 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x110, count), _super), 0) * x673);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x677 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x114, addr), _super), 0));
    ExtVal x678 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x114, cycle), _super), 0));
    ExtVal x679 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x114, dataLow), _super), 0));
    ExtVal x680 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x114, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x681 = (((x677 + x678) + x679) + x680);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x682 = (x681 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x683 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x114, count), _super), 0) * inv_0(x682));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x684 = (((x662 + x667) + x674) + x683);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x684);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x685 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x686 = (((x685 * (x675 * x682)) - (x676 * x682)) -
                   ((x666 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x113, count), _super), 0)) * x682));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x686 - (x675 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x114, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x687 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x115, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x688 = (x687 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x689 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x115, count), _super), 0) * inv_0(x688));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x690 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x118, addr), _super), 0));
    ExtVal x691 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x118, cycle), _super), 0));
    ExtVal x692 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x118, dataLow), _super), 0));
    ExtVal x693 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x118, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x694 = (((x690 + x691) + x692) + x693);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x695 = (x694 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x696 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x118, count), _super), 0) * inv_0(x695));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x697 = (x688 * x695);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x698 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x115, count), _super), 0) * x695);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x699 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x119, addr), _super), 0));
    ExtVal x700 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x119, cycle), _super), 0));
    ExtVal x701 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x119, dataLow), _super), 0));
    ExtVal x702 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x119, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x703 = (((x699 + x700) + x701) + x702);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x704 = (x703 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x705 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x119, count), _super), 0) * inv_0(x704));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x706 = (((x684 + x689) + x696) + x705);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x706);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x707 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x708 = (((x707 * (x697 * x704)) - (x698 * x704)) -
                   ((x688 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x118, count), _super), 0)) * x704));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x708 - (x697 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x119, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x709 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x120, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x710 = (x709 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x711 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x120, count), _super), 0) * inv_0(x710));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x712 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x713 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x714 = ((LOAD_EXT(x9, 0) * LOAD(x713, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x715 = ((x706 + x711) + (LOAD(x712, 0) * inv_0(x714)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x716 = (x710 * x714);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x717 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x120, count), _super), 0) * x714);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x718 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x719 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x720 = ((LOAD_EXT(x9, 0) * LOAD(x719, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x721 = (x715 + (LOAD(x718, 0) * inv_0(x720)));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x721);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x722 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x723 = (((x722 * (x716 * x720)) - (x717 * x720)) - ((x710 * LOAD(x712, 0)) * x720));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x723 - (x716 * LOAD(x718, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x724 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x725 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x726 = ((LOAD_EXT(x9, 0) * LOAD(x725, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x727 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x728 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x729 = ((LOAD_EXT(x9, 0) * LOAD(x728, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x730 = ((x721 + (LOAD(x724, 0) * inv_0(x726))) + (LOAD(x727, 0) * inv_0(x729)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x731 = (x726 * x729);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x732 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x733 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x121, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x734 = ((LOAD_EXT(x9, 0) * LOAD(x733, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x735 = (x730 + (LOAD(x732, 0) * inv_0(x734)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x736 = ((LOAD(x724, 0) * x729) * x734);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x735);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x737 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x738 = (((x737 * (x731 * x734)) - x736) - ((x726 * LOAD(x727, 0)) * x734));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x738 - (x731 * LOAD(x732, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), x735);
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x739 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x739, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 3), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x740 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x126, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x741 = (x740 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x742 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x126, count), _super), 0) * inv_0(x741));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x743 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x742);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x744 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x128, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x745 = (x744 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x746 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x128, count), _super), 0) * inv_0(x745));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x747 = (x741 * x745);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x748 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x126, count), _super), 0) * x745);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x749 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x129, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x750 = (x749 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x751 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x129, count), _super), 0) * inv_0(x750));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x752 = ((x743 + x746) + x751);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x752);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x753 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x754 = (((x753 * (x747 * x750)) - (x748 * x750)) -
                   ((x741 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x128, count), _super), 0)) * x750));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x754 - (x747 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x129, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x755 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x132, addr), _super), 0));
    ExtVal x756 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x132, cycle), _super), 0));
    ExtVal x757 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x132, dataLow), _super), 0));
    ExtVal x758 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x132, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x759 = (((x755 + x756) + x757) + x758);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x760 = (x759 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x761 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x132, count), _super), 0) * inv_0(x760));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x762 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x133, addr), _super), 0));
    ExtVal x763 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x133, cycle), _super), 0));
    ExtVal x764 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x133, dataLow), _super), 0));
    ExtVal x765 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x133, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x766 = (((x762 + x763) + x764) + x765);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x767 = (x766 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x768 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x133, count), _super), 0) * inv_0(x767));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x769 = (x760 * x767);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x770 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x132, count), _super), 0) * x767);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x771 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x134, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x772 = (x771 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x773 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x134, count), _super), 0) * inv_0(x772));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x774 = (((x752 + x761) + x768) + x773);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x774);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x775 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x776 = (((x775 * (x769 * x772)) - (x770 * x772)) -
                   ((x760 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x133, count), _super), 0)) * x772));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x776 - (x769 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x134, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x777 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x137, addr), _super), 0));
    ExtVal x778 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x137, cycle), _super), 0));
    ExtVal x779 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x137, dataLow), _super), 0));
    ExtVal x780 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x137, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x781 = (((x777 + x778) + x779) + x780);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x782 = (x781 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x783 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x137, count), _super), 0) * inv_0(x782));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x784 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x138, addr), _super), 0));
    ExtVal x785 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x138, cycle), _super), 0));
    ExtVal x786 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x138, dataLow), _super), 0));
    ExtVal x787 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x138, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x788 = (((x784 + x785) + x786) + x787);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x789 = (x788 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x790 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x138, count), _super), 0) * inv_0(x789));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x791 = (x782 * x789);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x792 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x137, count), _super), 0) * x789);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x793 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x139, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x794 = (x793 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x795 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x139, count), _super), 0) * inv_0(x794));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x796 = (((x774 + x783) + x790) + x795);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x796);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x797 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x798 = (((x797 * (x791 * x794)) - (x792 * x794)) -
                   ((x782 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x138, count), _super), 0)) * x794));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x798 - (x791 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x139, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x799 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x142, addr), _super), 0));
    ExtVal x800 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x142, cycle), _super), 0));
    ExtVal x801 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x142, dataLow), _super), 0));
    ExtVal x802 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x142, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x803 = (((x799 + x800) + x801) + x802);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x804 = (x803 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x805 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x142, count), _super), 0) * inv_0(x804));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x806 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x143, addr), _super), 0));
    ExtVal x807 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x143, cycle), _super), 0));
    ExtVal x808 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x143, dataLow), _super), 0));
    ExtVal x809 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x143, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x810 = (((x806 + x807) + x808) + x809);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x811 = (x810 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x812 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x143, count), _super), 0) * inv_0(x811));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x813 = (x804 * x811);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x814 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x142, count), _super), 0) * x811);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x815 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x144, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x816 = (x815 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x817 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x144, count), _super), 0) * inv_0(x816));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x818 = (((x796 + x805) + x812) + x817);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x818);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x819 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x820 = (((x819 * (x813 * x816)) - (x814 * x816)) -
                   ((x804 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x143, count), _super), 0)) * x816));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x820 - (x813 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x144, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x821 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x822 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x823 = ((LOAD_EXT(x9, 0) * LOAD(x822, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x824 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x825 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x826 = ((LOAD_EXT(x9, 0) * LOAD(x825, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x827 = ((x818 + (LOAD(x821, 0) * inv_0(x823))) + (LOAD(x824, 0) * inv_0(x826)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x828 = (x823 * x826);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x829 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x830 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x831 = ((LOAD_EXT(x9, 0) * LOAD(x830, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x832 = (x827 + (LOAD(x829, 0) * inv_0(x831)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x833 = ((LOAD(x821, 0) * x826) * x831);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x832);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x834 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x835 = (((x834 * (x828 * x831)) - x833) - ((x823 * LOAD(x824, 0)) * x831));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x835 - (x828 * LOAD(x829, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x836 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x837 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x838 = ((LOAD_EXT(x9, 0) * LOAD(x837, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x839 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x840 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x841 = ((LOAD_EXT(x9, 0) * LOAD(x840, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x842 = ((x832 + (LOAD(x836, 0) * inv_0(x838))) + (LOAD(x839, 0) * inv_0(x841)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x843 = (x838 * x841);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x844 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x845 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x146, 5), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x846 = ((LOAD_EXT(x9, 0) * LOAD(x845, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x847 = (x842 + (LOAD(x844, 0) * inv_0(x846)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x848 = ((LOAD(x836, 0) * x841) * x846);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x847);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x849 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x850 = (((x849 * (x843 * x846)) - x848) - ((x838 * LOAD(x839, 0)) * x846));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x850 - (x843 * LOAD(x844, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x851 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x852 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x853 = ((LOAD_EXT(x123, 0) * LOAD(x852, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x854 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x855 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x856 = ((LOAD_EXT(x123, 0) * LOAD(x855, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x857 = ((x847 + (LOAD(x851, 0) * inv_0(x853))) + (LOAD(x854, 0) * inv_0(x856)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x858 = (x853 * x856);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x859 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x860 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x861 = ((LOAD_EXT(x123, 0) * LOAD(x860, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x862 = (x857 + (LOAD(x859, 0) * inv_0(x861)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x863 = ((LOAD(x851, 0) * x856) * x861);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x862);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x864 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x865 = (((x864 * (x858 * x861)) - x863) - ((x853 * LOAD(x854, 0)) * x861));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x865 - (x858 * LOAD(x859, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x866 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x867 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x868 = ((LOAD_EXT(x123, 0) * LOAD(x867, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x869 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x870 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x871 = ((LOAD_EXT(x123, 0) * LOAD(x870, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x872 = ((x862 + (LOAD(x866, 0) * inv_0(x868))) + (LOAD(x869, 0) * inv_0(x871)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x873 = (x868 * x871);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x874 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x875 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 5), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x876 = ((LOAD_EXT(x123, 0) * LOAD(x875, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x877 = (x872 + (LOAD(x874, 0) * inv_0(x876)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x878 = ((LOAD(x866, 0) * x871) * x876);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x877);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x879 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x880 = (((x879 * (x873 * x876)) - x878) - ((x868 * LOAD(x869, 0)) * x876));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x880 - (x873 * LOAD(x874, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x881 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x882 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 6), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x883 = ((LOAD_EXT(x123, 0) * LOAD(x882, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x884 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x885 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 7), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x886 = ((LOAD_EXT(x123, 0) * LOAD(x885, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x887 = ((x877 + (LOAD(x881, 0) * inv_0(x883))) + (LOAD(x884, 0) * inv_0(x886)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x888 = (x883 * x886);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x889 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x890 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 8), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x891 = ((LOAD_EXT(x123, 0) * LOAD(x890, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x892 = (x887 + (LOAD(x889, 0) * inv_0(x891)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x893 = ((LOAD(x881, 0) * x886) * x891);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 8), x892);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x894 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x895 = (((x894 * (x888 * x891)) - x893) - ((x883 * LOAD(x884, 0)) * x891));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x895 - (x888 * LOAD(x889, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x896 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 9), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x897 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 9), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x898 = ((LOAD_EXT(x123, 0) * LOAD(x897, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x899 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 10), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x900 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 10), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x901 = ((LOAD_EXT(x123, 0) * LOAD(x900, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x902 = ((x892 + (LOAD(x896, 0) * inv_0(x898))) + (LOAD(x899, 0) * inv_0(x901)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x903 = (x898 * x901);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x904 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 11), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x905 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 11), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x906 = ((LOAD_EXT(x123, 0) * LOAD(x905, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x907 = (x902 + (LOAD(x904, 0) * inv_0(x906)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x908 = ((LOAD(x896, 0) * x901) * x906);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 9), x907);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x909 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x910 = (((x909 * (x903 * x906)) - x908) - ((x898 * LOAD(x899, 0)) * x906));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x910 - (x903 * LOAD(x904, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x911 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 12), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x912 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x147, 12), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x913 = ((LOAD_EXT(x123, 0) * LOAD(x912, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x914 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x150, addr), _super), 0));
    ExtVal x915 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x150, cycle), _super), 0));
    ExtVal x916 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x150, dataLow), _super), 0));
    ExtVal x917 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x150, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x918 = (((x914 + x915) + x916) + x917);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x919 = (x918 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x920 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x150, count), _super), 0) * inv_0(x919));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x921 = (x913 * x919);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x922 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x151, addr), _super), 0));
    ExtVal x923 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x151, cycle), _super), 0));
    ExtVal x924 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x151, dataLow), _super), 0));
    ExtVal x925 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x151, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x926 = (((x922 + x923) + x924) + x925);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x927 = (x926 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x928 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x151, count), _super), 0) * inv_0(x927));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x929 = (((x907 + (LOAD(x911, 0) * inv_0(x913))) + x920) + x928);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x930 = ((LOAD(x911, 0) * x919) * x927);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 10), x929);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x931 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x932 = (((x931 * (x921 * x927)) - x930) -
                   ((x913 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x150, count), _super), 0)) * x927));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x932 - (x921 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x151, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x933 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x152, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x934 = (x933 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x935 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x152, count), _super), 0) * inv_0(x934));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x936 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x154, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x937 = (x936 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x938 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x154, count), _super), 0) * inv_0(x937));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x939 = (x934 * x937);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x940 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x152, count), _super), 0) * x937);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x941 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x155, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x942 = (x941 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x943 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x155, count), _super), 0) * inv_0(x942));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x944 = (((x929 + x935) + x938) + x943);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 11), x944);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x945 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x946 = (((x945 * (x939 * x942)) - (x940 * x942)) -
                   ((x934 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x154, count), _super), 0)) * x942));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x946 - (x939 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x155, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), x944);
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x947 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x947, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 4), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x948 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x159, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x949 = (x948 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x950 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x159, count), _super), 0) * inv_0(x949));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x951 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x950);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x952 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x161, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x953 = (x952 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x954 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x161, count), _super), 0) * inv_0(x953));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x955 = (x949 * x953);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x956 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x159, count), _super), 0) * x953);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x957 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x162, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x958 = (x957 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x959 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x162, count), _super), 0) * inv_0(x958));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x960 = ((x951 + x954) + x959);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x960);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x961 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x962 = (((x961 * (x955 * x958)) - (x956 * x958)) -
                   ((x949 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x161, count), _super), 0)) * x958));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x962 - (x955 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x162, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x963 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x165, addr), _super), 0));
    ExtVal x964 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x165, cycle), _super), 0));
    ExtVal x965 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x165, dataLow), _super), 0));
    ExtVal x966 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x165, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x967 = (((x963 + x964) + x965) + x966);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x968 = (x967 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x969 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x165, count), _super), 0) * inv_0(x968));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x970 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x166, addr), _super), 0));
    ExtVal x971 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x166, cycle), _super), 0));
    ExtVal x972 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x166, dataLow), _super), 0));
    ExtVal x973 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x166, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x974 = (((x970 + x971) + x972) + x973);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x975 = (x974 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x976 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x166, count), _super), 0) * inv_0(x975));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x977 = (x968 * x975);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x978 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x165, count), _super), 0) * x975);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x979 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x167, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x980 = (x979 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x981 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x167, count), _super), 0) * inv_0(x980));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x982 = (((x960 + x969) + x976) + x981);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x982);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x983 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x984 = (((x983 * (x977 * x980)) - (x978 * x980)) -
                   ((x968 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x166, count), _super), 0)) * x980));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x984 - (x977 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x167, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x985 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x170, addr), _super), 0));
    ExtVal x986 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x170, cycle), _super), 0));
    ExtVal x987 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x170, dataLow), _super), 0));
    ExtVal x988 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x170, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x989 = (((x985 + x986) + x987) + x988);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x990 = (x989 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x991 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x170, count), _super), 0) * inv_0(x990));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x992 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x171, addr), _super), 0));
    ExtVal x993 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x171, cycle), _super), 0));
    ExtVal x994 = (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x171, dataLow), _super), 0));
    ExtVal x995 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x171, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x996 = (((x992 + x993) + x994) + x995);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x997 = (x996 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x998 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x171, count), _super), 0) * inv_0(x997));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x999 = (x990 * x997);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1000 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x170, count), _super), 0) * x997);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1001 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x172, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1002 = (x1001 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1003 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x172, count), _super), 0) * inv_0(x1002));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1004 = (((x982 + x991) + x998) + x1003);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x1004);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1005 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1006 = (((x1005 * (x999 * x1002)) - (x1000 * x1002)) -
                    ((x990 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x171, count), _super), 0)) * x1002));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1006 - (x999 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x172, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1007 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x175, addr), _super), 0));
    ExtVal x1008 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x175, cycle), _super), 0));
    ExtVal x1009 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x175, dataLow), _super), 0));
    ExtVal x1010 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x175, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1011 = (((x1007 + x1008) + x1009) + x1010);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1012 = (x1011 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1013 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x175, count), _super), 0) * inv_0(x1012));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1014 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x176, addr), _super), 0));
    ExtVal x1015 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x176, cycle), _super), 0));
    ExtVal x1016 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x176, dataLow), _super), 0));
    ExtVal x1017 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x176, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1018 = (((x1014 + x1015) + x1016) + x1017);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1019 = (x1018 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1020 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x176, count), _super), 0) * inv_0(x1019));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1021 = (x1012 * x1019);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1022 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x175, count), _super), 0) * x1019);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1023 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x177, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1024 = (x1023 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1025 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x177, count), _super), 0) * inv_0(x1024));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1026 = (((x1004 + x1013) + x1020) + x1025);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x1026);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1027 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1028 = (((x1027 * (x1021 * x1024)) - (x1022 * x1024)) -
                    ((x1012 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x176, count), _super), 0)) * x1024));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1028 - (x1021 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x177, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1029 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1030 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1031 = ((LOAD_EXT(x9, 0) * LOAD(x1030, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1032 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1033 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1034 = ((LOAD_EXT(x9, 0) * LOAD(x1033, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1035 = ((x1026 + (LOAD(x1029, 0) * inv_0(x1031))) + (LOAD(x1032, 0) * inv_0(x1034)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1036 = (x1031 * x1034);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1037 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1038 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1039 = ((LOAD_EXT(x9, 0) * LOAD(x1038, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1040 = (x1035 + (LOAD(x1037, 0) * inv_0(x1039)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1041 = ((LOAD(x1029, 0) * x1034) * x1039);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x1040);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1042 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1043 = (((x1042 * (x1036 * x1039)) - x1041) - ((x1031 * LOAD(x1032, 0)) * x1039));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1043 - (x1036 * LOAD(x1037, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1044 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1045 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1046 = ((LOAD_EXT(x9, 0) * LOAD(x1045, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1047 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1048 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1049 = ((LOAD_EXT(x9, 0) * LOAD(x1048, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1050 = ((x1040 + (LOAD(x1044, 0) * inv_0(x1046))) + (LOAD(x1047, 0) * inv_0(x1049)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1051 = (x1046 * x1049);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1052 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1053 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 5), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1054 = ((LOAD_EXT(x9, 0) * LOAD(x1053, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1055 = (x1050 + (LOAD(x1052, 0) * inv_0(x1054)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1056 = ((LOAD(x1044, 0) * x1049) * x1054);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x1055);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1057 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1058 = (((x1057 * (x1051 * x1054)) - x1056) - ((x1046 * LOAD(x1047, 0)) * x1054));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1058 - (x1051 * LOAD(x1052, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1059 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1060 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 6), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1061 = ((LOAD_EXT(x9, 0) * LOAD(x1060, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1062 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1063 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 7), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1064 = ((LOAD_EXT(x9, 0) * LOAD(x1063, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1065 = ((x1055 + (LOAD(x1059, 0) * inv_0(x1061))) + (LOAD(x1062, 0) * inv_0(x1064)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1066 = (x1061 * x1064);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1067 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1068 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x179, 8), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1069 = ((LOAD_EXT(x9, 0) * LOAD(x1068, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1070 = (x1065 + (LOAD(x1067, 0) * inv_0(x1069)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1071 = ((LOAD(x1059, 0) * x1064) * x1069);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x1070);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1072 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1073 = (((x1072 * (x1066 * x1069)) - x1071) - ((x1061 * LOAD(x1062, 0)) * x1069));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1073 - (x1066 * LOAD(x1067, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1074 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1075 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1076 = ((LOAD_EXT(x123, 0) * LOAD(x1075, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1077 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1078 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1079 = ((LOAD_EXT(x123, 0) * LOAD(x1078, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1080 = ((x1070 + (LOAD(x1074, 0) * inv_0(x1076))) + (LOAD(x1077, 0) * inv_0(x1079)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1081 = (x1076 * x1079);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1082 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1083 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1084 = ((LOAD_EXT(x123, 0) * LOAD(x1083, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1085 = (x1080 + (LOAD(x1082, 0) * inv_0(x1084)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1086 = ((LOAD(x1074, 0) * x1079) * x1084);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x1085);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1087 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1088 = (((x1087 * (x1081 * x1084)) - x1086) - ((x1076 * LOAD(x1077, 0)) * x1084));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1088 - (x1081 * LOAD(x1082, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1089 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1090 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1091 = ((LOAD_EXT(x123, 0) * LOAD(x1090, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1092 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1093 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1094 = ((LOAD_EXT(x123, 0) * LOAD(x1093, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1095 = ((x1085 + (LOAD(x1089, 0) * inv_0(x1091))) + (LOAD(x1092, 0) * inv_0(x1094)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1096 = (x1091 * x1094);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1097 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1098 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 5), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1099 = ((LOAD_EXT(x123, 0) * LOAD(x1098, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1100 = (x1095 + (LOAD(x1097, 0) * inv_0(x1099)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1101 = ((LOAD(x1089, 0) * x1094) * x1099);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 8), x1100);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1102 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1103 = (((x1102 * (x1096 * x1099)) - x1101) - ((x1091 * LOAD(x1092, 0)) * x1099));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1103 - (x1096 * LOAD(x1097, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1104 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1105 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 6), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1106 = ((LOAD_EXT(x123, 0) * LOAD(x1105, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1107 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1108 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 7), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1109 = ((LOAD_EXT(x123, 0) * LOAD(x1108, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1110 = ((x1100 + (LOAD(x1104, 0) * inv_0(x1106))) + (LOAD(x1107, 0) * inv_0(x1109)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1111 = (x1106 * x1109);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1112 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1113 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 8), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1114 = ((LOAD_EXT(x123, 0) * LOAD(x1113, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1115 = (x1110 + (LOAD(x1112, 0) * inv_0(x1114)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1116 = ((LOAD(x1104, 0) * x1109) * x1114);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 9), x1115);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1117 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1118 = (((x1117 * (x1111 * x1114)) - x1116) - ((x1106 * LOAD(x1107, 0)) * x1114));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1118 - (x1111 * LOAD(x1112, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1119 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 9), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1120 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 9), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1121 = ((LOAD_EXT(x123, 0) * LOAD(x1120, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1122 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 10), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1123 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 10), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1124 = ((LOAD_EXT(x123, 0) * LOAD(x1123, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1125 = ((x1115 + (LOAD(x1119, 0) * inv_0(x1121))) + (LOAD(x1122, 0) * inv_0(x1124)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1126 = (x1121 * x1124);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1127 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 11), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1128 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 11), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1129 = ((LOAD_EXT(x123, 0) * LOAD(x1128, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1130 = (x1125 + (LOAD(x1127, 0) * inv_0(x1129)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1131 = ((LOAD(x1119, 0) * x1124) * x1129);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 10), x1130);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1132 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1133 = (((x1132 * (x1126 * x1129)) - x1131) - ((x1121 * LOAD(x1122, 0)) * x1129));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1133 - (x1126 * LOAD(x1127, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1134 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 12), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1135 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x180, 12), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1136 = ((LOAD_EXT(x123, 0) * LOAD(x1135, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1137 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x183, addr), _super), 0));
    ExtVal x1138 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x183, cycle), _super), 0));
    ExtVal x1139 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x183, dataLow), _super), 0));
    ExtVal x1140 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x183, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1141 = (((x1137 + x1138) + x1139) + x1140);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1142 = (x1141 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1143 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x183, count), _super), 0) * inv_0(x1142));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1144 = (x1136 * x1142);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1145 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x184, addr), _super), 0));
    ExtVal x1146 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x184, cycle), _super), 0));
    ExtVal x1147 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x184, dataLow), _super), 0));
    ExtVal x1148 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x184, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1149 = (((x1145 + x1146) + x1147) + x1148);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1150 = (x1149 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1151 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x184, count), _super), 0) * inv_0(x1150));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1152 = (((x1130 + (LOAD(x1134, 0) * inv_0(x1136))) + x1143) + x1151);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1153 = ((LOAD(x1134, 0) * x1142) * x1150);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 11), x1152);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1154 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1155 = (((x1154 * (x1144 * x1150)) - x1153) -
                    ((x1136 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x183, count), _super), 0)) * x1150));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1155 - (x1144 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x184, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1156 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x185, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1157 = (x1156 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1158 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x185, count), _super), 0) * inv_0(x1157));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1159 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x187, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1160 = (x1159 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1161 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x187, count), _super), 0) * inv_0(x1160));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1162 = (x1157 * x1160);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1163 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x185, count), _super), 0) * x1160);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1164 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x188, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1165 = (x1164 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1166 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x188, count), _super), 0) * inv_0(x1165));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1167 = (((x1152 + x1158) + x1161) + x1166);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 12), x1167);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1168 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 12), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1169 = (((x1168 * (x1162 * x1165)) - (x1163 * x1165)) -
                    ((x1157 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x187, count), _super), 0)) * x1165));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1169 - (x1162 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x188, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), x1167);
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x1170 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 12), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x1170, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 5), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1171 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x192, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1172 = (x1171 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1173 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x192, count), _super), 0) * inv_0(x1172));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1174 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x1173);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1175 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x194, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1176 = (x1175 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1177 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x194, count), _super), 0) * inv_0(x1176));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1178 = (x1172 * x1176);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1179 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x192, count), _super), 0) * x1176);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1180 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x195, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1181 = (x1180 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1182 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x195, count), _super), 0) * inv_0(x1181));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1183 = ((x1174 + x1177) + x1182);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x1183);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1184 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1185 = (((x1184 * (x1178 * x1181)) - (x1179 * x1181)) -
                    ((x1172 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x194, count), _super), 0)) * x1181));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1185 - (x1178 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x195, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1186 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x198, addr), _super), 0));
    ExtVal x1187 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x198, cycle), _super), 0));
    ExtVal x1188 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x198, dataLow), _super), 0));
    ExtVal x1189 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x198, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1190 = (((x1186 + x1187) + x1188) + x1189);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1191 = (x1190 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1192 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x198, count), _super), 0) * inv_0(x1191));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1193 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x199, addr), _super), 0));
    ExtVal x1194 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x199, cycle), _super), 0));
    ExtVal x1195 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x199, dataLow), _super), 0));
    ExtVal x1196 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x199, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1197 = (((x1193 + x1194) + x1195) + x1196);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1198 = (x1197 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1199 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x199, count), _super), 0) * inv_0(x1198));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1200 = (x1191 * x1198);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1201 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x198, count), _super), 0) * x1198);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1202 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x200, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1203 = (x1202 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1204 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x200, count), _super), 0) * inv_0(x1203));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1205 = (((x1183 + x1192) + x1199) + x1204);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x1205);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1206 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1207 = (((x1206 * (x1200 * x1203)) - (x1201 * x1203)) -
                    ((x1191 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x199, count), _super), 0)) * x1203));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1207 - (x1200 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x200, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1208 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x203, addr), _super), 0));
    ExtVal x1209 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x203, cycle), _super), 0));
    ExtVal x1210 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x203, dataLow), _super), 0));
    ExtVal x1211 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x203, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1212 = (((x1208 + x1209) + x1210) + x1211);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1213 = (x1212 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1214 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x203, count), _super), 0) * inv_0(x1213));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1215 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x204, addr), _super), 0));
    ExtVal x1216 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x204, cycle), _super), 0));
    ExtVal x1217 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x204, dataLow), _super), 0));
    ExtVal x1218 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x204, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1219 = (((x1215 + x1216) + x1217) + x1218);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1220 = (x1219 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1221 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x204, count), _super), 0) * inv_0(x1220));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1222 = (x1213 * x1220);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1223 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x203, count), _super), 0) * x1220);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1224 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x205, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1225 = (x1224 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1226 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x205, count), _super), 0) * inv_0(x1225));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1227 = (((x1205 + x1214) + x1221) + x1226);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x1227);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1228 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1229 = (((x1228 * (x1222 * x1225)) - (x1223 * x1225)) -
                    ((x1213 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x204, count), _super), 0)) * x1225));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1229 - (x1222 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x205, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1230 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x207, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1231 = (x1230 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1232 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x207, count), _super), 0) * inv_0(x1231));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1233 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x208, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1234 = (x1233 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1235 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x208, count), _super), 0) * inv_0(x1234));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1236 = (x1231 * x1234);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1237 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x207, count), _super), 0) * x1234);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1238 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x210, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1239 = (x1238 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1240 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x210, count), _super), 0) * inv_0(x1239));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1241 = (((x1227 + x1232) + x1235) + x1240);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x1241);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1242 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1243 = (((x1242 * (x1236 * x1239)) - (x1237 * x1239)) -
                    ((x1231 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x208, count), _super), 0)) * x1239));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1243 - (x1236 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x210, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1244 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x211, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1245 = (x1244 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1246 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x211, count), _super), 0) * inv_0(x1245));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1247 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x214, addr), _super), 0));
    ExtVal x1248 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x214, cycle), _super), 0));
    ExtVal x1249 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x214, dataLow), _super), 0));
    ExtVal x1250 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x214, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1251 = (((x1247 + x1248) + x1249) + x1250);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1252 = (x1251 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1253 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x214, count), _super), 0) * inv_0(x1252));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1254 = (x1245 * x1252);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1255 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x211, count), _super), 0) * x1252);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1256 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x215, addr), _super), 0));
    ExtVal x1257 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x215, cycle), _super), 0));
    ExtVal x1258 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x215, dataLow), _super), 0));
    ExtVal x1259 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x215, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1260 = (((x1256 + x1257) + x1258) + x1259);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1261 = (x1260 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1262 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x215, count), _super), 0) * inv_0(x1261));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1263 = (((x1241 + x1246) + x1253) + x1262);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x1263);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1264 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1265 = (((x1264 * (x1254 * x1261)) - (x1255 * x1261)) -
                    ((x1245 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x214, count), _super), 0)) * x1261));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1265 - (x1254 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x215, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1266 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x216, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1267 = (x1266 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1268 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x216, count), _super), 0) * inv_0(x1267));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1269 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x217, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1270 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x217, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1271 = ((LOAD_EXT(x123, 0) * LOAD(x1270, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1272 = ((x1263 + x1268) + (LOAD(x1269, 0) * inv_0(x1271)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1273 = (x1267 * x1271);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1274 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x216, count), _super), 0) * x1271);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1275 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x217, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1276 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x217, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1277 = ((LOAD_EXT(x123, 0) * LOAD(x1276, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1278 = (x1272 + (LOAD(x1275, 0) * inv_0(x1277)));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x1278);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1279 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1280 =
        (((x1279 * (x1273 * x1277)) - (x1274 * x1277)) - ((x1267 * LOAD(x1269, 0)) * x1277));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1280 - (x1273 * LOAD(x1275, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1281 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x217, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1282 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x217, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1283 = ((LOAD_EXT(x123, 0) * LOAD(x1282, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1284 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x220, addr), _super), 0));
    ExtVal x1285 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x220, cycle), _super), 0));
    ExtVal x1286 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x220, dataLow), _super), 0));
    ExtVal x1287 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x220, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1288 = (((x1284 + x1285) + x1286) + x1287);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1289 = (x1288 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1290 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x220, count), _super), 0) * inv_0(x1289));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1291 = (x1283 * x1289);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1292 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x221, addr), _super), 0));
    ExtVal x1293 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x221, cycle), _super), 0));
    ExtVal x1294 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x221, dataLow), _super), 0));
    ExtVal x1295 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x221, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1296 = (((x1292 + x1293) + x1294) + x1295);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1297 = (x1296 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1298 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x221, count), _super), 0) * inv_0(x1297));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1299 = (((x1278 + (LOAD(x1281, 0) * inv_0(x1283))) + x1290) + x1298);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1300 = ((LOAD(x1281, 0) * x1289) * x1297);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x1299);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1301 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1302 = (((x1301 * (x1291 * x1297)) - x1300) -
                    ((x1283 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x220, count), _super), 0)) * x1297));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1302 - (x1291 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x221, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1303 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x222, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1304 = (x1303 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1305 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x222, count), _super), 0) * inv_0(x1304));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1306 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x224, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1307 = (x1306 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1308 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x224, count), _super), 0) * inv_0(x1307));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1309 = (x1304 * x1307);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1310 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x222, count), _super), 0) * x1307);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1311 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x225, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1312 = (x1311 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1313 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x225, count), _super), 0) * inv_0(x1312));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1314 = (((x1299 + x1305) + x1308) + x1313);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x1314);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1315 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1316 = (((x1315 * (x1309 * x1312)) - (x1310 * x1312)) -
                    ((x1304 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x224, count), _super), 0)) * x1312));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1316 - (x1309 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x225, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), x1314);
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x1317 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x1317, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 6), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1318 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x229, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1319 = (x1318 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1320 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x229, count), _super), 0) * inv_0(x1319));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1321 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x1320);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1322 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x231, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1323 = (x1322 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1324 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x231, count), _super), 0) * inv_0(x1323));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1325 = (x1319 * x1323);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1326 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x229, count), _super), 0) * x1323);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1327 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x232, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1328 = (x1327 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1329 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x232, count), _super), 0) * inv_0(x1328));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1330 = ((x1321 + x1324) + x1329);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x1330);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1331 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1332 = (((x1331 * (x1325 * x1328)) - (x1326 * x1328)) -
                    ((x1319 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x231, count), _super), 0)) * x1328));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1332 - (x1325 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x232, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1333 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x235, addr), _super), 0));
    ExtVal x1334 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x235, cycle), _super), 0));
    ExtVal x1335 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x235, dataLow), _super), 0));
    ExtVal x1336 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x235, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1337 = (((x1333 + x1334) + x1335) + x1336);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1338 = (x1337 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1339 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x235, count), _super), 0) * inv_0(x1338));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1340 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x236, addr), _super), 0));
    ExtVal x1341 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x236, cycle), _super), 0));
    ExtVal x1342 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x236, dataLow), _super), 0));
    ExtVal x1343 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x236, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1344 = (((x1340 + x1341) + x1342) + x1343);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1345 = (x1344 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1346 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x236, count), _super), 0) * inv_0(x1345));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1347 = (x1338 * x1345);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1348 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x235, count), _super), 0) * x1345);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1349 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x237, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1350 = (x1349 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1351 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x237, count), _super), 0) * inv_0(x1350));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1352 = (((x1330 + x1339) + x1346) + x1351);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x1352);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1353 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1354 = (((x1353 * (x1347 * x1350)) - (x1348 * x1350)) -
                    ((x1338 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x236, count), _super), 0)) * x1350));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1354 - (x1347 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x237, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1355 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x240, addr), _super), 0));
    ExtVal x1356 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x240, cycle), _super), 0));
    ExtVal x1357 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x240, dataLow), _super), 0));
    ExtVal x1358 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x240, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1359 = (((x1355 + x1356) + x1357) + x1358);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1360 = (x1359 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1361 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x240, count), _super), 0) * inv_0(x1360));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1362 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x241, addr), _super), 0));
    ExtVal x1363 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x241, cycle), _super), 0));
    ExtVal x1364 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x241, dataLow), _super), 0));
    ExtVal x1365 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x241, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1366 = (((x1362 + x1363) + x1364) + x1365);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1367 = (x1366 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1368 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x241, count), _super), 0) * inv_0(x1367));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1369 = (x1360 * x1367);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1370 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x240, count), _super), 0) * x1367);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1371 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x242, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1372 = (x1371 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1373 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x242, count), _super), 0) * inv_0(x1372));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1374 = (((x1352 + x1361) + x1368) + x1373);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x1374);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1375 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1376 = (((x1375 * (x1369 * x1372)) - (x1370 * x1372)) -
                    ((x1360 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x241, count), _super), 0)) * x1372));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1376 - (x1369 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x242, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1377 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x245, addr), _super), 0));
    ExtVal x1378 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x245, cycle), _super), 0));
    ExtVal x1379 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x245, dataLow), _super), 0));
    ExtVal x1380 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x245, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1381 = (((x1377 + x1378) + x1379) + x1380);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1382 = (x1381 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1383 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x245, count), _super), 0) * inv_0(x1382));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1384 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x246, addr), _super), 0));
    ExtVal x1385 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x246, cycle), _super), 0));
    ExtVal x1386 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x246, dataLow), _super), 0));
    ExtVal x1387 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x246, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1388 = (((x1384 + x1385) + x1386) + x1387);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1389 = (x1388 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1390 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x246, count), _super), 0) * inv_0(x1389));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1391 = (x1382 * x1389);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1392 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x245, count), _super), 0) * x1389);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1393 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x247, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1394 = (x1393 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1395 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x247, count), _super), 0) * inv_0(x1394));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1396 = (((x1374 + x1383) + x1390) + x1395);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x1396);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1397 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1398 = (((x1397 * (x1391 * x1394)) - (x1392 * x1394)) -
                    ((x1382 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x246, count), _super), 0)) * x1394));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1398 - (x1391 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x247, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1399 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x249, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1400 = (x1399 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1401 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x249, count), _super), 0) * inv_0(x1400));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1402 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x250, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1403 = (x1402 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1404 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x250, count), _super), 0) * inv_0(x1403));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1405 = (x1400 * x1403);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1406 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x249, count), _super), 0) * x1403);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1407 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x252, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1408 = (x1407 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1409 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x252, count), _super), 0) * inv_0(x1408));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1410 = (((x1396 + x1401) + x1404) + x1409);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x1410);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1411 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1412 = (((x1411 * (x1405 * x1408)) - (x1406 * x1408)) -
                    ((x1400 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x250, count), _super), 0)) * x1408));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1412 - (x1405 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x252, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1413 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x253, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1414 = (x1413 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1415 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x253, count), _super), 0) * inv_0(x1414));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1416 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x256, addr), _super), 0));
    ExtVal x1417 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x256, cycle), _super), 0));
    ExtVal x1418 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x256, dataLow), _super), 0));
    ExtVal x1419 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x256, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1420 = (((x1416 + x1417) + x1418) + x1419);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1421 = (x1420 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1422 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x256, count), _super), 0) * inv_0(x1421));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1423 = (x1414 * x1421);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1424 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x253, count), _super), 0) * x1421);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1425 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x257, addr), _super), 0));
    ExtVal x1426 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x257, cycle), _super), 0));
    ExtVal x1427 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x257, dataLow), _super), 0));
    ExtVal x1428 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x257, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1429 = (((x1425 + x1426) + x1427) + x1428);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1430 = (x1429 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1431 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x257, count), _super), 0) * inv_0(x1430));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1432 = (((x1410 + x1415) + x1422) + x1431);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x1432);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1433 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1434 = (((x1433 * (x1423 * x1430)) - (x1424 * x1430)) -
                    ((x1414 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x256, count), _super), 0)) * x1430));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1434 - (x1423 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x257, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1435 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x258, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1436 = (x1435 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1437 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x258, count), _super), 0) * inv_0(x1436));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1438 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1439 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1440 = ((LOAD_EXT(x123, 0) * LOAD(x1439, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1441 = ((x1432 + x1437) + (LOAD(x1438, 0) * inv_0(x1440)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1442 = (x1436 * x1440);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1443 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x258, count), _super), 0) * x1440);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1444 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1445 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1446 = ((LOAD_EXT(x123, 0) * LOAD(x1445, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1447 = (x1441 + (LOAD(x1444, 0) * inv_0(x1446)));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x1447);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1448 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1449 =
        (((x1448 * (x1442 * x1446)) - (x1443 * x1446)) - ((x1436 * LOAD(x1438, 0)) * x1446));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1449 - (x1442 * LOAD(x1444, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1450 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1451 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1452 = ((LOAD_EXT(x123, 0) * LOAD(x1451, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1453 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1454 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x259, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1455 = ((LOAD_EXT(x123, 0) * LOAD(x1454, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1456 = ((x1447 + (LOAD(x1450, 0) * inv_0(x1452))) + (LOAD(x1453, 0) * inv_0(x1455)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1457 = (x1452 * x1455);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1458 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x262, addr), _super), 0));
    ExtVal x1459 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x262, cycle), _super), 0));
    ExtVal x1460 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x262, dataLow), _super), 0));
    ExtVal x1461 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x262, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1462 = (((x1458 + x1459) + x1460) + x1461);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1463 = (x1462 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1464 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x262, count), _super), 0) * inv_0(x1463));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1465 = (x1456 + x1464);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1466 = ((LOAD(x1450, 0) * x1455) * x1463);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x1465);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1467 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1468 = (((x1467 * (x1457 * x1463)) - x1466) - ((x1452 * LOAD(x1453, 0)) * x1463));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1468 - (x1457 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x262, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1469 = (LOAD_EXT(x11, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x263, addr), _super), 0));
    ExtVal x1470 = (LOAD_EXT(x12, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x263, cycle), _super), 0));
    ExtVal x1471 =
        (LOAD_EXT(x13, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x263, dataLow), _super), 0));
    ExtVal x1472 =
        (LOAD_EXT(x14, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x263, dataHigh), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1473 = (((x1469 + x1470) + x1471) + x1472);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1474 = (x1473 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1475 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x263, count), _super), 0) * inv_0(x1474));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1476 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x264, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1477 = (x1476 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1478 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x264, count), _super), 0) * inv_0(x1477));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1479 = (x1474 * x1477);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1480 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x263, count), _super), 0) * x1477);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1481 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x266, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1482 = (x1481 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1483 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x266, count), _super), 0) * inv_0(x1482));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1484 = (((x1465 + x1475) + x1478) + x1483);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 8), x1484);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1485 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1486 = (((x1485 * (x1479 * x1482)) - (x1480 * x1482)) -
                    ((x1474 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x264, count), _super), 0)) * x1482));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1486 - (x1479 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x266, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1487 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x267, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1488 = (x1487 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1489 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x267, count), _super), 0) * inv_0(x1488));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 9), (x1484 + x1489));
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1490 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1491 = ((x1490 * x1488) - LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x267, count), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ(x1491, "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x1492 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x1492, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 7), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1493 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x269, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1494 = (x1493 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1495 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x269, count), _super), 0) * inv_0(x1494));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1496 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x1495);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1497 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1498 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 0), addr), _super);
    BoundLayout<Reg> x1499 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 0), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1500 = ((LOAD_EXT(x11, 0) * LOAD(x1498, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1499, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1501 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 0), dataLow), _super);
    BoundLayout<Reg> x1502 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 0), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1503 =
        ((x1500 + (LOAD_EXT(x13, 0) * LOAD(x1501, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1502, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1504 = (x1503 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1505 = (x1494 * x1504);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1506 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x269, count), _super), 0) * x1504);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1507 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1508 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 1), addr), _super);
    BoundLayout<Reg> x1509 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 1), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1510 = ((LOAD_EXT(x11, 0) * LOAD(x1508, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1509, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1511 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 1), dataLow), _super);
    BoundLayout<Reg> x1512 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 1), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1513 =
        ((x1510 + (LOAD_EXT(x13, 0) * LOAD(x1511, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1512, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1514 = (x1513 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1515 = ((x1496 + (LOAD(x1497, 0) * inv_0(x1504))) + (LOAD(x1507, 0) * inv_0(x1514)));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x1515);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1516 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1517 =
        (((x1516 * (x1505 * x1514)) - (x1506 * x1514)) - ((x1494 * LOAD(x1497, 0)) * x1514));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1517 - (x1505 * LOAD(x1507, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1518 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1519 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 2), addr), _super);
    BoundLayout<Reg> x1520 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 2), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1521 = ((LOAD_EXT(x11, 0) * LOAD(x1519, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1520, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1522 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 2), dataLow), _super);
    BoundLayout<Reg> x1523 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 2), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1524 =
        ((x1521 + (LOAD_EXT(x13, 0) * LOAD(x1522, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1523, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1525 = (x1524 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1526 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1527 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 3), addr), _super);
    BoundLayout<Reg> x1528 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 3), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1529 = ((LOAD_EXT(x11, 0) * LOAD(x1527, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1528, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1530 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 3), dataLow), _super);
    BoundLayout<Reg> x1531 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 3), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1532 =
        ((x1529 + (LOAD_EXT(x13, 0) * LOAD(x1530, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1531, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1533 = (x1532 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1534 = ((x1515 + (LOAD(x1518, 0) * inv_0(x1525))) + (LOAD(x1526, 0) * inv_0(x1533)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1535 = (x1525 * x1533);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1536 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1537 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 4), addr), _super);
    BoundLayout<Reg> x1538 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 4), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1539 = ((LOAD_EXT(x11, 0) * LOAD(x1537, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1538, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1540 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 4), dataLow), _super);
    BoundLayout<Reg> x1541 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 4), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1542 =
        ((x1539 + (LOAD_EXT(x13, 0) * LOAD(x1540, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1541, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1543 = (x1542 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1544 = (x1534 + (LOAD(x1536, 0) * inv_0(x1543)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1545 = ((LOAD(x1518, 0) * x1533) * x1543);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x1544);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1546 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1547 = (((x1546 * (x1535 * x1543)) - x1545) - ((x1525 * LOAD(x1526, 0)) * x1543));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1547 - (x1535 * LOAD(x1536, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1548 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1549 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 5), addr), _super);
    BoundLayout<Reg> x1550 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 5), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1551 = ((LOAD_EXT(x11, 0) * LOAD(x1549, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1550, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1552 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 5), dataLow), _super);
    BoundLayout<Reg> x1553 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 5), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1554 =
        ((x1551 + (LOAD_EXT(x13, 0) * LOAD(x1552, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1553, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1555 = (x1554 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1556 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1557 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 6), addr), _super);
    BoundLayout<Reg> x1558 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 6), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1559 = ((LOAD_EXT(x11, 0) * LOAD(x1557, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1558, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1560 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 6), dataLow), _super);
    BoundLayout<Reg> x1561 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 6), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1562 =
        ((x1559 + (LOAD_EXT(x13, 0) * LOAD(x1560, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1561, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1563 = (x1562 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1564 = ((x1544 + (LOAD(x1548, 0) * inv_0(x1555))) + (LOAD(x1556, 0) * inv_0(x1563)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1565 = (x1555 * x1563);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1566 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1567 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 7), addr), _super);
    BoundLayout<Reg> x1568 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 7), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1569 = ((LOAD_EXT(x11, 0) * LOAD(x1567, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1568, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1570 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 7), dataLow), _super);
    BoundLayout<Reg> x1571 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 7), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1572 =
        ((x1569 + (LOAD_EXT(x13, 0) * LOAD(x1570, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1571, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1573 = (x1572 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1574 = (x1564 + (LOAD(x1566, 0) * inv_0(x1573)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1575 = ((LOAD(x1548, 0) * x1563) * x1573);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x1574);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1576 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1577 = (((x1576 * (x1565 * x1573)) - x1575) - ((x1555 * LOAD(x1556, 0)) * x1573));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1577 - (x1565 * LOAD(x1566, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1578 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1579 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 8), addr), _super);
    BoundLayout<Reg> x1580 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 8), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1581 = ((LOAD_EXT(x11, 0) * LOAD(x1579, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1580, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1582 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 8), dataLow), _super);
    BoundLayout<Reg> x1583 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 8), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1584 =
        ((x1581 + (LOAD_EXT(x13, 0) * LOAD(x1582, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1583, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1585 = (x1584 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1586 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 9), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1587 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 9), addr), _super);
    BoundLayout<Reg> x1588 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 9), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1589 = ((LOAD_EXT(x11, 0) * LOAD(x1587, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1588, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1590 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 9), dataLow), _super);
    BoundLayout<Reg> x1591 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 9), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1592 =
        ((x1589 + (LOAD_EXT(x13, 0) * LOAD(x1590, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1591, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1593 = (x1592 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1594 = ((x1574 + (LOAD(x1578, 0) * inv_0(x1585))) + (LOAD(x1586, 0) * inv_0(x1593)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1595 = (x1585 * x1593);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1596 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 10), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1597 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 10), addr), _super);
    BoundLayout<Reg> x1598 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 10), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1599 = ((LOAD_EXT(x11, 0) * LOAD(x1597, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1598, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1600 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 10), dataLow), _super);
    BoundLayout<Reg> x1601 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 10), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1602 =
        ((x1599 + (LOAD_EXT(x13, 0) * LOAD(x1600, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1601, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1603 = (x1602 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1604 = (x1594 + (LOAD(x1596, 0) * inv_0(x1603)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1605 = ((LOAD(x1578, 0) * x1593) * x1603);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x1604);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1606 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1607 = (((x1606 * (x1595 * x1603)) - x1605) - ((x1585 * LOAD(x1586, 0)) * x1603));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1607 - (x1595 * LOAD(x1596, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1608 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 11), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1609 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 11), addr), _super);
    BoundLayout<Reg> x1610 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 11), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1611 = ((LOAD_EXT(x11, 0) * LOAD(x1609, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1610, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1612 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 11), dataLow), _super);
    BoundLayout<Reg> x1613 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 11), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1614 =
        ((x1611 + (LOAD_EXT(x13, 0) * LOAD(x1612, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1613, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1615 = (x1614 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1616 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 12), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1617 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 12), addr), _super);
    BoundLayout<Reg> x1618 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 12), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1619 = ((LOAD_EXT(x11, 0) * LOAD(x1617, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1618, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1620 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 12), dataLow), _super);
    BoundLayout<Reg> x1621 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 12), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1622 =
        ((x1619 + (LOAD_EXT(x13, 0) * LOAD(x1620, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1621, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1623 = (x1622 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1624 = ((x1604 + (LOAD(x1608, 0) * inv_0(x1615))) + (LOAD(x1616, 0) * inv_0(x1623)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1625 = (x1615 * x1623);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1626 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 13), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1627 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 13), addr), _super);
    BoundLayout<Reg> x1628 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 13), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1629 = ((LOAD_EXT(x11, 0) * LOAD(x1627, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1628, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1630 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 13), dataLow), _super);
    BoundLayout<Reg> x1631 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 13), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1632 =
        ((x1629 + (LOAD_EXT(x13, 0) * LOAD(x1630, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1631, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1633 = (x1632 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1634 = (x1624 + (LOAD(x1626, 0) * inv_0(x1633)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1635 = ((LOAD(x1608, 0) * x1623) * x1633);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x1634);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1636 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1637 = (((x1636 * (x1625 * x1633)) - x1635) - ((x1615 * LOAD(x1616, 0)) * x1633));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1637 - (x1625 * LOAD(x1626, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1638 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 14), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1639 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 14), addr), _super);
    BoundLayout<Reg> x1640 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 14), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1641 = ((LOAD_EXT(x11, 0) * LOAD(x1639, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1640, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1642 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 14), dataLow), _super);
    BoundLayout<Reg> x1643 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 14), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1644 =
        ((x1641 + (LOAD_EXT(x13, 0) * LOAD(x1642, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1643, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1645 = (x1644 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1646 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 15), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1647 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 15), addr), _super);
    BoundLayout<Reg> x1648 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 15), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1649 = ((LOAD_EXT(x11, 0) * LOAD(x1647, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1648, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1650 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 15), dataLow), _super);
    BoundLayout<Reg> x1651 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x271, 15), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1652 =
        ((x1649 + (LOAD_EXT(x13, 0) * LOAD(x1650, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1651, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1653 = (x1652 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1654 = ((x1634 + (LOAD(x1638, 0) * inv_0(x1645))) + (LOAD(x1646, 0) * inv_0(x1653)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1655 = (x1645 * x1653);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1656 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1657 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 0), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1658 = ((LOAD_EXT(x15, 0) * LOAD(x1657, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1659 = (x1654 + (LOAD(x1656, 0) * inv_0(x1658)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1660 = ((LOAD(x1638, 0) * x1653) * x1658);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x1659);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1661 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1662 = (((x1661 * (x1655 * x1658)) - x1660) - ((x1645 * LOAD(x1646, 0)) * x1658));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1662 - (x1655 * LOAD(x1656, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1663 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1664 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 1), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1665 = ((LOAD_EXT(x15, 0) * LOAD(x1664, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1666 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1667 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 2), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1668 = ((LOAD_EXT(x15, 0) * LOAD(x1667, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1669 = ((x1659 + (LOAD(x1663, 0) * inv_0(x1665))) + (LOAD(x1666, 0) * inv_0(x1668)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1670 = (x1665 * x1668);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1671 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1672 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 3), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1673 = ((LOAD_EXT(x15, 0) * LOAD(x1672, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1674 = (x1669 + (LOAD(x1671, 0) * inv_0(x1673)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1675 = ((LOAD(x1663, 0) * x1668) * x1673);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x1674);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1676 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1677 = (((x1676 * (x1670 * x1673)) - x1675) - ((x1665 * LOAD(x1666, 0)) * x1673));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1677 - (x1670 * LOAD(x1671, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1678 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1679 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 4), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1680 = ((LOAD_EXT(x15, 0) * LOAD(x1679, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1681 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1682 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 5), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1683 = ((LOAD_EXT(x15, 0) * LOAD(x1682, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1684 = ((x1674 + (LOAD(x1678, 0) * inv_0(x1680))) + (LOAD(x1681, 0) * inv_0(x1683)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1685 = (x1680 * x1683);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1686 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1687 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 6), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1688 = ((LOAD_EXT(x15, 0) * LOAD(x1687, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1689 = (x1684 + (LOAD(x1686, 0) * inv_0(x1688)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1690 = ((LOAD(x1678, 0) * x1683) * x1688);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x1689);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1691 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1692 = (((x1691 * (x1685 * x1688)) - x1690) - ((x1680 * LOAD(x1681, 0)) * x1688));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1692 - (x1685 * LOAD(x1686, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1693 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1694 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x272, 7), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1695 = ((LOAD_EXT(x15, 0) * LOAD(x1694, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1696 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1697 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1698 = ((LOAD_EXT(x9, 0) * LOAD(x1697, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1699 = ((x1689 + (LOAD(x1693, 0) * inv_0(x1695))) + (LOAD(x1696, 0) * inv_0(x1698)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1700 = (x1695 * x1698);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1701 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1702 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1703 = ((LOAD_EXT(x9, 0) * LOAD(x1702, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1704 = (x1699 + (LOAD(x1701, 0) * inv_0(x1703)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1705 = ((LOAD(x1693, 0) * x1698) * x1703);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 8), x1704);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1706 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1707 = (((x1706 * (x1700 * x1703)) - x1705) - ((x1695 * LOAD(x1696, 0)) * x1703));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1707 - (x1700 * LOAD(x1701, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1708 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1709 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1710 = ((LOAD_EXT(x9, 0) * LOAD(x1709, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1711 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1712 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1713 = ((LOAD_EXT(x9, 0) * LOAD(x1712, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1714 = ((x1704 + (LOAD(x1708, 0) * inv_0(x1710))) + (LOAD(x1711, 0) * inv_0(x1713)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1715 = (x1710 * x1713);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1716 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1717 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1718 = ((LOAD_EXT(x9, 0) * LOAD(x1717, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1719 = (x1714 + (LOAD(x1716, 0) * inv_0(x1718)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1720 = ((LOAD(x1708, 0) * x1713) * x1718);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 9), x1719);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1721 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1722 = (((x1721 * (x1715 * x1718)) - x1720) - ((x1710 * LOAD(x1711, 0)) * x1718));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1722 - (x1715 * LOAD(x1716, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1723 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1724 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 5), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1725 = ((LOAD_EXT(x9, 0) * LOAD(x1724, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1726 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1727 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 6), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1728 = ((LOAD_EXT(x9, 0) * LOAD(x1727, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1729 = ((x1719 + (LOAD(x1723, 0) * inv_0(x1725))) + (LOAD(x1726, 0) * inv_0(x1728)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1730 = (x1725 * x1728);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1731 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1732 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 7), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1733 = ((LOAD_EXT(x9, 0) * LOAD(x1732, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1734 = (x1729 + (LOAD(x1731, 0) * inv_0(x1733)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1735 = ((LOAD(x1723, 0) * x1728) * x1733);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 10), x1734);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1736 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1737 = (((x1736 * (x1730 * x1733)) - x1735) - ((x1725 * LOAD(x1726, 0)) * x1733));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1737 - (x1730 * LOAD(x1731, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1738 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1739 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 8), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1740 = ((LOAD_EXT(x9, 0) * LOAD(x1739, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1741 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 9), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1742 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 9), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1743 = ((LOAD_EXT(x9, 0) * LOAD(x1742, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1744 = ((x1734 + (LOAD(x1738, 0) * inv_0(x1740))) + (LOAD(x1741, 0) * inv_0(x1743)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1745 = (x1740 * x1743);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1746 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 10), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1747 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 10), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1748 = ((LOAD_EXT(x9, 0) * LOAD(x1747, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1749 = (x1744 + (LOAD(x1746, 0) * inv_0(x1748)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1750 = ((LOAD(x1738, 0) * x1743) * x1748);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 11), x1749);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1751 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1752 = (((x1751 * (x1745 * x1748)) - x1750) - ((x1740 * LOAD(x1741, 0)) * x1748));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1752 - (x1745 * LOAD(x1746, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1753 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 11), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1754 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 11), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1755 = ((LOAD_EXT(x9, 0) * LOAD(x1754, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1756 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 12), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1757 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 12), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1758 = ((LOAD_EXT(x9, 0) * LOAD(x1757, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1759 = ((x1749 + (LOAD(x1753, 0) * inv_0(x1755))) + (LOAD(x1756, 0) * inv_0(x1758)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1760 = (x1755 * x1758);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1761 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 13), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1762 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 13), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1763 = ((LOAD_EXT(x9, 0) * LOAD(x1762, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1764 = (x1759 + (LOAD(x1761, 0) * inv_0(x1763)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1765 = ((LOAD(x1753, 0) * x1758) * x1763);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 12), x1764);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1766 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 12), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1767 = (((x1766 * (x1760 * x1763)) - x1765) - ((x1755 * LOAD(x1756, 0)) * x1763));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1767 - (x1760 * LOAD(x1761, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1768 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 14), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1769 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 14), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1770 = ((LOAD_EXT(x9, 0) * LOAD(x1769, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1771 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 15), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1772 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x273, 15), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1773 = ((LOAD_EXT(x9, 0) * LOAD(x1772, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1774 = ((x1764 + (LOAD(x1768, 0) * inv_0(x1770))) + (LOAD(x1771, 0) * inv_0(x1773)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1775 = (x1770 * x1773);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1776 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1777 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1778 = ((LOAD_EXT(x123, 0) * LOAD(x1777, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1779 = (x1774 + (LOAD(x1776, 0) * inv_0(x1778)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1780 = ((LOAD(x1768, 0) * x1773) * x1778);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 13), x1779);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1781 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 13), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 12), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1782 = (((x1781 * (x1775 * x1778)) - x1780) - ((x1770 * LOAD(x1771, 0)) * x1778));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1782 - (x1775 * LOAD(x1776, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1783 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1784 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1785 = ((LOAD_EXT(x123, 0) * LOAD(x1784, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1786 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1787 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1788 = ((LOAD_EXT(x123, 0) * LOAD(x1787, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1789 = ((x1779 + (LOAD(x1783, 0) * inv_0(x1785))) + (LOAD(x1786, 0) * inv_0(x1788)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1790 = (x1785 * x1788);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1791 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1792 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1793 = ((LOAD_EXT(x123, 0) * LOAD(x1792, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1794 = (x1789 + (LOAD(x1791, 0) * inv_0(x1793)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1795 = ((LOAD(x1783, 0) * x1788) * x1793);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 14), x1794);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1796 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 14), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 13), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1797 = (((x1796 * (x1790 * x1793)) - x1795) - ((x1785 * LOAD(x1786, 0)) * x1793));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1797 - (x1790 * LOAD(x1791, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1798 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1799 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1800 = ((LOAD_EXT(x123, 0) * LOAD(x1799, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1801 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1802 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 5), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1803 = ((LOAD_EXT(x123, 0) * LOAD(x1802, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1804 = ((x1794 + (LOAD(x1798, 0) * inv_0(x1800))) + (LOAD(x1801, 0) * inv_0(x1803)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1805 = (x1800 * x1803);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1806 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1807 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 6), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1808 = ((LOAD_EXT(x123, 0) * LOAD(x1807, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1809 = (x1804 + (LOAD(x1806, 0) * inv_0(x1808)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1810 = ((LOAD(x1798, 0) * x1803) * x1808);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 15), x1809);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1811 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 15), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 14), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1812 = (((x1811 * (x1805 * x1808)) - x1810) - ((x1800 * LOAD(x1801, 0)) * x1808));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1812 - (x1805 * LOAD(x1806, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1813 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1814 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 7), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1815 = ((LOAD_EXT(x123, 0) * LOAD(x1814, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1816 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1817 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 8), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1818 = ((LOAD_EXT(x123, 0) * LOAD(x1817, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1819 = ((x1809 + (LOAD(x1813, 0) * inv_0(x1815))) + (LOAD(x1816, 0) * inv_0(x1818)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1820 = (x1815 * x1818);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1821 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 9), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1822 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 9), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1823 = ((LOAD_EXT(x123, 0) * LOAD(x1822, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1824 = (x1819 + (LOAD(x1821, 0) * inv_0(x1823)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1825 = ((LOAD(x1813, 0) * x1818) * x1823);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 16), x1824);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1826 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 16), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 15), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1827 = (((x1826 * (x1820 * x1823)) - x1825) - ((x1815 * LOAD(x1816, 0)) * x1823));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1827 - (x1820 * LOAD(x1821, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1828 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 10), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1829 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 10), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1830 = ((LOAD_EXT(x123, 0) * LOAD(x1829, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1831 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 11), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1832 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 11), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1833 = ((LOAD_EXT(x123, 0) * LOAD(x1832, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1834 = ((x1824 + (LOAD(x1828, 0) * inv_0(x1830))) + (LOAD(x1831, 0) * inv_0(x1833)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1835 = (x1830 * x1833);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1836 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 12), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1837 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 12), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1838 = ((LOAD_EXT(x123, 0) * LOAD(x1837, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1839 = (x1834 + (LOAD(x1836, 0) * inv_0(x1838)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1840 = ((LOAD(x1828, 0) * x1833) * x1838);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 17), x1839);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1841 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 17), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 16), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1842 = (((x1841 * (x1835 * x1838)) - x1840) - ((x1830 * LOAD(x1831, 0)) * x1838));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1842 - (x1835 * LOAD(x1836, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1843 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 13), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1844 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 13), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1845 = ((LOAD_EXT(x123, 0) * LOAD(x1844, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1846 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 14), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1847 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 14), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1848 = ((LOAD_EXT(x123, 0) * LOAD(x1847, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1849 = ((x1839 + (LOAD(x1843, 0) * inv_0(x1845))) + (LOAD(x1846, 0) * inv_0(x1848)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1850 = (x1845 * x1848);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1851 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 15), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1852 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x274, 15), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1853 = ((LOAD_EXT(x123, 0) * LOAD(x1852, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1854 = ((LOAD(x1843, 0) * x1848) * x1853);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), (x1849 + (LOAD(x1851, 0) * inv_0(x1853))));
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1855 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 17), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1856 = (((x1855 * (x1850 * x1853)) - x1854) - ((x1845 * LOAD(x1846, 0)) * x1853));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1856 - (x1850 * LOAD(x1851, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 8), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1857 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x277, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1858 = (x1857 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1859 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x277, count), _super), 0) * inv_0(x1858));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1860 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x1859);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1861 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x278, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1862 = (x1861 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1863 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x278, count), _super), 0) * inv_0(x1862));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1864 = (x1858 * x1862);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1865 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x277, count), _super), 0) * x1862);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1866 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1867 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 0), addr), _super);
    BoundLayout<Reg> x1868 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 0), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1869 = ((LOAD_EXT(x11, 0) * LOAD(x1867, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1868, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1870 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 0), dataLow), _super);
    BoundLayout<Reg> x1871 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 0), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1872 =
        ((x1869 + (LOAD_EXT(x13, 0) * LOAD(x1870, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1871, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1873 = (x1872 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1874 = ((x1860 + x1863) + (LOAD(x1866, 0) * inv_0(x1873)));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x1874);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1875 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1876 = (((x1875 * (x1864 * x1873)) - (x1865 * x1873)) -
                    ((x1858 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x278, count), _super), 0)) * x1873));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1876 - (x1864 * LOAD(x1866, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1877 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1878 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 1), addr), _super);
    BoundLayout<Reg> x1879 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 1), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1880 = ((LOAD_EXT(x11, 0) * LOAD(x1878, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1879, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1881 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 1), dataLow), _super);
    BoundLayout<Reg> x1882 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 1), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1883 =
        ((x1880 + (LOAD_EXT(x13, 0) * LOAD(x1881, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1882, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1884 = (x1883 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1885 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1886 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 2), addr), _super);
    BoundLayout<Reg> x1887 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 2), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1888 = ((LOAD_EXT(x11, 0) * LOAD(x1886, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1887, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1889 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 2), dataLow), _super);
    BoundLayout<Reg> x1890 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 2), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1891 =
        ((x1888 + (LOAD_EXT(x13, 0) * LOAD(x1889, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1890, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1892 = (x1891 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1893 = ((x1874 + (LOAD(x1877, 0) * inv_0(x1884))) + (LOAD(x1885, 0) * inv_0(x1892)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1894 = (x1884 * x1892);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1895 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1896 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 3), addr), _super);
    BoundLayout<Reg> x1897 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 3), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1898 = ((LOAD_EXT(x11, 0) * LOAD(x1896, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1897, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1899 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 3), dataLow), _super);
    BoundLayout<Reg> x1900 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 3), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1901 =
        ((x1898 + (LOAD_EXT(x13, 0) * LOAD(x1899, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1900, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1902 = (x1901 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1903 = (x1893 + (LOAD(x1895, 0) * inv_0(x1902)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1904 = ((LOAD(x1877, 0) * x1892) * x1902);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x1903);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1905 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1906 = (((x1905 * (x1894 * x1902)) - x1904) - ((x1884 * LOAD(x1885, 0)) * x1902));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1906 - (x1894 * LOAD(x1895, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1907 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1908 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 4), addr), _super);
    BoundLayout<Reg> x1909 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 4), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1910 = ((LOAD_EXT(x11, 0) * LOAD(x1908, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1909, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1911 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 4), dataLow), _super);
    BoundLayout<Reg> x1912 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 4), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1913 =
        ((x1910 + (LOAD_EXT(x13, 0) * LOAD(x1911, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1912, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1914 = (x1913 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1915 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1916 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 5), addr), _super);
    BoundLayout<Reg> x1917 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 5), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1918 = ((LOAD_EXT(x11, 0) * LOAD(x1916, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1917, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1919 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 5), dataLow), _super);
    BoundLayout<Reg> x1920 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 5), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1921 =
        ((x1918 + (LOAD_EXT(x13, 0) * LOAD(x1919, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1920, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1922 = (x1921 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1923 = ((x1903 + (LOAD(x1907, 0) * inv_0(x1914))) + (LOAD(x1915, 0) * inv_0(x1922)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1924 = (x1914 * x1922);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1925 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1926 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 6), addr), _super);
    BoundLayout<Reg> x1927 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 6), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1928 = ((LOAD_EXT(x11, 0) * LOAD(x1926, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1927, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1929 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 6), dataLow), _super);
    BoundLayout<Reg> x1930 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 6), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1931 =
        ((x1928 + (LOAD_EXT(x13, 0) * LOAD(x1929, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1930, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1932 = (x1931 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1933 = (x1923 + (LOAD(x1925, 0) * inv_0(x1932)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1934 = ((LOAD(x1907, 0) * x1922) * x1932);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x1933);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1935 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1936 = (((x1935 * (x1924 * x1932)) - x1934) - ((x1914 * LOAD(x1915, 0)) * x1932));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1936 - (x1924 * LOAD(x1925, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1937 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1938 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 7), addr), _super);
    BoundLayout<Reg> x1939 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 7), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1940 = ((LOAD_EXT(x11, 0) * LOAD(x1938, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1939, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1941 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 7), dataLow), _super);
    BoundLayout<Reg> x1942 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x280, 7), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1943 =
        ((x1940 + (LOAD_EXT(x13, 0) * LOAD(x1941, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1942, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1944 = (x1943 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1945 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1946 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 0), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1947 = ((LOAD_EXT(x15, 0) * LOAD(x1946, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1948 = ((x1933 + (LOAD(x1937, 0) * inv_0(x1944))) + (LOAD(x1945, 0) * inv_0(x1947)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1949 = (x1944 * x1947);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1950 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1951 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 1), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1952 = ((LOAD_EXT(x15, 0) * LOAD(x1951, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1953 = (x1948 + (LOAD(x1950, 0) * inv_0(x1952)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1954 = ((LOAD(x1937, 0) * x1947) * x1952);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x1953);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1955 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1956 = (((x1955 * (x1949 * x1952)) - x1954) - ((x1944 * LOAD(x1945, 0)) * x1952));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1956 - (x1949 * LOAD(x1950, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1957 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1958 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 2), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1959 = ((LOAD_EXT(x15, 0) * LOAD(x1958, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1960 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1961 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x281, 3), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1962 = ((LOAD_EXT(x15, 0) * LOAD(x1961, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1963 = ((x1953 + (LOAD(x1957, 0) * inv_0(x1959))) + (LOAD(x1960, 0) * inv_0(x1962)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1964 = (x1959 * x1962);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1965 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x282, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1966 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x282, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1967 = ((LOAD_EXT(x9, 0) * LOAD(x1966, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1968 = (x1963 + (LOAD(x1965, 0) * inv_0(x1967)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1969 = ((LOAD(x1957, 0) * x1962) * x1967);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x1968);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1970 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1971 = (((x1970 * (x1964 * x1967)) - x1969) - ((x1959 * LOAD(x1960, 0)) * x1967));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1971 - (x1964 * LOAD(x1965, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1972 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x282, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1973 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x282, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1974 = ((LOAD_EXT(x9, 0) * LOAD(x1973, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1975 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x284, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1976 = (x1975 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1977 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x284, count), _super), 0) * inv_0(x1976));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x1978 = (x1974 * x1976);
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1979 = (LOAD_EXT(x9, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x285, val), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1980 = (x1979 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1981 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x285, count), _super), 0) * inv_0(x1980));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x1982 = (((x1968 + (LOAD(x1972, 0) * inv_0(x1974))) + x1977) + x1981);
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x1983 = ((LOAD(x1972, 0) * x1976) * x1980);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x1982);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1984 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1985 = (((x1984 * (x1978 * x1980)) - x1983) -
                    ((x1974 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x284, count), _super), 0)) * x1980));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x1985 - (x1978 * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x285, count), _super), 0))),
        "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x1986 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x286, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1987 = (x1986 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x1988 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x286, count), _super), 0) * inv_0(x1987));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), (x1982 + x1988));
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x1989 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x1990 = ((x1989 * x1987) - LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x286, count), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ(x1990, "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x1991 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x1991, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 9), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x1992 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1993 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 0), addr), _super);
    BoundLayout<Reg> x1994 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 0), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1995 = ((LOAD_EXT(x11, 0) * LOAD(x1993, 0)) + (LOAD_EXT(x12, 0) * LOAD(x1994, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x1996 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 0), dataLow), _super);
    BoundLayout<Reg> x1997 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 0), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x1998 =
        ((x1995 + (LOAD_EXT(x13, 0) * LOAD(x1996, 0))) + (LOAD_EXT(x14, 0) * LOAD(x1997, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x1999 = (x1998 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2000 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + (LOAD(x1992, 0) * inv_0(x1999)));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2001 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2002 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 1), addr), _super);
    BoundLayout<Reg> x2003 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 1), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2004 = ((LOAD_EXT(x11, 0) * LOAD(x2002, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2003, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2005 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 1), dataLow), _super);
    BoundLayout<Reg> x2006 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 1), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2007 =
        ((x2004 + (LOAD_EXT(x13, 0) * LOAD(x2005, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2006, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2008 = (x2007 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2009 = (x1999 * x2008);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2010 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2011 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 2), addr), _super);
    BoundLayout<Reg> x2012 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 2), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2013 = ((LOAD_EXT(x11, 0) * LOAD(x2011, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2012, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2014 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 2), dataLow), _super);
    BoundLayout<Reg> x2015 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 2), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2016 =
        ((x2013 + (LOAD_EXT(x13, 0) * LOAD(x2014, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2015, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2017 = (x2016 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2018 = ((x2000 + (LOAD(x2001, 0) * inv_0(x2008))) + (LOAD(x2010, 0) * inv_0(x2017)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2019 = ((LOAD(x1992, 0) * x2008) * x2017);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x2018);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2020 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2021 = (((x2020 * (x2009 * x2017)) - x2019) - ((x1999 * LOAD(x2001, 0)) * x2017));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2021 - (x2009 * LOAD(x2010, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2022 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2023 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 3), addr), _super);
    BoundLayout<Reg> x2024 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 3), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2025 = ((LOAD_EXT(x11, 0) * LOAD(x2023, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2024, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2026 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 3), dataLow), _super);
    BoundLayout<Reg> x2027 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 3), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2028 =
        ((x2025 + (LOAD_EXT(x13, 0) * LOAD(x2026, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2027, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2029 = (x2028 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2030 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2031 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 4), addr), _super);
    BoundLayout<Reg> x2032 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 4), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2033 = ((LOAD_EXT(x11, 0) * LOAD(x2031, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2032, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2034 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 4), dataLow), _super);
    BoundLayout<Reg> x2035 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 4), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2036 =
        ((x2033 + (LOAD_EXT(x13, 0) * LOAD(x2034, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2035, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2037 = (x2036 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2038 = ((x2018 + (LOAD(x2022, 0) * inv_0(x2029))) + (LOAD(x2030, 0) * inv_0(x2037)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2039 = (x2029 * x2037);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2040 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2041 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 5), addr), _super);
    BoundLayout<Reg> x2042 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 5), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2043 = ((LOAD_EXT(x11, 0) * LOAD(x2041, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2042, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2044 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 5), dataLow), _super);
    BoundLayout<Reg> x2045 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 5), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2046 =
        ((x2043 + (LOAD_EXT(x13, 0) * LOAD(x2044, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2045, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2047 = (x2046 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2048 = (x2038 + (LOAD(x2040, 0) * inv_0(x2047)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2049 = ((LOAD(x2022, 0) * x2037) * x2047);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 1), x2048);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2050 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2051 = (((x2050 * (x2039 * x2047)) - x2049) - ((x2029 * LOAD(x2030, 0)) * x2047));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2051 - (x2039 * LOAD(x2040, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2052 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2053 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 6), addr), _super);
    BoundLayout<Reg> x2054 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 6), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2055 = ((LOAD_EXT(x11, 0) * LOAD(x2053, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2054, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2056 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 6), dataLow), _super);
    BoundLayout<Reg> x2057 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 6), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2058 =
        ((x2055 + (LOAD_EXT(x13, 0) * LOAD(x2056, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2057, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2059 = (x2058 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2060 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2061 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 7), addr), _super);
    BoundLayout<Reg> x2062 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 7), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2063 = ((LOAD_EXT(x11, 0) * LOAD(x2061, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2062, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2064 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 7), dataLow), _super);
    BoundLayout<Reg> x2065 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 7), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2066 =
        ((x2063 + (LOAD_EXT(x13, 0) * LOAD(x2064, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2065, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2067 = (x2066 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2068 = ((x2048 + (LOAD(x2052, 0) * inv_0(x2059))) + (LOAD(x2060, 0) * inv_0(x2067)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2069 = (x2059 * x2067);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2070 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2071 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 8), addr), _super);
    BoundLayout<Reg> x2072 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 8), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2073 = ((LOAD_EXT(x11, 0) * LOAD(x2071, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2072, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2074 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 8), dataLow), _super);
    BoundLayout<Reg> x2075 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 8), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2076 =
        ((x2073 + (LOAD_EXT(x13, 0) * LOAD(x2074, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2075, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2077 = (x2076 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2078 = (x2068 + (LOAD(x2070, 0) * inv_0(x2077)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2079 = ((LOAD(x2052, 0) * x2067) * x2077);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 2), x2078);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2080 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 1), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2081 = (((x2080 * (x2069 * x2077)) - x2079) - ((x2059 * LOAD(x2060, 0)) * x2077));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2081 - (x2069 * LOAD(x2070, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2082 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 9), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2083 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 9), addr), _super);
    BoundLayout<Reg> x2084 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 9), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2085 = ((LOAD_EXT(x11, 0) * LOAD(x2083, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2084, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2086 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 9), dataLow), _super);
    BoundLayout<Reg> x2087 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 9), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2088 =
        ((x2085 + (LOAD_EXT(x13, 0) * LOAD(x2086, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2087, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2089 = (x2088 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2090 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 10), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2091 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 10), addr), _super);
    BoundLayout<Reg> x2092 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 10), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2093 = ((LOAD_EXT(x11, 0) * LOAD(x2091, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2092, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2094 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 10), dataLow), _super);
    BoundLayout<Reg> x2095 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 10), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2096 =
        ((x2093 + (LOAD_EXT(x13, 0) * LOAD(x2094, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2095, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2097 = (x2096 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2098 = ((x2078 + (LOAD(x2082, 0) * inv_0(x2089))) + (LOAD(x2090, 0) * inv_0(x2097)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2099 = (x2089 * x2097);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2100 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 11), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2101 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 11), addr), _super);
    BoundLayout<Reg> x2102 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 11), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2103 = ((LOAD_EXT(x11, 0) * LOAD(x2101, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2102, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2104 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 11), dataLow), _super);
    BoundLayout<Reg> x2105 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 11), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2106 =
        ((x2103 + (LOAD_EXT(x13, 0) * LOAD(x2104, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2105, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2107 = (x2106 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2108 = (x2098 + (LOAD(x2100, 0) * inv_0(x2107)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2109 = ((LOAD(x2082, 0) * x2097) * x2107);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 3), x2108);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2110 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 2), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2111 = (((x2110 * (x2099 * x2107)) - x2109) - ((x2089 * LOAD(x2090, 0)) * x2107));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2111 - (x2099 * LOAD(x2100, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2112 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 12), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2113 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 12), addr), _super);
    BoundLayout<Reg> x2114 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 12), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2115 = ((LOAD_EXT(x11, 0) * LOAD(x2113, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2114, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2116 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 12), dataLow), _super);
    BoundLayout<Reg> x2117 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 12), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2118 =
        ((x2115 + (LOAD_EXT(x13, 0) * LOAD(x2116, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2117, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2119 = (x2118 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2120 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 13), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2121 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 13), addr), _super);
    BoundLayout<Reg> x2122 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 13), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2123 = ((LOAD_EXT(x11, 0) * LOAD(x2121, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2122, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2124 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 13), dataLow), _super);
    BoundLayout<Reg> x2125 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 13), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2126 =
        ((x2123 + (LOAD_EXT(x13, 0) * LOAD(x2124, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2125, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2127 = (x2126 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2128 = ((x2108 + (LOAD(x2112, 0) * inv_0(x2119))) + (LOAD(x2120, 0) * inv_0(x2127)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2129 = (x2119 * x2127);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2130 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 14), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2131 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 14), addr), _super);
    BoundLayout<Reg> x2132 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 14), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2133 = ((LOAD_EXT(x11, 0) * LOAD(x2131, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2132, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2134 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 14), dataLow), _super);
    BoundLayout<Reg> x2135 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 14), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2136 =
        ((x2133 + (LOAD_EXT(x13, 0) * LOAD(x2134, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2135, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2137 = (x2136 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2138 = (x2128 + (LOAD(x2130, 0) * inv_0(x2137)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2139 = ((LOAD(x2112, 0) * x2127) * x2137);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 4), x2138);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2140 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 3), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2141 = (((x2140 * (x2129 * x2137)) - x2139) - ((x2119 * LOAD(x2120, 0)) * x2137));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2141 - (x2129 * LOAD(x2130, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2142 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 15), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2143 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 15), addr), _super);
    BoundLayout<Reg> x2144 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 15), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2145 = ((LOAD_EXT(x11, 0) * LOAD(x2143, 0)) + (LOAD_EXT(x12, 0) * LOAD(x2144, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2146 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 15), dataLow), _super);
    BoundLayout<Reg> x2147 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x289, 15), dataHigh), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:157
    ExtVal x2148 =
        ((x2145 + (LOAD_EXT(x13, 0) * LOAD(x2146, 0))) + (LOAD_EXT(x14, 0) * LOAD(x2147, 0)));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2149 = (x2148 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2150 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2151 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 0), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2152 = ((LOAD_EXT(x15, 0) * LOAD(x2151, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2153 = ((x2138 + (LOAD(x2142, 0) * inv_0(x2149))) + (LOAD(x2150, 0) * inv_0(x2152)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2154 = (x2149 * x2152);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2155 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2156 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 1), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2157 = ((LOAD_EXT(x15, 0) * LOAD(x2156, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2158 = (x2153 + (LOAD(x2155, 0) * inv_0(x2157)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2159 = ((LOAD(x2142, 0) * x2152) * x2157);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 5), x2158);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2160 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 4), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2161 = (((x2160 * (x2154 * x2157)) - x2159) - ((x2149 * LOAD(x2150, 0)) * x2157));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2161 - (x2154 * LOAD(x2155, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2162 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2163 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 2), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2164 = ((LOAD_EXT(x15, 0) * LOAD(x2163, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2165 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2166 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 3), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2167 = ((LOAD_EXT(x15, 0) * LOAD(x2166, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2168 = ((x2158 + (LOAD(x2162, 0) * inv_0(x2164))) + (LOAD(x2165, 0) * inv_0(x2167)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2169 = (x2164 * x2167);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2170 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2171 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 4), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2172 = ((LOAD_EXT(x15, 0) * LOAD(x2171, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2173 = (x2168 + (LOAD(x2170, 0) * inv_0(x2172)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2174 = ((LOAD(x2162, 0) * x2167) * x2172);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 6), x2173);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2175 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 5), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2176 = (((x2175 * (x2169 * x2172)) - x2174) - ((x2164 * LOAD(x2165, 0)) * x2172));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2176 - (x2169 * LOAD(x2170, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2177 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2178 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 5), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2179 = ((LOAD_EXT(x15, 0) * LOAD(x2178, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2180 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2181 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 6), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2182 = ((LOAD_EXT(x15, 0) * LOAD(x2181, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2183 = ((x2173 + (LOAD(x2177, 0) * inv_0(x2179))) + (LOAD(x2180, 0) * inv_0(x2182)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2184 = (x2179 * x2182);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2185 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2186 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x290, 7), cycle), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2187 = ((LOAD_EXT(x15, 0) * LOAD(x2186, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2188 = (x2183 + (LOAD(x2185, 0) * inv_0(x2187)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2189 = ((LOAD(x2177, 0) * x2182) * x2187);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 7), x2188);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2190 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 6), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2191 = (((x2190 * (x2184 * x2187)) - x2189) - ((x2179 * LOAD(x2180, 0)) * x2187));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2191 - (x2184 * LOAD(x2185, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2192 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2193 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2194 = ((LOAD_EXT(x9, 0) * LOAD(x2193, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2195 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2196 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2197 = ((LOAD_EXT(x9, 0) * LOAD(x2196, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2198 = ((x2188 + (LOAD(x2192, 0) * inv_0(x2194))) + (LOAD(x2195, 0) * inv_0(x2197)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2199 = (x2194 * x2197);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2200 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 2), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2201 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 2), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2202 = ((LOAD_EXT(x9, 0) * LOAD(x2201, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2203 = (x2198 + (LOAD(x2200, 0) * inv_0(x2202)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2204 = ((LOAD(x2192, 0) * x2197) * x2202);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 8), x2203);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2205 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 7), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2206 = (((x2205 * (x2199 * x2202)) - x2204) - ((x2194 * LOAD(x2195, 0)) * x2202));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2206 - (x2199 * LOAD(x2200, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2207 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 3), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2208 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 3), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2209 = ((LOAD_EXT(x9, 0) * LOAD(x2208, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2210 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 4), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2211 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 4), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2212 = ((LOAD_EXT(x9, 0) * LOAD(x2211, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2213 = ((x2203 + (LOAD(x2207, 0) * inv_0(x2209))) + (LOAD(x2210, 0) * inv_0(x2212)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2214 = (x2209 * x2212);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2215 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 5), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2216 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 5), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2217 = ((LOAD_EXT(x9, 0) * LOAD(x2216, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2218 = (x2213 + (LOAD(x2215, 0) * inv_0(x2217)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2219 = ((LOAD(x2207, 0) * x2212) * x2217);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 9), x2218);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2220 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 8), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2221 = (((x2220 * (x2214 * x2217)) - x2219) - ((x2209 * LOAD(x2210, 0)) * x2217));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2221 - (x2214 * LOAD(x2215, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2222 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 6), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2223 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 6), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2224 = ((LOAD_EXT(x9, 0) * LOAD(x2223, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2225 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 7), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2226 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 7), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2227 = ((LOAD_EXT(x9, 0) * LOAD(x2226, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2228 = ((x2218 + (LOAD(x2222, 0) * inv_0(x2224))) + (LOAD(x2225, 0) * inv_0(x2227)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2229 = (x2224 * x2227);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2230 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 8), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2231 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 8), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2232 = ((LOAD_EXT(x9, 0) * LOAD(x2231, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2233 = (x2228 + (LOAD(x2230, 0) * inv_0(x2232)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2234 = ((LOAD(x2222, 0) * x2227) * x2232);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 10), x2233);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2235 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 9), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2236 = (((x2235 * (x2229 * x2232)) - x2234) - ((x2224 * LOAD(x2225, 0)) * x2232));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2236 - (x2229 * LOAD(x2230, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2237 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 9), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2238 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 9), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2239 = ((LOAD_EXT(x9, 0) * LOAD(x2238, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2240 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 10), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2241 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 10), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2242 = ((LOAD_EXT(x9, 0) * LOAD(x2241, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2243 = ((x2233 + (LOAD(x2237, 0) * inv_0(x2239))) + (LOAD(x2240, 0) * inv_0(x2242)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2244 = (x2239 * x2242);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2245 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 11), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2246 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 11), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2247 = ((LOAD_EXT(x9, 0) * LOAD(x2246, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2248 = (x2243 + (LOAD(x2245, 0) * inv_0(x2247)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2249 = ((LOAD(x2237, 0) * x2242) * x2247);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 11), x2248);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2250 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 10), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2251 = (((x2250 * (x2244 * x2247)) - x2249) - ((x2239 * LOAD(x2240, 0)) * x2247));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2251 - (x2244 * LOAD(x2245, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2252 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 12), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2253 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 12), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2254 = ((LOAD_EXT(x9, 0) * LOAD(x2253, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2255 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 13), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2256 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 13), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2257 = ((LOAD_EXT(x9, 0) * LOAD(x2256, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2258 = ((x2248 + (LOAD(x2252, 0) * inv_0(x2254))) + (LOAD(x2255, 0) * inv_0(x2257)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2259 = (x2254 * x2257);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2260 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 14), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2261 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 14), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2262 = ((LOAD_EXT(x9, 0) * LOAD(x2261, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2263 = (x2258 + (LOAD(x2260, 0) * inv_0(x2262)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2264 = ((LOAD(x2252, 0) * x2257) * x2262);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 12), x2263);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2265 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 12), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 11), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2266 = (((x2265 * (x2259 * x2262)) - x2264) - ((x2254 * LOAD(x2255, 0)) * x2262));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2266 - (x2259 * LOAD(x2260, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2267 =
        LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 15), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2268 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x291, 15), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2269 = ((LOAD_EXT(x9, 0) * LOAD(x2268, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2270 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x292, 0), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2271 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x292, 0), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2272 = ((LOAD_EXT(x123, 0) * LOAD(x2271, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2273 = ((x2263 + (LOAD(x2267, 0) * inv_0(x2269))) + (LOAD(x2270, 0) * inv_0(x2272)));
    // zirgen/dsl/passes/GenerateAccum.cpp:217
    ExtVal x2274 = (x2269 * x2272);
    // zirgen/dsl/passes/GenerateAccum.cpp:234
    BoundLayout<Reg> x2275 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x292, 1), count), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:144
    BoundLayout<Reg> x2276 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x292, 1), val), _super);
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2277 = ((LOAD_EXT(x123, 0) * LOAD(x2276, 0)) + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2278 = (x2273 + (LOAD(x2275, 0) * inv_0(x2277)));
    // zirgen/dsl/passes/GenerateAccum.cpp:223
    ExtVal x2279 = ((LOAD(x2267, 0) * x2272) * x2277);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 13), x2278);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2280 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 13), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 12), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2281 = (((x2280 * (x2274 * x2277)) - x2279) - ((x2269 * LOAD(x2270, 0)) * x2277));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ((x2281 - (x2274 * LOAD(x2275, 0))), "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x2282 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x293, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2283 = (x2282 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x2284 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x293, count), _super), 0) * inv_0(x2283));
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 14), (x2278 + x2284));
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2285 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 14), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 13), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2286 = ((x2285 * x2283) - LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x293, count), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ(x2286, "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 14), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x2287 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 14), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x2287, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 10), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:146
    ExtVal x2288 = (LOAD_EXT(x15, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x294, cycle), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:238
    ExtVal x2289 = (x2288 + LOAD_EXT(x16, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:240
    ExtVal x2290 = (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x294, count), _super), 0) * inv_0(x2289));
    // zirgen/dsl/passes/GenerateAccum.cpp:241
    ExtVal x2291 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1) + x2290);
    // zirgen/dsl/passes/GenerateAccum.cpp:189
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 0), x2291);
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    ExtVal x2292 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 1));
    // zirgen/dsl/passes/GenerateAccum.cpp:180
    ExtVal x2293 = ((x2292 * x2289) - LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x294, count), _super), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:182
    EQZ(x2293, "zirgen/dsl/passes/GenerateAccum.cpp:182");
    // zirgen/dsl/passes/GenerateAccum.cpp:122
    STORE_EXT(LAYOUT_SUBSCRIPT(x8, 18), LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:124
    ExtVal x2294 = (LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 18), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(x8, 0), 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:125
    EQZ(x2294, "zirgen/dsl/passes/GenerateAccum.cpp:125");
    x295 = x51;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  return x51;
}
void step_TopAccum(ExecContext& ctx, MutableBuf accum0, MutableBuf data1, GlobalBuf mix2) {
  // zirgen/dsl/passes/GenerateAccum.cpp:526
  BoundLayout<TopLayout> x3 = BIND_LAYOUT(kLayout_Top, data1);
  BoundLayout<LayoutAccumLayout> x4 = BIND_LAYOUT(kLayout_TopAccum, accum0);
  ComponentStruct x5 = exec_TopAccum(ctx, x3, x4, mix2);
  return;
}
