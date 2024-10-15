NondetRegStruct
back_NondetReg(ExecContext& ctx, Index distance0, BoundLayout<NondetRegLayout> layout1) {
  return NondetRegStruct{._super = LOAD(LAYOUT_LOOKUP(layout1, _super), distance0)};
}
NondetRegStruct exec_NondetReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  BoundLayout<Reg> x2 = LAYOUT_LOOKUP(layout1, _super);
  STORE(x2, arg0);
  return NondetRegStruct{._super = LOAD(x2, 0)};
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
OneHotStruct exec_OneHot(ExecContext& ctx, Val arg0, BoundLayout<OneHotLayout> layout1) {
  // OneHot(zirgen/dsl/examples/calculator.zir:14)
  NondetRegStruct2Array x2 =
      map(Val2Array{Val(0), Val(1)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val2Array::value_type x3,
                            BoundLayout<NondetRegLayout2LayoutArray::value_type> x4) {
            NondetRegStruct x5 = exec_NondetReg(ctx, isz((x3 - arg0)), x4);
            return x5;
          }));
  // OneHot(zirgen/dsl/examples/calculator.zir:16)
  Val x6 = x2[0]._super;
  EQZ((x6 * (Val(1) - x6)), "OneHot(zirgen/dsl/examples/calculator.zir:16)");
  Val x7 = x2[1]._super;
  EQZ((x7 * (Val(1) - x7)), "OneHot(zirgen/dsl/examples/calculator.zir:16)");
  // OneHot(zirgen/dsl/examples/calculator.zir:18)
  EQZ(((x6 + x7) - Val(1)), "OneHot(zirgen/dsl/examples/calculator.zir:18)");
  // OneHot(zirgen/dsl/examples/calculator.zir:20)
  EQZ((x7 - arg0), "OneHot(zirgen/dsl/examples/calculator.zir:20)");
  return OneHotStruct{._super = x2};
}
ComponentStruct
exec_MyArgument(ExecContext& ctx, Val arg0, Val arg1, BoundLayout<MyArgumentLayout> layout2) {
  // MyArgument(zirgen/dsl/examples/calculator.zir:34)
  NondetRegStruct x3 = exec_NondetReg(ctx, arg0, LAYOUT_LOOKUP(layout2, a));
  // MyArgument(zirgen/dsl/examples/calculator.zir:35)
  NondetRegStruct x4 = exec_NondetReg(ctx, arg1, LAYOUT_LOOKUP(layout2, b));
  return ComponentStruct{};
}
TopStruct exec_Top(ExecContext& ctx, BoundLayout<TopLayout> layout0, GlobalBuf global1) {
  // Top(zirgen/dsl/examples/calculator.zir:42)
  BoundLayout<TopResultLayout> x2 = LAYOUT_LOOKUP(layout0, result);
  // Top(zirgen/dsl/examples/calculator.zir:53)
  BoundLayout<Top__0Layout> x3 = LAYOUT_LOOKUP(layout0, _3);
  // GetValFromUser(zirgen/dsl/examples/calculator.zir:5)
  // Top(zirgen/dsl/examples/calculator.zir:39)
  Val x4 = INVOKE_EXTERN(ctx, getValFromUser);
  NondetRegStruct x5 = exec_NondetReg(ctx, x4, LAYOUT_LOOKUP(layout0, op));
  // GetValFromUser(zirgen/dsl/examples/calculator.zir:5)
  // Top(zirgen/dsl/examples/calculator.zir:40)
  Val x6 = INVOKE_EXTERN(ctx, getValFromUser);
  NondetRegStruct x7 = exec_NondetReg(ctx, x6, LAYOUT_LOOKUP(layout0, left));
  // GetValFromUser(zirgen/dsl/examples/calculator.zir:5)
  // Top(zirgen/dsl/examples/calculator.zir:41)
  Val x8 = INVOKE_EXTERN(ctx, getValFromUser);
  NondetRegStruct x9 = exec_NondetReg(ctx, x8, LAYOUT_LOOKUP(layout0, right));
  // Top(zirgen/dsl/examples/calculator.zir:42)
  OneHotStruct x10 = exec_OneHot(ctx, x5._super, LAYOUT_LOOKUP(layout0, _0));
  NondetRegStruct2Array x11 = x10._super;
  BoundLayout<NondetRegLayout> x12 = LAYOUT_LOOKUP(x2, _super);
  // Top(zirgen/dsl/examples/calculator.zir:40)
  Val x13 = x7._super;
  // Top(zirgen/dsl/examples/calculator.zir:41)
  Val x14 = x9._super;
  if (to_size_t(x11[0]._super)) {
    // Top(zirgen/dsl/examples/calculator.zir:43)
    NondetRegStruct x15 = exec_Reg(ctx, (x13 + x14), LAYOUT_LOOKUP(x2, arm0));
  }
  if (to_size_t(x11[1]._super)) {
    // Top(zirgen/dsl/examples/calculator.zir:44)
    NondetRegStruct x16 = exec_Reg(ctx, (x13 - x14), LAYOUT_LOOKUP(x2, arm1));
  }
  // Top(zirgen/dsl/examples/calculator.zir:42)
  NondetRegStruct x17 = back_Reg(ctx, 0, x12);
  // Log(<preamble>:16)
  // Top(zirgen/dsl/examples/calculator.zir:46)
  INVOKE_EXTERN(ctx, log, "result = %u", std::initializer_list<Val>{x17._super});
  // Top(zirgen/dsl/examples/calculator.zir:42)
  NondetRegStruct x18 = back_Reg(ctx, 0, x12);
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:29)
  // Top(zirgen/dsl/examples/calculator.zir:47)
  BoundLayout<_globalLayout> x19 = BIND_LAYOUT(kLayoutGlobal, global1);
  // SetGlobalResult(zirgen/dsl/examples/calculator.zir:30)
  BoundLayout<NondetRegLayout> x20 = LAYOUT_LOOKUP(x19, result);
  NondetRegStruct x21 = exec_Reg(ctx, x18._super, x20);
  // GetGlobalResult(zirgen/dsl/examples/calculator.zir:25)
  // Top(zirgen/dsl/examples/calculator.zir:48)
  NondetRegStruct x22 = back_Reg(ctx, 0, x20);
  // Top(zirgen/dsl/examples/calculator.zir:42)
  NondetRegStruct x23 = back_Reg(ctx, 0, x12);
  // Top(zirgen/dsl/examples/calculator.zir:48)
  EQZ((x22._super - x23._super), "Top(zirgen/dsl/examples/calculator.zir:48)");
  // Top(zirgen/dsl/examples/calculator.zir:42)
  NondetRegStruct x24 = back_Reg(ctx, 0, x12);
  // OutputToUser(zirgen/dsl/examples/calculator.zir:6)
  // Top(zirgen/dsl/examples/calculator.zir:49)
  INVOKE_EXTERN(ctx, outputToUser, x24._super);
  // GetCycle(zirgen/dsl/examples/calculator.zir:7)
  // Top(zirgen/dsl/examples/calculator.zir:51)
  Val x25 = INVOKE_EXTERN(ctx, getCycle);
  // Top(zirgen/dsl/examples/calculator.zir:52)
  Val x26 = INVOKE_EXTERN(ctx, getCycle);
  Val x27 = bitAnd(x26, Val(1));
  // Top(zirgen/dsl/examples/calculator.zir:53)
  NondetRegStruct x28 = exec_Reg(ctx, Val(1), LAYOUT_LOOKUP(layout0, _1));
  NondetRegStruct x29 = exec_Reg(ctx, Val(0), LAYOUT_LOOKUP(layout0, _2));
  if (to_size_t(x28._super)) {
    // Top(zirgen/dsl/examples/calculator.zir:54)
    ComponentStruct x30 =
        exec_MyArgument(ctx, (x25 - x27), ((x27 * Val(2)) - Val(1)), LAYOUT_LOOKUP(x3, arm0));
  }
  if (to_size_t(x29._super)) {
    // Top(zirgen/dsl/examples/calculator.zir:55)
    NondetRegStruct x31 = exec_Reg(ctx, Val(0), LAYOUT_LOOKUP(x3, arm1));
  }
  return TopStruct{};
}
void step_Top(ExecContext& ctx, MutableBuf data0, GlobalBuf global1) {
  // Top(zirgen/dsl/examples/calculator.zir:38)
  BoundLayout<TopLayout> x2 = BIND_LAYOUT(kLayout_Top, data0);
  TopStruct x3 = exec_Top(ctx, x2, global1);
  return;
}
ComponentStruct exec_TopAccum(ExecContext& ctx,
                              BoundLayout<TopLayout> arg0,
                              BoundLayout<Reg1LayoutArray> layout1,
                              GlobalBuf mix2) {
  BoundLayout<_mixLayout> x3 = BIND_LAYOUT(kLayoutMix, mix2);
  BoundLayout<_accumLayout> x4 = LAYOUT_LOOKUP(x3, randomness);
  BoundLayout<Top__0Layout> x5 = LAYOUT_LOOKUP(arg0, _3);
  BoundLayout<NondetRegLayout2LayoutArray> x6 = LAYOUT_LOOKUP(x5, _selector);
  BoundLayout<MyArgumentLayout> x7 = LAYOUT_LOOKUP(x5, arm0);
  if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 0), _super), 0))) {
    ExtVal x8 = ((LOAD_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, myArgument), b), 0) *
                  LOAD_AS_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, b), _super), 0)) +
                 LOAD_EXT(LAYOUT_LOOKUP(x4, _offset), 0));
    STORE_EXT(LAYOUT_SUBSCRIPT(layout1, 0),
              (LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1) +
               (LOAD_AS_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, a), _super), 0) * inv(x8))));
    EQZ((((LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1)) *
          x8) -
         LOAD_AS_EXT(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, a), _super), 0)),
        "Top(zirgen/dsl/examples/calculator.zir:38)");
  }
  if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 1), _super), 0))) {
    STORE_EXT(LAYOUT_SUBSCRIPT(layout1, 0), LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1));
    EQZ((LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1)),
        "Top(zirgen/dsl/examples/calculator.zir:38)");
  }
  return ComponentStruct{};
}
void step_TopAccum(ExecContext& ctx, MutableBuf accum0, MutableBuf data1, GlobalBuf mix2) {
  BoundLayout<TopLayout> x3 = BIND_LAYOUT(kLayout_Top, data1);
  BoundLayout<Reg1LayoutArray> x4 = BIND_LAYOUT(kLayout_TopAccum, accum0);
  ComponentStruct x5 = exec_TopAccum(ctx, x3, x4, mix2);
  return;
}
