NondetRegStruct
back_NondetReg(ExecContext& ctx, Index distance0, BoundLayout<NondetRegLayout> layout1) {
  NondetRegStruct x2 = NondetRegStruct{._super = LOAD(LAYOUT_LOOKUP(layout1, _super), distance0),
                                       ._layout = layout1};
  return x2;
}
NondetRegStruct exec_NondetReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  BoundLayout<Reg> x2 = LAYOUT_LOOKUP(layout1, _super);
  STORE(x2, arg0);
  return NondetRegStruct{._super = LOAD(x2, 0), ._layout = layout1};
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
  // OneHot(zirgen/dsl/examples/calculator/calculator.zir:13)
  NondetRegStruct2Array x2 =
      map(Val2Array{Val(0), Val(1)},
          LAYOUT_LOOKUP(layout1, _super),
          std::function([&](Val2Array::value_type x3,
                            BoundLayout<NondetRegLayout2LayoutArray::value_type> x4) {
            NondetRegStruct x5 = exec_NondetReg(ctx, isz((x3 - arg0)), x4);
            return x5;
          }));
  // OneHot(zirgen/dsl/examples/calculator/calculator.zir:15)
  Val x6 = x2[0]._super;
  EQZ((x6 * (Val(1) - x6)), "OneHot(zirgen/dsl/examples/calculator/calculator.zir:15)");
  Val x7 = x2[1]._super;
  EQZ((x7 * (Val(1) - x7)), "OneHot(zirgen/dsl/examples/calculator/calculator.zir:15)");
  // OneHot(zirgen/dsl/examples/calculator/calculator.zir:17)
  EQZ(((x6 + x7) - Val(1)), "OneHot(zirgen/dsl/examples/calculator/calculator.zir:17)");
  // OneHot(zirgen/dsl/examples/calculator/calculator.zir:19)
  EQZ((x7 - arg0), "OneHot(zirgen/dsl/examples/calculator/calculator.zir:19)");
  return OneHotStruct{._super = x2};
}
TopStruct exec_Top(ExecContext& ctx, BoundLayout<TopLayout> layout0, GlobalBuf global1) {
  // Top(zirgen/dsl/examples/calculator/calculator.zir:41)
  BoundLayout<NondetRegLayout> x2 = LAYOUT_LOOKUP(layout0, arg1);
  // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
  BoundLayout<TopResultLayout> x3 = LAYOUT_LOOKUP(layout0, result);
  // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:40)
  Val x4 = INVOKE_EXTERN(ctx, getValFromUser);
  NondetRegStruct x5 = exec_NondetReg(ctx, x4, LAYOUT_LOOKUP(layout0, op));
  // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:41)
  Val x6 = INVOKE_EXTERN(ctx, getValFromUser);
  NondetRegStruct x7 = exec_NondetReg(ctx, x6, x2);
  // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:42)
  Val x8 = INVOKE_EXTERN(ctx, getValFromUser);
  NondetRegStruct x9 = exec_NondetReg(ctx, x8, LAYOUT_LOOKUP(layout0, arg2));
  // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:43)
  Val x10 = INVOKE_EXTERN(ctx, getValFromUser);
  NondetRegStruct x11 = exec_NondetReg(ctx, x10, LAYOUT_LOOKUP(layout0, expected));
  // Top(zirgen/dsl/examples/calculator/calculator.zir:40)
  Val x12 = x5._super;
  // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
  OneHotStruct x13 = exec_OneHot(ctx, x12, LAYOUT_LOOKUP(layout0, _0));
  NondetRegStruct2Array x14 = x13._super;
  BoundLayout<NondetRegLayout> x15 = LAYOUT_LOOKUP(x3, _super);
  // Top(zirgen/dsl/examples/calculator/calculator.zir:41)
  Val x16 = x7._super;
  // Top(zirgen/dsl/examples/calculator/calculator.zir:42)
  Val x17 = x9._super;
  NondetRegStruct x18;
  if (to_size_t(x14[0]._super)) {
    // Top(zirgen/dsl/examples/calculator/calculator.zir:46)
    NondetRegStruct x19 = exec_Reg(ctx, (x16 + x17), LAYOUT_LOOKUP(x3, arm0));
    x18 = x19;
  } else if (to_size_t(x14[1]._super)) {
    // Top(zirgen/dsl/examples/calculator/calculator.zir:48)
    NondetRegStruct x20 = exec_Reg(ctx, (x16 - x17), LAYOUT_LOOKUP(x3, arm1));
    x18 = x20;
  } else {
    assert(0 && "Reached unreachable mux arm");
  }
  // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
  NondetRegStruct x21 = back_Reg(ctx, 0, x15);
  // Top(zirgen/dsl/examples/calculator/calculator.zir:43)
  Val x22 = x11._super;
  // Log(<preamble>:16)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:50)
  INVOKE_EXTERN(ctx,
                log,
                "op = %u, arg1 = %u arg2 = %u result = %u expected = %u",
                std::initializer_list<Val>{x12, x16, x17, x21._super, x22});
  // BackOne(zirgen/dsl/examples/calculator/calculator.zir:36)
  // BackOneFoo(zirgen/dsl/examples/calculator/calculator.zir:32)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:51)
  NondetRegStruct x23 = back_NondetReg(ctx, 1, x7._layout);
  // Top(zirgen/dsl/examples/calculator/calculator.zir:52)
  NondetRegStruct x24 = back_NondetReg(ctx, 1, x2);
  EQZ((x23._super - x24._super), "Top(zirgen/dsl/examples/calculator/calculator.zir:52)");
  // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
  NondetRegStruct x25 = back_Reg(ctx, 0, x15);
  // Top(zirgen/dsl/examples/calculator/calculator.zir:53)
  EQZ((x25._super - x22), "Top(zirgen/dsl/examples/calculator/calculator.zir:53)");
  // SetGlobalResult(zirgen/dsl/examples/calculator/calculator.zir:23)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:54)
  BoundLayout<_globalLayout> x26 = BIND_LAYOUT(kLayoutGlobal, global1);
  // SetGlobalResult(zirgen/dsl/examples/calculator/calculator.zir:24)
  NondetRegStruct x27 = exec_Reg(ctx, Val(5), LAYOUT_LOOKUP(x26, result));
  // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
  NondetRegStruct x28 = back_Reg(ctx, 0, x15);
  // OutputToUser(zirgen/dsl/examples/calculator/calculator.zir:6)
  // Top(zirgen/dsl/examples/calculator/calculator.zir:55)
  INVOKE_EXTERN(ctx, outputToUser, x28._super);
  return TopStruct{};
}
void step_Top(ExecContext& ctx, MutableBuf data0, GlobalBuf global1) {
  // Top(zirgen/dsl/examples/calculator/calculator.zir:39)
  BoundLayout<TopLayout> x2 = BIND_LAYOUT(kLayout_Top, data0);
  TopStruct x3 = exec_Top(ctx, x2, global1);
  return;
}
