NondetRegStruct exec_NondetReg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  BoundLayout<Reg> x2 = LAYOUT_LOOKUP(layout1, _super);
  STORE(x2, arg0);
  return NondetRegStruct{._super = LOAD(x2, 0)};
}
NondetRegStruct exec_Reg(ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  // Reg(<preamble>:4)
  NondetRegStruct x2 = exec_NondetReg(ctx, arg0, layout1);
  // Reg(<preamble>:5)
  EQZ((arg0 - x2._super), "Reg(<preamble>:5)");
  return x2;
}
NondetMyArgumentStruct exec_NondetMyArgument(ExecContext& ctx,
                                             Val arg0,
                                             Val arg1,
                                             BoundLayout<NondetMyArgumentLayout> layout2) {
  // NondetMyArgument(zirgen/dsl/test/simple-accum.zir:5)
  NondetRegStruct x3 = exec_NondetReg(ctx, arg0, LAYOUT_LOOKUP(layout2, a));
  // NondetMyArgument(zirgen/dsl/test/simple-accum.zir:6)
  NondetRegStruct x4 = exec_NondetReg(ctx, arg1, LAYOUT_LOOKUP(layout2, b));
  return NondetMyArgumentStruct{.a = x3, .b = x4};
}
MyArgumentStruct
exec_MyArgument(ExecContext& ctx, Val arg0, Val arg1, BoundLayout<NondetMyArgumentLayout> layout2) {
  // MyArgument(zirgen/dsl/test/simple-accum.zir:10)
  NondetMyArgumentStruct x3 = exec_NondetMyArgument(ctx, arg0, arg1, layout2);
  // MyArgument(zirgen/dsl/test/simple-accum.zir:11)
  EQZ((arg0 - x3.a._super), "MyArgument(zirgen/dsl/test/simple-accum.zir:11)");
  // MyArgument(zirgen/dsl/test/simple-accum.zir:12)
  EQZ((arg1 - x3.b._super), "MyArgument(zirgen/dsl/test/simple-accum.zir:12)");
  return MyArgumentStruct{};
}
TopStruct exec_Top(ExecContext& ctx, BoundLayout<TopLayout> layout0, GlobalBuf global1) {
  // Top(zirgen/dsl/test/simple-accum.zir:16)
  BoundLayout<_globalLayout> x2 = BIND_LAYOUT(kLayoutGlobal, global1);
  // Top(zirgen/dsl/test/simple-accum.zir:19)
  BoundLayout<Top__0Layout> x3 = LAYOUT_LOOKUP(layout0, _1);
  // Top(zirgen/dsl/test/simple-accum.zir:17)
  NondetRegStruct x4 = exec_Reg(ctx, Val(11), LAYOUT_LOOKUP(x2, globzero));
  // GetCycle(zirgen/dsl/test/simple-accum.zir:2)
  // Top(zirgen/dsl/test/simple-accum.zir:18)
  Val x5 = INVOKE_EXTERN(ctx, getCycle);
  NondetRegStruct x6 = exec_NondetReg(ctx, bitAnd(x5, Val(1)), LAYOUT_LOOKUP(layout0, odd));
  Val x7 = x6._super;
  // Top(zirgen/dsl/test/simple-accum.zir:19)
  NondetRegStruct x8 = exec_Reg(ctx, (Val(1) - x7), LAYOUT_LOOKUP(layout0, _0));
  if (to_size_t(x7)) {
    MyArgumentStruct x9 = exec_MyArgument(ctx, Val(2), Val(3), LAYOUT_LOOKUP(x3, arm0));
  }
  if (to_size_t(x8._super)) {
    MyArgumentStruct x10 = exec_MyArgument(ctx, Val(2013265919), Val(3), LAYOUT_LOOKUP(x3, arm1));
  }
  return TopStruct{};
}
void step_Top(ExecContext& ctx, MutableBuf data0, GlobalBuf global1) {
  // Top(zirgen/dsl/test/simple-accum.zir:16)
  BoundLayout<TopLayout> x2 = BIND_LAYOUT(kLayout_Top, data0);
  TopStruct x3 = exec_Top(ctx, x2, global1);
  return;
}
ComponentStruct exec_TopAccum(ExecContext& ctx,
                              BoundLayout<TopLayout> arg0,
                              BoundLayout<Reg1LayoutArray> layout1,
                              GlobalBuf mix2) {
  // zirgen/dsl/passes/GenerateAccum.cpp:400
  BoundLayout<_mixLayout> x3 = BIND_LAYOUT(kLayoutMix, mix2);
  // zirgen/dsl/passes/GenerateAccum.cpp:407
  BoundLayout<_accumLayout> x4 = LAYOUT_LOOKUP(x3, randomness);
  // zirgen/dsl/passes/GenerateAccum.cpp:413
  BoundLayout<Top__0Layout> x5 = LAYOUT_LOOKUP(arg0, _1);
  // zirgen/dsl/passes/GenerateAccum.cpp:422
  BoundLayout<NondetRegLayout2LayoutArray> x6 = LAYOUT_LOOKUP(x5, _selector);
  // zirgen/dsl/passes/GenerateAccum.cpp:450
  BoundLayout<NondetMyArgumentLayout> x7 = LAYOUT_LOOKUP(x5, arm0);
  // zirgen/dsl/passes/GenerateAccum.cpp:50
  BoundLayout<Reg> x8 = LAYOUT_LOOKUP(LAYOUT_LOOKUP(x4, nondetMyArgument), b);
  // zirgen/dsl/passes/GenerateAccum.cpp:80
  BoundLayout<Reg> x9 = LAYOUT_LOOKUP(x4, _offset);
  if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 0), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:233
    ExtVal x10 = ((LOAD_EXT(x8, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, b), _super), 0)) +
                  LOAD_EXT(x9, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:184
    STORE_EXT(LAYOUT_SUBSCRIPT(layout1, 0),
              (LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1) +
               (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, a), _super), 0) * inv(x10))));
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    EQZ((((LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1)) *
          x10) -
         LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x7, a), _super), 0)),
        "zirgen/dsl/passes/GenerateAccum.cpp:177");
  }
  // zirgen/dsl/passes/GenerateAccum.cpp:450
  BoundLayout<NondetMyArgumentLayout> x11 = LAYOUT_LOOKUP(x5, arm1);
  if (to_size_t(LOAD(LAYOUT_LOOKUP(LAYOUT_SUBSCRIPT(x6, 1), _super), 0))) {
    // zirgen/dsl/passes/GenerateAccum.cpp:233
    ExtVal x12 = ((LOAD_EXT(x8, 0) * LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, b), _super), 0)) +
                  LOAD_EXT(x9, 0));
    // zirgen/dsl/passes/GenerateAccum.cpp:184
    STORE_EXT(LAYOUT_SUBSCRIPT(layout1, 0),
              (LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1) +
               (LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, a), _super), 0) * inv(x12))));
    // zirgen/dsl/passes/GenerateAccum.cpp:177
    EQZ((((LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 0) - LOAD_EXT(LAYOUT_SUBSCRIPT(layout1, 0), 1)) *
          x12) -
         LOAD(LAYOUT_LOOKUP(LAYOUT_LOOKUP(x11, a), _super), 0)),
        "zirgen/dsl/passes/GenerateAccum.cpp:177");
  }
  return ComponentStruct{};
}
void step_TopAccum(ExecContext& ctx, MutableBuf accum0, MutableBuf data1, GlobalBuf mix2) {
  // zirgen/dsl/passes/GenerateAccum.cpp:400
  BoundLayout<TopLayout> x3 = BIND_LAYOUT(kLayout_Top, data1);
  BoundLayout<Reg1LayoutArray> x4 = BIND_LAYOUT(kLayout_TopAccum, accum0);
  ComponentStruct x5 = exec_TopAccum(ctx, x3, x4, mix2);
  return;
}
