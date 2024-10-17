pub fn exec_nondet_reg(
    ctx: &ExecContext,
    arg0: Val,
    layout1: &BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<NondetRegStruct> {
    let x2: &BoundLayout<Reg, _> = (&layout1.map(|c| c._super));
    x2.store(ctx, arg0);
    return Ok(NondetRegStruct {
        _super: x2.load(ctx, 0),
    });
}
pub fn exec_reg(
    ctx: &ExecContext,
    arg0: Val,
    layout1: &BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<NondetRegStruct> {
    // Reg(<preamble>:4)
    let x2: NondetRegStruct = exec_nondet_reg(ctx, arg0, layout1)?;
    // Reg(<preamble>:5)
    eqz!((arg0 - x2._super), "Reg(<preamble>:5)");
    return Ok(x2);
}
pub fn exec_my_argument(
    ctx: &ExecContext,
    arg0: Val,
    arg1: Val,
    layout2: &BoundLayout<MyArgumentLayout, impl BufferRow<ValType = Val>>,
) -> Result<MyArgumentStruct> {
    // MyArgument(zirgen/dsl/test/simple-accum.zir:4)
    let x3: NondetRegStruct = exec_nondet_reg(ctx, arg0, (&layout2.map(|c| c.a)))?;
    // MyArgument(zirgen/dsl/test/simple-accum.zir:5)
    let x4: NondetRegStruct = exec_nondet_reg(ctx, arg1, (&layout2.map(|c| c.b)))?;
    return Ok(MyArgumentStruct {});
}
pub fn exec_top(
    ctx: &ExecContext,
    layout0: &BoundLayout<TopLayout, impl BufferRow<ValType = Val>>,
    global1: &impl BufferRow<ValType = Val>,
) -> Result<TopStruct> {
    // Top(zirgen/dsl/test/simple-accum.zir:8)
    let x2: &BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // Top(zirgen/dsl/test/simple-accum.zir:11)
    let x3: &BoundLayout<Top__0Layout, _> = (&layout0.map(|c| c._1));
    // Top(zirgen/dsl/test/simple-accum.zir:9)
    let x4: NondetRegStruct = exec_reg(ctx, Val::new(0), (&x2.map(|c| c.globzero)))?;
    // GetCycle(zirgen/dsl/test/simple-accum.zir:1)
    // Top(zirgen/dsl/test/simple-accum.zir:10)
    let x5: Val = invoke_extern!(ctx, get_cycle);
    let x6: NondetRegStruct =
        exec_nondet_reg(ctx, bit_and(x5, Val::new(1))?, (&layout0.map(|c| c.odd)))?;
    let x7: Val = x6._super;
    // Top(zirgen/dsl/test/simple-accum.zir:11)
    let x8: NondetRegStruct = exec_reg(ctx, (Val::new(1) - x7), (&layout0.map(|c| c._0)))?;
    if is_nonzero(x7) {
        let x9: MyArgumentStruct =
            exec_my_argument(ctx, Val::new(1), Val::new(1), (&x3.map(|c| c.arm0)))?;
    }
    if is_nonzero(x8._super) {
        let x10: MyArgumentStruct = exec_my_argument(
            ctx,
            Val::new(2013265920),
            Val::new(1),
            (&x3.map(|c| c.arm1)),
        )?;
    }
    return Ok(TopStruct {});
}
pub fn step_top(
    ctx: &ExecContext,
    data0: &impl BufferRow<ValType = Val>,
    global1: &impl BufferRow<ValType = Val>,
) -> Result<()> {
    // Top(zirgen/dsl/test/simple-accum.zir:8)
    let x2: &BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data0);
    let x3: TopStruct = exec_top(ctx, x2, global1)?;
    return Ok(());
}
pub fn exec_top_accum(
    ctx: &ExecContext,
    arg0: &BoundLayout<TopLayout, impl BufferRow<ValType = Val>>,
    layout1: &BoundLayout<Reg1LayoutArray, impl BufferRow<ValType = Val>>,
    mix2: &impl BufferRow<ValType = Val>,
) -> Result<ComponentStruct> {
    // zirgen/dsl/passes/GenerateAccum.cpp:401
    let x3: &BoundLayout<_mixLayout, _> = bind_layout!(LAYOUT_MIX, mix2);
    // zirgen/dsl/passes/GenerateAccum.cpp:408
    let x4: &BoundLayout<_accumLayout, _> = (&x3.map(|c| c.randomness));
    // zirgen/dsl/passes/GenerateAccum.cpp:414
    let x5: &BoundLayout<Top__0Layout, _> = (&arg0.map(|c| c._1));
    // zirgen/dsl/passes/GenerateAccum.cpp:423
    let x6: &BoundLayout<NondetRegLayout2LayoutArray, _> = (&x5.map(|c| c._selector));
    // zirgen/dsl/passes/GenerateAccum.cpp:451
    let x7: &BoundLayout<MyArgumentLayout, _> = (&x5.map(|c| c.arm0));
    // zirgen/dsl/passes/GenerateAccum.cpp:50
    let x8: &BoundLayout<Reg, _> = (&(&x4.map(|c| c.my_argument)).map(|c| c.b));
    // zirgen/dsl/passes/GenerateAccum.cpp:81
    let x9: &BoundLayout<Reg, _> = (&x4.map(|c| c._offset));
    if is_nonzero((&(&x6.map(|c| c[to_usize(0)])).map(|c| c._super)).load(ctx, 0)) {
        // zirgen/dsl/passes/GenerateAccum.cpp:234
        let x10: ExtVal = ((x8.load_ext::<ExtVal>(ctx, 0)
            * (&(&x7.map(|c| c.b)).map(|c| c._super)).load(ctx, 0))
            + x9.load_ext::<ExtVal>(ctx, 0));
        // zirgen/dsl/passes/GenerateAccum.cpp:185
        (&layout1.map(|c| c[to_usize(0)])).store_ext(
            ctx,
            ((&layout1.map(|c| c[to_usize(0)])).load_ext::<ExtVal>(ctx, 1)
                + ((&(&x7.map(|c| c.a)).map(|c| c._super)).load(ctx, 0) * inv(x10)?)),
        );
        // zirgen/dsl/passes/GenerateAccum.cpp:178
        eqz!(
            ((((&layout1.map(|c| c[to_usize(0)])).load_ext::<ExtVal>(ctx, 0)
                - (&layout1.map(|c| c[to_usize(0)])).load_ext::<ExtVal>(ctx, 1))
                * x10)
                - (&(&x7.map(|c| c.a)).map(|c| c._super)).load(ctx, 0)),
            "zirgen/dsl/passes/GenerateAccum.cpp:178"
        );
    }
    // zirgen/dsl/passes/GenerateAccum.cpp:451
    let x11: &BoundLayout<MyArgumentLayout, _> = (&x5.map(|c| c.arm1));
    if is_nonzero((&(&x6.map(|c| c[to_usize(1)])).map(|c| c._super)).load(ctx, 0)) {
        // zirgen/dsl/passes/GenerateAccum.cpp:234
        let x12: ExtVal = ((x8.load_ext::<ExtVal>(ctx, 0)
            * (&(&x11.map(|c| c.b)).map(|c| c._super)).load(ctx, 0))
            + x9.load_ext::<ExtVal>(ctx, 0));
        // zirgen/dsl/passes/GenerateAccum.cpp:185
        (&layout1.map(|c| c[to_usize(0)])).store_ext(
            ctx,
            ((&layout1.map(|c| c[to_usize(0)])).load_ext::<ExtVal>(ctx, 1)
                + ((&(&x11.map(|c| c.a)).map(|c| c._super)).load(ctx, 0) * inv(x12)?)),
        );
        // zirgen/dsl/passes/GenerateAccum.cpp:178
        eqz!(
            ((((&layout1.map(|c| c[to_usize(0)])).load_ext::<ExtVal>(ctx, 0)
                - (&layout1.map(|c| c[to_usize(0)])).load_ext::<ExtVal>(ctx, 1))
                * x12)
                - (&(&x11.map(|c| c.a)).map(|c| c._super)).load(ctx, 0)),
            "zirgen/dsl/passes/GenerateAccum.cpp:178"
        );
    }
    return Ok(ComponentStruct {});
}
pub fn step_top_accum(
    ctx: &ExecContext,
    accum0: &impl BufferRow<ValType = Val>,
    data1: &impl BufferRow<ValType = Val>,
    mix2: &impl BufferRow<ValType = Val>,
) -> Result<()> {
    // zirgen/dsl/passes/GenerateAccum.cpp:401
    let x3: &BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data1);
    let x4: &BoundLayout<Reg1LayoutArray, _> = bind_layout!(LAYOUT_TOP_ACCUM, accum0);
    let x5: ComponentStruct = exec_top_accum(ctx, x3, x4, mix2)?;
    return Ok(());
}
