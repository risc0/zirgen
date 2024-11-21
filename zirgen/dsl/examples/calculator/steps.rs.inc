pub fn back_nondet_reg<'a>(
    ctx: &'a ExecContext,
    distance0: Index,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct<'a>> {
    let x2: NondetRegStruct = NondetRegStruct {
        _super: (layout1.map(|c| c._super)).load(ctx, distance0),
        _layout: layout1,
    };
    return Ok(x2);
}
pub fn exec_nondet_reg<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct<'a>> {
    let x2: BoundLayout<Reg, _> = (layout1.map(|c| c._super));
    x2.store(ctx, arg0);
    return Ok(NondetRegStruct {
        _super: x2.load(ctx, 0),
        _layout: layout1,
    });
}
pub fn back_reg<'a>(
    ctx: &'a ExecContext,
    distance0: Index,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct<'a>> {
    // Reg(<preamble>:4)
    let x2: NondetRegStruct = back_nondet_reg(ctx, distance0, layout1)?;
    return Ok(x2);
}
pub fn exec_reg<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct<'a>> {
    let x2: NondetRegStruct = exec_nondet_reg(ctx, arg0, layout1)?;
    // Reg(<preamble>:5)
    eqz!((arg0 - x2._super), "Reg(<preamble>:5)");
    return Ok(x2);
}
pub fn exec_one_hot<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, OneHotLayout, Val>,
) -> Result<OneHotStruct<'a>> {
    // OneHot(zirgen/dsl/examples/calculator/calculator.zir:13)
    let x2: NondetRegStruct2Array = map_layout(
        [Val::new(0), Val::new(1)],
        (layout1.map(|c| c._super)),
        |x3, x4| {
            let x5: NondetRegStruct = exec_nondet_reg(ctx, isz((x3 - arg0))?, x4)?;
            return Ok(x5);
        },
    )?;
    // OneHot(zirgen/dsl/examples/calculator/calculator.zir:15)
    let x6: Val = x2[to_usize(Val::new(0))]._super;
    eqz!(
        (x6 * (Val::new(1) - x6)),
        "OneHot(zirgen/dsl/examples/calculator/calculator.zir:15)"
    );
    let x7: Val = x2[to_usize(Val::new(1))]._super;
    eqz!(
        (x7 * (Val::new(1) - x7)),
        "OneHot(zirgen/dsl/examples/calculator/calculator.zir:15)"
    );
    // OneHot(zirgen/dsl/examples/calculator/calculator.zir:17)
    eqz!(
        ((x6 + x7) - Val::new(1)),
        "OneHot(zirgen/dsl/examples/calculator/calculator.zir:17)"
    );
    // OneHot(zirgen/dsl/examples/calculator/calculator.zir:19)
    eqz!(
        (x7 - arg0),
        "OneHot(zirgen/dsl/examples/calculator/calculator.zir:19)"
    );
    return Ok(OneHotStruct { _super: x2 });
}
pub fn exec_top<'a>(
    ctx: &'a ExecContext,
    layout0: BoundLayout<'a, TopLayout, Val>,
    global1: BufferRow<Val>,
) -> Result<TopStruct> {
    // Top(zirgen/dsl/examples/calculator/calculator.zir:41)
    let x2: BoundLayout<NondetRegLayout, _> = (layout0.map(|c| c.arg1));
    // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
    let x3: BoundLayout<TopResultLayout, _> = (layout0.map(|c| c.result));
    // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:40)
    let x4: Val = invoke_extern!(ctx, get_val_from_user);
    let x5: NondetRegStruct = exec_nondet_reg(ctx, x4, (layout0.map(|c| c.op)))?;
    // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:41)
    let x6: Val = invoke_extern!(ctx, get_val_from_user);
    let x7: NondetRegStruct = exec_nondet_reg(ctx, x6, x2)?;
    // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:42)
    let x8: Val = invoke_extern!(ctx, get_val_from_user);
    let x9: NondetRegStruct = exec_nondet_reg(ctx, x8, (layout0.map(|c| c.arg2)))?;
    // GetValFromUser(zirgen/dsl/examples/calculator/calculator.zir:5)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:43)
    let x10: Val = invoke_extern!(ctx, get_val_from_user);
    let x11: NondetRegStruct = exec_nondet_reg(ctx, x10, (layout0.map(|c| c.expected)))?;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:40)
    let x12: Val = x5._super;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
    let x13: OneHotStruct = exec_one_hot(ctx, x12, (layout0.map(|c| c._0)))?;
    let x14: NondetRegStruct2Array = x13._super;
    let x15: BoundLayout<NondetRegLayout, _> = (x3.map(|c| c._super));
    // Top(zirgen/dsl/examples/calculator/calculator.zir:41)
    let x16: Val = x7._super;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:42)
    let x17: Val = x9._super;
    let x18: NondetRegStruct;
    if is_true(x14[to_usize(Val::new(0))]._super) {
        // Top(zirgen/dsl/examples/calculator/calculator.zir:46)
        let x19: NondetRegStruct = exec_reg(ctx, (x16 + x17), (x3.map(|c| c.arm0)))?;
        x18 = x19;
    } else if is_true(x14[to_usize(Val::new(1))]._super) {
        // Top(zirgen/dsl/examples/calculator/calculator.zir:48)
        let x20: NondetRegStruct = exec_reg(ctx, (x16 - x17), (x3.map(|c| c.arm1)))?;
        x18 = x20;
    } else {
        bail!("Reached unreachable mux arm")
    } // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
    let x21: NondetRegStruct = back_reg(ctx, 0, x15)?;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:43)
    let x22: Val = x11._super;
    // Log(<preamble>:22)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:50)
    invoke_extern!(
        ctx,
        log,
        "op = %u, arg1 = %u arg2 = %u result = %u expected = %u",
        [x12, x16, x17, x21._super, x22]
    );
    // BackOne(zirgen/dsl/examples/calculator/calculator.zir:36)
    // BackOneFoo(zirgen/dsl/examples/calculator/calculator.zir:32)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:51)
    let x23: NondetRegStruct = back_nondet_reg(ctx, 1, x7._layout)?;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:52)
    let x24: NondetRegStruct = back_nondet_reg(ctx, 1, x2)?;
    eqz!(
        (x23._super - x24._super),
        "Top(zirgen/dsl/examples/calculator/calculator.zir:52)"
    );
    // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
    let x25: NondetRegStruct = back_reg(ctx, 0, x15)?;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:53)
    eqz!(
        (x25._super - x22),
        "Top(zirgen/dsl/examples/calculator/calculator.zir:53)"
    );
    // SetGlobalResult(zirgen/dsl/examples/calculator/calculator.zir:23)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:54)
    let x26: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // SetGlobalResult(zirgen/dsl/examples/calculator/calculator.zir:24)
    let x27: NondetRegStruct = exec_reg(ctx, Val::new(5), (x26.map(|c| c.result)))?;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:44)
    let x28: NondetRegStruct = back_reg(ctx, 0, x15)?;
    // OutputToUser(zirgen/dsl/examples/calculator/calculator.zir:6)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:55)
    invoke_extern!(ctx, output_to_user, x28._super);
    return Ok(TopStruct {});
}
pub fn step_top<'a>(
    ctx: &'a ExecContext,
    data0: BufferRow<Val>,
    global1: BufferRow<Val>,
) -> Result<()> {
    // Top(zirgen/dsl/examples/calculator/calculator.zir:39)
    let x2: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data0);
    let x3: TopStruct = exec_top(ctx, x2, global1)?;
    return Ok(());
}
