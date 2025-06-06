pub fn calculator<'a>(
    ctx: &ValidityCtx,
    data0: BufferRow<Val>,
    global1: BufferRow<Val>,
) -> Result<MixState> {
    // All Constraints
    let x2: MixState = trivial_constraint()?;
    // builtin NondetReg
    // Top(zirgen/dsl/examples/calculator/calculator.zir:30)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:28)
    let x3: Val = get(ctx, data0, 1, 0)?;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:31)
    let x4: Val = get(ctx, data0, 2, 0)?;
    // OneHot(zirgen/dsl/examples/calculator/calculator.zir:13)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:33)
    let x5: Val = get(ctx, data0, 4, 0)?;
    let x6: Val = get(ctx, data0, 5, 0)?;
    // OneHot(zirgen/dsl/examples/calculator/calculator.zir:17)
    let x7: MixState = and_eqz(
        ctx,
        and_eqz(
            ctx,
            and_eqz(ctx, x2, (x5 * (Val::new(1) - x5)))?,
            (x6 * (Val::new(1) - x6)),
        )?,
        ((x5 + x6) - Val::new(1)),
    )?;
    // builtin NondetReg
    // Reg(<preamble>:5)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:35)
    let x8: Val = get(ctx, data0, 6, 0)?;
    // Top(zirgen/dsl/examples/calculator/calculator.zir:33)
    let x9: MixState = and_cond(
        and_cond(
            and_eqz(ctx, x7, (x6 - get(ctx, data0, 0, 0)?))?,
            x5,
            and_eqz(ctx, x2, ((x3 + x4) - x8))?,
        )?,
        x6,
        and_eqz(ctx, x2, ((x3 - x4) - x8))?,
    )?;
    // Reg(<preamble>:6)
    // SetGlobalResult(zirgen/dsl/examples/calculator/calculator.zir:24)
    // Top(zirgen/dsl/examples/calculator/calculator.zir:41)
    let x10: MixState = and_eqz(
        ctx,
        and_eqz(ctx, x9, (x8 - get(ctx, data0, 3, 0)?))?,
        (Val::new(5) - get_global(global1, 0)?),
    )?;
    return Ok(x10);
}
