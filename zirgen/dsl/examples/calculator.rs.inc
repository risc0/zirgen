set_field!(BabyBear);
define_buffer_list! {
all: [accum,code,data,global,mix,],
rows: [accum,code,data,],
taps: [accum,code,data,],
globals: [global,mix,],}
define_tap_buffer! {accum, /*count=*/1, /*groupId=*/0}
define_tap_buffer! {code, /*count=*/1, /*groupId=*/1}
define_tap_buffer! {data, /*count=*/6, /*groupId=*/2}
define_global_buffer! {global, /*count=*/4}
define_global_buffer! {mix, /*count=*/4}
pub type Tap8Array = [Tap; 8];
pub struct NondetRegLayout {
    pub _super: &'static Reg,
}
impl risc0_zkp::layout::Component for NondetRegLayout {
    fn ty_name(&self) -> &'static str {
        "NondetRegLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub type NondetRegLayout2LayoutArray = [&'static NondetRegLayout; 2];
pub struct OneHotLayout {
    pub _super: &'static NondetRegLayout2LayoutArray,
}
impl risc0_zkp::layout::Component for OneHotLayout {
    fn ty_name(&self) -> &'static str {
        "OneHotLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        Ok(())
    }
}
pub struct TopResultLayout {
    pub _super: &'static NondetRegLayout,
    pub arm0: &'static NondetRegLayout,
    pub arm1: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for TopResultLayout {
    fn ty_name(&self) -> &'static str {
        "TopResultLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub struct TopLayout {
    pub op: &'static NondetRegLayout,
    pub left: &'static NondetRegLayout,
    pub right: &'static NondetRegLayout,
    pub _0: &'static OneHotLayout,
    pub result: &'static TopResultLayout,
}
impl risc0_zkp::layout::Component for TopLayout {
    fn ty_name(&self) -> &'static str {
        "TopLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("op", self.op)?;
        v.visit_component("left", self.left)?;
        v.visit_component("right", self.right)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("result", self.result)?;
        Ok(())
    }
}
pub struct _globalLayout {
    pub result: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for _globalLayout {
    fn ty_name(&self) -> &'static str {
        "_globalLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("result", self.result)?;
        Ok(())
    }
}
#[derive(Copy, Clone, Debug)]
pub struct NondetRegStruct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct ComponentStruct {}
#[derive(Copy, Clone, Debug)]
pub struct RegStruct {
    pub _super: NondetRegStruct,
    pub reg: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct GetValFromUserStruct {
    pub _super: Val,
}
pub type Val2Array = [Val; 2];
pub type NondetRegStruct2Array = [NondetRegStruct; 2];
pub type ComponentStruct2Array = [ComponentStruct; 2];
#[derive(Copy, Clone, Debug)]
pub struct OneHot__0_SuperStruct {
    pub _super: Val,
}
pub type OneHot__0_SuperStruct2Array = [OneHot__0_SuperStruct; 2];
#[derive(Copy, Clone, Debug)]
pub struct OneHotStruct {
    pub _super: NondetRegStruct2Array,
    pub n: Val,
    pub bits: NondetRegStruct2Array,
}
#[derive(Copy, Clone, Debug)]
pub struct TopStruct {
    pub _super: ComponentStruct,
    pub op: NondetRegStruct,
    pub left: NondetRegStruct,
    pub right: NondetRegStruct,
    pub result: RegStruct,
}
pub const TAP_LIST: Tap8Array = [
    make_tap!(0, 0, 0),
    make_tap!(1, 0, 0),
    make_tap!(2, 0, 0),
    make_tap!(2, 1, 0),
    make_tap!(2, 2, 0),
    make_tap!(2, 3, 0),
    make_tap!(2, 4, 0),
    make_tap!(2, 5, 0),
];
pub const LAYOUT_TOP: &TopLayout = &TopLayout {
    op: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    left: &NondetRegLayout {
        _super: &Reg { offset: 1 },
    },
    right: &NondetRegLayout {
        _super: &Reg { offset: 2 },
    },
    _0: &OneHotLayout {
        _super: &[
            &NondetRegLayout {
                _super: &Reg { offset: 3 },
            },
            &NondetRegLayout {
                _super: &Reg { offset: 4 },
            },
        ],
    },
    result: &TopResultLayout {
        _super: &NondetRegLayout {
            _super: &Reg { offset: 5 },
        },
        arm0: &NondetRegLayout {
            _super: &Reg { offset: 5 },
        },
        arm1: &NondetRegLayout {
            _super: &Reg { offset: 5 },
        },
    },
};
pub const LAYOUT_GLOBAL: &_globalLayout = &_globalLayout {
    result: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
};
pub fn exec_isz(ctx: &ExecContext, arg0: Val) -> Result<Val> {
    return Ok(isz(arg0)?);
}
pub fn exec_add(ctx: &ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 + arg1));
}
pub fn exec_sub(ctx: &ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 - arg1));
}
pub fn exec_mul(ctx: &ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 * arg1));
}
pub fn back_nondet_reg(
    ctx: &ExecContext,
    distance0: Index,
    layout1: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<NondetRegStruct> {
    return Ok(NondetRegStruct {
        _super: load!(layout_lookup!(layout1, _super), distance0),
    });
}
pub fn exec_nondet_reg(
    ctx: &ExecContext,
    arg0: Val,
    layout1: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<NondetRegStruct> {
    let x2: BoundLayout<Reg, _> = layout_lookup!(layout1, _super);
    store!(x2, arg0);
    return Ok(NondetRegStruct {
        _super: load!(x2, 0),
    });
}
pub fn exec_component(ctx: &ExecContext) -> Result<ComponentStruct> {
    return Ok(ComponentStruct {});
}
pub fn back_reg(
    ctx: &ExecContext,
    distance0: Index,
    layout1: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<RegStruct> {
    // Reg(<preamble>:4)
    let x2: NondetRegStruct = back_nondet_reg(ctx, distance0, layout1)?;
    return Ok(RegStruct {
        _super: x2.clone(),
        reg: x2,
    });
}
pub fn exec_reg(
    ctx: &ExecContext,
    arg0: Val,
    layout1: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<RegStruct> {
    let x2: NondetRegStruct = exec_nondet_reg(ctx, arg0, layout1)?;
    // Reg(<preamble>:5)
    eqz!((arg0 - x2._super), "Reg(<preamble>:5)");
    return Ok(RegStruct {
        _super: x2.clone(),
        reg: x2,
    });
}
pub fn exec_log(ctx: &ExecContext, arg0: &str, arg1: &[Val]) -> Result<ComponentStruct> {
    // Log(<preamble>:16)
    invoke_extern!(ctx, log, arg0, arg1);
    return Ok(ComponentStruct {});
}
pub fn exec_get_val_from_user(ctx: &ExecContext) -> Result<GetValFromUserStruct> {
    // GetValFromUser(zirgen/dsl/examples/calculator.zir:5)
    let x0: Val = invoke_extern!(ctx, get_val_from_user);
    return Ok(GetValFromUserStruct { _super: x0 });
}
pub fn exec_output_to_user(ctx: &ExecContext, arg0: Val) -> Result<ComponentStruct> {
    // OutputToUser(zirgen/dsl/examples/calculator.zir:6)
    invoke_extern!(ctx, output_to_user, arg0);
    return Ok(ComponentStruct {});
}
pub fn exec_one_hot(
    ctx: &ExecContext,
    arg0: Val,
    layout1: BoundLayout<OneHotLayout, impl BufferRow<ValType = Val>>,
) -> Result<OneHotStruct> {
    // OneHot(zirgen/dsl/examples/calculator.zir:13)
    let x2: Val2Array = [Val::new(0), Val::new(1)];
    let x3: NondetRegStruct2Array = map_layout(x2, layout_lookup!(layout1, _super), |x4, x5| {
        let x6: Val = exec_sub(ctx, x4, arg0)?;
        let x7: Val = exec_isz(ctx, x6)?;
        let x8: NondetRegStruct = exec_nondet_reg(ctx, x7, x5)?;
        return Ok(x8);
    })?;
    // OneHot(zirgen/dsl/examples/calculator.zir:15)
    let x9: ComponentStruct2Array = map(x3, |x10| {
        let x11: Val = x10._super;
        let x12: Val = exec_sub(ctx, Val::new(1), x11)?;
        let x13: Val = exec_mul(ctx, x11, x12)?;
        eqz!(x13, "OneHot(zirgen/dsl/examples/calculator.zir:15)");
        let x14: ComponentStruct = exec_component(ctx)?;
        return Ok(x14);
    })?;
    // OneHot(zirgen/dsl/examples/calculator.zir:17)
    let x15: Val = reduce(x3, Val::new(0), |x16, x17| {
        let x18: Val = exec_add(ctx, x16, x17._super)?;
        return Ok(x18);
    })?;
    eqz!(
        (x15 - Val::new(1)),
        "OneHot(zirgen/dsl/examples/calculator.zir:17)"
    );
    // OneHot(zirgen/dsl/examples/calculator.zir:19)
    let x19: OneHot__0_SuperStruct2Array = map(x2, |x20| {
        let x21: Val = exec_mul(ctx, x3[u64::from(x20) as usize]._super, x20)?;
        return Ok(OneHot__0_SuperStruct { _super: x21 });
    })?;
    let x22: Val = reduce(x19, Val::new(0), |x23, x24| {
        let x25: Val = exec_add(ctx, x23, x24._super)?;
        return Ok(x25);
    })?;
    eqz!(
        (x22 - arg0),
        "OneHot(zirgen/dsl/examples/calculator.zir:19)"
    );
    return Ok(OneHotStruct {
        _super: x3.clone(),
        n: Val::new(2),
        bits: x3,
    });
}
pub fn exec_get_global_result(
    ctx: &ExecContext,
    global0: &impl BufferRow<ValType = Val>,
) -> Result<RegStruct> {
    // GetGlobalResult(zirgen/dsl/examples/calculator.zir:23)
    let x1: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global0);
    // GetGlobalResult(zirgen/dsl/examples/calculator.zir:24)
    let x2: RegStruct = back_reg(ctx, 0, layout_lookup!(x1, result))?;
    return Ok(x2);
}
pub fn exec_set_global_result(
    ctx: &ExecContext,
    arg0: Val,
    global1: &impl BufferRow<ValType = Val>,
) -> Result<ComponentStruct> {
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:29)
    let x3: RegStruct = exec_reg(ctx, arg0, layout_lookup!(x2, result))?;
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    let x4: ComponentStruct = exec_component(ctx)?;
    return Ok(x4);
}
pub fn exec_top(
    ctx: &ExecContext,
    layout0: BoundLayout<TopLayout, impl BufferRow<ValType = Val>>,
    global1: &impl BufferRow<ValType = Val>,
) -> Result<TopStruct> {
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x2: BoundLayout<TopResultLayout, _> = layout_lookup!(layout0, result);
    // Top(zirgen/dsl/examples/calculator.zir:33)
    let x3: GetValFromUserStruct = exec_get_val_from_user(ctx)?;
    let x4: NondetRegStruct = exec_nondet_reg(ctx, x3._super, layout_lookup!(layout0, op))?;
    // Top(zirgen/dsl/examples/calculator.zir:34)
    let x5: GetValFromUserStruct = exec_get_val_from_user(ctx)?;
    let x6: NondetRegStruct = exec_nondet_reg(ctx, x5._super, layout_lookup!(layout0, left))?;
    // Top(zirgen/dsl/examples/calculator.zir:35)
    let x7: GetValFromUserStruct = exec_get_val_from_user(ctx)?;
    let x8: NondetRegStruct = exec_nondet_reg(ctx, x7._super, layout_lookup!(layout0, right))?;
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x9: OneHotStruct = exec_one_hot(ctx, x4._super, layout_lookup!(layout0, _0))?;
    let x10: Val2Array = map(x9._super, |x11| {
        return Ok(x11._super);
    })?;
    let x12: BoundLayout<NondetRegLayout, _> = layout_lookup!(x2, _super);
    let x13: RegStruct;
    if is_nonzero(x10[0]) {
        // Top(zirgen/dsl/examples/calculator.zir:37)
        let x14: Val = exec_add(ctx, x6._super, x8._super)?;
        let x15: RegStruct = exec_reg(ctx, x14, layout_lookup!(x2, arm0))?;
        x13 = x15;
    } else if is_nonzero(x10[1]) {
        // Top(zirgen/dsl/examples/calculator.zir:38)
        let x16: Val = exec_sub(ctx, x6._super, x8._super)?;
        let x17: RegStruct = exec_reg(ctx, x16, layout_lookup!(x2, arm1))?;
        x13 = x17;
    } else {
        bail!("Reached unreachable mux arm")
    } // Top(zirgen/dsl/examples/calculator.zir:36)
    let x18: RegStruct = back_reg(ctx, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:40)
    let x19: ComponentStruct = exec_log(ctx, "result = %u", &[x18._super._super])?;
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x20: RegStruct = back_reg(ctx, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:41)
    let x21: ComponentStruct = exec_set_global_result(ctx, x20._super._super, global1)?;
    // Top(zirgen/dsl/examples/calculator.zir:42)
    let x22: RegStruct = exec_get_global_result(ctx, global1)?;
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x23: RegStruct = back_reg(ctx, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:42)
    eqz!(
        (x22._super._super - x23._super._super),
        "Top(zirgen/dsl/examples/calculator.zir:42)"
    );
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x24: RegStruct = back_reg(ctx, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:43)
    let x25: ComponentStruct = exec_output_to_user(ctx, x24._super._super)?;
    // Top(zirgen/dsl/examples/calculator.zir:32)
    let x26: ComponentStruct = exec_component(ctx)?;
    return Ok(TopStruct {
        _super: x26,
        op: x4,
        left: x6,
        right: x8,
        result: x18,
    });
}
pub fn step_top(
    ctx: &ExecContext,
    data0: &impl BufferRow<ValType = Val>,
    global1: &impl BufferRow<ValType = Val>,
) -> Result<()> {
    let x2: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data0);
    let x3: TopStruct = exec_top(ctx, x2, global1)?;
    return Ok(());
}
pub fn validity_taps(
    ctx: &ValidityTapsContext,
    taps0: &impl BufferRow<ValType = ExtVal>,
    poly_mix1: PolyMix,
    global2: &impl BufferRow<ValType = Val>,
) -> Result<MixState> {
    let x3: ExtVal = get(taps0, 3, 0)?;
    let x4: ExtVal = get(taps0, 4, 0)?;
    let x5: ExtVal = get(taps0, 5, 0)?;
    let x6: ExtVal = get(taps0, 6, 0)?;
    let x7: ExtVal = get(taps0, 7, 0)?;
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x8: MixState = trivial_constraint()?;
    let x9: MixState = and_cond_ext(
        and_cond_ext(
            and_eqz_ext(
                poly_mix1,
                and_eqz_ext(
                    poly_mix1,
                    and_eqz_ext(
                        poly_mix1,
                        and_eqz_ext(
                            poly_mix1,
                            x8,
                            (x5 * (ExtVal::new(
                                Val::new(1),
                                Val::new(0),
                                Val::new(0),
                                Val::new(0),
                            ) - x5)),
                        )?,
                        (x6 * (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0))
                            - x6)),
                    )?,
                    ((x5 + x6) - ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0))),
                )?,
                (x6 - get(taps0, 2, 0)?),
            )?,
            x5,
            and_eqz_ext(poly_mix1, x8, ((x3 + x4) - x7))?,
        )?,
        x6,
        and_eqz_ext(poly_mix1, x8, ((x3 - x4) - x7))?,
    )?;
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    // Top(zirgen/dsl/examples/calculator.zir:41)
    let x10: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    return Ok(and_eqz_ext(
        poly_mix1,
        and_eqz_ext(
            poly_mix1,
            x9,
            (x7 - load_as_ext!(layout_lookup!(layout_lookup!(x10, result), _super), 0)),
        )?,
        (load_as_ext!(layout_lookup!(layout_lookup!(x10, result), _super), 0) - x7),
    )?);
}
pub fn validity_regs(
    ctx: &ValidityRegsContext,
    poly_mix0: PolyMix,
    data1: &impl BufferRow<ValType = Val>,
    global2: &impl BufferRow<ValType = Val>,
) -> Result<MixState> {
    let x3: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data1);
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x4: BoundLayout<TopResultLayout, _> = layout_lookup!(x3, result);
    // OneHot(zirgen/dsl/examples/calculator.zir:13)
    let x5: BoundLayout<NondetRegLayout2LayoutArray, _> =
        layout_lookup!(layout_lookup!(x3, _0), _super);
    // OneHot(zirgen/dsl/examples/calculator.zir:15)
    let x6: MixState = and_eqz(
        poly_mix0,
        trivial_constraint()?,
        (load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0)
            * (Val::new(1) - load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0))),
    )?;
    // OneHot(zirgen/dsl/examples/calculator.zir:17)
    let x7: MixState = and_eqz(
        poly_mix0,
        and_eqz(
            poly_mix0,
            x6,
            (load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0)
                * (Val::new(1) - load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0))),
        )?,
        ((load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0)
            + load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0))
            - Val::new(1)),
    )?;
    let x8: BoundLayout<NondetRegLayout, _> = layout_lookup!(x4, _super);
    // Reg(<preamble>:5)
    // Top(zirgen/dsl/examples/calculator.zir:37)
    let x9: MixState = and_eqz(
        poly_mix0,
        trivial_constraint()?,
        ((load!(layout_lookup!(layout_lookup!(x3, left), _super), 0)
            + load!(layout_lookup!(layout_lookup!(x3, right), _super), 0))
            - load!(layout_lookup!(layout_lookup!(x4, arm0), _super), 0)),
    )?;
    // Top(zirgen/dsl/examples/calculator.zir:38)
    let x10: MixState = and_eqz(
        poly_mix0,
        trivial_constraint()?,
        ((load!(layout_lookup!(layout_lookup!(x3, left), _super), 0)
            - load!(layout_lookup!(layout_lookup!(x3, right), _super), 0))
            - load!(layout_lookup!(layout_lookup!(x4, arm1), _super), 0)),
    )?;
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    // Top(zirgen/dsl/examples/calculator.zir:41)
    let x11: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    // Reg(<preamble>:5)
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:29)
    let x12: MixState = and_eqz(
        poly_mix0,
        and_cond(
            and_cond(
                and_eqz(
                    poly_mix0,
                    x7,
                    (load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0)
                        - load!(layout_lookup!(layout_lookup!(x3, op), _super), 0)),
                )?,
                load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0),
                x9,
            )?,
            load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0),
            x10,
        )?,
        (load!(layout_lookup!(x8, _super), 0)
            - load!(layout_lookup!(layout_lookup!(x11, result), _super), 0)),
    )?;
    // GetGlobalResult(zirgen/dsl/examples/calculator.zir:23)
    // Top(zirgen/dsl/examples/calculator.zir:42)
    let x13: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    return Ok(and_eqz(
        poly_mix0,
        x12,
        (load!(layout_lookup!(layout_lookup!(x13, result), _super), 0)
            - load!(layout_lookup!(x8, _super), 0)),
    )?);
}
