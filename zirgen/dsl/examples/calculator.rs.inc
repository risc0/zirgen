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
pub type ExtVal8Array = [ExtVal; 8];
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
        _super: make_ref!(0),
    },
    left: &NondetRegLayout {
        _super: make_ref!(1),
    },
    right: &NondetRegLayout {
        _super: make_ref!(2),
    },
    _0: &OneHotLayout {
        _super: &[
            &NondetRegLayout {
                _super: make_ref!(3),
            },
            &NondetRegLayout {
                _super: make_ref!(4),
            },
        ],
    },
    result: &TopResultLayout {
        _super: &NondetRegLayout {
            _super: make_ref!(5),
        },
        arm0: &NondetRegLayout {
            _super: make_ref!(5),
        },
        arm1: &NondetRegLayout {
            _super: make_ref!(5),
        },
    },
};
pub const LAYOUT_GLOBAL: &_globalLayout = &_globalLayout {
    result: &NondetRegLayout {
        _super: make_ref!(0),
    },
};
pub fn exec_isz(ctx0: &ExecContext, arg1: Val) -> Result<Val> {
    return Ok(isz(arg1)?);
}
pub fn exec_add(ctx0: &ExecContext, arg1: Val, arg2: Val) -> Result<Val> {
    return Ok((arg1 + arg2));
}
pub fn exec_sub(ctx0: &ExecContext, arg1: Val, arg2: Val) -> Result<Val> {
    return Ok((arg1 - arg2));
}
pub fn exec_mul(ctx0: &ExecContext, arg1: Val, arg2: Val) -> Result<Val> {
    return Ok((arg1 * arg2));
}
pub fn back_nondet_reg(
    ctx0: &ExecContext,
    distance1: Index,
    layout2: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<NondetRegStruct> {
    return Ok(NondetRegStruct {
        _super: load!(layout_lookup!(layout2, _super), distance1),
    });
}
pub fn exec_nondet_reg(
    ctx0: &ExecContext,
    arg1: Val,
    layout2: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<NondetRegStruct> {
    let x3: BoundLayout<Reg, _> = layout_lookup!(layout2, _super);
    store!(x3, arg1);
    return Ok(NondetRegStruct {
        _super: load!(x3, 0),
    });
}
pub fn exec_component(ctx0: &ExecContext) -> Result<ComponentStruct> {
    return Ok(ComponentStruct {});
}
pub fn back_reg(
    ctx0: &ExecContext,
    distance1: Index,
    layout2: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<RegStruct> {
    // Reg(<preamble>:4)
    let x3: NondetRegStruct = back_nondet_reg(ctx0, distance1, layout2)?;
    return Ok(RegStruct {
        _super: x3.clone(),
        reg: x3,
    });
}
pub fn exec_reg(
    ctx0: &ExecContext,
    arg1: Val,
    layout2: BoundLayout<NondetRegLayout, impl BufferRow<ValType = Val>>,
) -> Result<RegStruct> {
    let x3: NondetRegStruct = exec_nondet_reg(ctx0, arg1, layout2)?;
    // Reg(<preamble>:5)
    eqz!((arg1 - x3._super), "Reg(<preamble>:5)");
    return Ok(RegStruct {
        _super: x3.clone(),
        reg: x3,
    });
}
pub fn exec_log(ctx0: &ExecContext, arg1: &str, arg2: &[Val]) -> Result<ComponentStruct> {
    // Log(<preamble>:16)
    invoke_extern!(ctx0, log, arg1, arg2);
    return Ok(ComponentStruct {});
}
pub fn exec_get_val_from_user(ctx0: &ExecContext) -> Result<GetValFromUserStruct> {
    // GetValFromUser(zirgen/dsl/examples/calculator.zir:5)
    let x1: Val = invoke_extern!(ctx0, get_val_from_user);
    return Ok(GetValFromUserStruct { _super: x1 });
}
pub fn exec_output_to_user(ctx0: &ExecContext, arg1: Val) -> Result<ComponentStruct> {
    // OutputToUser(zirgen/dsl/examples/calculator.zir:6)
    invoke_extern!(ctx0, output_to_user, arg1);
    return Ok(ComponentStruct {});
}
pub fn exec_one_hot(
    ctx0: &ExecContext,
    arg1: Val,
    layout2: BoundLayout<OneHotLayout, impl BufferRow<ValType = Val>>,
) -> Result<OneHotStruct> {
    // OneHot(zirgen/dsl/examples/calculator.zir:13)
    let x3: Val2Array = [make_val!(0), make_val!(1)];
    let x4: NondetRegStruct2Array = map_layout(x3, layout_lookup!(layout2, _super), |x5, x6| {
        let x7: Val = exec_sub(ctx0, x5, arg1)?;
        let x8: Val = exec_isz(ctx0, x7)?;
        let x9: NondetRegStruct = exec_nondet_reg(ctx0, x8, x6)?;
        return Ok(x9);
    })?;
    // OneHot(zirgen/dsl/examples/calculator.zir:15)
    let x10: ComponentStruct2Array = map(x4, |x11| {
        let x12: Val = x11._super;
        let x13: Val = exec_sub(ctx0, make_val!(1), x12)?;
        let x14: Val = exec_mul(ctx0, x12, x13)?;
        eqz!(x14, "OneHot(zirgen/dsl/examples/calculator.zir:15)");
        let x15: ComponentStruct = exec_component(ctx0)?;
        return Ok(x15);
    })?;
    // OneHot(zirgen/dsl/examples/calculator.zir:17)
    let x16: Val = reduce(x4, make_val!(0), |x17, x18| {
        let x19: Val = exec_add(ctx0, x17, x18._super)?;
        return Ok(x19);
    })?;
    eqz!(
        (x16 - make_val!(1)),
        "OneHot(zirgen/dsl/examples/calculator.zir:17)"
    );
    // OneHot(zirgen/dsl/examples/calculator.zir:19)
    let x20: OneHot__0_SuperStruct2Array = map(x3, |x21| {
        let x22: Val = exec_mul(ctx0, x4[u64::from(x21) as usize]._super, x21)?;
        return Ok(OneHot__0_SuperStruct { _super: x22 });
    })?;
    let x23: Val = reduce(x20, make_val!(0), |x24, x25| {
        let x26: Val = exec_add(ctx0, x24, x25._super)?;
        return Ok(x26);
    })?;
    eqz!(
        (x23 - arg1),
        "OneHot(zirgen/dsl/examples/calculator.zir:19)"
    );
    return Ok(OneHotStruct {
        _super: x4.clone(),
        n: make_val!(2),
        bits: x4,
    });
}
pub fn exec_get_global_result(ctx0: &ExecContext) -> Result<RegStruct> {
    // GetGlobalResult(zirgen/dsl/examples/calculator.zir:23)
    let x1: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx0, global));
    // GetGlobalResult(zirgen/dsl/examples/calculator.zir:24)
    let x2: RegStruct = back_reg(ctx0, 0, layout_lookup!(x1, result))?;
    return Ok(x2);
}
pub fn exec_set_global_result(ctx0: &ExecContext, arg1: Val) -> Result<ComponentStruct> {
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx0, global));
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:29)
    let x3: RegStruct = exec_reg(ctx0, arg1, layout_lookup!(x2, result))?;
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    let x4: ComponentStruct = exec_component(ctx0)?;
    return Ok(x4);
}
pub fn exec_top(
    ctx0: &ExecContext,
    layout1: BoundLayout<TopLayout, impl BufferRow<ValType = Val>>,
) -> Result<TopStruct> {
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x2: BoundLayout<TopResultLayout, _> = layout_lookup!(layout1, result);
    // Top(zirgen/dsl/examples/calculator.zir:33)
    let x3: GetValFromUserStruct = exec_get_val_from_user(ctx0)?;
    let x4: NondetRegStruct = exec_nondet_reg(ctx0, x3._super, layout_lookup!(layout1, op))?;
    // Top(zirgen/dsl/examples/calculator.zir:34)
    let x5: GetValFromUserStruct = exec_get_val_from_user(ctx0)?;
    let x6: NondetRegStruct = exec_nondet_reg(ctx0, x5._super, layout_lookup!(layout1, left))?;
    // Top(zirgen/dsl/examples/calculator.zir:35)
    let x7: GetValFromUserStruct = exec_get_val_from_user(ctx0)?;
    let x8: NondetRegStruct = exec_nondet_reg(ctx0, x7._super, layout_lookup!(layout1, right))?;
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x9: OneHotStruct = exec_one_hot(ctx0, x4._super, layout_lookup!(layout1, _0))?;
    let x10: Val2Array = map(x9._super, |x11| {
        return Ok(x11._super);
    })?;
    let x12: BoundLayout<NondetRegLayout, _> = layout_lookup!(x2, _super);
    let x13: RegStruct;
    if is_nonzero(x10[0]) {
        // Top(zirgen/dsl/examples/calculator.zir:37)
        let x14: Val = exec_add(ctx0, x6._super, x8._super)?;
        let x15: RegStruct = exec_reg(ctx0, x14, layout_lookup!(x2, arm0))?;
        x13 = x15;
    } else if is_nonzero(x10[1]) {
        // Top(zirgen/dsl/examples/calculator.zir:38)
        let x16: Val = exec_sub(ctx0, x6._super, x8._super)?;
        let x17: RegStruct = exec_reg(ctx0, x16, layout_lookup!(x2, arm1))?;
        x13 = x17;
    } else {
        bail!("Reached unreachable mux arm")
    } // Top(zirgen/dsl/examples/calculator.zir:36)
    let x18: RegStruct = back_reg(ctx0, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:40)
    let x19: ComponentStruct = exec_log(ctx0, "result = %u", &[x18._super._super])?;
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x20: RegStruct = back_reg(ctx0, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:41)
    let x21: ComponentStruct = exec_set_global_result(ctx0, x20._super._super)?;
    // Top(zirgen/dsl/examples/calculator.zir:42)
    let x22: RegStruct = exec_get_global_result(ctx0)?;
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x23: RegStruct = back_reg(ctx0, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:42)
    eqz!(
        (x22._super._super - x23._super._super),
        "Top(zirgen/dsl/examples/calculator.zir:42)"
    );
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x24: RegStruct = back_reg(ctx0, 0, x12)?;
    // Top(zirgen/dsl/examples/calculator.zir:43)
    let x25: ComponentStruct = exec_output_to_user(ctx0, x24._super._super)?;
    // Top(zirgen/dsl/examples/calculator.zir:32)
    let x26: ComponentStruct = exec_component(ctx0)?;
    return Ok(TopStruct {
        _super: x26,
        op: x4,
        left: x6,
        right: x8,
        result: x18,
    });
}
pub fn step_top(ctx0: &ExecContext) -> Result<()> {
    let x1: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, get_buffer!(ctx0, data));
    let x2: TopStruct = exec_top(ctx0, x1)?;
    return Ok(());
}
pub fn validity_taps_(
    ctx0: &ValidityTapsContext,
    poly_mix1: PolyMix,
    taps2: &ExtVal8Array,
) -> Result<MixState> {
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x3: MixState = trivial_constraint()?;
    // OneHot(zirgen/dsl/examples/calculator.zir:19)
    let x4: MixState = and_eqz_ext(
        poly_mix1,
        and_eqz_ext(
            poly_mix1,
            and_eqz_ext(
                poly_mix1,
                and_eqz_ext(
                    poly_mix1,
                    x3,
                    (taps2[5] * (make_val_ext!(1, 0, 0, 0) - taps2[5])),
                )?,
                (taps2[6] * (make_val_ext!(1, 0, 0, 0) - taps2[6])),
            )?,
            ((taps2[5] + taps2[6]) - make_val_ext!(1, 0, 0, 0)),
        )?,
        (taps2[6] - taps2[2]),
    )?;
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    // Top(zirgen/dsl/examples/calculator.zir:41)
    let x5: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx0, global));
    // Top(zirgen/dsl/examples/calculator.zir:42)
    let x6: MixState = and_eqz_ext(
        poly_mix1,
        and_eqz_ext(
            poly_mix1,
            and_cond_ext(
                and_cond_ext(
                    x4,
                    taps2[5],
                    and_eqz_ext(poly_mix1, x3, ((taps2[3] + taps2[4]) - taps2[7]))?,
                )?,
                taps2[6],
                and_eqz_ext(poly_mix1, x3, ((taps2[3] - taps2[4]) - taps2[7]))?,
            )?,
            (taps2[7] - load_as_ext!(layout_lookup!(layout_lookup!(x5, result), _super), 0)),
        )?,
        (load_as_ext!(layout_lookup!(layout_lookup!(x5, result), _super), 0) - taps2[7]),
    )?;
    return Ok(x6);
}
pub fn validity_regs_(ctx0: &ValidityRegsContext, poly_mix1: PolyMix) -> Result<MixState> {
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x2: MixState = trivial_constraint()?;
    let x3: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, get_buffer!(ctx0, data));
    // Top(zirgen/dsl/examples/calculator.zir:36)
    let x4: BoundLayout<TopResultLayout, _> = layout_lookup!(x3, result);
    // OneHot(zirgen/dsl/examples/calculator.zir:13)
    let x5: BoundLayout<NondetRegLayout2LayoutArray, _> =
        layout_lookup!(layout_lookup!(x3, _0), _super);
    // OneHot(zirgen/dsl/examples/calculator.zir:15)
    let x6: MixState = and_eqz(
        poly_mix1,
        and_eqz(
            poly_mix1,
            x2,
            (load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0)
                * (make_val!(1) - load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0))),
        )?,
        (load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0)
            * (make_val!(1) - load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0))),
    )?;
    let x7: MixState = and_cond(
        and_eqz(
            poly_mix1,
            and_eqz(
                poly_mix1,
                x6,
                ((load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0)
                    + load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0))
                    - make_val!(1)),
            )?,
            (load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0)
                - load!(layout_lookup!(layout_lookup!(x3, op), _super), 0)),
        )?,
        load!(layout_lookup!(layout_subscript!(x5, 0), _super), 0),
        and_eqz(
            poly_mix1,
            x2,
            ((load!(layout_lookup!(layout_lookup!(x3, left), _super), 0)
                + load!(layout_lookup!(layout_lookup!(x3, right), _super), 0))
                - load!(layout_lookup!(layout_lookup!(x4, arm0), _super), 0)),
        )?,
    )?;
    // SetGlobalResult(zirgen/dsl/examples/calculator.zir:28)
    // Top(zirgen/dsl/examples/calculator.zir:41)
    let x8: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx0, global));
    // Top(zirgen/dsl/examples/calculator.zir:42)
    let x9: MixState = and_eqz(
        poly_mix1,
        and_eqz(
            poly_mix1,
            and_cond(
                x7,
                load!(layout_lookup!(layout_subscript!(x5, 1), _super), 0),
                and_eqz(
                    poly_mix1,
                    x2,
                    ((load!(layout_lookup!(layout_lookup!(x3, left), _super), 0)
                        - load!(layout_lookup!(layout_lookup!(x3, right), _super), 0))
                        - load!(layout_lookup!(layout_lookup!(x4, arm1), _super), 0)),
                )?,
            )?,
            (load!(layout_lookup!(layout_lookup!(x4, _super), _super), 0)
                - load!(layout_lookup!(layout_lookup!(x8, result), _super), 0)),
        )?,
        (load!(layout_lookup!(layout_lookup!(x8, result), _super), 0)
            - load!(layout_lookup!(layout_lookup!(x4, _super), _super), 0)),
    )?;
    return Ok(x9);
}
