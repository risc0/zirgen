set_field!(BabyBear);
define_buffer_list! {
all: [accum,code,data,global,mix,],
rows: [accum,code,data,],
taps: [accum,code,data,],
globals: [global,mix,],}
define_tap_buffer! {accum, /*count=*/1, /*groupId=*/0}
define_tap_buffer! {code, /*count=*/1, /*groupId=*/1}
define_tap_buffer! {data, /*count=*/8, /*groupId=*/2}
define_global_buffer! {global, /*count=*/5}
define_global_buffer! {mix, /*count=*/4}
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
pub struct IsZeroLayout {
    pub _super: &'static NondetRegLayout,
    pub inv: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for IsZeroLayout {
    fn ty_name(&self) -> &'static str {
        "IsZeroLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("inv", self.inv)?;
        Ok(())
    }
}
pub struct CycleCounterLayout {
    pub _super: &'static NondetRegLayout,
    pub is_first_cycle: &'static IsZeroLayout,
}
impl risc0_zkp::layout::Component for CycleCounterLayout {
    fn ty_name(&self) -> &'static str {
        "CycleCounterLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("is_first_cycle", self.is_first_cycle)?;
        Ok(())
    }
}
pub type Tap13Array = [Tap; 13];
pub struct TopLayout {
    pub cycle: &'static CycleCounterLayout,
    pub d2: &'static NondetRegLayout,
    pub d3: &'static NondetRegLayout,
    pub d1: &'static NondetRegLayout,
    pub terminate: &'static IsZeroLayout,
}
impl risc0_zkp::layout::Component for TopLayout {
    fn ty_name(&self) -> &'static str {
        "TopLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("cycle", self.cycle)?;
        v.visit_component("d2", self.d2)?;
        v.visit_component("d3", self.d3)?;
        v.visit_component("d1", self.d1)?;
        v.visit_component("terminate", self.terminate)?;
        Ok(())
    }
}
pub struct _globalLayout {
    pub f0: &'static NondetRegLayout,
    pub f1: &'static NondetRegLayout,
    pub f_last: &'static NondetRegLayout,
    pub steps: &'static NondetRegLayout,
    pub total_cycles: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for _globalLayout {
    fn ty_name(&self) -> &'static str {
        "_globalLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("f0", self.f0)?;
        v.visit_component("f1", self.f1)?;
        v.visit_component("f_last", self.f_last)?;
        v.visit_component("steps", self.steps)?;
        v.visit_component("total_cycles", self.total_cycles)?;
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
pub struct GetCycleStruct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct IsZeroStruct {
    pub _super: NondetRegStruct,
    pub is_zero: NondetRegStruct,
    pub inv: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct CycleCounterStruct {
    pub _super: NondetRegStruct,
    pub cycle: NondetRegStruct,
    pub is_first_cycle: IsZeroStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct TopStruct {
    pub _super: ComponentStruct,
    pub cycle: CycleCounterStruct,
    pub first: IsZeroStruct,
    pub d1: RegStruct,
    pub d2: RegStruct,
    pub d3: RegStruct,
    pub terminate: IsZeroStruct,
}
pub type ExtVal13Array = [ExtVal; 13];
pub const LAYOUT__0: &CycleCounterLayout = &CycleCounterLayout {
    _super: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    is_first_cycle: &IsZeroLayout {
        _super: &NondetRegLayout {
            _super: &Reg { offset: 1 },
        },
        inv: &NondetRegLayout {
            _super: &Reg { offset: 2 },
        },
    },
};
pub const TAP_LIST: Tap13Array = [
    make_tap!(0, 0, 0),
    make_tap!(1, 0, 0),
    make_tap!(2, 0, 0),
    make_tap!(2, 0, 1),
    make_tap!(2, 1, 0),
    make_tap!(2, 2, 0),
    make_tap!(2, 3, 0),
    make_tap!(2, 3, 1),
    make_tap!(2, 4, 0),
    make_tap!(2, 4, 1),
    make_tap!(2, 5, 0),
    make_tap!(2, 6, 0),
    make_tap!(2, 7, 0),
];
pub const LAYOUT_TOP: &TopLayout = &TopLayout {
    cycle: LAYOUT__0,
    d2: &NondetRegLayout {
        _super: &Reg { offset: 3 },
    },
    d3: &NondetRegLayout {
        _super: &Reg { offset: 4 },
    },
    d1: &NondetRegLayout {
        _super: &Reg { offset: 5 },
    },
    terminate: &IsZeroLayout {
        _super: &NondetRegLayout {
            _super: &Reg { offset: 6 },
        },
        inv: &NondetRegLayout {
            _super: &Reg { offset: 7 },
        },
    },
};
pub const LAYOUT_GLOBAL: &_globalLayout = &_globalLayout {
    f0: &NondetRegLayout {
        _super: &Reg { offset: 0 },
    },
    f1: &NondetRegLayout {
        _super: &Reg { offset: 1 },
    },
    f_last: &NondetRegLayout {
        _super: &Reg { offset: 2 },
    },
    steps: &NondetRegLayout {
        _super: &Reg { offset: 3 },
    },
    total_cycles: &NondetRegLayout {
        _super: &Reg { offset: 4 },
    },
};
pub fn exec_inv(ctx: &ExecContext, arg0: Val) -> Result<Val> {
    return Ok(inv(arg0)?);
}
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
pub fn exec_get_cycle(ctx: &ExecContext) -> Result<GetCycleStruct> {
    // GetCycle(zirgen/dsl/examples/fibonacci.zir:2)
    let x0: Val = invoke_extern!(ctx, get_cycle);
    return Ok(GetCycleStruct { _super: x0 });
}
pub fn exec_is_zero(
    ctx: &ExecContext,
    arg0: Val,
    layout1: BoundLayout<IsZeroLayout, impl BufferRow<ValType = Val>>,
) -> Result<IsZeroStruct> {
    // IsZero(zirgen/dsl/examples/fibonacci.zir:6)
    let x2: Val = exec_isz(ctx, arg0)?;
    let x3: NondetRegStruct = exec_nondet_reg(ctx, x2, layout_lookup!(layout1, _super))?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:9)
    let x4: Val = exec_inv(ctx, arg0)?;
    let x5: NondetRegStruct = exec_nondet_reg(ctx, x4, layout_lookup!(layout1, inv))?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:6)
    let x6: Val = x3._super;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x7: Val = exec_sub(ctx, Val::new(1), x6)?;
    let x8: Val = exec_mul(ctx, x6, x7)?;
    eqz!(x8, "IsZero(zirgen/dsl/examples/fibonacci.zir:12)");
    // IsZero(zirgen/dsl/examples/fibonacci.zir:9)
    let x9: Val = x5._super;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x10: Val = exec_mul(ctx, arg0, x9)?;
    let x11: Val = exec_sub(ctx, Val::new(1), x6)?;
    eqz!((x10 - x11), "IsZero(zirgen/dsl/examples/fibonacci.zir:14)");
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x12: Val = exec_mul(ctx, x6, arg0)?;
    eqz!(x12, "IsZero(zirgen/dsl/examples/fibonacci.zir:16)");
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x13: Val = exec_mul(ctx, x6, x9)?;
    eqz!(x13, "IsZero(zirgen/dsl/examples/fibonacci.zir:18)");
    return Ok(IsZeroStruct {
        _super: x3.clone(),
        is_zero: x3,
        inv: x5,
    });
}
pub fn exec_cycle_counter(
    ctx: &ExecContext,
    layout0: BoundLayout<CycleCounterLayout, impl BufferRow<ValType = Val>>,
) -> Result<CycleCounterStruct> {
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:28)
    let x1: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx, global));
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    let x2: BoundLayout<NondetRegLayout, _> = layout_lookup!(layout0, _super);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:29)
    let x3: NondetRegStruct = exec_nondet_reg(ctx, Val::new(6), layout_lookup!(x1, total_cycles))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    let x4: GetCycleStruct = exec_get_cycle(ctx)?;
    let x5: NondetRegStruct = exec_nondet_reg(ctx, x4._super, x2)?;
    let x6: Val = x5._super;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    let x7: IsZeroStruct = exec_is_zero(ctx, x6, layout_lookup!(layout0, is_first_cycle))?;
    let x8: Val = x7._super._super;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x9: Val = exec_sub(ctx, Val::new(1), x8)?;
    let x10: ComponentStruct;
    if is_nonzero(x8) {
        let x11: ComponentStruct = exec_component(ctx)?;
        x10 = x11;
    } else if is_nonzero(x9) {
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
        let x12: NondetRegStruct = back_nondet_reg(ctx, 1, x2)?;
        let x13: Val = exec_add(ctx, x12._super, Val::new(1))?;
        eqz!(
            (x6 - x13),
            "CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)"
        );
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:37)
        let x14: ComponentStruct = exec_component(ctx)?;
        x10 = x14;
    } else {
        bail!("Reached unreachable mux arm")
    }
    return Ok(CycleCounterStruct {
        _super: x5.clone(),
        cycle: x5,
        is_first_cycle: x7,
    });
}
pub fn exec_top(
    ctx: &ExecContext,
    layout0: BoundLayout<TopLayout, impl BufferRow<ValType = Val>>,
) -> Result<TopStruct> {
    // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x1: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx, global));
    // Top(zirgen/dsl/examples/fibonacci.zir:53)
    let x2: BoundLayout<NondetRegLayout, _> = layout_lookup!(layout0, d2);
    // Top(zirgen/dsl/examples/fibonacci.zir:54)
    let x3: BoundLayout<NondetRegLayout, _> = layout_lookup!(layout0, d3);
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x4: CycleCounterStruct = exec_cycle_counter(ctx, layout_lookup!(layout0, cycle))?;
    let x5: IsZeroStruct = x4.is_first_cycle;
    let x6: Val = x5._super._super;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x7: Val = exec_sub(ctx, Val::new(1), x6)?;
    let x8: RegStruct;
    if is_nonzero(x6) {
        // Top(zirgen/dsl/examples/fibonacci.zir:45)
        let x9: RegStruct = back_reg(ctx, 0, layout_lookup!(x1, f0))?;
        x8 = x9;
    } else if is_nonzero(x7) {
        // Top(zirgen/dsl/examples/fibonacci.zir:55)
        let x10: RegStruct = back_reg(ctx, 1, x2)?;
        x8 = x10;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x11: RegStruct = exec_reg(ctx, x8._super._super, layout_lookup!(layout0, d1))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x12: Val = exec_sub(ctx, Val::new(1), x6)?;
    let x13: RegStruct;
    if is_nonzero(x6) {
        // Top(zirgen/dsl/examples/fibonacci.zir:46)
        let x14: RegStruct = back_reg(ctx, 0, layout_lookup!(x1, f1))?;
        x13 = x14;
    } else if is_nonzero(x12) {
        // Top(zirgen/dsl/examples/fibonacci.zir:56)
        let x15: RegStruct = back_reg(ctx, 1, x3)?;
        x13 = x15;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x16: RegStruct = exec_reg(ctx, x13._super._super, x2)?;
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x17: Val = exec_add(ctx, x11._super._super, x16._super._super)?;
    let x18: RegStruct = exec_reg(ctx, x17, x3)?;
    // Top(zirgen/dsl/examples/fibonacci.zir:47)
    let x19: RegStruct = back_reg(ctx, 0, layout_lookup!(x1, steps))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x20: Val = exec_sub(ctx, x4._super._super, x19._super._super)?;
    let x21: Val = exec_add(ctx, x20, Val::new(1))?;
    let x22: IsZeroStruct = exec_is_zero(ctx, x21, layout_lookup!(layout0, terminate))?;
    let x23: Val = x22._super._super;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x24: Val = exec_sub(ctx, Val::new(1), x23)?;
    let x25: ComponentStruct;
    if is_nonzero(x23) {
        // Top(zirgen/dsl/examples/fibonacci.zir:64)
        let x26: BoundLayout<NondetRegLayout, _> = layout_lookup!(x1, f_last);
        let x27: RegStruct = exec_reg(ctx, x18._super._super, x26)?;
        let x28: RegStruct = back_reg(ctx, 0, x26)?;
        // Top(zirgen/dsl/examples/fibonacci.zir:65)
        let x29: ComponentStruct = exec_log(ctx, "f_last = %u", &[x28._super._super])?;
        // Top(zirgen/dsl/examples/fibonacci.zir:63)
        let x30: ComponentStruct = exec_component(ctx)?;
        x25 = x30;
    } else if is_nonzero(x24) {
        // Top(zirgen/dsl/examples/fibonacci.zir:66)
        let x31: ComponentStruct = exec_component(ctx)?;
        x25 = x31;
    } else {
        bail!("Reached unreachable mux arm")
    } // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x32: ComponentStruct = exec_component(ctx)?;
    return Ok(TopStruct {
        _super: x32,
        cycle: x4,
        first: x5,
        d1: x11,
        d2: x16,
        d3: x18,
        terminate: x22,
    });
}
pub fn step_top(ctx: &ExecContext) -> Result<()> {
    let x0: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, get_buffer!(ctx, data));
    let x1: TopStruct = exec_top(ctx, x0)?;
    return Ok(());
}
pub fn validity_taps_(
    ctx: &ValidityTapsContext,
    poly_mix0: PolyMix,
    taps1: &ExtVal13Array,
) -> Result<MixState> {
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x2: MixState = trivial_constraint()?;
    let x3: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx, global));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x4: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - taps1[4]);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x5: MixState = and_eqz_ext(
        poly_mix0,
        and_eqz_ext(
            poly_mix0,
            and_eqz_ext(
                poly_mix0,
                and_eqz_ext(poly_mix0, x2, (taps1[4] * x4))?,
                ((taps1[2] * taps1[5]) - x4),
            )?,
            (taps1[4] * taps1[2]),
        )?,
        (taps1[4] * taps1[5]),
    )?;
    // Reg(<preamble>:5)
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x6: ExtVal = (((load_as_ext!(layout_lookup!(layout_lookup!(x3, f0), _super), 0)
        * taps1[4])
        + (taps1[7] * x4))
        - taps1[10]);
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x7: ExtVal = (((load_as_ext!(layout_lookup!(layout_lookup!(x3, f1), _super), 0)
        * taps1[4])
        + (taps1[9] * x4))
        - taps1[6]);
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x8: ExtVal = ((taps1[2]
        - load_as_ext!(layout_lookup!(layout_lookup!(x3, steps), _super), 0))
        + ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x9: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - taps1[11]);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x10: MixState = and_eqz_ext(
        poly_mix0,
        and_eqz_ext(
            poly_mix0,
            and_eqz_ext(
                poly_mix0,
                and_eqz_ext(
                    poly_mix0,
                    and_eqz_ext(
                        poly_mix0,
                        and_cond_ext(
                            x5,
                            x4,
                            and_eqz_ext(
                                poly_mix0,
                                x2,
                                (taps1[2]
                                    - (taps1[3]
                                        + ExtVal::new(
                                            Val::new(1),
                                            Val::new(0),
                                            Val::new(0),
                                            Val::new(0),
                                        ))),
                            )?,
                        )?,
                        x6,
                    )?,
                    x7,
                )?,
                ((taps1[10] + taps1[6]) - taps1[8]),
            )?,
            (taps1[11] * x9),
        )?,
        ((x8 * taps1[12]) - x9),
    )?;
    return Ok(and_cond_ext(
        and_eqz_ext(
            poly_mix0,
            and_eqz_ext(poly_mix0, x10, (taps1[11] * x8))?,
            (taps1[11] * taps1[12]),
        )?,
        taps1[11],
        and_eqz_ext(
            poly_mix0,
            x2,
            (taps1[8] - load_as_ext!(layout_lookup!(layout_lookup!(x3, f_last), _super), 0)),
        )?,
    )?);
}
pub fn validity_regs_(ctx: &ValidityRegsContext, poly_mix0: PolyMix) -> Result<MixState> {
    let x1: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, get_buffer!(ctx, data));
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, get_buffer!(ctx, global));
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x3: BoundLayout<CycleCounterLayout, _> = layout_lookup!(x1, cycle);
    // Top(zirgen/dsl/examples/fibonacci.zir:53)
    let x4: BoundLayout<NondetRegLayout, _> = layout_lookup!(x1, d2);
    // Top(zirgen/dsl/examples/fibonacci.zir:54)
    let x5: BoundLayout<NondetRegLayout, _> = layout_lookup!(x1, d3);
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x6: BoundLayout<IsZeroLayout, _> = layout_lookup!(x1, terminate);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x7: BoundLayout<NondetRegLayout, _> = layout_lookup!(x3, _super);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    let x8: BoundLayout<IsZeroLayout, _> = layout_lookup!(x3, is_first_cycle);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x9: MixState = and_eqz(
        poly_mix0,
        trivial_constraint()?,
        (load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0)
            * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))),
    )?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x10: MixState = and_eqz(
        poly_mix0,
        and_eqz(
            poly_mix0,
            and_eqz(
                poly_mix0,
                x9,
                ((load!(layout_lookup!(x7, _super), 0)
                    * load!(layout_lookup!(layout_lookup!(x8, inv), _super), 0))
                    - (Val::new(1) - load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))),
            )?,
            (load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0)
                * load!(layout_lookup!(x7, _super), 0)),
        )?,
        (load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0)
            * load!(layout_lookup!(layout_lookup!(x8, inv), _super), 0)),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x11: Val = ((load!(layout_lookup!(layout_lookup!(x2, f0), _super), 0)
        * load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))
        + (load!(layout_lookup!(x4, _super), 1)
            * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))));
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x12: Val = ((load!(layout_lookup!(layout_lookup!(x2, f1), _super), 0)
        * load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))
        + (load!(layout_lookup!(x5, _super), 1)
            * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))));
    // Reg(<preamble>:5)
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x13: MixState = and_eqz(
        poly_mix0,
        and_eqz(
            poly_mix0,
            and_eqz(
                poly_mix0,
                and_cond(
                    x10,
                    (Val::new(1) - load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0)),
                    and_eqz(
                        poly_mix0,
                        trivial_constraint()?,
                        (load!(layout_lookup!(x7, _super), 0)
                            - (load!(layout_lookup!(x7, _super), 1) + Val::new(1))),
                    )?,
                )?,
                (x11 - load!(layout_lookup!(layout_lookup!(x1, d1), _super), 0)),
            )?,
            (x12 - load!(layout_lookup!(x4, _super), 0)),
        )?,
        ((load!(layout_lookup!(layout_lookup!(x1, d1), _super), 0)
            + load!(layout_lookup!(x4, _super), 0))
            - load!(layout_lookup!(x5, _super), 0)),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x14: Val = ((load!(layout_lookup!(x7, _super), 0)
        - load!(layout_lookup!(layout_lookup!(x2, steps), _super), 0))
        + Val::new(1));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x15: MixState = and_eqz(
        poly_mix0,
        and_eqz(
            poly_mix0,
            x13,
            (load!(layout_lookup!(layout_lookup!(x6, _super), _super), 0)
                * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x6, _super), _super), 0))),
        )?,
        ((x14 * load!(layout_lookup!(layout_lookup!(x6, inv), _super), 0))
            - (Val::new(1) - load!(layout_lookup!(layout_lookup!(x6, _super), _super), 0))),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x16: MixState = and_cond(
        and_eqz(
            poly_mix0,
            and_eqz(
                poly_mix0,
                x15,
                (load!(layout_lookup!(layout_lookup!(x6, _super), _super), 0) * x14),
            )?,
            (load!(layout_lookup!(layout_lookup!(x6, _super), _super), 0)
                * load!(layout_lookup!(layout_lookup!(x6, inv), _super), 0)),
        )?,
        load!(layout_lookup!(layout_lookup!(x6, _super), _super), 0),
        and_eqz(
            poly_mix0,
            trivial_constraint()?,
            (load!(layout_lookup!(x5, _super), 0)
                - load!(layout_lookup!(layout_lookup!(x2, f_last), _super), 0)),
        )?,
    )?;
    return Ok(x16);
}
