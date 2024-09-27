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
    global1: &impl BufferRow<ValType = Val>,
) -> Result<CycleCounterStruct> {
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:28)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    let x3: BoundLayout<NondetRegLayout, _> = layout_lookup!(layout0, _super);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:29)
    let x4: NondetRegStruct = exec_nondet_reg(ctx, Val::new(6), layout_lookup!(x2, total_cycles))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    let x5: GetCycleStruct = exec_get_cycle(ctx)?;
    let x6: NondetRegStruct = exec_nondet_reg(ctx, x5._super, x3)?;
    let x7: Val = x6._super;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    let x8: IsZeroStruct = exec_is_zero(ctx, x7, layout_lookup!(layout0, is_first_cycle))?;
    let x9: Val = x8._super._super;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x10: Val = exec_sub(ctx, Val::new(1), x9)?;
    let x11: ComponentStruct;
    if is_nonzero(x9) {
        let x12: ComponentStruct = exec_component(ctx)?;
        x11 = x12;
    } else if is_nonzero(x10) {
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
        let x13: NondetRegStruct = back_nondet_reg(ctx, 1, x3)?;
        let x14: Val = exec_add(ctx, x13._super, Val::new(1))?;
        eqz!(
            (x7 - x14),
            "CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)"
        );
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:37)
        let x15: ComponentStruct = exec_component(ctx)?;
        x11 = x15;
    } else {
        bail!("Reached unreachable mux arm")
    }
    return Ok(CycleCounterStruct {
        _super: x6.clone(),
        cycle: x6,
        is_first_cycle: x8,
    });
}
pub fn exec_top(
    ctx: &ExecContext,
    layout0: BoundLayout<TopLayout, impl BufferRow<ValType = Val>>,
    global1: &impl BufferRow<ValType = Val>,
) -> Result<TopStruct> {
    // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // Top(zirgen/dsl/examples/fibonacci.zir:53)
    let x3: BoundLayout<NondetRegLayout, _> = layout_lookup!(layout0, d2);
    // Top(zirgen/dsl/examples/fibonacci.zir:54)
    let x4: BoundLayout<NondetRegLayout, _> = layout_lookup!(layout0, d3);
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x5: CycleCounterStruct = exec_cycle_counter(ctx, layout_lookup!(layout0, cycle), global1)?;
    let x6: IsZeroStruct = x5.is_first_cycle;
    let x7: Val = x6._super._super;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x8: Val = exec_sub(ctx, Val::new(1), x7)?;
    let x9: RegStruct;
    if is_nonzero(x7) {
        // Top(zirgen/dsl/examples/fibonacci.zir:45)
        let x10: RegStruct = back_reg(ctx, 0, layout_lookup!(x2, f0))?;
        x9 = x10;
    } else if is_nonzero(x8) {
        // Top(zirgen/dsl/examples/fibonacci.zir:55)
        let x11: RegStruct = back_reg(ctx, 1, x3)?;
        x9 = x11;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x12: RegStruct = exec_reg(ctx, x9._super._super, layout_lookup!(layout0, d1))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x13: Val = exec_sub(ctx, Val::new(1), x7)?;
    let x14: RegStruct;
    if is_nonzero(x7) {
        // Top(zirgen/dsl/examples/fibonacci.zir:46)
        let x15: RegStruct = back_reg(ctx, 0, layout_lookup!(x2, f1))?;
        x14 = x15;
    } else if is_nonzero(x13) {
        // Top(zirgen/dsl/examples/fibonacci.zir:56)
        let x16: RegStruct = back_reg(ctx, 1, x4)?;
        x14 = x16;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x17: RegStruct = exec_reg(ctx, x14._super._super, x3)?;
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x18: Val = exec_add(ctx, x12._super._super, x17._super._super)?;
    let x19: RegStruct = exec_reg(ctx, x18, x4)?;
    // Top(zirgen/dsl/examples/fibonacci.zir:47)
    let x20: RegStruct = back_reg(ctx, 0, layout_lookup!(x2, steps))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x21: Val = exec_sub(ctx, x5._super._super, x20._super._super)?;
    let x22: Val = exec_add(ctx, x21, Val::new(1))?;
    let x23: IsZeroStruct = exec_is_zero(ctx, x22, layout_lookup!(layout0, terminate))?;
    let x24: Val = x23._super._super;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x25: Val = exec_sub(ctx, Val::new(1), x24)?;
    let x26: ComponentStruct;
    if is_nonzero(x24) {
        // Top(zirgen/dsl/examples/fibonacci.zir:64)
        let x27: BoundLayout<NondetRegLayout, _> = layout_lookup!(x2, f_last);
        let x28: RegStruct = exec_reg(ctx, x19._super._super, x27)?;
        let x29: RegStruct = back_reg(ctx, 0, x27)?;
        // Top(zirgen/dsl/examples/fibonacci.zir:65)
        let x30: ComponentStruct = exec_log(ctx, "f_last = %u", &[x29._super._super])?;
        // Top(zirgen/dsl/examples/fibonacci.zir:63)
        let x31: ComponentStruct = exec_component(ctx)?;
        x26 = x31;
    } else if is_nonzero(x25) {
        // Top(zirgen/dsl/examples/fibonacci.zir:66)
        let x32: ComponentStruct = exec_component(ctx)?;
        x26 = x32;
    } else {
        bail!("Reached unreachable mux arm")
    } // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x33: ComponentStruct = exec_component(ctx)?;
    return Ok(TopStruct {
        _super: x33,
        cycle: x5,
        first: x6,
        d1: x12,
        d2: x17,
        d3: x19,
        terminate: x23,
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
    let x3: ExtVal = get(taps0, 2, 0)?;
    let x4: ExtVal = get(taps0, 4, 0)?;
    let x5: ExtVal = get(taps0, 5, 0)?;
    let x6: ExtVal = get(taps0, 6, 0)?;
    let x7: ExtVal = get(taps0, 8, 0)?;
    let x8: ExtVal = get(taps0, 10, 0)?;
    let x9: ExtVal = get(taps0, 11, 0)?;
    let x10: ExtVal = get(taps0, 12, 0)?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x11: MixState = trivial_constraint()?;
    let x12: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x13: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - x4);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x14: MixState = and_cond_ext(
        and_eqz_ext(
            poly_mix1,
            and_eqz_ext(
                poly_mix1,
                and_eqz_ext(
                    poly_mix1,
                    and_eqz_ext(poly_mix1, x11, (x4 * x13))?,
                    ((x3 * x5) - x13),
                )?,
                (x4 * x3),
            )?,
            (x4 * x5),
        )?,
        x13,
        and_eqz_ext(
            poly_mix1,
            x11,
            (x3 - (get(taps0, 3, 0)?
                + ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)))),
        )?,
    )?;
    // Reg(<preamble>:5)
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x15: ExtVal = (((load_as_ext!(layout_lookup!(layout_lookup!(x12, f0), _super), 0) * x4)
        + (get(taps0, 7, 0)? * x13))
        - x8);
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x16: ExtVal = (((load_as_ext!(layout_lookup!(layout_lookup!(x12, f1), _super), 0) * x4)
        + (get(taps0, 9, 0)? * x13))
        - x6);
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x17: ExtVal = ((x3 - load_as_ext!(layout_lookup!(layout_lookup!(x12, steps), _super), 0))
        + ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x18: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - x9);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x19: MixState = and_eqz_ext(
        poly_mix1,
        and_eqz_ext(
            poly_mix1,
            and_eqz_ext(
                poly_mix1,
                and_eqz_ext(
                    poly_mix1,
                    and_eqz_ext(poly_mix1, and_eqz_ext(poly_mix1, x14, x15)?, x16)?,
                    ((x8 + x6) - x7),
                )?,
                (x9 * x18),
            )?,
            ((x17 * x10) - x18),
        )?,
        (x9 * x17),
    )?;
    return Ok(and_cond_ext(
        and_eqz_ext(poly_mix1, x19, (x9 * x10))?,
        x9,
        and_eqz_ext(
            poly_mix1,
            x11,
            (x7 - load_as_ext!(layout_lookup!(layout_lookup!(x12, f_last), _super), 0)),
        )?,
    )?);
}
pub fn validity_regs(
    ctx: &ValidityRegsContext,
    poly_mix0: PolyMix,
    data1: &impl BufferRow<ValType = Val>,
    global2: &impl BufferRow<ValType = Val>,
) -> Result<MixState> {
    let x3: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data1);
    let x4: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x5: BoundLayout<CycleCounterLayout, _> = layout_lookup!(x3, cycle);
    // Top(zirgen/dsl/examples/fibonacci.zir:53)
    let x6: BoundLayout<NondetRegLayout, _> = layout_lookup!(x3, d2);
    // Top(zirgen/dsl/examples/fibonacci.zir:54)
    let x7: BoundLayout<NondetRegLayout, _> = layout_lookup!(x3, d3);
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x8: BoundLayout<IsZeroLayout, _> = layout_lookup!(x3, terminate);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x9: BoundLayout<NondetRegLayout, _> = layout_lookup!(x5, _super);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    let x10: BoundLayout<IsZeroLayout, _> = layout_lookup!(x5, is_first_cycle);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x11: MixState = and_eqz(
        poly_mix0,
        trivial_constraint()?,
        (load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0)
            * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0))),
    )?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x12: MixState = and_eqz(
        poly_mix0,
        and_eqz(
            poly_mix0,
            and_eqz(
                poly_mix0,
                x11,
                ((load!(layout_lookup!(x9, _super), 0)
                    * load!(layout_lookup!(layout_lookup!(x10, inv), _super), 0))
                    - (Val::new(1)
                        - load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0))),
            )?,
            (load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0)
                * load!(layout_lookup!(x9, _super), 0)),
        )?,
        (load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0)
            * load!(layout_lookup!(layout_lookup!(x10, inv), _super), 0)),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x13: Val = ((load!(layout_lookup!(layout_lookup!(x4, f0), _super), 0)
        * load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0))
        + (load!(layout_lookup!(x6, _super), 1)
            * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0))));
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x14: Val = ((load!(layout_lookup!(layout_lookup!(x4, f1), _super), 0)
        * load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0))
        + (load!(layout_lookup!(x7, _super), 1)
            * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0))));
    // Reg(<preamble>:5)
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x15: MixState = and_eqz(
        poly_mix0,
        and_eqz(
            poly_mix0,
            and_eqz(
                poly_mix0,
                and_cond(
                    x12,
                    (Val::new(1) - load!(layout_lookup!(layout_lookup!(x10, _super), _super), 0)),
                    and_eqz(
                        poly_mix0,
                        trivial_constraint()?,
                        (load!(layout_lookup!(x9, _super), 0)
                            - (load!(layout_lookup!(x9, _super), 1) + Val::new(1))),
                    )?,
                )?,
                (x13 - load!(layout_lookup!(layout_lookup!(x3, d1), _super), 0)),
            )?,
            (x14 - load!(layout_lookup!(x6, _super), 0)),
        )?,
        ((load!(layout_lookup!(layout_lookup!(x3, d1), _super), 0)
            + load!(layout_lookup!(x6, _super), 0))
            - load!(layout_lookup!(x7, _super), 0)),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x16: Val = ((load!(layout_lookup!(x9, _super), 0)
        - load!(layout_lookup!(layout_lookup!(x4, steps), _super), 0))
        + Val::new(1));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x17: MixState = and_eqz(
        poly_mix0,
        and_eqz(
            poly_mix0,
            x15,
            (load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0)
                * (Val::new(1) - load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))),
        )?,
        ((x16 * load!(layout_lookup!(layout_lookup!(x8, inv), _super), 0))
            - (Val::new(1) - load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0))),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x18: MixState = and_cond(
        and_eqz(
            poly_mix0,
            and_eqz(
                poly_mix0,
                x17,
                (load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0) * x16),
            )?,
            (load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0)
                * load!(layout_lookup!(layout_lookup!(x8, inv), _super), 0)),
        )?,
        load!(layout_lookup!(layout_lookup!(x8, _super), _super), 0),
        and_eqz(
            poly_mix0,
            trivial_constraint()?,
            (load!(layout_lookup!(x7, _super), 0)
                - load!(layout_lookup!(layout_lookup!(x4, f_last), _super), 0)),
        )?,
    )?;
    return Ok(x18);
}
