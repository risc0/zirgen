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
pub struct LogStruct {}
#[derive(Copy, Clone, Debug)]
pub struct GetCycleStruct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct CycleCounterStruct {
    pub _super: NondetRegStruct,
    pub is_first_cycle: NondetRegStruct,
}
#[derive(Copy, Clone, Debug)]
pub struct TopStruct {
    pub terminate: NondetRegStruct,
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
pub fn exec_inv<'a>(ctx: &'a ExecContext, arg0: Val) -> Result<Val> {
    return Ok(inv_0(arg0)?);
}
pub fn exec_isz<'a>(ctx: &'a ExecContext, arg0: Val) -> Result<Val> {
    return Ok(isz(arg0)?);
}
pub fn exec_add<'a>(ctx: &'a ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 + arg1));
}
pub fn exec_sub<'a>(ctx: &'a ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 - arg1));
}
pub fn exec_mul<'a>(ctx: &'a ExecContext, arg0: Val, arg1: Val) -> Result<Val> {
    return Ok((arg0 * arg1));
}
pub fn back_nondet_reg<'a>(
    ctx: &'a ExecContext,
    distance0: Index,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct> {
    // builtin NondetReg
    let x2: NondetRegStruct = NondetRegStruct {
        _super: (layout1.map(|c| c._super)).load(ctx, distance0),
    };
    return Ok(x2);
}
pub fn exec_nondet_reg<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct> {
    (layout1.map(|c| c._super)).store(ctx, arg0);
    let x2: NondetRegStruct = NondetRegStruct {
        _super: (layout1.map(|c| c._super)).load(ctx, 0),
    };
    return Ok(x2);
}
pub fn exec_component<'a>(ctx: &'a ExecContext) -> Result<ComponentStruct> {
    return Ok(ComponentStruct {});
}
pub fn back_reg<'a>(
    ctx: &'a ExecContext,
    distance0: Index,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct> {
    // Reg(<preamble>:5)
    let x2: NondetRegStruct = back_nondet_reg(ctx, distance0, layout1)?;
    return Ok(x2);
}
pub fn exec_reg<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct> {
    let x2: NondetRegStruct = exec_nondet_reg(ctx, arg0, layout1)?;
    // Reg(<preamble>:6)
    eqz!((arg0 - x2._super), "Reg(<preamble>:6)");
    return Ok(x2);
}
pub fn exec_log<'a>(ctx: &'a ExecContext, arg0: &str, arg1: &[Val]) -> Result<LogStruct> {
    // Log(<preamble>:25)
    invoke_extern!(ctx, log, arg0, arg1);
    return Ok(LogStruct {});
}
pub fn exec_get_cycle<'a>(ctx: &'a ExecContext) -> Result<GetCycleStruct> {
    // GetCycle(zirgen/dsl/examples/fibonacci.zir:2)
    let x0: Val = invoke_extern!(ctx, get_cycle);
    return Ok(GetCycleStruct { _super: x0 });
}
pub fn exec_is_zero<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, IsZeroLayout, Val>,
) -> Result<NondetRegStruct> {
    // IsZero(zirgen/dsl/examples/fibonacci.zir:6)
    let x2: Val = exec_isz(ctx, arg0)?;
    let x3: NondetRegStruct = exec_nondet_reg(ctx, x2, (layout1.map(|c| c._super)))?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:9)
    let x4: Val = exec_inv(ctx, arg0)?;
    let x5: NondetRegStruct = exec_nondet_reg(ctx, x4, (layout1.map(|c| c.inv)))?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x6: Val = exec_sub(ctx, Val::new(1), x3._super)?;
    let x7: Val = exec_mul(ctx, x3._super, x6)?;
    eqz!(x7, "IsZero(zirgen/dsl/examples/fibonacci.zir:12)");
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x8: Val = exec_mul(ctx, arg0, x5._super)?;
    let x9: Val = exec_sub(ctx, Val::new(1), x3._super)?;
    eqz!((x8 - x9), "IsZero(zirgen/dsl/examples/fibonacci.zir:14)");
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x10: Val = exec_mul(ctx, x3._super, arg0)?;
    eqz!(x10, "IsZero(zirgen/dsl/examples/fibonacci.zir:16)");
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x11: Val = exec_mul(ctx, x3._super, x5._super)?;
    eqz!(x11, "IsZero(zirgen/dsl/examples/fibonacci.zir:18)");
    return Ok(x3);
}
pub fn exec_cycle_counter<'a>(
    ctx: &'a ExecContext,
    layout0: BoundLayout<'a, CycleCounterLayout, Val>,
    global1: BufferRow<Val>,
) -> Result<CycleCounterStruct> {
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:28)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:29)
    let x3: NondetRegStruct = exec_nondet_reg(ctx, Val::new(6), (x2.map(|c| c.total_cycles)))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    let x4: GetCycleStruct = exec_get_cycle(ctx)?;
    let x5: NondetRegStruct = exec_nondet_reg(ctx, x4._super, (layout0.map(|c| c._super)))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    let x6: NondetRegStruct = exec_is_zero(ctx, x5._super, (layout0.map(|c| c.is_first_cycle)))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x7: Val = exec_sub(ctx, Val::new(1), x6._super)?;
    let x8: ComponentStruct = exec_component(ctx)?;
    let x9: ComponentStruct;
    if is_true(x6._super) {
        x9 = x8;
    } else if is_true(x7) {
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
        let x10: NondetRegStruct = back_nondet_reg(ctx, 1, (layout0.map(|c| c._super)))?;
        let x11: Val = exec_add(ctx, x10._super, Val::new(1))?;
        eqz!(
            (x5._super - x11),
            "CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)"
        );
        x9 = x8;
    } else {
        bail!("Reached unreachable mux arm")
    }
    return Ok(CycleCounterStruct {
        _super: x5,
        is_first_cycle: x6,
    });
}
pub fn exec_top<'a>(
    ctx: &'a ExecContext,
    layout0: BoundLayout<'a, TopLayout, Val>,
    global1: BufferRow<Val>,
) -> Result<TopStruct> {
    // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x3: CycleCounterStruct = exec_cycle_counter(ctx, (layout0.map(|c| c.cycle)), global1)?;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x4: Val = exec_sub(ctx, Val::new(1), x3.is_first_cycle._super)?;
    let x5: NondetRegStruct;
    if is_true(x3.is_first_cycle._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:45)
        let x6: NondetRegStruct = back_reg(ctx, 0, (x2.map(|c| c.f0)))?;
        x5 = x6;
    } else if is_true(x4) {
        // Top(zirgen/dsl/examples/fibonacci.zir:55)
        let x7: NondetRegStruct = back_reg(ctx, 1, (layout0.map(|c| c.d2)))?;
        x5 = x7;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x8: NondetRegStruct = exec_reg(ctx, x5._super, (layout0.map(|c| c.d1)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x9: Val = exec_sub(ctx, Val::new(1), x3.is_first_cycle._super)?;
    let x10: NondetRegStruct;
    if is_true(x3.is_first_cycle._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:46)
        let x11: NondetRegStruct = back_reg(ctx, 0, (x2.map(|c| c.f1)))?;
        x10 = x11;
    } else if is_true(x9) {
        // Top(zirgen/dsl/examples/fibonacci.zir:56)
        let x12: NondetRegStruct = back_reg(ctx, 1, (layout0.map(|c| c.d3)))?;
        x10 = x12;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x13: NondetRegStruct = exec_reg(ctx, x10._super, (layout0.map(|c| c.d2)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x14: Val = exec_add(ctx, x8._super, x13._super)?;
    let x15: NondetRegStruct = exec_reg(ctx, x14, (layout0.map(|c| c.d3)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:47)
    let x16: NondetRegStruct = back_reg(ctx, 0, (x2.map(|c| c.steps)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x17: Val = exec_sub(ctx, x3._super._super, x16._super)?;
    let x18: Val = exec_add(ctx, x17, Val::new(1))?;
    let x19: NondetRegStruct = exec_is_zero(ctx, x18, (layout0.map(|c| c.terminate)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x20: Val = exec_sub(ctx, Val::new(1), x19._super)?;
    let x21: ComponentStruct = exec_component(ctx)?;
    let x22: ComponentStruct;
    if is_true(x19._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:64)
        let x23: NondetRegStruct = exec_reg(ctx, x15._super, (x2.map(|c| c.f_last)))?;
        let x24: NondetRegStruct = back_reg(ctx, 0, (x2.map(|c| c.f_last)))?;
        // Top(zirgen/dsl/examples/fibonacci.zir:65)
        let x25: LogStruct = exec_log(ctx, "f_last = %u", &[x24._super])?;
        x22 = x21;
    } else if is_true(x20) {
        x22 = x21;
    } else {
        bail!("Reached unreachable mux arm")
    } // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x26: ComponentStruct = exec_component(ctx)?;
    return Ok(TopStruct { terminate: x19 });
}
pub fn step_top<'a>(
    ctx: &'a ExecContext,
    data0: BufferRow<Val>,
    global1: BufferRow<Val>,
) -> Result<()> {
    let x2: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data0);
    let x3: TopStruct = exec_top(ctx, x2, global1)?;
    return Ok(());
}
pub fn validity_taps<'a>(
    ctx: &'a ValidityTapsContext,
    taps0: BufferRow<ExtVal>,
    poly_mix1: PolyMix,
    global2: BufferRow<Val>,
) -> Result<MixState> {
    // All Constraints
    let x3: ExtVal = get(ctx, taps0, 2, 0)?;
    let x4: ExtVal = get(ctx, taps0, 4, 0)?;
    let x5: ExtVal = get(ctx, taps0, 5, 0)?;
    let x6: ExtVal = get(ctx, taps0, 6, 0)?;
    let x7: ExtVal = get(ctx, taps0, 8, 0)?;
    let x8: ExtVal = get(ctx, taps0, 10, 0)?;
    let x9: ExtVal = get(ctx, taps0, 11, 0)?;
    let x10: ExtVal = get(ctx, taps0, 12, 0)?;
    let x11: MixState = trivial_constraint()?;
    // Top(zirgen/dsl/examples/fibonacci.zir:44)
    // Top(zirgen/dsl/examples/fibonacci.zir:44)
    let x12: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    // builtin Add
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x13: ExtVal = ((x3 - ((x12.map(|c| c.steps)).map(|c| c._super)).load(ctx, 0))
        + ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)));
    // builtin Sub
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x14: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - x9);
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x15: ExtVal = (ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)) - x4);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x16: MixState = and_eqz_ext(ctx, and_eqz_ext(ctx, x11, (x4 * x15))?, ((x3 * x5) - x15))?;
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x17: MixState = and_cond_ext(
        and_eqz_ext(ctx, and_eqz_ext(ctx, x16, (x4 * x3))?, (x4 * x5))?,
        x15,
        and_eqz_ext(
            ctx,
            x11,
            (x3 - (get(ctx, taps0, 3, 0)?
                + ExtVal::new(Val::new(1), Val::new(0), Val::new(0), Val::new(0)))),
        )?,
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x18: ExtVal = (((x12.map(|c| c.f0)).map(|c| c._super)).load_unchecked(ctx, 0) * x4);
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x19: ExtVal = (((x12.map(|c| c.f1)).map(|c| c._super)).load_unchecked(ctx, 0) * x4);
    // Reg(<preamble>:6)
    let x20: MixState = and_eqz_ext(
        ctx,
        and_eqz_ext(ctx, x17, ((x18 + (get(ctx, taps0, 7, 0)? * x15)) - x8))?,
        ((x19 + (get(ctx, taps0, 9, 0)? * x15)) - x6),
    )?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x21: MixState = and_eqz_ext(
        ctx,
        and_eqz_ext(ctx, and_eqz_ext(ctx, x20, ((x8 + x6) - x7))?, (x9 * x14))?,
        ((x13 * x10) - x14),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x22: MixState = and_cond_ext(
        and_eqz_ext(ctx, and_eqz_ext(ctx, x21, (x9 * x13))?, (x9 * x10))?,
        x9,
        and_eqz_ext(
            ctx,
            x11,
            (x7 - ((x12.map(|c| c.f_last)).map(|c| c._super)).load(ctx, 0)),
        )?,
    )?;
    return Ok(x22);
}
pub fn validity_regs<'a>(
    ctx: &'a ValidityRegsContext,
    poly_mix0: PolyMix,
    data1: BufferRow<Val>,
    global2: BufferRow<Val>,
) -> Result<MixState> {
    let x3: BoundLayout<TopLayout, _> = bind_layout!(LAYOUT_TOP, data1);
    let x4: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global2);
    // builtin Add
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x5: Val =
        ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 1) + Val::new(1));
    let x6: Val = ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0) - x5);
    // builtin Add
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x7: Val = (((x3.map(|c| c.d1)).map(|c| c._super)).load(ctx, 0)
        + ((x3.map(|c| c.d2)).map(|c| c._super)).load(ctx, 0));
    // builtin Sub
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x8: Val = ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        - ((x4.map(|c| c.steps)).map(|c| c._super)).load(ctx, 0));
    // builtin Add
    let x9: Val = (x8 + Val::new(1));
    // builtin Sub
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x10: Val = (Val::new(1)
        - (((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    // builtin Mul
    let x11: Val =
        ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0) * x10);
    // builtin Sub
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x12: Val = (Val::new(1)
        - (((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    // builtin Mul
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x13: Val =
        ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0) * x9);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x14: Val = ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        * (((x3.map(|c| c.terminate)).map(|c| c.inv)).map(|c| c._super)).load(ctx, 0));
    // Reg(<preamble>:6)
    // Top(zirgen/dsl/examples/fibonacci.zir:64)
    let x15: Val = (((x3.map(|c| c.d3)).map(|c| c._super)).load(ctx, 0)
        - ((x4.map(|c| c.f_last)).map(|c| c._super)).load(ctx, 0));
    // builtin Sub
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x16: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    // builtin Mul
    let x17: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * x16);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x18: Val = ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c.inv)).map(|c| c._super))
            .load(ctx, 0));
    // builtin Sub
    let x19: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x20: MixState = and_eqz(ctx, and_eqz(ctx, trivial_constraint()?, x17)?, (x18 - x19))?;
    // builtin Mul
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x21: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * (((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x22: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c.inv)).map(|c| c._super))
            .load(ctx, 0));
    // builtin Sub
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x23: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x24: MixState = and_cond(
        and_eqz(ctx, and_eqz(ctx, x20, x21)?, x22)?,
        x23,
        and_eqz(ctx, trivial_constraint()?, x6)?,
    )?;
    // builtin Sub
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x25: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x26: Val = (((x4.map(|c| c.f0)).map(|c| c._super)).load_unchecked(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x27: Val = (((x3.map(|c| c.d2)).map(|c| c._super)).load_unchecked(ctx, 1) * x25);
    // Reg(<preamble>:6)
    let x28: Val = ((x26 + x27) - ((x3.map(|c| c.d1)).map(|c| c._super)).load(ctx, 0));
    // builtin Sub
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x29: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x30: Val = (((x4.map(|c| c.f1)).map(|c| c._super)).load_unchecked(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x31: Val = (((x3.map(|c| c.d3)).map(|c| c._super)).load_unchecked(ctx, 1) * x29);
    // Reg(<preamble>:6)
    let x32: Val = ((x30 + x31) - ((x3.map(|c| c.d2)).map(|c| c._super)).load(ctx, 0));
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x33: MixState = and_eqz(
        ctx,
        and_eqz(ctx, and_eqz(ctx, x24, x28)?, x32)?,
        (x7 - ((x3.map(|c| c.d3)).map(|c| c._super)).load(ctx, 0)),
    )?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x34: MixState = and_eqz(
        ctx,
        and_eqz(ctx, x33, x11)?,
        ((x9 * (((x3.map(|c| c.terminate)).map(|c| c.inv)).map(|c| c._super)).load(ctx, 0)) - x12),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x35: MixState = and_cond(
        and_eqz(ctx, and_eqz(ctx, x34, x13)?, x14)?,
        (((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0),
        and_eqz(ctx, trivial_constraint()?, x15)?,
    )?;
    return Ok(x35);
}
