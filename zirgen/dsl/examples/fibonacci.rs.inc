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
pub fn back_reg<'a>(
    ctx: &'a ExecContext,
    distance0: Index,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct> {
    // Reg(<preamble>:5)
    let x2: NondetRegStruct = NondetRegStruct {
        _super: (layout1.map(|c| c._super)).load(ctx, distance0),
    };
    return Ok(x2);
}
pub fn exec_reg<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, NondetRegLayout, Val>,
) -> Result<NondetRegStruct> {
    (layout1.map(|c| c._super)).store(ctx, arg0);
    let x2: NondetRegStruct = NondetRegStruct {
        _super: (layout1.map(|c| c._super)).load(ctx, 0),
    };
    // Reg(<preamble>:6)
    eqz!(
        (arg0 - (layout1.map(|c| c._super)).load(ctx, 0)),
        "Reg(<preamble>:6)"
    );
    return Ok(x2);
}
pub fn exec_is_zero<'a>(
    ctx: &'a ExecContext,
    arg0: Val,
    layout1: BoundLayout<'a, IsZeroLayout, Val>,
) -> Result<NondetRegStruct> {
    // builtin NondetReg
    // IsZero(zirgen/dsl/examples/fibonacci.zir:6)
    ((layout1.map(|c| c._super)).map(|c| c._super)).store(ctx, isz(arg0)?);
    let x2: NondetRegStruct = NondetRegStruct {
        _super: ((layout1.map(|c| c._super)).map(|c| c._super)).load(ctx, 0),
    };
    // IsZero(zirgen/dsl/examples/fibonacci.zir:9)
    ((layout1.map(|c| c.inv)).map(|c| c._super)).store(ctx, inv_0(arg0)?);
    // builtin Sub
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    let x3: Val = (Val::new(1) - ((layout1.map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    // builtin Mul
    let x4: Val = (((layout1.map(|c| c._super)).map(|c| c._super)).load(ctx, 0) * x3);
    eqz!(x4, "IsZero(zirgen/dsl/examples/fibonacci.zir:12)");
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    eqz!(
        ((arg0 * ((layout1.map(|c| c.inv)).map(|c| c._super)).load(ctx, 0)) - x3),
        "IsZero(zirgen/dsl/examples/fibonacci.zir:14)"
    );
    // builtin Mul
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x5: Val = (((layout1.map(|c| c._super)).map(|c| c._super)).load(ctx, 0) * arg0);
    eqz!(x5, "IsZero(zirgen/dsl/examples/fibonacci.zir:16)");
    // builtin Mul
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x6: Val = (((layout1.map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        * ((layout1.map(|c| c.inv)).map(|c| c._super)).load(ctx, 0));
    eqz!(x6, "IsZero(zirgen/dsl/examples/fibonacci.zir:18)");
    return Ok(x2);
}
pub fn exec_cycle_counter<'a>(
    ctx: &'a ExecContext,
    layout0: BoundLayout<'a, CycleCounterLayout, Val>,
    global1: BufferRow<Val>,
) -> Result<CycleCounterStruct> {
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:28)
    let x2: BoundLayout<_globalLayout, _> = bind_layout!(LAYOUT_GLOBAL, global1);
    // builtin NondetReg
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:29)
    ((x2.map(|c| c.total_cycles)).map(|c| c._super)).store(ctx, Val::new(6));
    // GetCycle(zirgen/dsl/examples/fibonacci.zir:2)
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:31)
    let x3: Val = invoke_extern!(ctx, get_cycle);
    // builtin NondetReg
    ((layout0.map(|c| c._super)).map(|c| c._super)).store(ctx, x3);
    let x4: NondetRegStruct = NondetRegStruct {
        _super: ((layout0.map(|c| c._super)).map(|c| c._super)).load(ctx, 0),
    };
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    let x5: NondetRegStruct = exec_is_zero(
        ctx,
        ((layout0.map(|c| c._super)).map(|c| c._super)).load(ctx, 0),
        (layout0.map(|c| c.is_first_cycle)),
    )?;
    // builtin Component
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x6: ComponentStruct = ComponentStruct {};
    let x7: ComponentStruct;
    if is_true(x5._super) {
        x7 = x6;
    } else if is_true((Val::new(1) - x5._super)) {
        // CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)
        let x8: NondetRegStruct = back_nondet_reg(ctx, 1, (layout0.map(|c| c._super)))?;
        let x9: Val = (((layout0.map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
            - (x8._super + Val::new(1)));
        eqz!(x9, "CycleCounter(zirgen/dsl/examples/fibonacci.zir:39)");
        x7 = x6;
    } else {
        bail!("Reached unreachable mux arm")
    }
    return Ok(CycleCounterStruct {
        _super: x4,
        is_first_cycle: x5,
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
    // builtin Sub
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x4: Val = (Val::new(1) - x3.is_first_cycle._super);
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
    let x9: NondetRegStruct;
    if is_true(x3.is_first_cycle._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:46)
        let x10: NondetRegStruct = back_reg(ctx, 0, (x2.map(|c| c.f1)))?;
        x9 = x10;
    } else if is_true(x4) {
        // Top(zirgen/dsl/examples/fibonacci.zir:56)
        let x11: NondetRegStruct = back_reg(ctx, 1, (layout0.map(|c| c.d3)))?;
        x9 = x11;
    } else {
        bail!("Reached unreachable mux arm")
    }
    let x12: NondetRegStruct = exec_reg(ctx, x9._super, (layout0.map(|c| c.d2)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x13: NondetRegStruct = exec_reg(ctx, (x8._super + x12._super), (layout0.map(|c| c.d3)))?;
    // Top(zirgen/dsl/examples/fibonacci.zir:47)
    let x14: NondetRegStruct = back_reg(ctx, 0, (x2.map(|c| c.steps)))?;
    // builtin Sub
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x15: Val = (x3._super._super - x14._super);
    let x16: NondetRegStruct =
        exec_is_zero(ctx, (x15 + Val::new(1)), (layout0.map(|c| c.terminate)))?;
    // Log(<preamble>:22)
    // Top(zirgen/dsl/examples/fibonacci.zir:65)
    let x17: ComponentStruct = ComponentStruct {};
    let x18: ComponentStruct;
    if is_true(x16._super) {
        // Top(zirgen/dsl/examples/fibonacci.zir:64)
        let x19: NondetRegStruct = exec_reg(ctx, x13._super, (x2.map(|c| c.f_last)))?;
        let x20: NondetRegStruct = back_reg(ctx, 0, (x2.map(|c| c.f_last)))?;
        // Log(<preamble>:22)
        // Top(zirgen/dsl/examples/fibonacci.zir:65)
        invoke_extern!(ctx, log, "f_last = %u", [x20._super]);
        x18 = x17;
    } else if is_true((Val::new(1) - x16._super)) {
        x18 = x17;
    } else {
        bail!("Reached unreachable mux arm")
    }
    return Ok(TopStruct { terminate: x16 });
}
pub fn step_top<'a>(
    ctx: &'a ExecContext,
    data0: BufferRow<Val>,
    global1: BufferRow<Val>,
) -> Result<()> {
    // Top(zirgen/dsl/examples/fibonacci.zir:44)
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
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x12: Val =
        ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0) * x9);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x13: Val = ((((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        * (((x3.map(|c| c.terminate)).map(|c| c.inv)).map(|c| c._super)).load(ctx, 0));
    // Reg(<preamble>:6)
    // Top(zirgen/dsl/examples/fibonacci.zir:64)
    let x14: Val = (((x3.map(|c| c.d3)).map(|c| c._super)).load(ctx, 0)
        - ((x4.map(|c| c.f_last)).map(|c| c._super)).load(ctx, 0));
    // builtin Sub
    // IsZero(zirgen/dsl/examples/fibonacci.zir:12)
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:32)
    // Top(zirgen/dsl/examples/fibonacci.zir:49)
    let x15: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    // builtin Mul
    let x16: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * x15);
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    let x17: Val = ((((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c.inv)).map(|c| c._super))
            .load(ctx, 0));
    let x18: MixState = and_eqz(ctx, and_eqz(ctx, trivial_constraint()?, x16)?, (x17 - x15))?;
    // builtin Mul
    // IsZero(zirgen/dsl/examples/fibonacci.zir:16)
    let x19: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * (((x3.map(|c| c.cycle)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0));
    // IsZero(zirgen/dsl/examples/fibonacci.zir:18)
    let x20: Val = (((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
        .map(|c| c._super))
    .load(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c.inv)).map(|c| c._super))
            .load(ctx, 0));
    // builtin Sub
    // CycleCounter(zirgen/dsl/examples/fibonacci.zir:34)
    let x21: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x22: MixState = and_cond(
        and_eqz(ctx, and_eqz(ctx, x18, x19)?, x20)?,
        x21,
        and_eqz(ctx, trivial_constraint()?, x6)?,
    )?;
    // builtin Sub
    // Top(zirgen/dsl/examples/fibonacci.zir:55)
    let x23: Val = (Val::new(1)
        - ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x24: Val = (((x4.map(|c| c.f0)).map(|c| c._super)).load_unchecked(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x25: Val = (((x3.map(|c| c.d2)).map(|c| c._super)).load_unchecked(ctx, 1) * x23);
    // Reg(<preamble>:6)
    let x26: Val = ((x24 + x25) - ((x3.map(|c| c.d1)).map(|c| c._super)).load(ctx, 0));
    // Top(zirgen/dsl/examples/fibonacci.zir:56)
    let x27: Val = (((x4.map(|c| c.f1)).map(|c| c._super)).load_unchecked(ctx, 0)
        * ((((x3.map(|c| c.cycle)).map(|c| c.is_first_cycle)).map(|c| c._super))
            .map(|c| c._super))
        .load(ctx, 0));
    let x28: Val = (((x3.map(|c| c.d3)).map(|c| c._super)).load_unchecked(ctx, 1) * x23);
    // Reg(<preamble>:6)
    let x29: Val = ((x27 + x28) - ((x3.map(|c| c.d2)).map(|c| c._super)).load(ctx, 0));
    // Top(zirgen/dsl/examples/fibonacci.zir:59)
    let x30: MixState = and_eqz(
        ctx,
        and_eqz(ctx, and_eqz(ctx, x22, x26)?, x29)?,
        (x7 - ((x3.map(|c| c.d3)).map(|c| c._super)).load(ctx, 0)),
    )?;
    // IsZero(zirgen/dsl/examples/fibonacci.zir:14)
    // Top(zirgen/dsl/examples/fibonacci.zir:62)
    let x31: MixState = and_eqz(
        ctx,
        and_eqz(ctx, x30, x11)?,
        ((x9 * (((x3.map(|c| c.terminate)).map(|c| c.inv)).map(|c| c._super)).load(ctx, 0)) - x10),
    )?;
    // Top(zirgen/dsl/examples/fibonacci.zir:63)
    let x32: MixState = and_cond(
        and_eqz(ctx, and_eqz(ctx, x31, x12)?, x13)?,
        (((x3.map(|c| c.terminate)).map(|c| c._super)).map(|c| c._super)).load(ctx, 0),
        and_eqz(ctx, trivial_constraint()?, x14)?,
    )?;
    return Ok(x32);
}
