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
pub struct NondetMyArgumentLayout {
    pub a: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for NondetMyArgumentLayout {
    fn ty_name(&self) -> &'static str {
        "NondetMyArgumentLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("a", self.a)?;
        v.visit_component("b", self.b)?;
        Ok(())
    }
}
pub type NondetRegLayout2LayoutArray = [&'static NondetRegLayout; 2];
pub struct Top__0Layout {
    pub _super: &'static NondetMyArgumentLayout,
    pub _selector: &'static NondetRegLayout2LayoutArray,
    pub arm0: &'static NondetMyArgumentLayout,
    pub arm1: &'static NondetMyArgumentLayout,
}
impl risc0_zkp::layout::Component for Top__0Layout {
    fn ty_name(&self) -> &'static str {
        "Top__0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_super", self._super)?;
        v.visit_component("_selector", self._selector)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type Reg1LayoutArray = [&'static Reg; 1];
pub struct TopLayout {
    pub odd: &'static NondetRegLayout,
    pub _0: &'static NondetRegLayout,
    pub _1: &'static Top__0Layout,
}
impl risc0_zkp::layout::Component for TopLayout {
    fn ty_name(&self) -> &'static str {
        "TopLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("odd", self.odd)?;
        v.visit_component("_0", self._0)?;
        v.visit_component("_1", self._1)?;
        Ok(())
    }
}
pub struct _globalLayout {
    pub globzero: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for _globalLayout {
    fn ty_name(&self) -> &'static str {
        "_globalLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("globzero", self.globzero)?;
        Ok(())
    }
}
pub struct Arg_NondetMyArgumentLayout {
    pub b: &'static Reg,
}
impl risc0_zkp::layout::Component for Arg_NondetMyArgumentLayout {
    fn ty_name(&self) -> &'static str {
        "Arg_NondetMyArgumentLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("b", self.b)?;
        Ok(())
    }
}
pub struct _accumLayout {
    pub nondet_my_argument: &'static Arg_NondetMyArgumentLayout,
    pub _offset: &'static Reg,
}
impl risc0_zkp::layout::Component for _accumLayout {
    fn ty_name(&self) -> &'static str {
        "_accumLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("nondet_my_argument", self.nondet_my_argument)?;
        v.visit_component("_offset", self._offset)?;
        Ok(())
    }
}
pub struct _mixLayout {
    pub randomness: &'static _accumLayout,
}
impl risc0_zkp::layout::Component for _mixLayout {
    fn ty_name(&self) -> &'static str {
        "_mixLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("randomness", self.randomness)?;
        Ok(())
    }
}
#[derive(Copy, Clone, Debug)]
pub struct NondetRegStruct {
    pub _super: Val,
}
#[derive(Copy, Clone, Debug)]
pub struct NondetMyArgumentStruct {}
#[derive(Copy, Clone, Debug)]
pub struct MyArgumentStruct {}
#[derive(Copy, Clone, Debug)]
pub struct TopStruct {}
#[derive(Copy, Clone, Debug)]
pub struct ComponentStruct {}
