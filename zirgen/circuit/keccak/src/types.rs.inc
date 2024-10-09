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
pub struct MyArgumentLayout {
    pub a: &'static NondetRegLayout,
    pub b: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for MyArgumentLayout {
    fn ty_name(&self) -> &'static str {
        "MyArgumentLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("a", self.a)?;
        v.visit_component("b", self.b)?;
        Ok(())
    }
}
pub struct Top__0Layout {
    pub _selector: &'static NondetRegLayout2LayoutArray,
    pub arm0: &'static MyArgumentLayout,
    pub arm1: &'static NondetRegLayout,
}
impl risc0_zkp::layout::Component for Top__0Layout {
    fn ty_name(&self) -> &'static str {
        "Top__0Layout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("_selector", self._selector)?;
        v.visit_component("arm0", self.arm0)?;
        v.visit_component("arm1", self.arm1)?;
        Ok(())
    }
}
pub type Reg1LayoutArray = [&'static Reg; 1];
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
    pub _1: &'static NondetRegLayout,
    pub _2: &'static NondetRegLayout,
    pub _3: &'static Top__0Layout,
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
        v.visit_component("_1", self._1)?;
        v.visit_component("_2", self._2)?;
        v.visit_component("_3", self._3)?;
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
pub struct Arg_MyArgumentLayout {
    pub b: &'static Reg,
}
impl risc0_zkp::layout::Component for Arg_MyArgumentLayout {
    fn ty_name(&self) -> &'static str {
        "Arg_MyArgumentLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("b", self.b)?;
        Ok(())
    }
}
pub struct _accumLayout {
    pub my_argument: &'static Arg_MyArgumentLayout,
    pub _offset: &'static Reg,
}
impl risc0_zkp::layout::Component for _accumLayout {
    fn ty_name(&self) -> &'static str {
        "_accumLayout"
    }
    #[allow(unused_variables)]
    fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
        v.visit_component("my_argument", self.my_argument)?;
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
pub type Val2Array = [Val; 2];
pub type NondetRegStruct2Array = [NondetRegStruct; 2];
#[derive(Copy, Clone, Debug)]
pub struct OneHotStruct {
    pub _super: NondetRegStruct2Array,
}
#[derive(Copy, Clone, Debug)]
pub struct ComponentStruct {}
#[derive(Copy, Clone, Debug)]
pub struct TopStruct {}
