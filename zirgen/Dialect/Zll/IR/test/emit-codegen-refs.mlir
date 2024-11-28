// RUN: zirgen-translate -rust-codegen %s | FileCheck %s --check-prefixes=CHECK

!emptyStruct = !zstruct.struct<Empty, <>>
!structWithEmpty = !zstruct.struct<WithEmpty, <"empty": !emptyStruct>>
// CHECK-LABEL: struct WithEmptyStruct
// CHECK-NEXT: empty: EmptyStruct,

!emptyLayout = !zstruct.layout<Empty, <>>
!layoutWithEmpty = !zstruct.layout<WithEmpty, <"empty": !emptyLayout>>
// CHECK-LABEL: struct WithEmptyLayout
// CHECK-NEXT: empty: &'static EmptyLayout,

zstruct.global_const @emptyLayout : !emptyLayout = #zstruct<bound_layout "empty" as #zstruct<struct {}> : !emptyLayout>

func.func @ref_clone_test(%arg0 : !zll.val<BabyBear>, %arg1 : !emptyLayout, %arg2: !emptyStruct, %arg4 : !layoutWithEmpty,   %globbuf : !zll.buffer<4, global>) -> !structWithEmpty {
  // CHECK-LABEL: fn ref_clone_test
  // CHECK-SAME: (arg0: Val, arg1: BoundLayout<'a, EmptyLayout, Val>, arg2: &EmptyStruct, arg3: BoundLayout<'a, WithEmptyLayout, Val>, arg4: BufferRow<Val>) -> Result<WithEmptyStruct>

  // arg1 is already a reference and must always be continued to be passed by reference.
  func.call @layout_black_box(%arg1) : (!emptyLayout) -> ()
  func.call @layout_black_box(%arg1) : (!emptyLayout) -> ()
  // CHECK: layout_black_box(arg1)
  // CHECK: layout_black_box(arg1)

  // Layouts should be passed by reference
  %glob0 = zstruct.bind_layout @emptyLayout -> !emptyLayout = %globbuf : <4, global>
  func.call @layout_black_box(%glob0) : (!emptyLayout) -> ()
  // CHECK: bind_layout
  // CHECK: layout_black_box(x

  // Passing arg2 on should just pass the reference.
  func.call @struct_black_box(%arg2) : (!emptyStruct) -> ()
  // CHECK: struct_black_box(arg2)
  func.call @struct_black_box(%arg2) : (!emptyStruct) -> ()
  // CHECK: struct_black_box(arg2)

  // Packing arg2 should clone it every time.
  %argclone0 = zstruct.pack(%arg2 : !emptyStruct) : !structWithEmpty
  func.call @struct_with_empty_black_box(%argclone0) : (!structWithEmpty) -> ()
  // CHECK: empty: arg2.clone()
  // CHECK: struct_with_empty_black_box
  %argclone1 = zstruct.pack(%arg2 : !emptyStruct) : !structWithEmpty
  func.call @struct_with_empty_black_box(%argclone0) : (!structWithEmpty) -> ()
  // CHECK: empty: arg2.clone()
  // CHECK: struct_with_empty_black_box

  %localclone0 = zstruct.pack() : !emptyStruct
  // CHECK: let [[EMPTY:x[0-9]+]] : EmptyStruct = EmptyStruct{
  // Locally constructed structure should be cloned all times except
  // the last one, and passed by reference.
  %localclone1 = zstruct.pack(%localclone0 : !emptyStruct) : !structWithEmpty
  func.call @struct_with_empty_black_box(%localclone1) : (!structWithEmpty) -> ()
  // CHECK: struct_with_empty_black_box(&WithEmptyStruct{
  // CHECK: empty: [[EMPTY]].clone(),
  %localclone2 = zstruct.pack(%localclone0 : !emptyStruct) : !structWithEmpty
  func.call @struct_with_empty_black_box(%localclone2) : (!structWithEmpty) -> ()
  func.call @struct_with_empty_black_box(%localclone2) : (!structWithEmpty) -> ()
  // CHECK: empty: [[EMPTY]],
  // CHECK: struct_with_empty_black_box(&
  // CHECK: struct_with_empty_black_box(&

  // Returning one of these should move the value, and should not clone it.
  func.return %localclone2 : !structWithEmpty
  // return Ok(x[[[0-9]+]])
}

func.func @layout_black_box(%arg0 : !emptyLayout) {
  func.return
}

func.func @struct_black_box(%0 : !emptyStruct) {
  func.return
}

func.func @struct_with_empty_black_box(%0 : !structWithEmpty) {
  func.return
}
