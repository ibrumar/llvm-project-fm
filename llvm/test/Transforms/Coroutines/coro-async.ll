; RUN: opt < %s -enable-coroutines -O2 -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { i8*, void (i8*, %async.task*, %async.actor*)* }

; The async callee.
@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(i8* %async.ctxt)

; The current async function (the caller).
; This struct describes an async function. The first field is the size needed
; for the async context of the current async function, the second field is the
; relative offset to the async function implementation.
@my_async_function_fp = constant <{ i32, i32 }>
  <{ i32 128,    ; Initial async context size without space for frame
     i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @my_async_function to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32)
  }>

; Function that implements the dispatch to the callee function.
define swiftcc void @my_async_function.my_other_async_function_fp.apply(i8* %async.ctxt, %async.task* %task, %async.actor* %actor) {
  musttail call swiftcc void @asyncSuspend(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)
  ret void
}

declare void @some_user(i64)
declare void @some_may_write(i64*)

define swiftcc void @my_async_function(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)  {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %proj.1 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 1

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i8* %async.ctxt, i8* bitcast (<{i32, i32}>* @my_async_function_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  store i64 0, i64* %proj.1, align 8
  store i64 1, i64* %proj.2, align 8
  call void @some_may_write(i64* %proj.1)

	; Begin lowering: apply %my_other_async_function(%args...)

  ; setup callee context
  %arg0 = bitcast %async.task* %task to i8*
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to i8*
  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* %arg0, i8* %arg1)
	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  ; store arguments ...
  ; ... (omitted)

  ; store the return continuation
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr

  ; store caller context into callee context
  %callee_context.caller_context.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr

  %res = call {i8*, i8*, i8*} (i8*, i8*, ...) @llvm.coro.suspend.async(
                                                  i8* %resume.func_ptr,
                                                  i8* %callee_context,
                                                  void (i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee_context, %async.task* %task, %async.actor *%actor)

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_task_arg = extractvalue {i8*, i8*, i8*} %res, 1
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*
  %val = load i64, i64* %proj.1
  call void @some_user(i64 %val)
  %val.2 = load i64, i64* %proj.2
  call void @some_user(i64 %val.2)

  tail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task.2, %async.actor* %actor)
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; Make sure we update the async function pointer
; CHECK: @my_async_function_fp = constant <{ i32, i32 }> <{ i32 168,
; CHECK: @my_async_function2_fp = constant <{ i32, i32 }> <{ i32 168,

; CHECK-LABEL: define swiftcc void @my_async_function(i8* %async.ctxt, %async.task* %task, %async.actor* %actor) {
; CHECK: entry:
; CHECK:   [[FRAMEPTR:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 128
; CHECK:   [[ACTOR_SPILL_ADDR:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 152
; CHECK:   [[CAST1:%.*]] = bitcast i8* [[ACTOR_SPILL_ADDR]] to %async.actor**
; CHECK:   store %async.actor* %actor, %async.actor** [[CAST1]]
; CHECK:   [[ADDR1:%.*]]  = getelementptr inbounds i8, i8* %async.ctxt, i64 144
; CHECK:   [[ASYNC_CTXT_SPILL_ADDR:%.*]] = bitcast i8* [[ADDR1]] to i8**
; CHECK:   store i8* %async.ctxt, i8** [[ASYNC_CTXT_SPILL_ADDR]]
; CHECK:   [[ALLOCA_PRJ1:%.*]] = bitcast i8* [[FRAMEPTR]] to i64*
; CHECK:   [[ALLOCA_PRJ2:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 136
; CHECK:   [[ADDR2:%.*]] = bitcast i8* [[ALLOCA_PRJ2]] to i64*
; CHECK:   store i64 0, i64* [[ALLOCA_PRJ1]]
; CHECK:   store i64 1, i64* [[ADDR2]]
; CHECK:   tail call void @some_may_write(i64* nonnull %proj.1)
; CHECK:   [[TASK:%.*]] = bitcast %async.task* %task to i8*
; CHECK:   [[CALLEE_CTXT:%.*]] = tail call i8* @llvm.coro.async.context.alloc(i8* [[TASK]], i8* bitcast (<{ i32, i32 }>* @my_other_async_function_fp to i8*))
; CHECK:   [[CALLEE_CTXT_SPILL:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 160
; CHECK:   [[CAST2:%.*]] = bitcast i8* [[CALLEE_CTXT_SPILL]] to i8**
; CHECK:   store i8* [[CALLEE_CTXT]], i8** [[CAST2]]
; CHECK:   [[TYPED_RETURN_TO_CALLER_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLEE_CTXT]], i64 8
; CHECK:   [[RETURN_TO_CALLER_ADDR:%.*]] = bitcast i8* [[TYPED_RETURN_TO_CALLER_ADDR]] to i8**
; CHECK:   store i8* bitcast (void (i8*, i8*, i8*)* @my_async_function.resume.0 to i8*), i8** [[RETURN_TO_CALLER_ADDR]]
; CHECK:   [[CALLER_CONTEXT_ADDR:%.*]] = bitcast i8* [[CALLEE_CTXT]] to i8**
; CHECK:   store i8* %async.ctxt, i8** [[CALLER_CONTEXT_ADDR]]
; CHECK:   musttail call swiftcc void @my_async_function.my_other_async_function_fp.apply(i8* [[CALLEE_CTXT]], %async.task* %task, %async.actor* %actor)
; CHECK:   ret void
; CHECK: }

; CHECK-LABEL: define internal swiftcc void @my_async_function.resume.0(i8* %0, i8* %1, i8* %2) {
; CHECK: entryresume.0:
; CHECK:   [[CALLER_CONTEXT_ADDR:%.*]] = bitcast i8* %0 to i8**
; CHECK:   [[CALLER_CONTEXT:%.*]] = load i8*, i8** [[CALLER_CONTEXT_ADDR]]
; CHECK:   [[FRAME_PTR:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 128
; CHECK:   [[CALLEE_CTXT_SPILL_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 160
; CHECK:   [[CAST1:%.*]] = bitcast i8* [[CALLEE_CTXT_SPILL_ADDR]] to i8**
; CHECK:   [[CALLEE_CTXT_RELOAD:%.*]] = load i8*, i8** [[CAST1]]
; CHECK:   [[ACTOR_RELOAD_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 152
; CHECK:   [[CAST2:%.*]] = bitcast i8* [[ACTOR_RELOAD_ADDR]] to %async.actor**
; CHECK:   [[ACTOR_RELOAD:%.*]] = load %async.actor*, %async.actor** [[CAST2]]
; CHECK:   [[ADDR1:%.*]] = getelementptr inbounds i8, i8* %4, i64 144
; CHECK:   [[ASYNC_CTXT_RELOAD_ADDR:%.*]] = bitcast i8* [[ADDR1]] to i8**
; CHECK:   [[ASYNC_CTXT_RELOAD:%.*]] = load i8*, i8** [[ASYNC_CTXT_RELOAD_ADDR]]
; CHECK:   [[ALLOCA_PRJ2:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 136
; CHECK:   [[ADDR2:%.*]] = bitcast i8* [[ALLOCA_PRJ2]] to i64*
; CHECK:   [[ALLOCA_PRJ1:%.*]] = bitcast i8* [[FRAME_PTR]] to i64*
; CHECK:   tail call void @llvm.coro.async.context.dealloc(i8* [[CALLEE_CTXT_RELOAD]])
; CHECK:   [[TASK_ARG:%.*]] = bitcast i8* %1 to %async.task*
; CHECK:   [[VAL1:%.*]] = load i64, i64* [[ALLOCA_PRJ1]]
; CHECK:   tail call void @some_user(i64 [[VAL1]])
; CHECK:   [[VAL2:%.*]] = load i64, i64* [[ADDR2]]
; CHECK:   tail call void @some_user(i64 [[VAL2]])
; CHECK:   tail call swiftcc void @asyncReturn(i8* [[ASYNC_CTXT_RELOAD]], %async.task* [[TASK_ARG]], %async.actor* [[ACTOR_RELOAD]])
; CHECK:   ret void
; CHECK: }

@my_async_function2_fp = constant <{ i32, i32 }>
  <{ i32 128,    ; Initial async context size without space for frame
     i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @my_async_function2 to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function2_fp, i32 0, i32 1) to i64)
       )
     to i32)
  }>

define swiftcc void @my_async_function2(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)  {
entry:

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i8* %async.ctxt, i8* bitcast (<{i32, i32}>* @my_async_function2_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  ; setup callee context
  %arg0 = bitcast %async.task* %task to i8*
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to i8*
  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* %arg0, i8* %arg1)

	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr
  %callee_context.caller_context.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr
  %res = call {i8*, i8*, i8*} (i8*, i8*, ...) @llvm.coro.suspend.async(
                                                  i8* %resume.func_ptr,
                                                  i8* %callee_context,
                                                  void (i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee_context, %async.task* %task, %async.actor *%actor)

  %continuation_task_arg = extractvalue {i8*, i8*, i8*} %res, 1
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*

	%callee_context.0.1 = bitcast i8* %callee_context to %async.ctxt*
  %callee_context.return_to_caller.addr.1 = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0.1, i32 0, i32 1
  %return_to_caller.addr.1 = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr.1 to i8**
  %resume.func_ptr.1 = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr.1, i8** %return_to_caller.addr.1
  %callee_context.caller_context.addr.1 = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0.1, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr.1
  %res.2 = call {i8*, i8*, i8*} (i8*, i8*, ...) @llvm.coro.suspend.async(
                                                  i8* %resume.func_ptr.1,
                                                  i8* %callee_context,
                                                  void (i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee_context, %async.task* %task, %async.actor *%actor)

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_actor_arg = extractvalue {i8*, i8*, i8*} %res.2, 2
  %actor.2 =  bitcast i8* %continuation_actor_arg to %async.actor*

  tail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task.2, %async.actor* %actor.2)
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define swiftcc void @my_async_function2(i8* %async.ctxt, %async.task* %task, %async.actor* %actor) {
; CHECK: store %async.actor* %actor,
; CHECK: store %async.task* %task,
; CHECK: store i8* %async.ctxt,
; CHECK: [[CALLEE_CTXT:%.*]] =  tail call i8* @llvm.coro.async.context.alloc(
; CHECK: store i8* [[CALLEE_CTXT]],
; CHECK: store i8* bitcast (void (i8*, i8*, i8*)* @my_async_function2.resume.0 to i8*),
; CHECK: store i8* %async.ctxt,
; CHECK: musttail call swiftcc void @my_async_function.my_other_async_function_fp.apply(i8* [[CALLEE_CTXT]], %async.task* %task, %async.actor* %actor)
; CHECK: ret void

; CHECK-LABEL: define internal swiftcc void @my_async_function2.resume.0(i8* %0, i8* %1, i8* %2) {
; CHECK: [[CALLEE_CTXT_ADDR:%.*]] = bitcast i8* %0 to i8**
; CHECK: [[CALLEE_CTXT:%.*]] = load i8*, i8** [[CALLEE_CTXT_ADDR]]
; CHECK: [[CALLEE_CTXT_SPILL_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLEE_CTXT]], i64 152
; CHECK: [[CALLEE_CTXT_SPILL_ADDR2:%.*]] = bitcast i8* [[CALLEE_CTXT_SPILL_ADDR]] to i8**
; CHECK: store i8* bitcast (void (i8*, i8*, i8*)* @my_async_function2.resume.1 to i8*),
; CHECK: [[CALLLE_CTXT_RELOAD:%.*]] = load i8*, i8** [[CALLEE_CTXT_SPILL_ADDR2]]
; CHECK: musttail call swiftcc void @my_async_function.my_other_async_function_fp.apply(i8* [[CALLEE_CTXT_RELOAD]]
; CHECK: ret void

; CHECK-LABEL: define internal swiftcc void @my_async_function2.resume.1(i8* %0, i8* %1, i8* %2) {
; CHECK: [[ACTOR_ARG:%.*]] = bitcast i8* %2
; CHECK: tail call swiftcc void @asyncReturn({{.*}}[[ACTOR_ARG]])
; CHECK: ret void

declare token @llvm.coro.id.async(i32, i32, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)
declare {i8*, i8*, i8*} @llvm.coro.suspend.async(i8*, i8*, ...)
declare i8* @llvm.coro.async.context.alloc(i8*, i8*)
declare void @llvm.coro.async.context.dealloc(i8*)
declare swiftcc void @asyncReturn(i8*, %async.task*, %async.actor*)
declare swiftcc void @asyncSuspend(i8*, %async.task*, %async.actor*)
declare i8* @llvm.coro.async.resume()