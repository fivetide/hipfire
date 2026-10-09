// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next exact AR lanes. Prompt IDs come from `qwen::render_qwen4_prompt`,
//! also used by `generate_qwen4_ar` and the native MTP singleton. Emission
//! reuses `generate_ar_with_forward`'s `QwenArSemanticProducer`, incremental
//! tokenizer decode, raw commit, committed-event, think-budget and done helpers.
//! Unlike the Qwen35 driver there is no loop guard or tool-markup parser when
//! tools are absent. A prompt's seed is held until its AR forward completes:
//! the singleton forwards the token BEFORE committing/classifying it. The
//! executor's AR pick is the following token, not the token just forwarded.
//! No lane session cache, promotion or parking is performed here.
//! Timing fields measure this execution and are not byte-identical wall-clock
//! values. `cached_tokens` is zero: S3 uses cold lane state, not singleton
//! session restore. Content/reasoning bytes and semantic finish classification
//! use the singleton code; compute-ID identity is the lane executor's oracle.

use crate::ar::*;
use crate::batch::{drain_qwen_batch_inbox, handoff_started_in_think, is_vmm_batch_request_eligible};
use crate::common::{emit_committed_event, fit_max_tokens_if};
use hipfire_arch_qwen4::lane::LaneStatus;
use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_loader::LoadedModel;
use hipfire_runtime::emit_text::ThinkBudget;
use hipfire_runtime::slot_batch::{BatchStepPlan, RequestEpoch, RequestRows, RequestStepKind, RowRange};
use rdna_compute::Gpu;
use std::collections::{HashSet, VecDeque};
use std::io::{Stdout, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const ROUTE: GenerationRoute = GenerationRoute::Qwen4Ar;
static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

struct LaneCtl {
    epoch: RequestEpoch,
    slot: usize,
    semantic: Option<QwenArSemanticProducer>,
    bytes: Vec<u8>,
    budget: ThinkBudget,
    forced: VecDeque<u32>,
    started_in_think: bool,
    max_tokens: usize,
    start: Instant,
    decode_start: Option<Instant>,
    prefill_ms: f64,
    cache: Option<(QwenArCacheAction, Vec<u32>)>,
}

fn release(model: &mut LoadedModel, ctl: &mut LaneCtl) -> Result<(), String> {
    if !ctl.epoch.is_admitted() {
        return Ok(());
    }
    model.qwen4_mut().ok_or("qwen4 bundle disappeared")?
        .release_lane(&ctl.epoch).map_err(|e| e.to_string())?;
    ctl.epoch.owner_generation = 0;
    Ok(())
}

fn lane_error(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    idx: usize,
    stdout: &mut Stdout,
    message: &str,
    class: &str,
) -> Result<(), String> {
    let (key, admission) = match &sched.lanes[idx] {
        BatchLane::Running(l) | BatchLane::Seeding(l) => (l.key.clone(), l.ticket.admission),
        BatchLane::AwaitingClient(l) => (l.key.clone(), l.ticket.admission),
        BatchLane::Empty { .. } => return Ok(()),
    };
    let released = controls[idx].as_mut().map_or(Ok(()), |c| release(model, c));
    let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
    emit_generation_error(ROUTE, stdout, Some(&key.id), message, class, false, released.is_ok());
    let _ = sched.abort_lane(idx, &key, admission);
    controls[idx] = None;
    released
}

fn cancel_lane(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    idx: usize,
    stdout: &mut Stdout,
) -> Result<(), String> {
    let (key, admission, generated) = match &sched.lanes[idx] {
        BatchLane::Running(l) | BatchLane::Seeding(l) =>
            (l.key.clone(), l.ticket.admission, l.streamed_tokens.len()),
        BatchLane::AwaitingClient(l) => (l.key.clone(), l.ticket.admission,
            l.pending_done.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize),
        BatchLane::Empty { .. } => return Ok(()),
    };
    if let Some(c) = controls[idx].as_mut() {
        release(model, c)?;
    }
    let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
    emit_generation_cancel(ROUTE, stdout, &key.id, generated);
    let _ = sched.abort_lane(idx, &key, admission);
    controls[idx] = None;
    Ok(())
}

fn render_request(
    model: &mut LoadedModel,
    request: &BatchPendingRequest,
) -> Result<(Vec<u32>, bool), (String, &'static str)> {
    if request.max_tokens == 0 {
        return Err(("max_tokens must be > 0".to_string(), "validation"));
    }
    let msg = &request.original_msg;
    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(
        msg.get("prompt").and_then(|v| v.as_str()).unwrap_or("Hello"),
    );
    let history = hipfire_engine::prompt::parse_generate_messages(msg)
        .map_err(|e| (format!("invalid messages field: {e}"), "validation"))?;
    let tools = msg.get("tools").and_then(|v| v.as_array())
        .filter(|a| !a.is_empty()).map(|a| a.as_slice());
    let raw_effort = msg.get("reasoning_effort").or_else(|| msg.get("thinking_mode"))
        .and_then(|v| v.as_str());
    let (enabled, effort) = hipfire_engine::prompt::qwen_jinja_reasoning(
        msg.get("thinking_enabled").and_then(|v| v.as_bool()),
        raw_effort, request.max_think_tokens,
    );
    crate::qwen::render_qwen4_prompt(model, crate::qwen::Qwen4PromptRoute::Ar,
        msg.get("system").and_then(|v| v.as_str()), &prompt,
        request.assistant_prefix, tools, history.as_deref(), enabled, effort.as_deref())
}

/// Build directly rather than using `BatchPlanner::forced_prefill_len`: that
/// planner charges exact singleton prefill tiles against the AR row budget,
/// so a tile larger than the budget could never run. Here only AR rows use
/// that budget; every prefilling owner gets at most one indivisible tile.
fn plan_step(
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    row_budget: usize,
    cursor: &mut usize,
    max_lanes: usize,
) -> Result<BatchStepPlan, String> {
    let bundle = model.qwen4_mut().ok_or("qwen4 bundle disappeared")?;
    let mut order: Vec<_> = controls.iter().flatten()
        .filter(|c| c.epoch.is_admitted()).map(|c| (c.slot, c.epoch)).collect();
    order.sort_by_key(|&(slot, _)| slot);
    let store = bundle.lanes().ok_or("qwen4 lanes not staged")?;
    let decoding: Vec<_> = order.iter().filter_map(|&(slot, epoch)| {
        store.lane(&epoch).filter(|l| l.host.status == LaneStatus::Decoding).map(|_| slot)
    }).collect();
    let count = decoding.len();
    let chosen: Vec<_> = (0..count.min(row_budget)).map(|j| decoding[(*cursor + j) % count]).collect();
    if count != 0 {
        *cursor = (*cursor + count.min(row_budget)) % count;
    }
    let mut rows = Vec::new();
    for (slot, epoch) in order {
        let lane = store.lane(&epoch).ok_or("admitted qwen4 lane disappeared")?;
        match lane.host.status {
            LaneStatus::Prefilling => rows.push((slot, epoch, RequestStepKind::Prefill,
                lane.host.prefilled, 0)),
            LaneStatus::Decoding if chosen.contains(&slot) => rows.push((slot, epoch, RequestStepKind::Ar,
                lane.state.position, 1)),
            LaneStatus::Decoding => {}
            LaneStatus::Poisoned => return Err("qwen4 lane poisoned outside a step".into()),
        }
    }
    for c in controls.iter_mut().flatten().filter(|c| chosen.contains(&c.slot)) {
        if let Some(token) = c.forced.pop_front() {
            bundle.force_lane_seed(&c.epoch, token).map_err(|e| e.to_string())?;
        }
    }
    {
        let executor = bundle.lane_executor().map_err(|e| e.to_string())?;
        for (_, epoch, kind, _, len) in &mut rows {
            if *kind == RequestStepKind::Prefill {
                *len = executor.exact_prefill_chunk_len(epoch)?;
            }
        }
    }
    let store = bundle.lanes().ok_or("qwen4 lanes not staged")?;
    let mut plan = BatchStepPlan::default();
    plan.batch.m_per_slot = vec![0; max_lanes];
    for (slot, epoch, kind, position, len) in rows {
        if len == 0 {
            return Err("qwen4 exact prefill tile is empty".into());
        }
        let lane = store.lane(&epoch).ok_or("admitted qwen4 lane disappeared")?;
        let begin = plan.batch.tokens.len();
        if kind == RequestStepKind::Prefill {
            let end = position.checked_add(len).ok_or("qwen4 tile position overflow")?;
            let slice = lane.host.prompt.get(position..end).ok_or("qwen4 prefill tile exceeds prompt")?;
            plan.batch.tokens.extend_from_slice(slice);
        } else {
            plan.batch.tokens.push(lane.host.pending_seed.ok_or("decoding lane has no pending seed")?);
        }
        for j in 0..len {
            plan.batch.positions.push(i32::try_from(position + j).map_err(|_| "lane position exceeds i32")?);
            plan.batch.row_slot.push(i32::try_from(slot).map_err(|_| "lane slot exceeds i32")?);
        }
        plan.batch.m_per_slot[slot] = len;
        plan.requests.push(RequestRows { epoch, rows: RowRange { begin, len }, kind });
        if kind == RequestStepKind::Ar { plan.decode_rows += len; } else { plan.prefill_rows += len; }
    }
    Ok(plan)
}

fn emit_forwarded(
    stdout: &mut Stdout,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    lane: &mut QwenBatchLane,
    ctl: &mut LaneCtl,
    token: u32,
    eos: u32,
    adv_length: bool,
) -> Result<Option<bool>, String> {
    let fed = ctl.bytes.len();
    tokenizer.decode_token_bytes_into(token, &mut ctl.bytes);
    let new_bytes = &ctl.bytes[fed..];
    let elapsed = ctl.start.elapsed().as_millis() as u64;
    lane.first_token_at.get_or_insert_with(Instant::now);
    let id = &lane.key.id;
    let stopped = ctl.semantic.as_mut().ok_or("lane producer missing")?.commit_and_classify(
        stdout, token,
        || {
            let pos = qwen_ar_raw_commit_token(&mut lane.conversation_tokens, &mut lane.streamed_tokens,
                &mut lane.seq_pos, token, QwenArRawCommitDisposition::ClassifiedVisible);
            (pos, new_bytes)
        },
        |pos, output| emit_committed_event(output, id, token, pos, elapsed),
    ).map_err(|e| e.to_string())?;
    if stopped || token == eos || tokenizer.is_terminator(token) {
        return Ok(Some(false));
    }
    match qwen_ar_think_budget_step(&mut ctl.budget, &ctl.bytes, ctl.started_in_think, tokenizer) {
        QwenArThinkBudgetAction::Continue => {}
        QwenArThinkBudgetAction::Close(tokens) => ctl.forced.extend(tokens),
        QwenArThinkBudgetAction::Stop => return Ok(Some(false)),
    }
    Ok((lane.streamed_tokens.len() >= ctl.max_tokens || adv_length).then_some(true))
}

fn finish_lane(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    controls: &mut [Option<LaneCtl>],
    idx: usize,
    stdout: &mut Stdout,
    length: bool,
) -> Result<(), String> {
    let ctl = controls[idx].as_mut().ok_or("lane control missing")?;
    let BatchLane::Running(lane) = &sched.lanes[idx] else { return Err("finished lane is not running".into()); };
    let key = lane.key.clone();
    let admission = lane.ticket.admission;
    let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
    let (finish, visible) = ctl.semantic.take().ok_or("lane producer missing")?
        .finish(stdout, length).map_err(|e| e.to_string())?;
    let now = Instant::now();
    let generated = lane.streamed_tokens.len();
    let total = now.duration_since(ctl.start).as_secs_f64();
    let decode = ctl.decode_start.map_or(0.0, |t| now.duration_since(t).as_secs_f64());
    let mut done = qwen_ar_done_value(&key.id, finish.finish_reason, generated,
        batch_finite_rate(generated, total), lane.prompt_len, ctl.prefill_ms,
        batch_finite_rate(lane.prompt_len, ctl.prefill_ms / 1000.0),
        batch_finite_rate(generated, decode), ctl.prefill_ms, 0, "");
    stage_terminal_tool_calls(&mut done, finish.finish_reason, &finish.wire_tool_calls);
    let action = qwen_ar_cache_action(&finish, &visible);
    if action.store {
        let tokenizer = model.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
        let im_end = tokenizer.encode("<|im_end|>");
        let im_end = if im_end.len() == 1 { Some(im_end[0]) } else { None };
        let nl = tokenizer.encode("\n").into_iter().collect();
        ctl.cache = Some((action, crate::qwen::qwen_dflash_cache_seq(&lane.streamed_tokens, im_end, &nl)));
    }
    release(model, ctl)?;
    let mut ready = done.clone();
    ready["type"] = serde_json::json!("commit_ready");
    if !sched.mark_awaiting_commit(idx, done) {
        return Err("qwen4 mark_awaiting_commit failed".into());
    }
    writeln!(stdout, "{ready}").and_then(|()| stdout.flush()).map_err(|e| e.to_string())
}

/// Serve concurrently admitted greedy Qwen4 requests using the exact lane
/// executor. All terminal ownership remains in `ContinuousBatchScheduler`.
pub fn drive_qwen4_lane_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut Gpu,
    m: &mut LoadedModel,
    params: VmmBatchParams,
    stdout: &mut Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let bundle = m.qwen4().ok_or_else(|| BatchDriveError::Gpu("lane model is not qwen4".into()))?;
    let store = bundle.lanes().ok_or_else(|| BatchDriveError::Gpu("qwen4 lanes not staged".into()))?;
    let max_lanes = store.max_lanes();
    let row_budget = store.row_budget().min(params.max_batch_tokens);
    let eos = bundle.config.eos_token_id;
    if row_budget == 0 || sched.max_batch > max_lanes {
        return Err(BatchDriveError::Gpu("qwen4 lane scheduler/budget does not match staged store".into()));
    }
    let mut controls: Vec<Option<LaneCtl>> = (0..sched.max_batch).map(|_| None).collect();
    let mut cursor = 0;
    let batch_size = sched.max_batch;
    let initial: HashSet<_> = sched.inbox.iter().cloned().collect();
    let result = (|| -> Result<(), String> {
        loop {
            // Terminal decisions must not block other lanes' GPU work.
            for idx in 0..sched.max_batch {
                let decision = match &sched.lanes[idx] {
                    BatchLane::AwaitingClient(t) => {
                        if batch_check_abort(&t.key.id, t.key.attempt_id, t.ticket.admission) || Instant::now() >= t.deadline {
                            Some(false)
                        } else if batch_poll_decision(&t.key.id, t.key.attempt_id, t.ticket.admission) == Some(ClientTerminalDecision::Commit) {
                            Some(true)
                        } else { None }
                    }
                    BatchLane::Running(l) | BatchLane::Seeding(l) if batch_check_abort(&l.key.id, l.key.attempt_id, l.ticket.admission) => Some(false),
                    _ => None,
                };
                match decision {
                    Some(false) => cancel_lane(sched, m, &mut controls, idx, stdout)?,
                    Some(true) => {
                        let BatchLane::AwaitingClient(t) = &sched.lanes[idx] else { unreachable!() };
                        let key = t.key.clone();
                        let admission = t.ticket.admission;
                        let done = t.pending_done.clone();
                        let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                        if !sched.commit_lane_retain_terminal(idx, &key, admission) {
                            lane_error(sched, m, &mut controls, idx, stdout, "qwen4 commit_lane failed", "internal")?;
                        } else {
                            if let Some(ctl) = controls[idx].as_mut() {
                                if let Some((action, tokens)) = ctl.cache.take() {
                                    let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
                                    qwen_ar_apply_cache_action(|fp, seq| {
                                        m.asst_turn_cache.insert(fp, crate::qwen::qwen_cached_turn_entry(tokenizer, seq, ctl.started_in_think));
                                    }, &action, tokens);
                                }
                            }
                            emit_generation_done_value(ROUTE, stdout, &done);
                            batch_clear_terminal_at_generation(&key.id, key.attempt_id, admission);
                            controls[idx] = None;
                        }
                    }
                    None => {}
                }
            }
            for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
                if let Some(req) = sched.pending.get(&key) {
                    let admission = req.admission;
                    if batch_check_abort(&key.id, key.attempt_id, admission) {
                        let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                        emit_generation_start(ROUTE, stdout, &key.id, false);
                        emit_generation_cancel(ROUTE, stdout, &key.id, 0);
                        sched.abort_queued(&key, admission);
                    }
                }
            }
            let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer missing")?;
            if let Some(barrier) = drain_qwen_batch_inbox(sched, m, batch_size,
                is_vmm_batch_request_eligible, tokenizer, m.chat_template.as_ref(), ROUTE, stdout, inbox, true)? {
                inbox.push_front(barrier);
            }
            while let Some((key, ticket)) = sched.try_assign_one() {
                let req = sched.pending.get(&key).cloned().ok_or("assigned request missing")?;
                if !initial.contains(&key) && sched.inbox.is_empty()
                    && !controls.iter().flatten().any(|c| c.epoch.is_admitted()) {
                    inbox.push_front(handoff_started_in_think(sched, ticket.lane, &key, &req, ROUTE)?);
                    continue;
                }
                let rendered = render_request(m, &req);
                let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, ticket.admission);
                let (prompt, started) = match rendered {
                    Ok(r) => r,
                    Err((error, class)) => {
                        lane_error(sched, m, &mut controls, ticket.lane, stdout, &error, class)?;
                        continue;
                    }
                };
                let max_tokens = fit_max_tokens_if(req.original_msg.get("max_tokens_fit").and_then(|v| v.as_bool()) == Some(true),
                    req.max_tokens, prompt.len() + 1, m.max_seq);
                let required = prompt.len().checked_add(max_tokens).and_then(|n| n.checked_add(1));
                let refusal = if req.max_tokens == 0 {
                    Some(("max_tokens must be > 0".to_string(), "validation"))
                } else if prompt.is_empty() {
                    Some(("qwen4 AR prompt rendered to zero tokens".to_string(), "validation"))
                } else if required.is_none_or(|n| n > m.max_seq) {
                    Some((format!("qwen4 request exceeds context window: prompt={} + max_tokens={} + trailer=1 > max_seq={}", prompt.len(), max_tokens, m.max_seq), "context_length"))
                } else { None };
                if let Some((message, class)) = refusal {
                    lane_error(sched, m, &mut controls, ticket.lane, stdout, &message, class)?;
                    continue;
                }
                let stop = hipfire_runtime::stop_sequence::parse_stop_field(req.original_msg.get("stop")).map_err(|e| e.to_string())?;
                let tag = NEXT_EPOCH.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                    .map_err(|_| "qwen4 lane epoch exhausted")?;
                let epoch = RequestEpoch { request_tag: tag, owner_generation: 1 };
                let slot = match m.qwen4_mut().ok_or("qwen4 bundle disappeared")?.admit_lane(gpu, epoch, prompt.clone()) {
                    Ok(slot) => slot,
                    Err(e) => {
                        lane_error(sched, m, &mut controls, ticket.lane, stdout, &e.to_string(), "context_length")?;
                        continue;
                    }
                };
                emit_generation_start(ROUTE, stdout, &key.id, started);
                // The singleton tools parser maps an empty array to None.
                let tools = req.original_msg.get("tools").and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty());
                controls[ticket.lane] = Some(LaneCtl {
                    epoch, slot, semantic: Some(QwenArSemanticProducer::new_with_tool_protocol(&key.id, started, tools).with_stop(&stop)),
                    bytes: Vec::new(), budget: ThinkBudget::new(req.max_think_tokens), forced: VecDeque::new(),
                    started_in_think: started, max_tokens, start: Instant::now(), decode_start: None, prefill_ms: 0.0, cache: None,
                });
                let BatchLane::Running(lane) = &mut sched.lanes[ticket.lane] else { return Err("assigned lane is not running".into()); };
                lane.prompt_len = prompt.len();
                lane.seq_pos = prompt.len();
                lane.conversation_tokens = prompt;
            }
            if sched.active_count() == 0 && sched.inbox.is_empty() { return Ok(()); }
            let plan = plan_step(m, &mut controls, row_budget, &mut cursor, max_lanes)?;
            if plan.requests.is_empty() {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            let stepped = {
                let bundle = m.qwen4_mut().ok_or("qwen4 bundle disappeared")?;
                let mut executor = bundle.lane_executor().map_err(|e| e.to_string())?;
                let result = executor.provision_step(gpu, &plan)
                    .and_then(|()| executor.forward_step(gpu, &plan))
                    .and_then(|out| executor.commit_step(gpu, &plan, out));
                if result.is_err() { executor.abort_step(&plan); }
                result
            };
            let advances = match stepped {
                Ok(a) => a,
                Err(error) => {
                    for request in &plan.requests {
                        if let Some(idx) = controls.iter().position(|c| c.as_ref().is_some_and(|c| c.epoch == request.epoch)) {
                            lane_error(sched, m, &mut controls, idx, stdout, &format!("qwen4 lane step failed: {error}"), "gpu")?;
                        }
                    }
                    continue;
                }
            };
            for advance in advances {
                let idx = controls.iter().position(|c| c.as_ref().is_some_and(|c| c.epoch == advance.epoch))
                    .ok_or("qwen4 advance has no request owner")?;
                let request = plan.requests.iter().find(|r| r.epoch == advance.epoch).ok_or("qwen4 advance not in plan")?;
                if request.kind == RequestStepKind::Prefill {
                    if !advance.committed_ids.is_empty() {
                        let ctl = controls[idx].as_mut().ok_or("lane control missing")?;
                        ctl.prefill_ms = ctl.start.elapsed().as_secs_f64() * 1000.0;
                        ctl.decode_start = Some(Instant::now());
                        if let BatchLane::Running(lane) = &mut sched.lanes[idx] { lane.prefill_done_at = ctl.decode_start; }
                    }
                    if advance.finish.as_deref() == Some("length") {
                        finish_lane(sched, m, &mut controls, idx, stdout, true)?;
                    }
                    continue;
                }
                // All step participants committed before any bytes are published.
                if let BatchLane::Running(lane) = &sched.lanes[idx] {
                    if batch_check_abort(&lane.key.id, lane.key.attempt_id, lane.ticket.admission) {
                        cancel_lane(sched, m, &mut controls, idx, stdout)?;
                        continue;
                    }
                }
                let BatchLane::Running(lane) = &mut sched.lanes[idx] else { return Err("advanced lane is not running".into()); };
                let _scope = BatchAttemptScope::enter_for_generation(&lane.key.id, lane.key.attempt_id, lane.ticket.admission);
                let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
                let emitted = emit_forwarded(stdout, tokenizer, lane, controls[idx].as_mut().ok_or("lane control missing")?,
                    plan.batch.tokens[request.rows.begin], eos, advance.finish.as_deref() == Some("length"));
                match emitted {
                    Ok(Some(length)) => {
                        if let Err(error) = finish_lane(sched, m, &mut controls, idx, stdout, length) {
                            lane_error(sched, m, &mut controls, idx, stdout, &error, "validation")?;
                        }
                    }
                    Ok(None) => {}
                    Err(error) => lane_error(sched, m, &mut controls, idx, stdout, &error, "validation")?,
                }
            }
        }
    })();
    if let Err(error) = result {
        let mut cleanup_error = None;
        for idx in 0..sched.max_batch {
            if let Err(e) = lane_error(sched, m, &mut controls, idx, stdout, &error, "gpu") { cleanup_error.get_or_insert(e); }
        }
        for (key, req) in &sched.pending {
            let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, req.admission);
            emit_generation_error(ROUTE, stdout, Some(&key.id), &error, "gpu", false, cleanup_error.is_none());
        }
        sched.fail_all_active();
        return Err(match cleanup_error {
            Some(e) => BatchDriveError::Poisoned(format!("{error}; lane release failed: {e}")),
            None => BatchDriveError::Gpu(error),
        });
    }
    Ok(())
}
