#!/usr/bin/env python3
"""Robot-side CPU bbox tracking with remote GPU ACT inference.

Standalone bbox processing; no LeRobot source edits or companion scripts required. No datasets are rewritten. Each invocation is one trial;
start at the agreed robot pose, initialize current object ROI, then run the policy.

로깅 v4: 기존 실행 옵션에 --experiment-name mix30k_positionA 만 추가하면 된다.
기본 폴더: act_bb_eval_runs/날짜_시각_실험명_입력모드/ (매 실행 새 폴더)
영상: comparison.mp4 (왼쪽 원본, 오른쪽 정책 입력), 기본 15fps.
--bb-video-fps 30 으로 변경하거나 --no-bb-save-video 로 끌 수 있다.

predictions.jsonl: 서버가 반환한 블렌딩 전 canonical 청크 + 요청 식별자/시각
queue_actions.jsonl: 블렌딩 후 큐에서 꺼낸 action (보간 전)
control.jsonl: tick별 관측 state, 보간 후 canonical action, 전송 목표값
motor_commands.jsonl: 드라이버 호출과 반환 목표값, hold 이벤트
async_events.jsonl: 청크 요청/수락/overlap/blend/큐 잔량
tracking.jsonl: 원래 박스/마진 적용 박스, 검출 점수/재검출 이벤트
inputs.jsonl, video_frames.jsonl: 입력 sequence/capture_time과 영상 프레임 연결
trial.json: 설정, 서버 artifact 식별자, 수동 결과, 루프 시간, 기록 누락 수

서버의 요청 capture_time을 inputs/video의 capture_time과 맞춰 비교한다.
영상에는 최신 입력 슬롯에 제출한 프레임이 들어가며 서버가 그 모두를 쓰지는 않는다.
모든 monotonic 시각은 로봇 PC 기준이며, 측정 state는 명령 전 관측값이다.
driver_reported_action은 전송 목표값이지 물리적으로 도달한 자세가 아니다.
S/F로 결과를 직접 표시하며, Q/시간 종료/Ctrl+C는 success=null 이다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
import importlib
import inspect
import json
import math
import re
from queue import Queue, Empty, Full
from pathlib import Path
import sys
from threading import Condition, Thread, Event, local
from types import SimpleNamespace
import time
import traceback

import numpy as np


def json_value(value):
    """로깅 전용 변환. 텐서는 작은 CPU action만 복사하고 영상은 별도 큐로 보낸다."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return {k: json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def action_copy(value):
    """제어 경로에서 쓰는 작은 숫자 배열/사전의 독립 스냅샷."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: action_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [action_copy(v) for v in value]
    return value


class TrialRecorder:
    """JSON과 영상 인코딩을 별도 스레드로 분리. 큐가 차면 기록만 누락시킨다.

    영상은 원본|정책입력을 한 파일에 저장한다. CFR mp4 재생시간 대신
    video_frames.jsonl의 capture_time을 기준으로 action/state와 정렬해야 한다.
    """
    def __init__(self, root, cv2, video_fps=15., save_video=True, video_queue=8):
        self.root, self.cv2 = root, cv2
        self.video_fps, self.save_video = video_fps, save_video
        self.rows = Queue(maxsize=8192)
        self.frames = Queue(maxsize=video_queue)
        self.stopping = Event()
        self.json_dropped = self.video_dropped = self.video_sampled_out = 0
        self.json_written = self.video_written = 0
        self.errors = []
        self._last_video_time = None
        self.io_thread = Thread(target=self._json_loop, daemon=True, name="Eval-JSON")
        self.video_thread = Thread(target=self._video_loop, daemon=True, name="Eval-Video")
        self.io_thread.start()
        if save_video:
            self.video_thread.start()

    def emit(self, stream, row):
        if self.stopping.is_set():
            return
        try:
            self.rows.put_nowait((stream, row))
        except Full:
            self.json_dropped += 1

    def video(self, packet, policy_rgb):
        if not self.save_video or self.stopping.is_set():
            return
        if (self._last_video_time is not None
                and packet.capture_time - self._last_video_time < 1 / self.video_fps - 1e-6):
            self.video_sampled_out += 1
            return
        self._last_video_time = packet.capture_time
        if self.frames.full():
            self.video_dropped += 1
            return
        # 기존 packet을 복사하며, 카메라를 추가로 읽지 않는다.
        item = (packet.sequence, packet.capture_time,
                packet.original_rgb.copy(), policy_rgb.copy())
        try:
            self.frames.put_nowait(item)
        except Full:
            self.video_dropped += 1

    def _json_loop(self):
        handles = {}
        try:
            flushed = time.monotonic()
            while not self.stopping.is_set() or not self.rows.empty():
                try:
                    stream, row = self.rows.get(timeout=.1)
                except Empty:
                    row = None
                if row is not None:
                    if stream not in handles:
                        handles[stream] = (self.root / (stream + ".jsonl")).open("w", buffering=262144)
                    handles[stream].write(json.dumps(json_value(row), ensure_ascii=False) + "\n")
                    self.json_written += 1
                if time.monotonic() - flushed > 1:
                    for handle in handles.values():
                        handle.flush()
                    flushed = time.monotonic()
        except Exception:
            self.errors.append("JSON writer: " + traceback.format_exc())
        finally:
            for handle in handles.values():
                handle.close()

    def _video_loop(self):
        writer = None
        try:
            with (self.root / "video_frames.jsonl").open("w") as timestamps:
                while not self.stopping.is_set() or not self.frames.empty():
                    try:
                        sequence, captured_at, original, policy = self.frames.get(timeout=.1)
                    except Empty:
                        continue
                    combined = np.concatenate([original, policy], axis=1)
                    if writer is None:
                        h, w = combined.shape[:2]
                        writer = self.cv2.VideoWriter(str(self.root / "comparison.mp4"),
                            self.cv2.VideoWriter_fourcc(*"mp4v"), self.video_fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError("mp4v VideoWriter could not open comparison.mp4")
                    writer.write(self.cv2.cvtColor(combined, self.cv2.COLOR_RGB2BGR))
                    timestamps.write(json.dumps({"video_frame_index": self.video_written,
                        "sequence": sequence, "capture_time": captured_at}) + "\n")
                    self.video_written += 1
        except Exception:
            self.errors.append("Video writer: " + traceback.format_exc())
        finally:
            if writer is not None:
                writer.release()

    def close(self):
        self.stopping.set()
        self.io_thread.join(timeout=10)
        if self.save_video:
            self.video_thread.join(timeout=10)
        return {"json_rows_written": self.json_written, "json_rows_dropped": self.json_dropped,
                "json_rows_pending": self.rows.qsize(), "video_frames_written": self.video_written,
                "video_frames_dropped": self.video_dropped,
                "video_frames_sampled_out": self.video_sampled_out,
                "video_frames_pending": self.frames.qsize(),
                "video_nominal_fps": self.video_fps, "video_enabled": self.save_video,
                "writers_still_running": self.io_thread.is_alive() or (
                    self.save_video and self.video_thread.is_alive()), "errors": list(self.errors)}


class InferenceLoggingHooks:
    """인스턴스 함수만 감싼다. 원본 결과/호출 횟수/dispatch gate를 변경하지 않는다."""
    def __init__(self, engine, wrapper, recorder):
        self.engine, self.wrapper, self.recorder = engine, wrapper, recorder
        self.tick = None
        self.phase = "initializing"
        self.start_time = None
        self.last_popped = None
        self.last_pop_tick = None
        self.command_events = []
        self.restores = []
        self._wrap(engine.client, "infer", self._infer)
        self._wrap(engine, "get_action", self._get_action)
        self._wrap(engine, "_event", self._event)
        self._wrap(wrapper, "send_action", self._send_action)
        self._wrap(wrapper, "hold", self._hold)

    def _wrap(self, obj, name, factory):
        original = getattr(obj, name)
        had_instance = name in vars(obj)
        self.restores.append((obj, name, original, had_instance))
        setattr(obj, name, factory(original))

    def _infer(self, original):
        def wrapped(request, *args, **kwargs):
            result = original(request, *args, **kwargs)
            # 서버 역정규화 후 canonical 공간. 블렌딩 전의 예측 청크이다.
            self.recorder.emit("predictions", {"received_time": time.monotonic(),
                "request_id": request.request_id, "generation": request.generation,
                "observation_id": request.observation.observation_id,
                "capture_time": request.observation.capture_time,
                "observation_state": action_copy(
                    getattr(request.observation, "features", {}).get("observation.state")),
                "observation_cursor": request.observation.action_cursor,
                "submission_cursor": request.continuation.cursor, "delay": request.delay,
                "canonical_actions_pre_blend": action_copy(result.canonical_actions),
                "model_actions": action_copy(result.model_actions),
                "server_durations": dict(result.server_durations)})
            return result
        return wrapped

    def _get_action(self, original):
        def wrapped(*args, **kwargs):
            action = original(*args, **kwargs)
            if action is not None:
                self.last_popped = action_copy(action)
                self.last_pop_tick = self.tick
                self.recorder.emit("queue_actions", {"tick": self.tick,
                    "pop_time": time.monotonic(),
                    "action_post_blend_pre_interpolation": self.last_popped.copy()})
            return action
        return wrapped

    def _event(self, original):
        def wrapped(event, **fields):
            self.recorder.emit("async_events", {"event": event,
                "time": time.monotonic(), **action_copy(fields)})
            return original(event, **fields)
        return wrapped

    def _send_action(self, original):
        def wrapped(action, *args, **kwargs):
            sent_at = time.monotonic()
            command = action_copy(action)
            try:
                result = original(action, *args, **kwargs)
            except Exception as error:
                self.recorder.emit("motor_commands", {"tick": self.tick, "kind": "policy_send",
                    "trial_phase": self.phase,
                    "send_time": sent_at, "command_to_driver": command, "submitted_ok": False,
                    "error": repr(error)})
                raise
            row = {"tick": self.tick, "kind": "policy_send", "send_time": sent_at,
                   "trial_phase": self.phase,
                   "returned_time": time.monotonic(), "command_to_driver": command,
                   "submitted_ok": True, "driver_reported_action": action_copy(result)}
            self.command_events.append(row)
            self.recorder.emit("motor_commands", row)
            return result
        return wrapped

    def _hold(self, original):
        def wrapped(*args, **kwargs):
            at = time.monotonic()
            result = original(*args, **kwargs)
            # ThreadSafeRobot.hold는 내부 드라이버를 직접 호출하므로 별도 기록한다.
            self.recorder.emit("motor_commands", {"tick": self.tick, "kind": "hold",
                "trial_phase": self.phase,
                "send_time": at, "returned_time": time.monotonic(),
                "driver_reported_action": action_copy(getattr(self.wrapper, "_last_applied_action", None))})
            return result
        return wrapped

    def close(self):
        for obj, name, original, had_instance in reversed(self.restores):
            if had_instance:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def clip_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError(f"Invalid bbox: {box}")
    x, y, w, h = map(float, box)
    x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    x1 = min(width, int(math.ceil(x + w)))
    y1 = min(height, int(math.ceil(y + h)))
    if w <= 0 or h <= 0 or x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"Empty/out-of-frame bbox: {box}")
    return (x0, y0, x1 - x0, y1 - y0)


def select_box(cv2, frame, label, *, allow_skip=False):
    """Use an explicit HighGUI event loop instead of the blocking selectROI helper."""
    window = "ROI selection"
    height, width = frame.shape[:2]
    state = {"start": None, "end": None, "dragging": False}

    def on_mouse(event, x, y, flags, param):
        point = (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(start=point, end=point, dragging=True)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = point
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state.update(end=point, dragging=False)

    # Use the source image size so mouse coordinates are also source coordinates.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(window, 80, 80)
    cv2.setMouseCallback(window, on_mouse)
    try:
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass  # Some window systems do not implement TOPMOST.
        skip_hint = "; S = absent / skip" if allow_skip else ""
        print(f"ROI [{label}]: drag a box; ENTER/SPACE confirm; C clear; ESC cancel{skip_hint}", flush=True)
        while True:
            display = frame.copy()
            cv2.putText(display, f"{label}: drag | ENTER confirm | C clear | ESC cancel",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            if allow_skip:
                cv2.putText(display, "S: absent / skip this label", (8, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            if state["start"] is not None and state["end"] is not None:
                cv2.rectangle(display, state["start"], state["end"], (0, 255, 0), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if allow_skip and key in (ord("s"), ord("S")):
                return None
            if key in (27, ord("q"), ord("Q")):
                raise RuntimeError(f"ROI selection cancelled: {label}")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError(f"ROI window closed: {label}")
            if key in (ord("c"), ord("C")):
                state.update(start=None, end=None, dragging=False)
            if (key in (10, 13, 32) and not state["dragging"]
                    and state["start"] is not None and state["end"] is not None):
                x0, y0 = state["start"]
                x1, y1 = state["end"]
                if abs(x1 - x0) >= 2 and abs(y1 - y0) >= 2:
                    return clip_box((min(x0, x1), min(y0, y1),
                                     abs(x1-x0), abs(y1-y0)), width, height)
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)  # Flush window destruction before selecting the next label.





def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0, min(ax+aw, bx+bw)-max(ax, bx)) * max(0, min(ay+ah, by+bh)-max(ay, by))
    return overlap / max(1, aw*ah+bw*bh-overlap)


def dark_mask(cv2, bgr, threshold):
    # Use maximum channel intensity, so saturated red mounts are not mistaken for black.
    maximum = bgr.max(axis=2)
    red = (bgr[:, :, 2].astype(np.float32) > 1.45 * np.maximum(bgr[:, :, 0], bgr[:, :, 1])) & (bgr[:, :, 2] > 40)
    mask = ((maximum <= threshold) & ~red).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))


def components(cv2, mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(tuple(map(int, stats[n, :4])), int(stats[n, 4]), n) for n in range(1, count)], labels


@dataclass
class Candidate:
    core: tuple
    box: tuple
    score: float
    shape_score: float
    appearance_score: float
    aspect_score: float


class TargetModel:
    def __init__(self, cv2, first, box, threshold, aspect_tolerance=2.5):
        self.cv2 = cv2
        self.threshold = threshold
        self.aspect_tolerance = aspect_tolerance
        self.seed_box = clip_box(box, first.shape[1], first.shape[0])
        x, y, w, h = self.seed_box
        roi = first[y:y+h, x:x+w]
        mask = dark_mask(cv2, roi, threshold)
        items, labels = components(cv2, mask)
        # Prefer the longest, slender component in the manually selected target ROI.
        options = [(rect, area, identity) for rect, area, identity in items
                   if area >= 12 and rect[3] >= max(12, h * .30) and rect[3] / max(1, rect[2]) >= 1.8]
        if not options:
            raise ValueError("No thin dark target found in the seed ROI. Select the black hook only, or adjust --dark-threshold (default 65).")
        rect, area, identity = max(options, key=lambda item: item[0][3] * math.sqrt(item[1]))
        cx, cy, cw, ch = rect
        self.seed_core = (x+cx, y+cy, cw, ch)
        isolated = (labels[cy:cy+ch, cx:cx+cw] == identity).astype(np.uint8) * 255
        self.binary, self.gray, self.aspect = self.canonical(first, isolated, (x+cx, y+cy, cw, ch))
        self.height = ch
        self.occupancy = area / (cw*ch)
        self.margins = (cx, cy, w-cx-cw, h-cy-ch)

    def canonical(self, frame, foreground, core):
        cv2 = self.cv2
        x, y, w, h = core
        padding = max(3, round(min(w, h) * .30))
        x0, y0 = max(0, x-padding), max(0, y-padding)
        x1, y1 = min(frame.shape[1], x+w+padding), min(frame.shape[0], y+h+padding)
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y-y0:y-y0+h, x-x0:x-x0+w] = foreground
        yy, xx = np.nonzero(mask)
        if len(xx) < 2:
            raise ValueError("Empty foreground for canonical target")
        coordinates = np.column_stack([xx, yy]).astype(np.float64)
        _, vectors = np.linalg.eigh(np.cov(coordinates.T))
        major = vectors[:, -1]
        if major[1] < 0:
            major = -major
        angle = math.degrees(math.atan2(major[1], major[0])) - 90
        side = int(math.ceil(math.hypot(*gray.shape))) + 8
        center = (float(xx.mean()), float(yy.mean()))
        transform = cv2.getRotationMatrix2D(center, angle, 1)
        transform[:, 2] += np.array([side/2-center[0], side/2-center[1]])
        rotated_mask = cv2.warpAffine(mask, transform, (side, side), flags=cv2.INTER_NEAREST)
        rotated_gray = cv2.warpAffine(gray, transform, (side, side), flags=cv2.INTER_LINEAR,
                                      borderValue=int(np.median(gray)))
        rx, ry, rw, rh = cv2.boundingRect(rotated_mask)
        binary = cv2.resize(rotated_mask[ry:ry+rh, rx:rx+rw], (40, 128), interpolation=cv2.INTER_NEAREST) > 0
        pad = max(2, round(rw*.3))
        patch = rotated_gray[max(0,ry-pad):min(side,ry+rh+pad), max(0,rx-pad):min(side,rx+rw+pad)]
        return binary, cv2.resize(patch, (48, 128), interpolation=cv2.INTER_AREA), rh/max(1, rw)

    def policy_box(self, core, shape):
        x, y, w, h = core
        left, top, right, bottom = self.margins
        scale = h / self.height
        return clip_box((x-left*scale, y-top*scale, w+(left+right)*scale, h+(top+bottom)*scale), shape[1], shape[0])

    def score_core(self, frame, mask, core):
        cv2 = self.cv2
        x, y, w, h = core
        binary, patch, aligned_aspect = self.canonical(frame, mask[y:y+h, x:x+w], core)
        shape_score = 2*np.logical_and(binary, self.binary).sum() / max(1, binary.sum()+self.binary.sum())
        # Canonical size compensates for changing scale; correlation is only a heuristic.
        correlation = float(cv2.matchTemplate(patch, self.gray, cv2.TM_CCOEFF_NORMED)[0, 0])
        appearance_score = max(0.0, min(1.0, correlation)) if math.isfinite(correlation) else 0.0
        ratio = aligned_aspect / self.aspect
        aspect_score = max(0.0, 1-abs(math.log(max(1e-6, ratio))) / math.log(self.aspect_tolerance))
        return float(shape_score), appearance_score, float(aspect_score)

    def candidates(self, frame, method, threshold=None):
        cv2 = self.cv2
        mask = dark_mask(cv2, frame, self.threshold if threshold is None else threshold)
        items, _ = components(cv2, mask)
        candidates = []
        for core, area, _ in items:
            x, y, w, h = core
            occupancy = area / max(1, w*h)
            if (area < 12 or not self.height*.35 <= h <= self.height*4
                    or h/max(1, w) < 1.8
                    or not .06 <= occupancy <= 1.0):
                continue
            shape, appearance, aspect = self.score_core(frame, mask, core)
            if aspect <= 0:
                continue
            score = (.65*shape+.35*aspect) if method == "shape" else (.45*shape+.40*appearance+.15*aspect)
            try:
                box = self.policy_box(core, frame.shape)
            except ValueError:
                continue
            candidates.append(Candidate(core, box, float(score), shape, appearance, aspect))
        candidates.sort(key=lambda value: value.score, reverse=True)
        return candidates, mask

    def score_tracking(self, frame, box, method):
        # Find a plausible component inside the current CSRT ROI, using the same model.
        x, y, w, h = box
        local, _ = self.candidates(frame[y:y+h, x:x+w], method)
        if not local:
            return 0.0
        left, top, right, bottom = self.margins
        scale = h / (self.height+top+bottom)
        expected = left*scale, top*scale
        # A component outside the ROI's expected core can be a nearby distractor.
        def adjusted(candidate):
            cx, cy, cw, ch = candidate.core
            displacement = math.hypot((cx-expected[0])/max(1, w), (cy-expected[1])/max(1, h))
            return candidate.score - .35*displacement
        return max(0.0, max(map(adjusted, local)))


class Reacquirer:
    """Own one tracker; emit no target bbox while detection is ambiguous or invalid."""
    def __init__(self, cv2, model, args, first):
        self.cv2, self.model, self.args = cv2, model, args
        self.create = getattr(cv2, "TrackerCSRT_create", None)
        if self.create is None:
            self.create = getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)
        if self.create is None:
            raise RuntimeError("CSRT unavailable. Use GUI opencv-contrib-python in your existing environment.")
        self.tracker = None
        self.box = None
        self.pending = None
        self.streak = 0
        self.ever_lost = False
        self.automatic_initializations = 0
        self.force_searches = 0
        self.initialize(first, model.seed_box)

    def initialize(self, frame, box):
        tracker = self.create()
        if tracker.init(frame, tuple(map(int, box))) is False:
            raise RuntimeError("CSRT init returned false")
        self.tracker, self.box = tracker, box
        self.pending, self.streak = None, 0

    def clear(self):
        self.tracker, self.box, self.pending, self.streak = None, None, None, 0
        self.ever_lost = True

    def step(self, frame, index):
        start = time.perf_counter()
        candidates, mask = self.model.candidates(frame, self.args.method)
        detection_ms = (time.perf_counter()-start)*1000
        best = candidates[0] if candidates else None
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        reliable = (best is not None and best.score >= self.args.score_threshold
                    and best.score-second_score >= self.args.score_margin)
        events = []
        tracking_score = None
        start = time.perf_counter()
        if index and self.tracker is not None:
            ok, box = self.tracker.update(frame)
            try:
                if not ok:
                    raise ValueError("CSRT update false")
                box = clip_box(box, frame.shape[1], frame.shape[0])
                tracking_score = self.model.score_tracking(frame, box, self.args.method)
                if reliable and iou(box, best.box) < .30:
                    # 명확한 후보와 추적 위치가 다르면 후보 위치로 복구
                    self.initialize(frame, best.box)
                else:
                    self.box = box
            except ValueError as exc:
                self.clear()
                events.append({"event": "lost", "reason": str(exc)})
        if self.tracker is None:
            if reliable:
                if self.pending is not None and iou(best.core, self.pending) >= self.args.confirm_iou:
                    self.streak += 1
                else:
                    self.streak = 1
                self.pending = best.core
                if self.streak >= self.args.confirm_frames:
                    self.initialize(frame, best.box)
                    self.automatic_initializations += 1
                    events.append({"event": "automatic_reinitialize", "score": best.score})
            else:
                self.pending, self.streak = None, 0
        state = "TRACKING" if self.tracker is not None else "CONFIRMING" if self.streak else "SEARCHING"
        return dict(state=state, box=self.box, candidates=candidates, mask=mask,
                    best_score=best.score if best else None, second_score=second_score,
                    reliable_candidate=bool(reliable), confirm_streak=self.streak,
                    tracking_score=tracking_score, events=events, detection_ms=detection_ms,
                    tracking_and_validation_ms=(time.perf_counter()-start)*1000)




@dataclass(frozen=True)
class FramePacket:
    observation: dict
    original_rgb: np.ndarray
    capture_time: float
    sequence: int
    boxes: dict
    processing_s: float


class Rby1PositionAdapter:
    """Local compatibility for the supplied joint-mode RBY1 driver only."""
    def __init__(self, robot):
        self._inner = robot
        cfg = robot._config
        if cfg.action_mode != "joint" or cfg.use_mobile_base:
            raise ValueError("RBY1 hold adapter requires joint mode and fixed base")
        features = robot.action_features
        self._key_map = {}
        for key in features:
            bare = key.removesuffix(".pos")
            if not (bare.startswith("left_arm_") or bare.startswith("right_arm_")
                    or bare in ("left_gripper_0", "right_gripper_0")):
                raise ValueError(f"Unsupported RBY1 position action: {key}")
            self._key_map[key] = bare + ".pos"
        if not features or len(set(self._key_map.values())) != len(features):
            raise ValueError("Empty or ambiguous RBY1 action features")
        self._reverse_keys = {canonical: raw for raw, canonical in self._key_map.items()}
        package = type(robot).__module__.rsplit(".", 1)[0]
        self._limits = importlib.import_module(package + ".command_builders")

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def action_features(self):
        return {canonical: self._inner.action_features[raw]
                for raw, canonical in self._key_map.items()}

    @property
    def observation_features(self):
        return {self._key_map.get(key, key): value
                for key, value in self._inner.observation_features.items()}

    def get_observation(self):
        return {self._key_map.get(key, key): value
                for key, value in self._inner.get_observation().items()}

    @property
    def supports_position_hold(self):
        return True

    def send_action(self, action):
        keys = self.action_features
        missing = set(keys) - set(action)
        if missing:
            raise ValueError(f"Missing RBY1 targets: {sorted(missing)}")
        targets = {key: float(action[key]) for key in keys}
        if not all(math.isfinite(v) for v in targets.values()):
            raise ValueError("RBY1 targets contain NaN/Inf")
        cb = self._limits
        for enabled, names, lower, upper in (
            (self._inner._config.use_right_arm, cb.RIGHT_ARM_NAMES, cb.RIGHT_ARM_Q_MIN, cb.RIGHT_ARM_Q_MAX),
            (self._inner._config.use_left_arm, cb.LEFT_ARM_NAMES, cb.LEFT_ARM_Q_MIN, cb.LEFT_ARM_Q_MAX),
        ):
            if enabled:
                values = np.clip([targets[name + ".pos"] for name in names], lower, upper)
                targets.update({name + ".pos": float(value) for name, value in zip(names, values)})
        for key in targets:
            if key in ("left_gripper_0.pos", "right_gripper_0.pos"):
                targets[key] = float(np.clip(targets[key], 0., 1.))
        # Preclip with the same joint limits used by the driver. Return only
        # after body and gripper submission both succeed; never hide failures.
        raw_targets = {self._reverse_keys[key]: value for key, value in targets.items()}
        self._inner.send_action(raw_targets)
        return targets


def build_remote_context_with_rby1_hold(cfg, shutdown):
    """Scope the wrapper factory override to this entrypoint's serial setup.

    No installed source files or global Rby1 class are modified. The ordinary
    ThreadSafeRobot still owns locking, validation, hold and fault handling.
    """
    import lerobot.rollout.context as context
    original = context.ThreadSafeRobot
    def wrap(robot):
        if getattr(robot, "name", None) == "rby1":
            robot = Rby1PositionAdapter(robot)
        return original(robot)
    context.ThreadSafeRobot = wrap
    try:
        return context.build_rollout_context(cfg, shutdown)
    finally:
        context.ThreadSafeRobot = original


class ImageRouter:
    """Map one physical RGB camera into checkpoint-declared image feature names."""
    def __init__(self, mode, source_key, output_keys=None):
        if mode not in ("original", "bbox", "original_bbox"):
            raise ValueError(f"Unsupported image mode: {mode}")
        self.mode, self.source_key = mode, source_key
        self.output_keys = list(output_keys or [])
        self.shape = None

    def configure_features(self, checkpoint_features, hardware_features):
        expected = {}
        for key, feature in checkpoint_features.items():
            kind = getattr(feature.type, "name", str(feature.type))
            if kind == "VISUAL":
                if not key.startswith("observation.images."):
                    raise ValueError(f"Unsupported visual feature name: {key}")
                expected[key.removeprefix("observation.images.")] = tuple(feature.shape)
        if len(expected) != 1 or self.source_key not in expected:
            raise ValueError("Use a one-image checkpoint with the physical camera key (left)")
        if not expected:
            raise ValueError("The checkpoint has no visual input")
        if self.mode == "original_bbox" and not self.output_keys:
            raise ValueError("original_bbox requires explicit --bb-output-keys ORIGINAL_KEY BBOX_KEY")
        keys = self.output_keys or list(expected)
        if len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise ValueError(f"Output keys must match checkpoint images exactly: {list(expected)}")
        if self.mode == "original_bbox" and len(keys) != 2:
            raise ValueError("original_bbox requires a two-image checkpoint")
        if self.mode != "original_bbox" and len(keys) not in (1, 2):
            raise ValueError("This single-camera experiment supports one or two image inputs")
        shape = hardware_features.get(self.source_key)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Physical RGB camera {self.source_key!r} missing: {hardware_features}")
        h, w, _ = shape
        for key in keys:
            if expected[key] != (3, h, w):
                raise ValueError(f"{key}: checkpoint expects {expected[key]}, camera gives {(3,h,w)}; configure the camera resolution/orientation")
        self.shape, self.output_keys = tuple(shape), keys
        # Preserve motor/state features, and declare virtual policy image slots.
        result = {k: v for k, v in hardware_features.items() if not isinstance(v, tuple)}
        result.update({key: self.shape for key in keys})
        return result

    def check_rgb(self, observation):
        rgb = observation[self.source_key]
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or tuple(rgb.shape) != self.shape:
            raise ValueError(f"Expected source RGB uint8 {self.shape}; got {getattr(rgb,'shape',None)}, {getattr(rgb,'dtype',None)}")
        return rgb

    def route(self, observation, rgb, bbox_rgb):
        output = dict(observation)
        if self.mode == "original_bbox":
            images = [rgb, bbox_rgb]
        else:
            images = [rgb if self.mode == "original" else bbox_rgb] * len(self.output_keys)
        for key, image in zip(self.output_keys, images, strict=True):
            if image is None:
                raise RuntimeError("bbox image missing")
            output[key] = image.copy()
        return output


class TargetInputGate:
    """Hide the initial target; enable once it returns after sustained absence."""
    def __init__(self, missing_frames=5):
        self.missing_frames = missing_frames
        self.missing = 0
        self.waiting_return = False
        self.activated = False

    def update(self, detected):
        self.missing = 0 if detected else self.missing + 1
        disappeared = self.missing == self.missing_frames
        if disappeared:
            self.waiting_return = True
        activate = detected and self.waiting_return and not self.activated
        if activate:
            self.activated = True
        return disappeared, activate


class BBoxWorker:
    """One owner for CSRT; a bounded pending slot and a latest complete snapshot.

    submit() accepts an already captured observation and NEVER reads the robot.
    The snapshot's state, RGB, and acquisition time travel together. A slow tracker
    replaces its pending input, rather than building an unbounded latency queue.
    """
    def __init__(self, router, cv2, first_rgb, boxes, args, on_failure=None, recorder=None):
        self.router, self.cv2 = router, cv2
        self.first_rgb, self.boxes = first_rgb.copy(), dict(boxes)
        self.on_failure = on_failure
        self.recorder = recorder
        self.args = args
        self.target_enabled = args.bb_target_initially_visible
        self.condition = Condition()
        self.pending = self.latest = None
        self.error = None
        self.stopping = False
        self.submitted = self.processed = self.dropped = self.duplicate_images = 0
        self.thread = None
        self.log_rows = []

    def start(self):
        self.thread = Thread(target=self._run, name="ACT-BBox", daemon=True)
        self.thread.start()

    def submit(self, observation, capture_time, sequence):
        if not math.isfinite(capture_time):
            raise ValueError("Non-finite capture time")
        owned = deepcopy(observation)
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            if self.stopping:
                return
            if self.pending is not None:
                self.dropped += 1
            self.pending = owned, float(capture_time), sequence
            self.submitted += 1
            self.condition.notify()

    def result(self):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            return self.latest

    def stop(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError("BBox worker did not stop within 3 seconds")

    def _run(self):
        try:
            cv2 = self.cv2
            trackers = {}
            target = None
            target_index = 0
            target_gate = TargetInputGate(self.args.bb_target_missing_frames)
            last_rgb = self.first_rgb
            if self.router.mode != "original":
                create = getattr(cv2, "TrackerCSRT_create", None)
                if create is None:
                    create = getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)
                if create is None:
                    raise RuntimeError("CSRT unavailable")
                first_bgr = cv2.cvtColor(self.first_rgb, cv2.COLOR_RGB2BGR)
                for label, box in self.boxes.items():
                    if label == "target":
                        model = TargetModel(cv2, first_bgr, box, self.args.bb_dark_threshold)
                        target = Reacquirer(cv2, model, SimpleNamespace(
                            method="hybrid", score_threshold=self.args.bb_score_threshold,
                            score_margin=self.args.bb_score_margin, confirm_frames=1,
                            confirm_iou=.35, tracking_threshold=0.0), first_bgr)
                        continue
                    tracker = create()
                    if tracker.init(first_bgr, box) is False:
                        raise RuntimeError(f"Tracker initialization failed: {label}")
                    trackers[label] = tracker
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or self.pending is not None)
                    if self.stopping:
                        return
                    observation, captured_at, sequence = self.pending
                    self.pending = None
                started = time.perf_counter()
                rgb = self.router.check_rgb(observation)
                bbox_rgb = None
                policy_boxes = {}
                target_diagnostics = None
                if self.router.mode != "original":
                    if np.array_equal(rgb, last_rgb):
                        self.duplicate_images += 1
                    else:
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if target is not None:
                            target_index += 1
                            result = target.step(bgr, target_index)
                            target_diagnostics = {k: result.get(k) for k in (
                                "reliable_candidate", "best_score", "second_score", "state",
                                "tracking_score", "events", "detection_ms", "tracking_and_validation_ms")}
                            target_diagnostics["candidates"] = [
                                {"box": c.box, "score": c.score} for c in result.get("candidates", [])[:3]]
                            disappeared, activate = target_gate.update(result["reliable_candidate"])
                            target_diagnostics.update(disappeared=disappeared, activated=activate)
                            if disappeared:
                                # CSRT can report success on background after the hook exits.
                                target.clear()
                                result["box"] = None
                                print("Target absent: waiting for automatic reacquisition", flush=True)
                            if activate:
                                target.initialize(bgr, result["candidates"][0].box)
                                result["box"] = target.box
                                self.target_enabled = True
                                print("Target reacquired: ACT target input automatically enabled", flush=True)
                            if result["box"] is None:
                                self.boxes.pop("target", None)
                            else:
                                self.boxes["target"] = result["box"]
                        for label, tracker in trackers.items():
                            ok, box = tracker.update(bgr)
                            if not ok:
                                if self.recorder:
                                    self.recorder.emit("tracking", {"sequence": sequence,
                                        "capture_time": captured_at, "lost_label": label,
                                        "tracking_ok": False})
                                raise RuntimeError(f"CSRT lost {label!r} at capture {sequence}; stop trial and reinitialize")
                            self.boxes[label] = clip_box(box, rgb.shape[1], rgb.shape[0])
                    bbox_rgb = np.full_like(rgb, 255)
                    for label, box in self.boxes.items():
                        if label == "target" and not self.target_enabled:
                            continue
                        x, y, w, h = map(int, box)
                        px = max(15, round(w * .20)) if label == "target" else 0
                        py = max(15, round(h * .10)) if label == "target" else 0
                        x1, y1 = max(0, x-px), max(0, y-py)
                        x2, y2 = min(rgb.shape[1], x+w+px), min(rgb.shape[0], y+h+py)
                        policy_boxes[label] = (x1, y1, x2-x1, y2-y1)
                        bbox_rgb[y1:y2, x1:x2] = rgb[y1:y2, x1:x2]
                    last_rgb = rgb.copy()
                converted = self.router.route(observation, rgb, bbox_rgb)
                elapsed = time.perf_counter() - started
                packet = FramePacket(converted, rgb, captured_at, sequence, dict(self.boxes), elapsed)
                with self.condition:
                    if self.stopping:
                        return
                    self.latest = packet
                    self.processed += 1
                tracking = {"sequence": sequence, "capture_time": captured_at,
                    "ready_time": time.monotonic(), "processing_s": elapsed,
                    "boxes": dict(self.boxes), "target_input_enabled": self.target_enabled,
                    "policy_boxes": policy_boxes, "tracking_ok": True,
                    "target_detection": target_diagnostics}
                if self.recorder:
                    self.recorder.emit("tracking", tracking)
                else:
                    self.log_rows.append(tracking)
        except Exception:
            failure = traceback.format_exc()
            # Revoke dispatch immediately; actual robot hold remains on control thread.
            if self.on_failure:
                try:
                    self.on_failure()
                except Exception:
                    failure += "\nFailure callback:\n" + traceback.format_exc()
            with self.condition:
                self.error = failure
                self.condition.notify_all()


def prepare_rois(cv2, rgb, args, router):
    if router.mode == "original":
        return {}
    path = args.bb_roi_file
    fixed = {}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["source_key"] != router.source_key or saved["frame_shape"] != list(rgb.shape):
            raise ValueError("Saved fixed ROI camera/shape differs from current camera")
        fixed = saved["fixed_boxes"]
    boxes = {}
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for label in args.bb_labels:
        if label in args.bb_fixed_labels and label in fixed:
            boxes[label] = clip_box(fixed[label], rgb.shape[1], rgb.shape[0])
            print(f"Reusing fixed ROI {label}: {boxes[label]}", flush=True)
        else:
            boxes[label] = select_box(cv2, bgr, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source_key": router.source_key, "frame_shape": list(rgb.shape),
                               "fixed_boxes": {k: v for k, v in boxes.items() if k in args.bb_fixed_labels}}, indent=2))
    print(f"Fixed ROIs saved: {path}. Variable object ROI is selected for every new trial.", flush=True)
    return boxes


def parse_bb_arguments():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--bb-help", action="store_true")
    parser.add_argument("--bb-mode", choices=["original", "bbox"], default="bbox")
    parser.add_argument("--bb-source-key", default="left", help="Physical robot observation key, default left")
    parser.add_argument("--bb-output-keys", nargs="+", help="Single checkpoint camera name without observation.images., default left")
    parser.add_argument("--bb-labels", nargs="+", default=["object", "gripper", "target"])
    parser.add_argument("--bb-fixed-labels", nargs="*", default=["gripper"])
    parser.add_argument("--bb-roi-file", type=Path, default=Path("act_bb_fixed_rois.json"))
    parser.add_argument("--bb-target-initially-visible", action="store_true",
                        help="Include target from start; default automatically enables on return after disappearance")
    parser.add_argument("--bb-score-threshold", type=float, default=.65)
    parser.add_argument("--bb-score-margin", type=float, default=.01)
    parser.add_argument("--bb-target-missing-frames", type=int, default=5,
                        help="Consecutive processed images without a reliable candidate before marking target absent")
    parser.add_argument("--bb-dark-threshold", type=int, default=65)
    parser.add_argument("--bb-no-preview", action="store_true")
    parser.add_argument("--bb-dry-run", action="store_true", help="Consume/log predictions without dispatching policy actions")
    parser.add_argument("--bb-run-dir", type=Path)
    parser.add_argument("--experiment-name", default=None, help="저장 폴더 실험명. 기본값은 서버 deployment")
    parser.add_argument("--bb-save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bb-video-fps", type=float, default=15., help="영상 저장 빈도. 제어/추론 fps와 별개")
    parser.add_argument("--bb-video-queue", type=int, default=8)
    parser.add_argument("--bb-plugin", action="append", default=[])
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.bb_video_fps) or args.bb_video_fps <= 0 or args.bb_video_queue < 1:
        parser.error("video fps and queue size must be positive")
    if args.bb_target_missing_frames < 1:
        parser.error("--bb-target-missing-frames must be at least 1")
    if args.bb_help:
        parser.print_help()
        raise SystemExit(0)
    if len(set(args.bb_labels)) != len(args.bb_labels) or not args.bb_labels:
        parser.error("ROI labels must be unique and nonempty")
    if not set(args.bb_fixed_labels).issubset(args.bb_labels):
        parser.error("Fixed labels must belong to --bb-labels")
    return args, remaining


def main():
    args, remaining = parse_bb_arguments()
    import cv2
    # This must run before importing LeRobot (which may import PyAV transitively).
    if args.bb_mode != "original" or not args.bb_no_preview:
        cv2.namedWindow("Initializing GUI", cv2.WINDOW_NORMAL)
        cv2.imshow("Initializing GUI", np.zeros((240, 480, 3), dtype=np.uint8))
        cv2.waitKey(300)
        cv2.destroyWindow("Initializing GUI")
        cv2.waitKey(1)

    from lerobot.configs import parser
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.rollout.context import build_rollout_context
    from lerobot.rollout.strategies.core import RolloutStrategy, send_next_action
    from lerobot.inference.factory import RemoteInferenceConfig
    from lerobot.utils import import_utils
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    register = getattr(import_utils, "register_third_party_plugins", None)
    if register is not None:
        register()
    for plugin in args.bb_plugin:
        importlib.import_module(plugin)
    if not any(a.startswith("--inference.type") for a in remaining):
        remaining.append("--inference.type=remote")
    sys.argv = [sys.argv[0], *remaining]

    def execute(cfg):
        if not isinstance(cfg.inference, RemoteInferenceConfig):
            raise ValueError("Use --inference.type=remote; checkpoint and CUDA belong on GPU server")
        if cfg.interactive or cfg.dataset is not None or cfg.strategy.type != "base":
            raise ValueError("First version supports one base-strategy trial per invocation; omit dataset/interactive flags")
        if cfg.use_torch_compile or cfg.rename_map or cfg.planner is not None:
            raise ValueError("First version requires compile=false, empty rename_map, and no external planner")
        if not math.isfinite(cfg.fps) or cfg.fps <= 0 or cfg.duration < 0:
            raise ValueError("Invalid FPS/duration")
        router = ImageRouter(args.bb_mode, args.bb_source_key, args.bb_output_keys)

        class EvalStrategy(RolloutStrategy):
            def run(self, ctx):
                run_trial(self, ctx)

            def teardown(self, ctx):
                self._teardown_hardware(ctx.hardware,
                    return_to_initial_position=cfg.return_to_initial_position and not args.bb_dry_run)

        experiment = re.sub(r"[^\w.-]+", "_", args.experiment_name or cfg.inference.deployment).strip("._") or "trial"
        run_dir = args.bb_run_dir or Path("act_bb_eval_runs") / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + experiment + "_" + args.bb_mode)
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        shutdown = Event()
        ctx = strategy = worker = hooks = None
        recorder = TrialRecorder(run_dir, cv2, args.bb_video_fps, args.bb_save_video, args.bb_video_queue)
        summary = {"mode": args.bb_mode, "source_key": args.bb_source_key,
                   "deployment": cfg.inference.deployment, "endpoint": cfg.inference.endpoint, "fps": cfg.fps,
                   "dry_run": args.bb_dry_run, "status": "initializing", "success": None,
                   "experiment_name": experiment, "created_at": datetime.now().astimezone().isoformat(),
                   "result_source": "manual_only", "client_file": str(Path(__file__).resolve()),
                   "client_arguments": json_value(vars(args)), "rollout_config": json_value(cfg),
                   "recording_notes": {"prediction_space": "canonical, post-server-postprocessing, pre-blend",
                       "driver_reported_action": "submitted target, NOT measured achieved pose",
                       "video": "original | policy input; capture_time in video_frames.jsonl is authoritative"}}

        def run_trial(strategy, ctx):
            nonlocal worker, hooks
            engine = ctx.policy.inference
            wrapper = ctx.hardware.robot_wrapper
            hooks = InferenceLoggingHooks(engine, wrapper, recorder)
            summary["action_names"] = list(ctx.data.ordered_action_keys)
            summary["artifact_identity"] = engine.client.artifact_identity
            summary["server_descriptor"] = json_value(engine.client.descriptor)
            summary["lerobot_version"] = importlib.import_module("lerobot").__version__
            clock_proxy = None
            if "capture_time" not in inspect.signature(engine.notify_observation).parameters:
                class CaptureClockProxy:
                    def __init__(self, inner):
                        self.inner, self.times = inner, local()
                    @property
                    def observation_time(self):
                        return getattr(self.times, "capture_time", self.inner.observation_time)
                    def __getattr__(self, name):
                        return getattr(self.inner, name)
                clock_proxy = CaptureClockProxy(engine._robot)
                engine._robot = clock_proxy
            first = wrapper.get_observation()
            first_rgb = router.check_rgb(first)
            boxes = prepare_rois(cv2, first_rgb, args, router)
            summary.update(output_keys=router.output_keys, initial_boxes=boxes)
            (run_dir / "trial.json").write_text(json.dumps(summary, indent=2))
            worker = BBoxWorker(router, cv2, first_rgb, boxes, args, on_failure=engine.pause, recorder=recorder)
            worker.start()
            engine.resume()
            control_interval = 1.0 / (cfg.fps * cfg.interpolation_multiplier)
            tick, last_published = 0, -1
            started = next_tick = time.monotonic()
            hooks.start_time = started
            hooks.phase = "running"
            summary.update(status="running", start_monotonic=started,
                           started_at=datetime.now().astimezone().isoformat())
            recorder.emit("async_events", {"event": "trial_start", "time": started})
            cached_processed = None
            loop_durations = []
            sent_actions = waiting_ticks = 0
            recorded_ticks = 0
            print(f"Running {args.bb_mode}: physical {router.source_key} -> {router.output_keys}", flush=True)
            print("Preview keys: T=toggle target input, S=success/end, F=failure/end, Q=stop. Ctrl+C also stops.", flush=True)
            if not args.bb_no_preview:
                cv2.namedWindow("ACT BB evaluation", cv2.WINDOW_NORMAL)
            try:
                while not shutdown.is_set():
                    now = time.monotonic()
                    if cfg.duration and now - started >= cfg.duration:
                        summary["status"] = "duration_reached"
                        break
                    # Match existing rollout: acquire fresh raw state for each motor tick.
                    hooks.tick = tick
                    hooks.command_events = []
                    obs_raw = wrapper.get_observation()
                    captured_at = wrapper.observation_time
                    if captured_at is None:
                        captured_at = time.monotonic()
                    # A 30 Hz policy/camera need not submit duplicate images at interpolated motor Hz.
                    if tick % cfg.interpolation_multiplier == 0:
                        worker.submit(obs_raw, captured_at, tick)
                    packet = worker.result()  # Raises before dispatch on tracking failure.
                    if packet is not None and packet.sequence != last_published:
                        cached_processed = ctx.processors.robot_observation_processor(packet.observation)
                        if "capture_time" in inspect.signature(engine.notify_observation).parameters:
                            engine.notify_observation(cached_processed, capture_time=packet.capture_time)
                        else:
                            # Existing RTC reads capture time from its robot wrapper.
                            clock_proxy.times.capture_time = packet.capture_time
                            try:
                                engine.notify_observation(cached_processed)
                            finally:
                                del clock_proxy.times.capture_time
                        last_published = packet.sequence
                        recorder.emit("inputs", {"sequence": packet.sequence,
                            "capture_time": packet.capture_time, "notify_time": time.monotonic(),
                            "processing_s": packet.processing_s, "boxes": packet.boxes,
                            "state": {k: action_copy(v) for k, v in packet.observation.items()
                                      if k in wrapper.action_features}})
                        recorder.video(packet, packet.observation[router.output_keys[-1]])
                        if not args.bb_no_preview:
                            original = packet.original_rgb
                            shown = packet.observation[router.output_keys[-1]]
                            comparison = np.concatenate([original, shown], axis=1)
                            cv2.imshow("ACT BB evaluation", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                    dispatched = None
                    if args.bb_dry_run:
                        # Observation-only evaluation: no send_next_action / robot target dispatch.
                        if engine.dispatch_allowed():
                            action = engine.get_action(None)
                            if action is not None and tick % 30 == 0:
                                print("Prediction:", action.detach().cpu().numpy(), flush=True)
                    elif cached_processed is not None:
                        dispatched = send_next_action(cached_processed, obs_raw, ctx, strategy._require_interpolator())
                        sent_actions += dispatched is not None
                        waiting_ticks += dispatched is None
                    recorded_at = time.monotonic()
                    recorder.emit("control", {"tick": tick, "time_s": recorded_at - started,
                        "observation_capture_time": captured_at, "control_record_time": recorded_at,
                        "obs_state": {k: action_copy(obs_raw[k]) for k in wrapper.action_features if k in obs_raw},
                        "input_sequence": last_published,
                        "input_capture_time": packet.capture_time if packet else None,
                        "policy_action_dispatched": dispatched is not None,
                        "canonical_action_post_interpolation": action_copy(dispatched),
                        "last_queue_action_post_blend": action_copy(hooks.last_popped),
                        "last_queue_action_pop_tick": hooks.last_pop_tick,
                        "driver_commands": list(hooks.command_events),
                        "queue_playback_s": engine.runtime.queue.qsize() * engine.runtime.interval,
                        "loop_so_far_s": recorded_at - now})
                    recorded_ticks += 1
                    if engine.failed:
                        raise RuntimeError(engine.failure_traceback or "Async engine failed")
                    engine.pump_query(cached_processed)
                    if not args.bb_no_preview:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("t"):
                            worker.target_enabled = not worker.target_enabled
                            print(f"Target input: {worker.target_enabled}", flush=True)
                        if key in (ord("s"), ord("f"), ord("q"), 27):
                            summary["success"] = True if key == ord("s") else False if key == ord("f") else None
                            summary["status"] = "operator_ended"
                            break
                        if cv2.getWindowProperty("ACT BB evaluation", cv2.WND_PROP_VISIBLE) < 1:
                            summary["status"] = "window_closed"
                            break
                    tick += 1
                    loop_durations.append(time.monotonic() - now)
                    next_tick += control_interval
                    delay = next_tick - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.monotonic()
            finally:
                ended = time.monotonic()
                summary.update(duration_s=ended - started, end_monotonic=ended,
                               ended_at=datetime.now().astimezone().isoformat())
                hooks.phase, hooks.tick = "shutdown", None
                engine.pause()
                strategy.hold_control_state(ctx.hardware)
                summary.update(control_ticks=recorded_ticks, dispatched_motor_ticks=sent_actions,
                               waiting_motor_ticks=waiting_ticks)
                if loop_durations:
                    summary.update(loop_mean_s=float(np.mean(loop_durations)),
                                   loop_p95_s=float(np.percentile(loop_durations, 95)),
                                   loop_over_budget_ticks=sum(t > control_interval for t in loop_durations))

        try:
            ctx = build_remote_context_with_rby1_hold(cfg, shutdown)
            visual_features = {}
            for spec in ctx.policy.inference.client.capabilities.features:
                if spec.kind == "rgb":
                    h, w, c = spec.shape
                    visual_features[spec.name] = SimpleNamespace(
                        type=SimpleNamespace(name="VISUAL"), shape=(c, h, w))
            router.configure_features(visual_features,
                                      ctx.hardware.robot_wrapper.inner.observation_features)
            strategy = EvalStrategy(cfg.strategy)
            strategy.setup(ctx)
            strategy.run(ctx)
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
        except Exception:
            summary["status"] = "error"
            summary["error"] = traceback.format_exc()
            raise
        finally:
            shutdown.set()
            cleanup_errors = []
            if ctx is not None:
                try:
                    ctx.policy.inference.pause()
                except Exception:
                    cleanup_errors.append(traceback.format_exc())
            if worker is not None:
                try:
                    worker.stop()
                except Exception:
                    cleanup_errors.append(traceback.format_exc())
                summary.update(submitted=worker.submitted, processed=worker.processed,
                               replaced_pending_frames=worker.dropped, duplicate_images=worker.duplicate_images)

            if strategy is not None and ctx is not None:
                try:
                    strategy.teardown(ctx)
                except Exception:
                    cleanup_errors.append(traceback.format_exc())
            if cleanup_errors:
                summary["cleanup_errors"] = cleanup_errors
                summary["status"] = "error"
            # worker 종료 후 훅을 해제한다. 진행 중인 콜백은 stopping 검사로 기록을 건너뛴다.
            if hooks is not None:
                hooks.close()
            summary["recording"] = recorder.close()
            (run_dir / "trial.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            if summary["recording"]["errors"] or summary["recording"]["writers_still_running"]:
                print("Recording incomplete: inspect trial.json recording field", flush=True)
            if args.bb_mode != "original" or not args.bb_no_preview:
                cv2.destroyAllWindows()
            print(f"Evaluation result: {run_dir / 'trial.json'}", flush=True)
            if cleanup_errors:
                raise RuntimeError("Evaluation cleanup failed; inspect trial.json")

    execute.__annotations__["cfg"] = RolloutConfig
    parser.wrap()(execute)()


if __name__ == "__main__":
    main()