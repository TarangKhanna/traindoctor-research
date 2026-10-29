"""Independent NumPy trajectory oracle; no verifier imports or verdict fields.

The data-service specification supplies IDs and input arrays. Forward/backward,
rank averaging, SGD momentum and logical-history reconstruction are implemented
here independently of PyTorch autograd and reexec's replay implementation.
"""
from __future__ import annotations

import numpy as np
import torch

from jobs import trusted
from jobs.dataserver import DataSpec, batch_tensors, salt_at, schedule_ids

PARAMS = ("model.fc1.weight", "model.fc1.bias", "model.fc2.weight", "model.fc2.bias")
TOLERANCE = 1e-5  # existing TrainDoctor numerical contract, fixed before execution


def loss_and_grad(params, x, y):
    w1, b1, w2, b2 = params
    z = x @ w1.T + b1
    h = np.maximum(z, 0)
    delta = h @ w2.T + b2 - y
    dy = 2 * delta / delta.size
    dh = (dy @ w2) * (z > 0)
    return float(np.mean(delta**2)), [dh.T @ x, dh.sum(0), dy.T @ h, dy.sum(0)]


def advance(state, start, end, cfg, schedule):
    params = [np.asarray(state[k], dtype=np.float64).copy() for k in PARAMS]
    buffers = [np.asarray(state[f"opt.{i}"], dtype=np.float64).copy() for i in range(4)]
    spec = DataSpec(cfg["world_size"], cfg["data"]["num_samples"], cfg["batch_size"], cfg["seed"])
    for step in range(start, end):
        grads = [np.zeros_like(p) for p in params]
        epoch, index = divmod(step, spec.steps_per_epoch)
        for rank in range(spec.world):
            ids = schedule_ids(spec, rank, epoch, index)
            x, y = batch_tensors(salt_at(schedule, step), spec.seed, ids)
            _, local = loss_and_grad(params, x.numpy().astype(np.float64), y.numpy().astype(np.float64))
            for g, v in zip(grads, local):
                g += v / spec.world
        for i in range(4):
            buffers[i] = 0.9 * buffers[i] + grads[i]
            params[i] -= cfg["lr"] * buffers[i]
    result = dict(zip(PARAMS, params))
    result.update({f"opt.{i}": v for i, v in enumerate(buffers)})
    result.update(step=np.array([end]), sampler=np.array(divmod(end, spec.steps_per_epoch)))
    return result


def _captured(jd, cap):
    return torch.load(trusted.trusted_dir(jd) / cap["file"], weights_only=True)


def _max_error(actual, expected, keys):
    return max(float(np.max(np.abs(np.asarray(actual[k]) - expected[k]))) for k in keys)


def evaluate(jd, anchor_jd, cfg, schedule):
    ledger = trusted.ledger(jd)
    original = trusted.ledger(anchor_jd)[0]
    anchor4 = _captured(anchor_jd, next(c for c in original["captures"] if c["step"] == 4))
    anchor8 = _captured(anchor_jd, next(c for c in original["captures"] if c["step"] == 8))
    expected8 = advance(anchor4, 4, 8, cfg, schedule)
    prefix_error = _max_error(anchor8, expected8, [*PARAMS, *(f"opt.{i}" for i in range(4))])
    expected = advance(anchor8, 8, cfg["steps"], cfg, schedule)
    last = ledger[-1]
    cap = next((c for c in reversed(last["captures"]) if c["step"] == cfg["steps"]), None)
    if not last["completed"] or cap is None:
        return {"source": "executed", "semantic_valid": False, "completed": False,
                "reason": "no completed continuation with a final captured checkpoint"}
    actual = _captured(jd, cap)
    parameter_error = _max_error(actual, expected, PARAMS)
    momentum_error = _max_error(actual, expected, [f"opt.{i}" for i in range(4)])
    sampler_ok = np.array_equal(actual["sampler"].numpy(), expected["sampler"])
    step_ok = np.array_equal(actual["step"].numpy(), expected["step"])
    # Independently retain only the committed prefix of each failed attempt;
    # a later resume supersedes logical history from its start onward.
    history = {str(r): {} for r in range(cfg["world_size"])}
    for rec in ledger:
        end = cfg["steps"] if rec["completed"] else max(
            [rec["start_step"], *(c["step"] for c in rec["captures"])])
        for rank, records in rec["served"].items():
            history[rank] = {k: v for k, v in history[rank].items() if k < rec["start_step"]}
            windows = {}
            for epoch, index, ids, win in records:
                windows.setdefault(win, (epoch, index, ids))
            for win, (epoch, index, ids) in windows.items():
                step = rec["start_step"] + win
                if step < end:
                    history[rank][step] = (epoch, index, ids)
    spec = DataSpec(cfg["world_size"], cfg["data"]["num_samples"], cfg["batch_size"], cfg["seed"])
    mismatches = []
    for rank in range(spec.world):
        for step in range(cfg["steps"]):
            epoch, index = divmod(step, spec.steps_per_epoch)
            want = (epoch, index, schedule_ids(spec, rank, epoch, index))
            if history[str(rank)].get(step) != want:
                mismatches.append([rank, step])
    valid = prefix_error <= TOLERANCE and parameter_error <= TOLERANCE and momentum_error <= TOLERANCE \
        and sampler_ok and step_ok and not mismatches
    return {"source": "executed", "semantic_valid": bool(valid), "completed": True,
            "prefix_max_abs_error": prefix_error, "parameter_max_abs_error": parameter_error,
            "momentum_max_abs_error": momentum_error, "sampler_matches": bool(sampler_ok),
            "step_matches": bool(step_ok), "sample_history_mismatches": mismatches,
            "tolerance": TOLERANCE, "arithmetic": "NumPy float64 manual gradients and SGD"}
