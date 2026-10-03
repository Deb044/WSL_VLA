#!/usr/bin/env python3
"""Probe which MuJoCo rendering backend works for LIBERO on this node.

Each backend runs in a fresh subprocess, because MUJOCO_GL is read once at
import time and a failed EGL context can leave the process unusable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CHILD = r"""
import json, os, sys, time
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

out_dir, steps = sys.argv[1], int(sys.argv[2])
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
t0 = time.time()
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
create_s = time.time() - t0
action = np.zeros(7, dtype=np.float32)
t0 = time.time()
for _ in range(steps):
    obs, _, _, _ = env.step(action)
step_s = (time.time() - t0) / steps
images = {k: np.asarray(obs[k]) for k in ("agentview_image", "robot0_eye_in_hand_image")}
env.close()
from PIL import Image
backend = os.environ["MUJOCO_GL"]
for key, image in images.items():
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(out_dir, f"{backend}_{key}.png"))
print("RESULT " + json.dumps({
    "env_create_seconds": round(create_s, 2),
    "seconds_per_step": round(step_s, 4),
    "image_shapes": {k: list(v.shape) for k, v in images.items()},
    "image_std": {k: round(float(v.std()), 2) for k, v in images.items()},
    "proprio_shapes": {k: list(np.asarray(obs[k]).shape) for k in
                       ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")},
}))
"""


def probe(backend: str, out_dir: Path, steps: int, timeout: int) -> dict:
    env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, str(out_dir), str(steps)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"backend": backend, "ok": False, "error": f"timed out after {timeout} s"}
    result = {"backend": backend, "ok": False, "returncode": completed.returncode}
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            result.update(json.loads(line[len("RESULT "):]))
            result["ok"] = min(result["image_std"].values()) > 1.0
    if not result["ok"]:
        result["stderr_tail"] = completed.stderr.strip().splitlines()[-15:]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["egl", "osmesa"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in args.backends:
        result = probe(backend, out_dir, args.steps, args.timeout)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
    (out_dir / "render_check.json").write_text(json.dumps(results, indent=2) + "\n")
    working = [r["backend"] for r in results if r["ok"]]
    print("WORKING BACKENDS:", " ".join(working) or "none")
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
