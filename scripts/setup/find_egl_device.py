#!/usr/bin/env python3
"""Print the index of the first EGL device that initializes headlessly.

On MIG slices the NVIDIA EGL driver has no graphics support, so jobs point EGL
at Mesa and must select its software device explicitly: Mesa lists the
unusable DRM cards first. MuJoCo's helper is loaded by path because importing
``mujoco`` with MUJOCO_GL=egl already fails when device 0 is unusable.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def working_egl_devices() -> list[int]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    root = importlib.util.find_spec("mujoco").submodule_search_locations[0]
    spec = importlib.util.spec_from_file_location(
        "_egl_ext", os.path.join(root, "egl", "egl_ext.py")
    )
    egl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(egl)
    working = []
    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        try:
            display = egl.eglGetPlatformDisplayEXT(egl.EGL_PLATFORM_DEVICE_EXT, device, None)
            if display != egl.EGL_NO_DISPLAY and egl.eglInitialize(display, None, None) == egl.EGL_TRUE:
                working.append(index)
        except Exception:  # PyOpenGL raises on a failed eglInitialize
            continue
    return working


def main() -> int:
    working = working_egl_devices()
    if not working:
        print("no EGL device initializes on this node", file=sys.stderr)
        return 1
    print(working[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
