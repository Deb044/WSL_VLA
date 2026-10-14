# Sourced by jobs that run LIBERO rollouts. Selects the MuJoCo backend from
# RENDER (pick one that 01_render_check reports as working):
#   egl       NVIDIA hardware EGL (whole GPUs only; MIG slices have no graphics)
#   mesa-egl  Mesa software EGL, for MIG slices
#   osmesa    software OSMesa (needs libOSMesa on the library path)
case "${RENDER:?set RENDER=egl, mesa-egl or osmesa from 01_render_check}" in
  egl|osmesa)
    export MUJOCO_GL=$RENDER PYOPENGL_PLATFORM=$RENDER ;;
  mesa-egl)
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export __EGL_VENDOR_LIBRARY_FILENAMES=${MESA_EGL_VENDOR:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}
    MUJOCO_EGL_DEVICE_ID=$(python scripts/setup/find_egl_device.py 2>/dev/null) \
      || { echo "mesa-egl: no working EGL device" >&2; exit 1; }
    export MUJOCO_EGL_DEVICE_ID ;;
  *) echo "unknown RENDER=$RENDER" >&2; exit 1 ;;
esac
# Software rendering threads per simulator process. For LIBERO's two 128x128
# cameras, 2 threads were 1.3x faster than 1 and more gave nothing (measured
# with Mesa llvmpipe), while several rollout processes share a slice's CPUs.
export LP_NUM_THREADS=${LP_NUM_THREADS:-2}
echo "RENDER=$RENDER MUJOCO_GL=$MUJOCO_GL MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-unset} LP_NUM_THREADS=$LP_NUM_THREADS"
