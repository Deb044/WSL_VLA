"""One preprocessing path for training batches, evidence, and rollouts.

Every model input - camera frames, the history window, the language task, the
action chunk, and all padding masks - is produced by the functions in this
module. Training, evidence extraction, and the rollout policy call the same
code, so train/evaluation agreement holds by construction and is asserted by
``tests/unit/test_data/test_preprocessing_agreement.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import canonical_sha256


PREPROCESSING_SCHEMA_VERSION = 1
IMAGE_TRANSFORMS = ("identity", "flip_vertical", "rotate_180")
# robosuite ``macros.IMAGE_CONVENTION`` values. LIBERO demonstration files record
# the convention they were generated with (``data.attrs.macros_image_convention``);
# "opengl" stores image rows bottom-up, so those frames are vertically flipped.
IMAGE_CONVENTIONS = ("opengl", "opencv")
RESIZE_METHOD = "pil_bilinear"

# checkpoint image key -> (LIBERO HDF5 observation key, LIBERO simulator key)
LIBERO_CAMERA_SOURCES = {
    "image_primary": ("agentview_rgb", "agentview_image"),
    "image_wrist": ("eye_in_hand_rgb", "robot0_eye_in_hand_image"),
}
BOOKKEEPING_OBSERVATIONS = ("timestep_pad_mask", "pad_mask_dict", "timestep", "task_completed")


@dataclass(frozen=True)
class CameraSpec:
    octo_key: str
    dataset_key: str
    simulator_key: str
    height: int
    width: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "octo_key": self.octo_key,
            "dataset_key": self.dataset_key,
            "simulator_key": self.simulator_key,
            "height": int(self.height),
            "width": int(self.width),
        }


@dataclass(frozen=True)
class PreprocessingSpec:
    """Hashable identity of the shared model-input pipeline."""

    window_size: int
    action_horizon: int
    action_dim: int
    cameras: tuple[CameraSpec, ...]
    source_height: int
    source_width: int
    source_image_convention: str
    image_transform: str
    simulator_alignment: str
    executed_actions_per_plan: int
    consumed_observations: tuple[str, ...]
    unconsumed_observations: tuple[str, ...]
    proprio_consumed: bool
    text_processor: str
    resize_method: str = RESIZE_METHOD
    schema_version: int = PREPROCESSING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if min(self.window_size, self.action_horizon, self.action_dim) <= 0:
            raise ValueError("window, action horizon, and action dimension must be positive")
        if not self.cameras:
            raise ValueError("the checkpoint consumes no camera this pipeline can provide")
        if min(self.source_height, self.source_width) <= 0:
            raise ValueError("source camera resolution must be positive")
        if self.source_image_convention not in IMAGE_CONVENTIONS:
            raise ValueError(f"source_image_convention must be one of {IMAGE_CONVENTIONS}")
        if self.image_transform not in IMAGE_TRANSFORMS:
            raise ValueError(f"image_transform must be one of {IMAGE_TRANSFORMS}")
        if self.simulator_alignment not in IMAGE_TRANSFORMS:
            raise ValueError(f"simulator_alignment must be one of {IMAGE_TRANSFORMS}")
        if not 1 <= self.executed_actions_per_plan <= self.action_horizon:
            raise ValueError("executed actions per plan must lie in [1, action_horizon]")
        if self.proprio_consumed:
            raise ValueError(
                "this pipeline has no shared preprocessing for a proprioception tokenizer"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "window_size": int(self.window_size),
            "action_horizon": int(self.action_horizon),
            "action_dim": int(self.action_dim),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "source_height": int(self.source_height),
            "source_width": int(self.source_width),
            "source_image_convention": self.source_image_convention,
            "image_transform": self.image_transform,
            "simulator_alignment": self.simulator_alignment,
            "resize_method": self.resize_method,
            "executed_actions_per_plan": int(self.executed_actions_per_plan),
            "consumed_observations": list(self.consumed_observations),
            "unconsumed_observations": list(self.unconsumed_observations),
            "proprio_consumed": bool(self.proprio_consumed),
            "text_processor": self.text_processor,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreprocessingSpec":
        if payload.get("schema_version") != PREPROCESSING_SCHEMA_VERSION:
            raise ValueError("unsupported preprocessing schema")
        return cls(
            window_size=int(payload["window_size"]),
            action_horizon=int(payload["action_horizon"]),
            action_dim=int(payload["action_dim"]),
            cameras=tuple(CameraSpec(**camera) for camera in payload["cameras"]),
            source_height=int(payload["source_height"]),
            source_width=int(payload["source_width"]),
            source_image_convention=str(payload["source_image_convention"]),
            image_transform=str(payload["image_transform"]),
            simulator_alignment=str(payload["simulator_alignment"]),
            resize_method=str(payload["resize_method"]),
            executed_actions_per_plan=int(payload["executed_actions_per_plan"]),
            consumed_observations=tuple(payload["consumed_observations"]),
            unconsumed_observations=tuple(payload["unconsumed_observations"]),
            proprio_consumed=bool(payload["proprio_consumed"]),
            text_processor=str(payload["text_processor"]),
        )

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def dataset_camera_keys(self) -> tuple[str, ...]:
        return tuple(camera.dataset_key for camera in self.cameras)

    @property
    def presents_upright_frames(self) -> bool:
        """Whether the checkpoint sees top-down (natural) image rows."""

        flipped = self.source_image_convention == "opengl"
        return flipped == (self.image_transform in {"flip_vertical", "rotate_180"})

    def assert_simulator_convention(self, simulator_convention: str) -> None:
        """Rollout renders must match how the demonstrations were generated.

        ``simulator_alignment`` is the declared transform from a simulator frame
        to the demonstration storage orientation. With ``identity`` the simulator
        must use the demonstrations' image convention; with ``flip_vertical`` it
        must use the other one.
        """

        if simulator_convention not in IMAGE_CONVENTIONS:
            raise ValueError(f"unknown simulator image convention: {simulator_convention!r}")
        same = simulator_convention == self.source_image_convention
        expected = "identity" if same else "flip_vertical"
        if self.simulator_alignment != expected:
            raise ValueError(
                f"camera orientation mismatch: demonstrations were generated with the "
                f"{self.source_image_convention!r} image convention, the simulator renders "
                f"{simulator_convention!r}, so simulator_alignment must be {expected!r} "
                f"but the preprocessing declares {self.simulator_alignment!r}"
            )


def consumed_observation_keys(
    model_config: Mapping[str, Any], example_observation: Mapping[str, Any]
) -> tuple[tuple[str, ...], bool]:
    """Observation keys the checkpoint's tokenizers actually read.

    Returns ``(keys, proprio_consumed)``. A key counts as proprioception when a
    non-image (low-dimensional) tokenizer matches it.
    """

    available = sorted(key for key in example_observation if key not in BOOKKEEPING_OBSERVATIONS)
    consumed: set[str] = set()
    proprio = False
    for name, spec in model_config.get("observation_tokenizers", {}).items():
        kwargs = spec.get("kwargs", {})
        if spec.get("name") == "ImageTokenizer":
            patterns = kwargs.get("obs_stack_keys", ("image_.*", "depth_.*"))
            consumed.update(key for key in available if any(re.match(p, key) for p in patterns))
        elif spec.get("name") in {"LowdimObsTokenizer", "BinTokenizer"}:
            matched = {
                key for key in available if any(re.match(p, key) for p in kwargs.get("obs_keys", ()))
            }
            proprio = proprio or bool(matched)
            consumed.update(matched)
        else:
            raise ValueError(f"unrecognized observation tokenizer {name}: {spec.get('name')}")
    return tuple(sorted(consumed)), proprio


TEXT_PROCESSOR_PROBES = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "open the top drawer and put the bowl inside",
    "turn on the stove",
    "put both the alphabet soup and the tomato sauce in the basket",
    "Close the bottom drawer of the cabinet, then open the top drawer.",
)


def text_processor_fingerprint(text_processor: Any, text_processor_config: Any) -> str:
    """Identity of the tokenizer by what it produces, not only by its name.

    Octo loads its tokenizer files from the Hugging Face hub by model name. The
    checkpoint pins the language *encoder weights* but not those files, so a
    changed tokenizer would silently feed different ids to the same weights.
    Hashing the ids of fixed probe instructions makes any such change a
    different preprocessing identity, which every artifact records.
    """

    encoded = None if text_processor is None else text_processor.encode(list(TEXT_PROCESSOR_PROBES))
    if isinstance(encoded, Mapping):
        probe = {
            key: np.asarray(value).astype(np.int64).tolist()
            for key, value in sorted(encoded.items())
        }
    elif encoded is None:
        probe = None
    else:
        probe = np.asarray(encoded).tolist()
    return canonical_sha256({"config": text_processor_config, "probe_encoding": probe})


def preprocessing_spec_from_checkpoint(
    *,
    model_config: Mapping[str, Any],
    example_batch: Mapping[str, Any],
    text_processor_config: Any,
    text_processor: Any,
    source_resolution: Sequence[int],
    source_image_convention: str,
    image_transform: str,
    simulator_alignment: str,
    executed_actions_per_plan: int = 1,
) -> PreprocessingSpec:
    """Derive the shared preprocessing identity from the pinned checkpoint."""

    observation = example_batch["observation"]
    if "timestep_pad_mask" not in observation:
        raise ValueError("Octo checkpoint has no timestep_pad_mask")
    consumed, proprio = consumed_observation_keys(model_config, observation)
    cameras = []
    for key in consumed:
        if key not in LIBERO_CAMERA_SOURCES:
            if proprio:
                continue
            raise ValueError(f"checkpoint consumes an observation LIBERO cannot supply: {key}")
        shape = np.asarray(observation[key]).shape
        dataset_key, simulator_key = LIBERO_CAMERA_SOURCES[key]
        cameras.append(
            CameraSpec(key, dataset_key, simulator_key, int(shape[-3]), int(shape[-2]))
        )
    action_shape = np.asarray(example_batch["action"]).shape
    unconsumed = tuple(
        sorted(
            key
            for key in observation
            if key not in BOOKKEEPING_OBSERVATIONS and key not in consumed
        )
    )
    return PreprocessingSpec(
        window_size=int(np.asarray(observation["timestep_pad_mask"]).shape[1]),
        action_horizon=int(action_shape[-2]),
        action_dim=int(action_shape[-1]),
        cameras=tuple(cameras),
        source_height=int(source_resolution[0]),
        source_width=int(source_resolution[1]),
        source_image_convention=str(source_image_convention),
        image_transform=str(image_transform),
        simulator_alignment=str(simulator_alignment),
        executed_actions_per_plan=int(executed_actions_per_plan),
        consumed_observations=consumed,
        unconsumed_observations=unconsumed,
        proprio_consumed=proprio,
        text_processor=text_processor_fingerprint(text_processor, text_processor_config),
    )


def apply_image_transform(frames: np.ndarray, transform: str) -> np.ndarray:
    """Orientation change applied identically to demonstrations and rollouts."""

    if transform == "identity":
        return frames
    if transform == "flip_vertical":
        return frames[..., ::-1, :, :]
    if transform == "rotate_180":
        return frames[..., ::-1, ::-1, :]
    raise ValueError(f"image_transform must be one of {IMAGE_TRANSFORMS}")


def resize_rgb(images: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of ``[time, height, width, 3]`` uint8 frames."""

    values = np.asarray(images)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("camera observations must have shape [time,height,width,3]")
    if values.shape[1:3] == tuple(size):
        return np.ascontiguousarray(values.astype(np.uint8, copy=False))
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - installed by official Octo
        raise RuntimeError("Pillow is required to resize LIBERO observations") from exc
    resample = getattr(Image, "Resampling", Image).BILINEAR
    resized = [
        np.asarray(Image.fromarray(frame.astype(np.uint8)).resize(size[::-1], resample))
        for frame in values
    ]
    return np.stack(resized).astype(np.uint8)


def camera_frames(
    raw: Any, camera: CameraSpec, spec: PreprocessingSpec, *, source: str
) -> np.ndarray:
    """Raw source frames -> the exact uint8 frames the checkpoint consumes.

    ``source`` is ``"dataset"`` for demonstration frames or ``"simulator"`` for
    rollout renders. Simulator frames are first brought into the demonstration
    storage orientation (``simulator_alignment``, measured by
    ``scripts/setup/verify_camera_orientation.py``); from there both sources go
    through the identical orientation and resize steps.
    """

    if source not in {"dataset", "simulator"}:
        raise ValueError("frame source must be 'dataset' or 'simulator'")
    values = np.asarray(raw)
    if values.ndim == 3:
        values = values[None]
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"{camera.octo_key} frames must have shape [time,height,width,3]")
    if values.shape[1:3] != (spec.source_height, spec.source_width):
        raise ValueError(
            f"{camera.octo_key} {source} frames are {values.shape[1:3]}, but the shared "
            f"preprocessing is locked to {(spec.source_height, spec.source_width)}; "
            "demonstrations and the simulator must use the same camera resolution"
        )
    if source == "simulator":
        values = apply_image_transform(values, spec.simulator_alignment)
    oriented = apply_image_transform(values, spec.image_transform)
    return resize_rgb(oriented, (camera.height, camera.width))


def observation_window(
    frames: Mapping[str, np.ndarray],
    spec: PreprocessingSpec,
    example_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Left-padded history window for one example (no batch axis).

    ``frames[octo_key]`` holds the most recent preprocessed frames, oldest
    first; at most ``window_size`` are used. Padding timesteps are zero-filled
    and marked False in every mask.
    """

    lengths = {len(frames[camera.octo_key]) for camera in spec.cameras}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("every camera needs the same non-empty frame history")
    window = spec.window_size
    count = min(lengths.pop(), window)
    start = window - count
    pad_mask = np.zeros(window, dtype=bool)
    pad_mask[start:] = True
    result: dict[str, Any] = {}
    for key, expected_value in example_observation.items():
        if key == "pad_mask_dict":
            result[key] = {name: pad_mask.copy() for name in expected_value}
            continue
        expected = np.asarray(expected_value)
        if key == "timestep_pad_mask":
            result[key] = pad_mask.copy()
        elif key == "timestep":
            result[key] = np.arange(window, dtype=expected.dtype)
        elif key == "task_completed":
            result[key] = np.zeros((window, *expected.shape[2:]), dtype=expected.dtype)
        elif key in {camera.octo_key for camera in spec.cameras}:
            selected = np.asarray(frames[key])[-count:]
            if selected.shape[1:] != expected.shape[2:]:
                raise ValueError(
                    f"{key} frames {selected.shape[1:]} differ from checkpoint {expected.shape[2:]}"
                )
            value = np.zeros((window, *expected.shape[2:]), dtype=expected.dtype)
            value[start:] = selected.astype(expected.dtype, copy=False)
            result[key] = value
        else:
            raise ValueError(f"unsupported checkpoint observation key for LIBERO: {key}")
    return result


def action_window(
    actions: np.ndarray, timestep: int, spec: PreprocessingSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded action chunks aligned with :func:`observation_window`.

    Window slot ``w`` holds the ``action_horizon`` commands starting at its
    source timestep; commands past the episode end are masked, never invented.
    """

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != spec.action_dim:
        raise ValueError("actions must have shape [time, action_dim]")
    if not 0 <= timestep < len(values):
        raise IndexError("timestep lies outside the episode")
    window, horizon = spec.window_size, spec.action_horizon
    start = max(0, timestep - window + 1)
    observed = timestep - start + 1
    chunk = np.zeros((window, horizon, spec.action_dim), dtype=np.float32)
    mask = np.zeros_like(chunk, dtype=bool)
    for slot, source in enumerate(range(start, timestep + 1), window - observed):
        end = min(len(values), source + horizon)
        chunk[slot, : end - source] = values[source:end]
        mask[slot, : end - source] = True
    return chunk, mask


def task_batch(
    instructions: Sequence[str], text_processor: Any, example_task: Mapping[str, Any]
) -> dict[str, Any]:
    """Language-conditioned task inputs; goal images are zero and masked out."""

    if not instructions or any(not str(text).strip() for text in instructions):
        raise ValueError("every example needs a non-empty language instruction")
    if text_processor is None:
        raise ValueError("the checkpoint's text processor is required for language tasks")
    batch = len(instructions)
    tasks: dict[str, Any] = {
        key: np.zeros((batch, *np.asarray(value).shape[1:]), dtype=np.asarray(value).dtype)
        for key, value in example_task.items()
        if key not in ("pad_mask_dict", "language_instruction")
    }
    tasks["language_instruction"] = text_processor.encode([str(text) for text in instructions])
    if "pad_mask_dict" in example_task:
        tasks["pad_mask_dict"] = {
            key: (
                np.ones(batch, dtype=bool)
                if key == "language_instruction"
                else np.zeros(batch, dtype=bool)
            )
            for key in example_task["pad_mask_dict"]
        }
    return tasks
