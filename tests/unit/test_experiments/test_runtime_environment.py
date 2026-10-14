"""GPU determinism settings that every job environment carries."""

from wsl_vla.experiments.runtime import (
    DETERMINISTIC_XLA_FLAG,
    deterministic_gpu_ops,
    with_deterministic_gpu_ops,
)


def test_deterministic_flag_is_added_once_and_keeps_other_xla_flags():
    plain = with_deterministic_gpu_ops({"PATH": "/bin"})
    assert plain["XLA_FLAGS"] == DETERMINISTIC_XLA_FLAG
    assert plain["PATH"] == "/bin"

    other = with_deterministic_gpu_ops({"XLA_FLAGS": "--xla_dump_to=/tmp/x"})
    assert other["XLA_FLAGS"].split() == ["--xla_dump_to=/tmp/x", DETERMINISTIC_XLA_FLAG]
    assert with_deterministic_gpu_ops(other) == other


def test_flag_detection_reads_whole_flags_only():
    assert deterministic_gpu_ops({"XLA_FLAGS": DETERMINISTIC_XLA_FLAG})
    assert not deterministic_gpu_ops({})
    assert not deterministic_gpu_ops({"XLA_FLAGS": "--xla_gpu_deterministic_ops=false"})
