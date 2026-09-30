import json

from wsl_vla.experiments.provenance import capture_environment_lock


def test_environment_lock_is_complete_serializable_and_secret_free():
    lock = capture_environment_lock()
    assert lock["schema_version"] == 1
    assert lock["python_version"]
    assert lock["distributions"]
    assert "pip" in lock["distributions"]
    serialized = json.dumps(lock, sort_keys=True)
    assert "https://" not in serialized
    assert "direct_url" not in serialized
