import pytest

from food_assistant.utils.process_lock import running_training, training_lock


def test_lock_prevents_overlap_and_releases(tmp_path):
    with training_lock("unit_fixture", tmp_path, check_processes=False):
        with pytest.raises(RuntimeError, match="holds"):
            with training_lock("unit_fixture", tmp_path, check_processes=False):
                pytest.fail("Concurrent lock must not be granted")
    with training_lock("unit_fixture", tmp_path, check_processes=False):
        pass


def test_exception_releases_lock(tmp_path):
    with pytest.raises(ValueError):
        with training_lock("unit_fixture", tmp_path, check_processes=False):
            raise ValueError("failure")
    with training_lock("unit_fixture", tmp_path, check_processes=False):
        pass


def test_unrelated_processes_not_treated_as_training(tmp_path):
    assert running_training("train_nonexistent_fixture.py", tmp_path) == []
