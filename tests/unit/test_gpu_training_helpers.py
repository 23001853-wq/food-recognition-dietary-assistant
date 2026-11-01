import pytest
import torch

from food_assistant.ml.train_utils import accumulation_window, numpy_rng_state, save_checkpoint


def test_partial_accumulation_window():
    assert [accumulation_window(i, 5, 4) for i in range(5)] == [
        (4, False),
        (4, False),
        (4, False),
        (4, True),
        (1, True),
    ]
    with pytest.raises(ValueError):
        accumulation_window(0, 5, 0)


def test_disabled_amp_accumulation_matches_full_batch():
    # Two tiny scalar regression updates exercise math, not a food-model training run.
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for i in range(2):
        with torch.amp.autocast("cpu", enabled=False):
            loss = (model(torch.ones(1, 1)) - 1).square().mean()
        count, step = accumulation_window(i, 2, 2)
        scaler.scale(loss / count).backward()
        if step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    assert model.weight.item() == pytest.approx(0.2)


def test_checkpoint_replace_is_complete(tmp_path):
    target = tmp_path / "resume.pth"
    save_checkpoint({"value": torch.tensor([1])}, target)
    assert torch.load(target, weights_only=True)["value"].item() == 1
    assert not target.with_name("resume.pth.pending").exists()


def test_resume_rng_payload_uses_weights_only_safe_types(tmp_path):
    import random

    target = tmp_path / "rng.pth"
    save_checkpoint(
        {
            "numpy_rng": numpy_rng_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
        },
        target,
    )
    loaded = torch.load(target, weights_only=True)
    assert loaded["numpy_rng"][0] == "MT19937"
    assert loaded["torch_rng"].dtype == torch.uint8
