"""Check temporal loss semantics and versioned runtime installation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

path = Path(__file__).resolve().parents[1] / "scripts/torch_pipeline/aging_stage2_runtime.py"
spec = importlib.util.spec_from_file_location("aging_stage2_runtime", path)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_nextcell_every_answer_token_is_supervised_and_context_is_not():
    class Tokenizer:
        special_tokens = {"<eoq>": 10, "<bos>": 11, "<eos>": 12}

        def create_loss_mask(self, token_ids, task_type, eoq_index, mask_bos_next_cell=True):
            assert task_type == "TimeBetweenCells"
            return [0] * (len(token_ids) - 2) + [1, 0]

    runtime._patch_tokenizer(Tokenizer)
    tokenizer = Tokenizer()
    # Includes earlier context BOS/EOS; only the answer after EOQ is supervised.
    tokens = [11, 101, 12, 20, 10, 11, 201, 202, 12]
    mask = tokenizer.create_loss_mask(tokens, "NextCell", 4)
    shifted_labels = tokens[1:] + [0]
    supervised = [token for token, active in zip(shifted_labels, mask) if active]
    assert supervised == [11, 201, 202, 12]
    assert sum(mask[:4]) == 0
    assert mask[-1] == 0
    assert tokenizer.create_loss_mask([11, 101, 12, 10, 11], "NextCell", 3) == [0] * 5
    # Calling repeatedly must not wrap or normalize a second time.
    first_method = Tokenizer.create_loss_mask
    runtime._patch_tokenizer(Tokenizer)
    assert Tokenizer.create_loss_mask is first_method
    tbc = [11, 101, 12, 10, 24]
    tbc_mask = tokenizer.create_loss_mask(tbc, "TimeBetweenCells", 3)
    assert [t for t, active in zip(tbc[1:] + [0], tbc_mask) if active] == [24]


@pytest.mark.parametrize("tokens,index", [
    ([10, 99, 12], 0),
    ([10, 11, 201], 0),
    ([99, 11, 12], 0),
])
def test_incomplete_or_malformed_nextcell_training_answer_rejected(tokens, index):
    class Tokenizer:
        special_tokens = {"<eoq>": 10, "<bos>": 11, "<eos>": 12}

        def create_loss_mask(self, *args, **kwargs):
            raise AssertionError("Unexpected fallback")

    runtime._patch_tokenizer(Tokenizer)
    with pytest.raises(ValueError):
        Tokenizer().create_loss_mask(tokens, "NextCell", index)


def test_headless_prediction_units_loss_and_gradient_match():
    class Headless:
        label_scalar = 200.0

        def headless_timelapse(self, hidden_states):
            # Same raw numeric expectation contract as upstream headless decode.
            return hidden_states, torch.tensor(0.125), torch.tensor(7)

    runtime._patch_headless(Headless)
    model = Headless()
    raw_expectation = torch.tensor([40.0, -10.0], requires_grad=True)
    expected, mass, argmax = model.headless_timelapse(raw_expectation)
    target = torch.tensor([20.0, -20.0])
    normalized_loss = torch.square(expected - target / model.label_scalar).mean()
    physical_loss = torch.square(raw_expectation - target).mean()
    assert torch.allclose(normalized_loss, physical_loss / model.label_scalar**2)
    assert torch.equal(expected * model.label_scalar, raw_expectation)
    assert mass.item() == 0.125 and argmax.item() == 7
    normalized_loss.backward()
    assert torch.allclose(raw_expectation.grad, 2 * (raw_expectation.detach() - target)
                          / target.numel() / model.label_scalar**2)
    first_method = Headless.headless_timelapse
    runtime._patch_headless(Headless)
    assert Headless.headless_timelapse is first_method
    assert torch.equal(model.headless_timelapse(raw_expectation)[0], expected)


def test_checkpoint_uses_current_validation_event_not_train_batch():
    class Callback:
        def __init__(self, **kwargs):
            self.options = kwargs

        def _save_checkpoint(self, trainer, filepath):
            return "saved"

    callbacks = SimpleNamespace(ModelCheckpoint=Callback)
    runtime._patch_checkpoint(callbacks)
    checkpoint = callbacks.ModelCheckpoint(
        every_n_train_steps=50, save_last=True, save_top_k=5,
        monitor="val_loss", always_save_context=True,
    )
    assert checkpoint.options == dict(
        every_n_train_steps=0, every_n_epochs=1, save_on_train_epoch_end=False,
        save_last=True, save_top_k=1, monitor="val_loss", always_save_context=True,
    )
    assert runtime.checkpoint_callbacks[-1] is checkpoint
    assert checkpoint._save_checkpoint(SimpleNamespace(global_step=100), "step=99.ckpt") == "saved"
    assert checkpoint.saved_optimizer_steps == {"step=99.ckpt": 100}
    first_class = callbacks.ModelCheckpoint
    runtime._patch_checkpoint(callbacks)
    assert callbacks.ModelCheckpoint is first_class


def test_unknown_runtime_rejected_before_importing_or_mutating_upstream():
    with pytest.raises(ValueError, match="Unsupported"):
        runtime.install("unrecognized-protocol")


def test_bound_upstream_install_collation_units_and_callback_persistence(tmp_path, monkeypatch):
    import pickle
    from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
    from bionemo.maxtoki.model import MaxTokiFineTuneModel
    import nemo.lightning.pytorch.callbacks as callbacks
    from lightning.pytorch.trainer.states import TrainerFn

    original_callback_class = callbacks.ModelCheckpoint
    runtime.install(runtime.PROTOCOL)
    runtime.install(runtime.PROTOCOL)
    assert callbacks.ModelCheckpoint is original_callback_class
    tokens = {"<pad>": 0, "<mask>": 1, "<bos>": 2, "<eos>": 3,
              "<eoq>": 4, "<boq>": 5, "ENSG1": 6, "ENSG2": 7,
              "0": 8, "20": 9}
    tokenizer = MaxTokiTokenizer(token_dictionary=tokens)
    batch = tokenizer.collate_batch_multitask(
        [{"input_ids": [2, 6, 3, 5, 9, 4, 2, 6, 7, 3]},
         {"input_ids": [2, 6, 3, 5, 7, 4, 9]}], padding_value=0)
    assert batch["labels"][0][batch["loss_mask"][0].bool()].tolist() == [2, 6, 7, 3]
    assert batch["labels"][1][batch["loss_mask"][1].bool()].tolist() == [9]
    holder = SimpleNamespace(
        label_scalar=200.0, numeric_mask=torch.tensor([True, True]),
        vocab_to_numeric_map=torch.tensor([-10., 30.]),
        _headless_timelapse=MaxTokiFineTuneModel._headless_timelapse)
    expected, _, argmax = MaxTokiFineTuneModel.headless_timelapse(
        holder, torch.log(torch.tensor([[0.25, 0.75]])))
    assert torch.allclose(expected * holder.label_scalar, torch.tensor([20.]))
    assert argmax.tolist() == [1]
    callback = callbacks.ModelCheckpoint(
        dirpath=tmp_path, monitor="val_loss", every_n_train_steps=100,
        filename="{step}-{val_loss:.2f}", save_last=True)
    restored = pickle.loads(pickle.dumps(callback))
    assert type(restored) is original_callback_class
    assert restored._every_n_train_steps == 0
    assert restored._every_n_epochs == 1
    saved = []
    monkeypatch.setattr(callback, "_save_topk_checkpoint",
                        lambda trainer, candidates: saved.append(float(candidates["val_loss"])))
    monkeypatch.setattr(callback, "_save_last_checkpoint", lambda *args: None)
    trainer = SimpleNamespace(
        global_step=100, current_epoch=0, fast_dev_run=False, sanity_checking=False,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        callback_metrics={"val_loss": torch.tensor(1.2345)})
    callback.on_train_batch_end(trainer, None, None, None, 99)
    assert saved == []
    callback.on_validation_end(trainer, None)
    assert saved == pytest.approx([1.2345])
