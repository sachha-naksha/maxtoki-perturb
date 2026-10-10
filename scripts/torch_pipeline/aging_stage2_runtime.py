"""Versioned corrections for newly trained aging SKM Stage 2 checkpoints.

Install only when a dataset/checkpoint manifest names PROTOCOL. The historical
upstream numeric scaling and answer masks remain available for old checkpoints.
"""
from __future__ import annotations

from functools import wraps
import math

PROTOCOL = "aging_stage2_runtime_v1"

# The caller records exact best/last paths and full-precision scores after fit.
checkpoint_callbacks = []

FIXES = {
    "nextcell_loss": "Supervise EOQ->BOS, BOS->first gene, and all remaining answer tokens through EOS",
    "regression_units": "Normalize numeric expectation and target together; return raw time-token units",
    "checkpoint_timing": "Save best and last checkpoints after the current validation finishes",
}


def _guard_patch(target):
    protocol = getattr(target, "_aging_stage2_protocol", None)
    if protocol is not None and protocol != PROTOCOL:
        raise RuntimeError(f"Incompatible Stage 2 runtime already installed: {protocol}")
    return protocol == PROTOCOL


def _patch_tokenizer(tokenizer_class):
    original = tokenizer_class.create_loss_mask
    if _guard_patch(original):
        return

    @wraps(original)
    def answer_loss_mask(self, token_ids, task_type, eoq_index, mask_bos_next_cell=True):
        if task_type != "NextCell":
            return original(self, token_ids, task_type, eoq_index, mask_bos_next_cell)
        # Generation rows contain only a BOS sentinel after EOQ. They have no
        # answer labels; run_headless_predict strips the sentinel before decode.
        if (eoq_index == len(token_ids) - 2
                and token_ids[eoq_index] == self.special_tokens["<eoq>"]
                and token_ids[eoq_index + 1] == self.special_tokens["<bos>"]):
            return [0] * len(token_ids)
        # Labels are shifted left by the collator: the logit AT EOQ must predict
        # BOS; the logit AT BOS must predict the first gene. Keep both losses.
        if not 0 <= eoq_index < len(token_ids) - 2:
            raise ValueError("NextCell training requires a complete answer after EOQ")
        if token_ids[eoq_index] != self.special_tokens["<eoq>"]:
            raise ValueError("NextCell EOQ index does not point to EOQ")
        if token_ids[eoq_index + 1] != self.special_tokens["<bos>"]:
            raise ValueError("NextCell answer must begin with BOS")
        if token_ids[-1] != self.special_tokens["<eos>"]:
            raise ValueError("NextCell training answer must end with EOS")
        return [0] * eoq_index + [1] * (len(token_ids) - eoq_index - 1) + [0]

    answer_loss_mask._aging_stage2_protocol = PROTOCOL
    tokenizer_class.create_loss_mask = answer_loss_mask


def _patch_headless(model_class):
    original = model_class.headless_timelapse
    if _guard_patch(original):
        return

    @wraps(original)
    def normalized_headless(self, hidden_states):
        scalar = float(self.label_scalar)
        if not math.isfinite(scalar) or scalar <= 0:
            raise ValueError("Stage 2 label_scalar must be finite and positive")
        expected, non_numeric_mass, numeric_argmax = original(self, hidden_states)
        # Upstream forward already divides numeric labels by label_scalar and
        # multiplies regression outputs by it. Normalize predictions here too,
        # so both MSE operands have the same units and output scaling cancels.
        return expected / scalar, non_numeric_mass, numeric_argmax

    normalized_headless._aging_stage2_protocol = PROTOCOL
    model_class.headless_timelapse = normalized_headless


def _patch_checkpoint(callbacks):
    # Preserve the importable NeMo callback class for context/pickle persistence.
    callback_class = callbacks.ModelCheckpoint
    original_init = callback_class.__init__
    if _guard_patch(original_init):
        return

    @wraps(original_init)
    def init(self, *args, **kwargs):
        kwargs.update(every_n_train_steps=0, every_n_epochs=1,
                      save_on_train_epoch_end=False, save_top_k=1)
        original_init(self, *args, **kwargs)
        self.saved_optimizer_steps = {}
        checkpoint_callbacks.append(self)

    original_save = callback_class._save_checkpoint

    @wraps(original_save)
    def save(self, trainer, filepath):
        result = original_save(self, trainer, filepath)
        self.saved_optimizer_steps[str(filepath)] = int(trainer.global_step)
        return result

    init._aging_stage2_protocol = PROTOCOL
    callback_class.__init__ = init
    callback_class._save_checkpoint = save


def install(protocol):
    """Install idempotent runtime corrections for a recognized new manifest.

    Call after aging_temporal.scaling(label_scalar), in both training and
    inference. Do not call this for historical manifests without runtime_protocol.
    """
    if protocol != PROTOCOL:
        raise ValueError(f"Unsupported aging Stage 2 runtime protocol: {protocol!r}")
    from bionemo.maxtoki.model import MaxTokiFineTuneModel
    from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
    import nemo.lightning.pytorch.callbacks as callbacks

    _patch_tokenizer(MaxTokiTokenizer)
    _patch_headless(MaxTokiFineTuneModel)
    _patch_checkpoint(callbacks)
    return {"runtime_protocol": PROTOCOL, "fixes": dict(FIXES)}
