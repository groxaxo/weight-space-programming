"""Active research boundary. Legacy entry points are retained, but disabled.

This prevents accidental CLI reuse; it is not a sandbox against arbitrary Python
or a substitute for the independent evaluation and resource boundaries.
"""


class MethodPolicyError(ValueError):
    """A historical operation is outside the active no-training objective."""


def blocked_legacy_operation(operation):
    raise MethodPolicyError(
        f"{operation} is disabled: the active objective is forward-only direct "
        "BF16 weight editing, without LoRA, training, backward passes, "
        "optimizers, or service preemption. Preserve historical artifacts. "
        "Use scripts/direct_weights.py and ops/direct_edit54.py; see AGENTS.md "
        "and docs/direct-weight-editing.md. There is no CLI/environment override."
    )
