# Paired OpenCode/Hermes output-budget follow-up

The user requested an effective Hermes output budget of **28,192 tokens** and a
fresh comparison of the official base against the already released MLP adapter.
This changes serving/request budgets, not model weights or training.

The installed Hermes one-shot path omits both output-limit fields. The frontend
now defaults to its configured output allowance instead of silently choosing
`min(1024, max_output)`. Explicit `max_tokens` and `max_completion_tokens` remain
honored, with the latter taking precedence, and cannot exceed the serving cap.
Three regressions include a real frontend-to-native HTTP control that accepts
a valid synthetic 1,500-token completion and forwards `n_predict=28192`.
These are transport controls, not model-quality scores.

Both arms use the authenticated official Qwen3.5-4B Q8 base and the existing
native runtime, 262,144-token context, Q8 K/V cache, one slot, non-thinking mode
and engine-verified Qwen instruct sampling. The output cap/default is 28,192;
the corresponding conservative input reservation is 233,952 tokens. The MLP is
the unchanged checkpoint-104 release, HF SHA-256
`0aa61a13cdb0f8f4c29c5367995944981021584965f4a9c4c2c1c82cdf6b2188`,
native F32 LoRA SHA-256
`b42d8f86daea1caad64dd7b137b41175976f097119f49015d7a2a5b5db6b1611`.

Run the same fourteen development cases, OpenCode and Hermes file edits, and
both contained two-module project clients for each arm in fresh directories.
Keep existing client versions, fixture tests, approval policies, 420-second
file and 1,000-second project deadlines, attribution, independent assertions and
scoped container cleanup. Capture every native request/output to verify effective
budgets, parsing and model/instance identity. Failures remain failures; incomplete
attribution cannot become a complete passing client result.

This is a new paired compatibility profile. The earlier HE9 measurements retain
their original 8,192-token cap and frozen source. They are not regenerated or
relabeled here, and their old identities cannot be substituted for the new
28,192-token client profile in the complete HE9 qualifier. No additional tuning,
adapter promotion or global provider switch is part of this follow-up.

The .54 supervisor owns the GPU window, pauses only the known Bonsai service
with the user's authorization, and restores its prior state on completion or
failure. Preserve Bonsai/Moss, canonical memory, personal VMs and all old results.
Publish both arms, raw request-budget diagnostics and remaining limitations in
the follow-up report and the Hugging Face model documentation.
