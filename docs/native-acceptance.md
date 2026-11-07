Q8 native profile acceptance contract, declared before GPU evaluation

The current user-selected Experimentos replay has a separate [HE9 contract](he9-acceptance.md), with training-aligned non-thinking generation, 262,144 context and Q8 K/V. Its selected-nine results cannot be substituted for the all-164 profile described below.

Q8 is a separate paired inference profile using the same official base revision, authenticated Q8 conversion, exact HF input token IDs, non-thinking generation, 1024 output-token budget, frozen development/HumanEval evaluators and pinned execution sandbox. On 7 November 2026 the user strengthened promotion for both NF4 and Q8: a HumanEval tie cannot promote an adapter.

Require complete paired development and all164HumanEval results with matching engine/protocol identities. Require no per-case development regressions, at least7/8code and6/6tools, a HumanEval solved count strictly higher than the Q8 base score, finite improved held-out training loss, and both actual client edit workflows passing independent completed assertions. A saturated development suite can still establish an independent broader gain through HumanEval. No strict HumanEval gain means selecting the unchanged base. Supplementary two-module project and latency results are reported separately and do not relax these requirements.

HumanEval is public and used for acceptance/selection; this is not an untouched final test set or evidence of frontier equivalence. Failed NF4 results remain in the comparison history.

The same thresholds apply to the separate Q8-base plus unmerged FP32-adapter profile. Its adapter layout transformations must preserve the factorized update, and the native response must confirm that the expected adapter is active. A protocol change requires fresh paired results; merged-profile results cannot be reused for an unmerged-profile acceptance decision.
