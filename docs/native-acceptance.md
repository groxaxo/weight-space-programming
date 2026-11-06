Q8 native profile acceptance contract, declared before GPU evaluation

The existing NF4 gate remains unchanged. Q8 is a separate paired inference profile using the same official base revision, authenticated Q8 conversion, exact HF input token IDs, non-thinking generation, 1024 output-token budget, frozen development/HumanEval evaluators and pinned execution sandbox.

Require complete paired development and all164HumanEval results with matching engine/protocol identities. Require no per-case development regressions, at least7/8code and6/6tools, HumanEval aggregate at least the Q8 base score, finite improved held-out training loss, and both actual client edit workflows passing independent completed assertions. Require a functional gain in development cases OR a strictly higher HumanEval solved count. A saturated development suite therefore does not conceal an independent broader gain. No gain means selecting the unchanged base. Supplementary two-module project and latency results are reported separately and do not relax these requirements.

HumanEval is public and used for acceptance/selection; this is not an untouched final test set or evidence of frontier equivalence. Failed NF4 results remain in the comparison history.

The same thresholds apply to the separate Q8-base plus unmerged FP32-adapter profile. Its adapter layout transformations must preserve the factorized update, and the native response must confirm that the expected adapter is active. A protocol change requires fresh paired results; merged-profile results cannot be reused for an unmerged-profile acceptance decision.
