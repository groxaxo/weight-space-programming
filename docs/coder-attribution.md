# Coding model inputs and attribution

The coding adapter is a modification of the language layers of
[Qwen3.5-4B by the Qwen team](https://huggingface.co/Qwen/Qwen3.5-4B), revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, published under Apache-2.0.
It is distributed separately and requires the original checkpoint. Training
changes the model's responses; it does not change the downloaded base files.

The supervised corpus combines selected and normalized examples from:

- **NVIDIA OpenCodeInstruct**, [dataset and terms](https://huggingface.co/datasets/nvidia/OpenCodeInstruct),
  revision `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`, licensed CC-BY-4.0.
  Credit: NVIDIA. Changes: selected examples using publisher test metadata and
  length limits, converted instruction/response fields into native chat
  conversations, and applied a deterministic training/validation split.
- **Nous Research Hermes function calling v1**, [dataset and terms](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1),
  revision `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`, licensed Apache-2.0.
  Credit: Nous Research and the dataset contributors. Changes: converted the
  ShareGPT roles and JSON function-call records into complete tool
  conversations, replaced the old protocol instructions with the model's
  native chat template, validated schemas, and selected bounded examples.

Retain this attribution and the corresponding upstream license requirements
when sharing the adapter or prepared data. These credits do not imply an
endorsement by Qwen, NVIDIA, or Nous Research.

[OpenAI HumanEval](https://huggingface.co/datasets/openai/openai_humaneval),
revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, is MIT-licensed and used
only for evaluation. The evaluation importer sends only the problem prompt to
the model and does not add reference solutions or test assertions to the
supervised corpus. Public benchmark results cannot rule out corpus overlap or
pretraining contamination.
