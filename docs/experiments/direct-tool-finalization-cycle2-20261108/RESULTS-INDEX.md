# Cycle 2 results and recovery index

Terminal result: **`STOP_semantic_specificity`**, five controls complete.
No weight candidate or independent model-quality result exists. B/P match
experimental predictions; S/C/N do not. This is not a coding-accuracy score.

- [Comprehensive report](../direct-tool-finalization-cycle2-20261108.md)
- [Machine-readable findings](FINDINGS.json) and [raw control responses](CONTROL-OUTPUTS.json)
- [Corrected full-episode gate](CORRECTED-REPRODUCTION-GATE.json) and
  [terminal semantic gate summary](SPECIFICITY-GATE-SUMMARY.json)
- [Source freeze before scores](SOURCE-FREEZE.json),
  [original cost ledger](BUDGET-LEDGER-PRIOR.json),
  [corrected qualification](QUALIFICATION-LEDGER.json) and
  [semantic qualification](SPECIFICITY-QUALIFICATION-LEDGER.json)
- [Linux and synthetic inference transcripts](QUALIFICATION-LOG.txt)
- [Portable evidence archive](CYCLE2-EVIDENCE.tar.gz),
  [filesystem/source inventory](EVIDENCE-INVENTORY.json) and
  [verified restoration receipt](EVIDENCE-RESTORE-RECEIPT.json)
- [Offline raw-evidence replay](OFFLINE-INTEGRITY-REPLAY.json),
  [byte-tampering rejection](TAMPER-NEGATIVE-CONTROL.json) and
  [preserved process/source state](FINAL-SAFETY.json)
- [Surgeon proposal](SURGEON-PROPOSAL.json),
  [post-fault stop-repair review](STOP-REPAIR-REVIEW.json) and
  [independent custodian's public interface](CUSTODIAN-INTERFACE.json)

`FILES.sha256` pins this directory's published files, excluding the checksum
file itself. The archive is 19,292,280 bytes, SHA-256
`98fc1f7c20f28cad1351d87e7c884e19bf7c02f23d82f45434aa52a586771df6`.
It has 509 regular members, no executable members or symlinks. Source snapshots
are `.txt` data, never imported by the replay checker. The inventory describes
460 original filesystem evidence/staging files and 47 source snapshots/preimages,
with hashes, sizes, ownership, dependencies, usefulness and recovery locations.
The original `.54` files and checkpoints remain in place. This archive is a
curated evidence copy, not an inventory of unrelated work or a model release.

To inspect the same terminal evidence without a GPU, model download, training
dependency, network request or further model generation, use this report's
matching repository source and the core NumPy environment:

```bash
cd docs/experiments/direct-tool-finalization-cycle2-20261108
sha256sum --check FILES.sha256
evidence_dir="$(mktemp -d)"
tar -xzf CYCLE2-EVIDENCE.tar.gz -C "$evidence_dir"
cd ../../..
python scripts/replay_tool_cycle2_evidence.py --evidence "$evidence_dir"
```

The checker independently reconstructs eligibility and classifications from raw
records using the reviewed repository code. It verifies all four corrected
episodes and five controls, their official-parent proof bindings, exact 426
parameter maps, token/stop data, numeric archives, source/protocol identities
and cumulative arithmetic. It does not freshly hash the multi-GB checkpoint on
the reader's machine, execute archived source, reproduce inference, or become
the independent final model evaluator. Changed repository producer sources fail
closed; use the historical source version containing this report.

The source freeze's `pushed:false` records its historical pre-generation state.
Later documentation/publication does not rewrite that receipt or the negative
gate. The original malformed-stop batch stays `STOP_harness`; its streams are
never trimmed or upgraded into valid trials. A byte-modified private copy fails
offline verification while the original remains unchanged.

Training is preserved stopped (`MainPID=0`, `Restart=no`, `ActiveState=failed`).
All thirteen model containers have confirmed exit. No original artifact/cache
cleanup or unrelated service control occurred. The two atomic config/ledger
staging prefixes are retained and inventoried, and specifically ignored by Git.
