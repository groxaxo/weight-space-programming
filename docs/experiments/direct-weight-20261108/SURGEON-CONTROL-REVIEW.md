# Sol xhigh final control interpretation

Reviewer: the user-selected `gpt-6.1-sol` surgeon, reasoning `xhigh`, in the same
recorded agent as the implementation. This is a read-only scientific
interpretation of exposed practice data, not an independent quality evaluation.
No files, model jobs, weights or holdout material were changed for this review.

The finite experiment completed, but its preregistered Stage 2 gate failed:
9 of 10 checks passed, with `reverse_incorrect` false. The required outcome is
`STOP_controls_failed`. All 426 parameter hashes remained unchanged.

The reverse arm produced two compatible observations. Its next greedy token was
`>=`, satisfying the operator-choice part. Its correct-minus-incorrect joint
continuation margin was +0.8176, failing the required negative margin. The
next-token choice compares the immediate operator tokens; joint scoring also
includes the conditional probabilities of the following ` end).` tokens. The
frozen criterion required both observations to favor the incorrect branch.
Greedy-token reversal alone cannot pass it.

There is positive evidence of conditional information carriage. Full transfer
changed the neutral margin from -3.9133 to +1.2290 and selected `>`; half strength
produced an intermediate effect. Fixed random and off-target controls met their
criteria, replay checks were exact, and magnitude/support matching passed.
Reverse transfer substantially reduced the semantic-context margin from +6.6345
to +0.8176 and changed the immediate operator to `>=`. Those findings support
influence of the intervened component. They do not establish the complete
bidirectional mechanism required by this protocol.

The proposed next mechanism stage remains `NOT_RUN` because the frozen gate did
not pass. These results do not admit layer subdivision, geometry for candidate
selection, strength tuning, a relaxed reverse criterion or weight editing. They
do not prove absence of causality, identify defective weights or establish
general coding improvement. Preserve this as a completed practice diagnostic
with a failed preregistered gate.
