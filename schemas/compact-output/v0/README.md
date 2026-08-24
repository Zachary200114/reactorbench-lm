# ReactorBench-LM compact output contract v0

This directory records the developmental `0.2.0` model-output language. The learned
sequence is a bounded, one-line `RB2` wire form; strict project target models remain
the source of truth and canonical JSON remains the audit/API representation.

Learned enum fields use the immutable single-atom code tables recorded in
`contract.json`. This keeps every task—including paired conclusions—within the
512-token control-model budget without dropping any strict target field. The compiler
expands every atom back to its allowlisted enum value; the decoder derives the same
tables without consulting truth.

The contract is frozen after the approved training/validation target-length inventory
and task-specific generation caps were recorded. Any later wire-format change requires
a new contract version; the historical Phase 6 v0.1 artifacts must not be overwritten.
