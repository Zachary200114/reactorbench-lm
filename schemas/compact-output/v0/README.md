# ReactorBench-LM compact output contract v0

I use this directory to define the developmental `0.2.0` model-output language. The
model learns a bounded, one-line `RB2` wire format, while strict target models remain
the source of truth and canonical JSON remains the audit and API representation.

I encode learned enum fields with the immutable single-atom tables in `contract.json`.
That keeps every task, including paired conclusions, inside the 512-token control
budget without dropping required fields. The compiler expands each atom back to its
allowlisted value, and the decoder derives the same tables without looking at target
truth.

I froze this contract after recording the approved training/validation target lengths
and task-specific generation caps. Any later wire-format change needs a new contract
version. Historical Phase 6 v0.1 artifacts are never overwritten.
