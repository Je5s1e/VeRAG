# Loops and invariants

A loop invariant must hold before the first iteration and at the end of every
loop body. At loop exit, combine the invariant with the negated loop condition
to prove the postcondition. The invariant i <= n permits i == n at termination.
An invariant i < n is too strong when the final iteration increments i to n.

## Example

```rust
{{#include ../references/count.rs:count}}
```
