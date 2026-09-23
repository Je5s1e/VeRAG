use vstd::prelude::*;
verus! {
/// A small loop invariant retrieval example, not a verifier test.
fn count_to(n: u64) -> (result: u64)
    ensures result == n,
{
    let mut i: u64 = 0;
    while i < n
        invariant i <= n,
        decreases n - i,
    {
        i = i + 1;
    }
    i
}
}
