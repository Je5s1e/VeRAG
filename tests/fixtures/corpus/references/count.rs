// ANCHOR: count
fn count_to(n: u64) -> (result: u64)
    ensures result == n,
{
    let mut i = 0;
    while i < n
        invariant i <= n,
    {
        i = i + 1;
    }
    i
}
// ANCHOR_END: count
