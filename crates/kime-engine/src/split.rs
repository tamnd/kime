//! How a set of questions is cut into device batches.
//!
//! A plan runs at its bucket's size, not the batch's, so a batch of 4,100 tokens in the 8,192
//! bucket spends half its GEMM work on padding. Splitting it into a full 4,096 batch and a small
//! one for the rest is cheaper, as long as the second run's fixed cost is smaller than the padding
//! it saves. The cost of a run is taken as [`OVERHEAD`] plus its bucket's tokens.

use kime_tensor::Bucket;

/// The fixed cost of one run in tokens of GEMM work: the launches and the copies in and out that
/// every run pays whatever its size. About 0.9 ms against 6 µs a token on an RTX 4090 in FP16.
pub(crate) const OVERHEAD: usize = 160;

/// Groups items, given as `(tokens, markers)` with one sequence each, into batches that each fit
/// a bucket. Every item is in exactly one batch. An item too big for the largest bucket gets a
/// batch of its own, and the device reports it.
pub(crate) fn split(buckets: &[Bucket], items: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut left: Vec<usize> = (0..items.len()).collect();
    let mut out = Vec::new();
    while !left.is_empty() {
        let all = pick(buckets, total(items, &left));
        // The largest bucket this work can nearly fill, and what does not fit in it.
        let t = total(items, &left).0;
        let fill = buckets.iter().rev().find(|b| b.tokens <= t).or(buckets.last());
        if let Some(&b) = fill {
            let (taken, rest) = fit(b, items, &left);
            if taken.is_empty() {
                // Nothing fits even the bucket it filled: a single item too big for any bucket.
                out.push(vec![left.remove(0)]);
                continue;
            }
            if !rest.is_empty() {
                let whole = all.map_or(usize::MAX, |a| OVERHEAD + a.tokens);
                let parts = OVERHEAD + b.tokens + cost(buckets, total(items, &rest));
                if parts < whole {
                    out.push(taken);
                    left = rest;
                    continue;
                }
            }
        }
        out.push(std::mem::take(&mut left));
    }
    out
}

/// Tokens, sequences and markers of some items.
fn total(items: &[(usize, usize)], at: &[usize]) -> (usize, usize, usize) {
    at.iter().fold((0, 0, 0), |(t, s, m), &i| (t + items[i].0, s + 1, m + items[i].1))
}

fn pick(buckets: &[Bucket], (t, s, m): (usize, usize, usize)) -> Option<Bucket> {
    buckets.iter().copied().find(|b| b.holds(t, s, m))
}

/// A rough cost for work that is split no further: one run if it fits a bucket, and a full
/// largest bucket per run if not.
fn cost(buckets: &[Bucket], n: (usize, usize, usize)) -> usize {
    match (pick(buckets, n), buckets.last()) {
        (Some(b), _) => OVERHEAD + b.tokens,
        (None, Some(l)) => n.0.div_ceil(l.tokens) * (OVERHEAD + l.tokens),
        (None, None) => usize::MAX / 4,
    }
}

/// First fit in order: the items that fit `b` together, and the rest.
fn fit(b: Bucket, items: &[(usize, usize)], left: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let (mut taken, mut rest) = (Vec::new(), Vec::new());
    let (mut t, mut s, mut m) = (0, 0, 0);
    for &i in left {
        let (it, im) = items[i];
        if b.holds(t + it, s + 1, m + im) {
            (t, s, m) = (t + it, s + 1, m + im);
            taken.push(i);
        } else {
            rest.push(i);
        }
    }
    (taken, rest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kime_tensor::Buckets;

    fn buckets() -> Vec<Bucket> {
        Buckets::default().stage("compat").to_vec()
    }

    fn padded(b: &[Bucket], items: &[(usize, usize)], plan: &[Vec<usize>]) -> usize {
        plan.iter().map(|p| cost(b, total(items, p))).sum()
    }

    fn check(b: &[Bucket], items: &[(usize, usize)], plan: &[Vec<usize>]) {
        let mut seen: Vec<usize> = plan.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..items.len()).collect::<Vec<_>>(), "each item exactly once");
        for p in plan {
            assert!(pick(b, total(items, p)).is_some(), "every batch fits a bucket");
        }
    }

    #[test]
    fn small_work_is_one_batch() {
        let b = buckets();
        let items = vec![(119, 3); 3];
        let plan = split(&b, &items);
        assert_eq!(plan, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn just_over_a_bucket_splits_off_the_rest() {
        // 41 questions of 100 tokens: 4,100 tokens would run in the 8,192 bucket.
        let b = buckets();
        let items = vec![(100, 3); 41];
        let plan = split(&b, &items);
        check(&b, &items, &plan);
        assert_eq!(plan.len(), 2);
        assert!(padded(&b, &items, &plan) < OVERHEAD + 8192);
    }

    #[test]
    fn never_worse_than_one_batch() {
        let b = buckets();
        let mut x = 7u64;
        for n in 1..300 {
            let items: Vec<(usize, usize)> = (0..n)
                .map(|_| {
                    x = x
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (20 + (x >> 33) as usize % 490, 2 + (x >> 20) as usize % 5)
                })
                .collect();
            let plan = split(&b, &items);
            check(&b, &items, &plan);
            if let Some(one) = pick(&b, total(&items, &(0..n).collect::<Vec<_>>())) {
                assert!(padded(&b, &items, &plan) <= OVERHEAD + one.tokens, "n {n}");
            }
        }
    }

    #[test]
    fn too_big_for_one_bucket() {
        let b = buckets();
        let items = vec![(500, 4); 100];
        let plan = split(&b, &items);
        check(&b, &items, &plan);
        let big = vec![(40_000, 1)];
        assert_eq!(split(&b, &big), vec![vec![0]]);
    }
}
