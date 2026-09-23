//! Rounding a distribution for the wire, per spec/03-api.md.
//!
//! Rounding each probability on its own can leave the rounded values summing to 0.99 or 1.01, and
//! jev-ultrafast rejects a response whose values miss 1 by more than 0.02. With 255 options that is
//! easy to miss by far more. So the values are rounded as a whole with the largest remainder
//! method: floor every scaled value, then hand the units that are left to the values that lost the
//! most to the floor. The result sums to exactly one unit of the chosen precision.

/// Round `p` to `digits` decimal places so that the results sum to exactly 1.
///
/// The input is renormalized first, so a distribution that sums to 0.9999999 after a softmax in
/// FP16 still comes out summing to 1. Ties on the remainder go to the larger unrounded value, then
/// to the earlier option, which keeps the argmax of the rounded values equal to the argmax of the
/// unrounded ones. Returns the scaled integers, so the caller decides how to print them without a
/// second round trip through floating point.
///
/// # Panics
///
/// Panics if `digits` is above 6, which is the most spec/03-api.md allows, or if `p` is empty or
/// holds a value that is negative or not finite. Validation rejects all of those long before a
/// distribution gets here.
#[must_use]
pub fn largest_remainder(p: &[f64], digits: u32) -> Vec<u64> {
    assert!(digits <= 6, "precision is 0 to 6 digits");
    assert!(!p.is_empty(), "a distribution has at least one value");
    assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0), "probabilities are finite and >= 0");

    let total: f64 = p.iter().sum();
    let scale = 10u64.pow(digits);
    let scaled: Vec<f64> = if total > 0.0 {
        p.iter().map(|x| x / total * scale as f64).collect()
    } else {
        vec![scale as f64 / p.len() as f64; p.len()]
    };

    let mut units: Vec<u64> = scaled.iter().map(|x| x.floor() as u64).collect();
    let used: u64 = units.iter().sum();
    let left = scale.saturating_sub(used) as usize;

    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| {
        let ra = scaled[a] - scaled[a].floor();
        let rb = scaled[b] - scaled[b].floor();
        rb.total_cmp(&ra).then(scaled[b].total_cmp(&scaled[a])).then(a.cmp(&b))
    });
    for &i in order.iter().take(left) {
        units[i] += 1;
    }
    units
}

/// [`largest_remainder`] as floating point values, which is what the JSON response carries.
#[must_use]
pub fn round_distribution(p: &[f64], digits: u32) -> Vec<f64> {
    let scale = 10u64.pow(digits) as f64;
    largest_remainder(p, digits).into_iter().map(|u| u as f64 / scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_worked_example_is_already_exact() {
        assert_eq!(largest_remainder(&[0.91, 0.02, 0.03, 0.04], 2), vec![91, 2, 3, 4]);
    }

    #[test]
    fn thirds_sum_to_one() {
        let units = largest_remainder(&[1.0 / 3.0; 3], 2);
        assert_eq!(units.iter().sum::<u64>(), 100);
        assert_eq!(units, vec![34, 33, 33]);
    }

    // The case that breaks independent rounding: 255 equal options at 2 digits round to 0.00 each
    // on their own, which sums to 0 and fails the jev-ultrafast check by a whole unit.
    #[test]
    fn many_small_values_still_sum_to_one() {
        for k in [7, 32, 255] {
            for digits in 0..=6 {
                let p = vec![1.0 / k as f64; k];
                let units = largest_remainder(&p, digits);
                assert_eq!(units.iter().sum::<u64>(), 10u64.pow(digits), "k={k} digits={digits}");
            }
        }
    }

    #[test]
    fn the_argmax_survives_rounding() {
        let p = [0.3349, 0.3351, 0.33];
        let units = largest_remainder(&p, 2);
        assert_eq!(units.iter().sum::<u64>(), 100);
        let top =
            units.iter().enumerate().max_by_key(|&(i, u)| (*u, std::cmp::Reverse(i))).unwrap();
        assert_eq!(top.0, 1);
    }

    #[test]
    fn an_unnormalized_softmax_is_renormalized() {
        let units = largest_remainder(&[0.4999999, 0.4999999], 2);
        assert_eq!(units, vec![50, 50]);
    }

    #[test]
    fn floats_come_back_on_the_grid() {
        let v = round_distribution(&[0.123456, 0.876544], 3);
        assert_eq!(v, vec![0.123, 0.877]);
    }
}
