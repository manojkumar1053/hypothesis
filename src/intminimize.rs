use std::cmp::min;

const SMALL: u64 = 5;

struct Minimizer<'a, F: 'a> {
    criterion: &'a mut F,
    best: u64,
}

impl<'a, F, T> Minimizer<'a, F>
where
    F: 'a + FnMut(u64) -> Result<bool, T>,
{
    fn test(&mut self, candidate: u64) -> Result<bool, T> {
        if candidate == self.best {
            return Ok(true);
        }
        if candidate > self.best {
            return Ok(false);
        }
        let result = (self.criterion)(candidate)?;
        if result {
            self.best = candidate;
        }
        Ok(result)
    }

    fn modify<G>(&mut self, g: G) -> Result<bool, T>
    where
        G: Fn(u64) -> u64,
    {
        let x = g(self.best);
        self.test(x)
    }
}

pub fn minimize_integer<F, T>(start: u64, mut criterion: F) -> Result<u64, T>
where
    F: FnMut(u64) -> Result<bool, T>,
{
    if start == 0 {
        return Ok(start);
    }

    for i in 0..min(start, SMALL) {
        if criterion(i)? {
            return Ok(i);
        }
    }
    if start <= SMALL {
        return Ok(start);
    }

    let mut minimizer = Minimizer {
        best: start,
        criterion: &mut criterion,
    };

    loop {
        if !minimizer.modify(|x| x >> 1)? {
            break;
        }
    }

    for i in 0..64 {
        minimizer.modify(|x| x ^ (1 << i))?;
    }

    assert!(minimizer.best >= SMALL);

    if !minimizer.modify(|x| x - 1)? {
        return Ok(minimizer.best);
    }

    let mut lo = 0;
    let mut hi = minimizer.best;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if minimizer.test(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Ok(minimizer.best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_failing_minimize<F>(start: u64, criterion: F) -> u64
    where
        F: Fn(u64) -> bool,
    {
        let r: Result<u64, ()> = minimize_integer(start, |x| Ok(criterion(x)));
        r.unwrap()
    }

    #[test]
    fn minimize_down_to() {
        let n = non_failing_minimize(100, |x| x >= 10);
        assert_eq!(n, 10);
    }

    #[test]
    fn unset_relevant_bits() {
        let x = 0b101010101010;
        let y = 0b111111111111;
        let n = non_failing_minimize(y, |k| k & x == x);
        assert_eq!(n, x);
    }
}
