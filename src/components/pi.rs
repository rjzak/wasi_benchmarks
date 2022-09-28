use crate::components::{BenchmarkTest, TestResult};
use fraction::Fraction;
use num_bigint::{BigInt, BigUint};
use num_traits::pow::Pow;
use std::ops::{Not, Sub};
use std::convert::TryFrom;
use std::time::SystemTime;


#[derive(Clone)]
pub struct PiCalcTest {
    precision: u64,
}

impl PiCalcTest {
    pub fn new() -> PiCalcTest {
        PiCalcTest {
            precision: 100
        }
    }
}

// Source:
// https://users.rust-lang.org/t/calculating-pi-to-1000-digit/30515/3
/// atan(x) = x - x^3/3 + x^5/5 - x^7/7 + x^9/9...
fn atan (x: Fraction, precision: u64) -> Fraction  {
    let end: BigUint =
        BigUint::from(10_u32)
            .pow(precision as u32)
        ;
    let target = Fraction::new(1.into(), end);

    let mut current_term = x.clone();
    let mut ret = Fraction::from(0);
    let mut sign = BigInt::from(1);
    let mut n = BigUint::from(1_u32);
    let mut x_pow_n = x.clone();
    let two = BigUint::from(2_u32);
    let x_square = &x * &x;

    while current_term.abs() > target {
        ret = ret + current_term;
        // eprintln!(
        //     "atan({}) ~ {}",
        //     x,
        //     ret.decimal(precision as usize),
        // );
        n += &two;
        sign = -sign;
        x_pow_n = x_pow_n * x_square;
        current_term = &x_pow_n * &Fraction::new(
            sign.clone(),
            n.clone(),
        );
    }
    ret
}

/// PI = 16 * atan(1/5) - 4 * atan(1/239)
fn pi (precision: u64) -> String {
    let precision_usize = usize::
    try_from(precision)
        .expect("Overflow")
        ;
    let pi_approx = Fraction::sub(
        Fraction::from(16) * atan(
            Fraction::new(1.into(), 5_u32.into()),
            precision
                .checked_add(2) // 16 -> 10 ^ 2
                .expect("Overflow"),
        ),
        Fraction::from(4) * atan(
            Fraction::new(1.into(), 239_u32.into()),
            precision + 1, // 4 -> 10 ^ 1
        ),
    );
    pi_approx.decimal(precision_usize)
}

impl BenchmarkTest for PiCalcTest {
    fn name(&self) -> String {
        "PiCalculator".to_string()
    }

    fn run(&self) -> TestResult {
        let start = SystemTime::now();

        let _ = pi(self.precision);

        let duration = start.elapsed().unwrap();

        TestResult {
            seconds: duration.as_secs_f64(),
            accuracy: 0.0,
        }
    }
}