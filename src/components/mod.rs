#[cfg(feature = "ML")]
pub mod ml;

#[cfg(feature = "ML")]
use ml::MachineLearningInferenceTest;
use std::fmt::{Display, Formatter};

pub struct TestResult {
    seconds: f64,
    accuracy: f32,
}

impl Display for TestResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut r = write!(f, " Seconds: {:.2}", self.seconds);
        if self.accuracy >= 0f32 {
            r = write!(f, ", Accuracy: {:2}%", self.accuracy * 100.0f32);
        }
        r
    }
}

pub trait BenchmarkTest {
    fn name(&self) -> String;
    fn run(&self) -> TestResult;
}

pub fn run() {
    #[cfg(feature = "ML")]
    {
        let test = MachineLearningInferenceTest::new().map_err(|x| eprintln!("{:?}", x)).unwrap();
        println!("{}: {}", test.name(), test.run());
    }
}
