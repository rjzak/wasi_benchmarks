use crate::components::{BenchmarkTest, TestResult};
use wasi_nn;

use std::time::SystemTime;

const ML_XML: &[u8] = include_bytes!("../../ML/model.xml");
const ML_WEIGHTS: &[u8] = include_bytes!("../../ML/model.bin");
const ML_DATA: &[u8] = include_bytes!("../../ML/iris_data.dat");
const ML_LABELS: &[u8] = include_bytes!("../../ML/iris_labels.dat");

pub struct MachineLearningInferenceTest {
    graph: wasi_nn::Graph,
    context: wasi_nn::GraphExecutionContext,
}

fn slice_to_f32(data: &[u8]) -> f32 {
    let mut temp_ints: [u8; 4] = [0; 4];
    temp_ints[0] = data[0];
    temp_ints[1] = data[1];
    temp_ints[2] = data[2];
    temp_ints[3] = data[3];
    f32::from_be_bytes(temp_ints)
}


fn get_iris_labels() -> Vec<f32> {
    const FLOATSIZE:usize = 4;
    let mut labels_vec = Vec::new();
    let mut counter = 0;
    loop {
        labels_vec.push(slice_to_f32(&ML_LABELS[counter..counter+FLOATSIZE]));
        counter += FLOATSIZE;
        if counter >= ML_LABELS.len() {
            break;
        }
    }
    labels_vec
}

impl MachineLearningInferenceTest {
    pub fn new() -> MachineLearningInferenceTest {
        let graph = unsafe { wasi_nn::load(&[&ML_XML, &ML_WEIGHTS], wasi_nn::GRAPH_ENCODING_OPENVINO, wasi_nn::EXECUTION_TARGET_CPU).unwrap() };
        let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
        MachineLearningInferenceTest {
            graph,
            context
        }
    }
}

impl BenchmarkTest for MachineLearningInferenceTest {
    fn name(&self) -> String {
        "MachineLearningInference".to_string()
    }

    fn run(&self) -> TestResult {
        let tensor = wasi_nn::Tensor { dimensions: &[150, 4], r#type: wasi_nn::TENSOR_TYPE_F32, data: &ML_DATA };

        let start = SystemTime::now();

        unsafe {
            wasi_nn::set_input(self.context, 0, tensor).unwrap();
            wasi_nn::compute(self.context).unwrap();
        }
        let mut output_buffer = vec![0f32; 150];
        unsafe {
            #[allow(unused_must_use)]
            wasi_nn::get_output(self.context, 0, &mut output_buffer[..] as *mut [f32] as *mut u8, (output_buffer.len() * 4).try_into().unwrap());
        }

        let mut correct = 0;
        let labels = get_iris_labels();
        if labels.len() != output_buffer.len() {
            eprintln!("Error, size from model is {}, but read {} labels from dataset.", output_buffer.len(), labels.len());
        } else {
            for index in 0..labels.len() {
                if labels.get(index) == output_buffer.get(index) {
                    correct += 1;
                }
            }
        }

        let duration = start.elapsed().unwrap();

        TestResult {
            seconds: duration.as_secs_f64(),
            accuracy: correct as f32 / labels.len() as f32
        }
    }
}