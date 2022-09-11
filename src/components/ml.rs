use crate::components::{BenchmarkTest, TestResult};
use anyhow::anyhow;
use std::time::SystemTime;
use wasi_nn;

const ML_XML: &str = include_str!("../../ML/model.xml");
const ML_WEIGHTS: &[u8] = include_bytes!("../../ML/model.bin");
const ML_DATA: &[u8] = include_bytes!("../../ML/iris_data.dat");
const ML_LABELS: &[u8] = include_bytes!("../../ML/iris_labels.dat");
const FLOATSIZE: usize = 4;
const DATASIZE: usize = 150;

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
    let mut labels_vec = Vec::new();
    let mut counter = 0;
    loop {
        labels_vec.push(slice_to_f32(&ML_LABELS[counter..counter + FLOATSIZE]));
        counter += FLOATSIZE;
        if counter >= ML_LABELS.len() {
            break;
        }
    }
    assert_eq!(ML_LABELS.len()/FLOATSIZE, DATASIZE);
    labels_vec
}

fn get_iris_data() -> Vec<Vec<f32>> {
    let mut data = Vec::new();
    let mut counter = 0;
    loop {
        let mut inner_counter = 0;
        let mut loop_counter = 0;
        let mut data_point = Vec::new();
        loop {
            data_point.push(slice_to_f32(&ML_DATA[inner_counter..inner_counter + FLOATSIZE]));
            inner_counter += FLOATSIZE;
            loop_counter += 1;
            if loop_counter > 3 {
                break;
            }
        }
        data.push(data_point);
        counter += 1;
        if counter >= DATASIZE {
            break;
        }
    }

    data
}

fn data_point_to_bytes(data: &Vec<f32>) -> Vec<u8> {
    let mut dp = Vec::new();
    for d in data {
        for b in d.to_le_bytes() {
            dp.push(b);
        }
    }
    dp
}

impl MachineLearningInferenceTest {
    pub fn new() -> anyhow::Result<MachineLearningInferenceTest> {
        let graph = unsafe {
            match wasi_nn::load(
                &[&ML_XML.to_string().into_bytes(), &ML_WEIGHTS],
                wasi_nn::GRAPH_ENCODING_OPENVINO,
                wasi_nn::EXECUTION_TARGET_CPU,
            ) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("Failed to load model files: {}", e);
                    return Err(anyhow!("Failed to load model files: {}", e));
                }
            }
        };
        eprintln!("ML graph created.");
        let context = unsafe {
            match wasi_nn::init_execution_context(graph) {
                Ok(x) => x,
                Err(e) => {
                    return Err(anyhow!("Failed to initialize execution graph: {}", e));
                }
            }
        };
        eprintln!("ML context created.");
        Ok(MachineLearningInferenceTest { graph, context })
    }
}

impl BenchmarkTest for MachineLearningInferenceTest {
    fn name(&self) -> String {
        "MachineLearningInference".to_string()
    }

    fn run(&self) -> TestResult {
        let data = get_iris_data();
        let labels = get_iris_labels();

        assert_eq!(data.len(), labels.len());
        let start = SystemTime::now();
        let mut counter = 0;
        let mut correct = 0;

        loop {

            let dp = data_point_to_bytes(&data[counter]);

            // Notes:
            // * Data has to be an array of bytes, as Wasi-NN::`TensorData` is just `&[u8]`
            // * Input seems to be unhappy with all the data, or just one data point
            // * Seems that incorrect assumptions were made inside Wasmtime-Wasi-NN:openvino.rs, using `Layout::NHWC`, which is [Batch, Height, Width, Channels], which assumes colour images. There more to ML that just looking at pictures! This was changed to `Layout::ANY`, which seemed to help loading the model, may have issues later.

            let tensor = wasi_nn::Tensor {
                dimensions: &[150, 4],
                r#type: wasi_nn::TENSOR_TYPE_F32,
                data: dp.as_slice(),
            };

            unsafe {
                wasi_nn::set_input(self.context, 0, tensor).unwrap();
                wasi_nn::compute(self.context).unwrap();
            }
            let mut output_buffer = vec![0f32; 150];
            unsafe {
                #[allow(unused_must_use)]
                wasi_nn::get_output(
                    self.context,
                    0,
                    &mut output_buffer[..] as *mut [f32] as *mut u8,
                    (output_buffer.len() * 4).try_into().unwrap(),
                );
            }

            if labels.len() != output_buffer.len() {
                eprintln!(
                    "Error, size from model is {}, but read {} labels from dataset.",
                    output_buffer.len(),
                    labels.len()
                );
            } else {
                for index in 0..labels.len() {
                    if labels.get(index) == output_buffer.get(index) {
                        correct += 1;
                    }
                }
            }

            counter += 1;
            if counter > data.len() {
                break;
            }
        }

        let duration = start.elapsed().unwrap();

        TestResult {
            seconds: duration.as_secs_f64(),
            accuracy: correct as f32 / labels.len() as f32,
        }
    }
}
