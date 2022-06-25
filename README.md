## [WASI](https://wasi.dev/) Benchmarks (WiP)

### Intention:
To develop a suite of benchmarks for measuring the performance of various Wasi runtime environments, and the underlying hardware used.

### Notes:
* Based on the [Wasi-nn tutorial](https://bytecodealliance.org/articles/using-wasi-nn-in-wasmtime) from the Bytecode Alliance
* This project doesn't yet work, unfortunately.

### Compiling:
1. Install [OpenVINO](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
   1. Also install the [dev tools](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html)
2. Optional: train the ML model if you don't want to use the included model file.
   1. Activate the Python virtual environment in the OpenVINO instructions.
   2. Change directory `cp ML`
   3. Run `./train.py`
   4. Convert the model to the OpenVINO format: `mo --input_model model.onnx`
3. Compile [Wasmtime](https://github.com/bytecodealliance/wasmtime)
   1. `git clone --recursive https://github.com/bytecodealliance/wasmtime.git`
   2. `OPENVINO_INSTALL_DIR=/opt/intel/openvino_2022 cargo build --release --features wasi-nn`
4. Compile this project:
   1. Run: `cargo build --release --features ML`
   2. Run with the new Wasmtime compiled in step 3.
