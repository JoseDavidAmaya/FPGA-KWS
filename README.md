# FPGA-KWS

FPGA Implementation of a recurrent neural network for Keyword Spotting

Tested on the [Nexys 4 DDR](https://google.com) FPGA board

Input data of the network is sent to the FPGA using the [FTDI C232HM-DDHSL-0](https://ftdichip.com/products/c232hm-ddhsl-0-2/) USB-Serial cable

The demo is built with [gradio](https://gradio.app/)

[bitstream](./bitstream/) files are available for a large and a small model. The [large model](./bitstream/NN_Controller_L_PQT.bit) achieves 94.78% accuracy on the test set of Google's speech commands dataset with an inference time of ~37ms, while the [small model](./bitstream/NN_Controller_SC_PQT.bit) achieves 92.35% with an inference time of ~5ms

# Requirements

## Software

- [Python 3](https://www.python.org/)
- [ffmpeg](https://ffmpeg.org/)
	- On Windows, executables directory should be added to PATH
- [Xilinx Vivado](https://www.xilinx.com/products/design-tools/vivado.html) (Not needed to only run the demo)
- [FTDI Cable D2XX drivers](https://ftdichip.com/drivers/d2xx-drivers/)
    - On Windows [extra steps are required](https://eblot.github.io/pyftdi/installation.html#windows)


## Python requirements

To install the python libraries run

```
pip install -r requirements.txt
```

# Project structure

[aconnect](./aconnect/) and [modules](./modules/) folders contain python modules with utilities to build the models, download the dataset, perform feature extraction, and run inference on the FPGA.

On the [scripts](./scripts/) folder:
- [downloadDataset.py](./scripts/downloadDataset.py) downloads the dataset (only has to be run once) so that when running any other script the dataset is ready
- [train.py](./scripts/train.py) trains a GRU model and saves it in the project folder
- [findQmn.py](./scripts/findQmn.py) quantizes a saved model with different Qm.n formats and reports the Qm.n format that gives the highest accuracy to use on the implementation
- [postQuantTrain.py](./scripts/postQuantTrain.py) perform post-quantization training on a model and saves it (before running this script, run [findQmn.py](./scripts/findQmn.py) to find the appropiate Qm.n format)
- [generateVerilogFiles.py](./scripts/generateVerilogFiles.py) quantizes and saves the parameters of the model in a .mem file, also updates a verilog header with some information about the model, (before running this script, run [findQmn.py](./scripts/findQmn.py) to find the appropiate Qm.n format)
- [randomSim.py](./scripts/randomSim.py) creates random input data for the model and calculates the output saving it in a .mem file to compare with the output of the simulation in Vivado
- [getFpgaAccuracy.py](./scripts/getFpgaAccuracy.py) performs inference with the test set of Google's speech commands dataset on the FPGA and reports the accuracy.

[KWSNN.srcs](./KWSNN.srcs/) contains sources for Vivado to run the implementation, the top module is [NN_Controller.v](./KWSNN.srcs/design/NN_Controller.v), the files in [verilogHeaders](./verilogHeaders/) **NNparams.vh** (generated with [generateVerilogFiles.py](./scripts/generateVerilogFiles.py)) and [NNprogram.vh](./verilogHeaders/NNprogram.vh) have to be imported also to the Vivado project

The [mem](./mem/) folder contains .mem files used by Vivado for the implementation

The [demo](./demo/) folder contains the script to run the demo

# Instructions

To run a script run

```
python -m scripts.scriptName
```

To run the demo

```
python -m demo.gradioDemo
```