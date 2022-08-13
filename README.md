Work in progress

# FPGA-KWS

FPGA Implementation of a recurrent neural network for Keyword Spotting

Tested on the [Nexys 4 DDR](https://google.com) FPGA board

Input data of the network is sent to the FPGA using the [FTDI C232HM-DDHSL-0](https://ftdichip.com/products/c232hm-ddhsl-0-2/) USB-Serial cable

The demo is built with [gradio](https://gradio.app/)

# TODO

Install requirements

```
pip install -r requirements.txt
```

To run a script run

```
python -m scripts.scriptName
```

To run the demo

```
python -m demo.gradioDemo
```