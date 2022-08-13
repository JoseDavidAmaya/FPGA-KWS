"""This script generates two files used for the FPGA implementation of a model
GRU.mem: which contains the quantized parameters of the model
NNparams.vh: which contains parameters used by the verilog code, like the size and width of memories, and some values used in NNprogram.vh 
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

import tensorflow as tf
import keras
import numpy as np

import pathlib

from aconnect.layers import GRUCell_AConnect
from modules.quant import q, qInteger, quantSigm, quantTanh
from modules.utils import pad, saveForVerilog

modelPath = "models/GRU_L"
integerBits = 3 # Value of m in Qm.n, found with the script findQmn.py
memwidth = 64 # Should be a power of 2 and a multiple of 8



# Load model
custom_objects = {"quantTanh": quantTanh, "quantSigm": quantSigm, "GRUCell_AConnect": GRUCell_AConnect}
model = keras.models.load_model(modelPath, custom_objects=custom_objects)
input_shape = model.input.shape[1:]

# Get model weights

    # Floating point versions
GRUkernel = model.layers[0].weights[0].numpy()
GRUreckernel = model.layers[0].weights[1].numpy()
GRUbias = model.layers[0].weights[2].numpy()
FCkernel = model.layers[1].weights[0].numpy()
FCbias = model.layers[1].weights[1].numpy()

    # Quantized, integer versions
GRUkernel_quant = qInteger(GRUkernel, integerBits)
GRUreckernel_quant = qInteger(GRUreckernel, integerBits)
GRUbias_quant = qInteger(GRUbias, integerBits)
FCkernel_quant = qInteger(FCkernel, integerBits)
FCbias_quant = qInteger(FCbias, integerBits)

# Pad parameters based on the memwidth
elemwidth = 8
m = int(memwidth/elemwidth)

names_quant_verilog = ["qgrukernel.mem", "qgrureckernel.mem", "qgrubias.mem", "qfckernel.mem", "qfcbias.mem"]

# -- GRUkernel, GRUreckernel and GRUbias are actually three sets of parameters, and each of the three sets has to be padded with zeroes instead of just the whole concatenation, because we sometimes do operations that only use one of the three sets
GRUkernel_quant_pad = np.concatenate([pad(x, m) for x in np.split(GRUkernel_quant, 3, axis=1)], axis=1)
GRUreckernel_quant_pad = np.concatenate([pad(x, m) for x in np.split(GRUreckernel_quant, 3, axis=1)], axis=1)
GRUbias_quant_pad = np.concatenate([pad(x, m) for x in np.split(GRUbias_quant, 3, axis=0)], axis=0)
# --

params_quant = [GRUkernel_quant_pad, GRUreckernel_quant_pad, GRUbias_quant_pad, FCkernel_quant, FCbias_quant]
template_memdepth_placeholders = ["{MEMDEPTH_GRU_KERNEL}", "{MEMDEPTH_GRU_REC_KERNEL}", "{MEMDEPTH_GRU_BIAS}", "{MEMDEPTH_FC_KERNEL}", "{MEMDEPTH_FC_BIAS}"]
template_dict = {}

memFilesFolder = "mem/"
verilogFilesFolder = "verilogHeaders/"

# Quant, verilog
for name, param, template_placeholder in zip(names_quant_verilog, params_quant, template_memdepth_placeholders):
    template_dict[template_placeholder] = saveForVerilog(param, memFilesFolder+name, m, padVariable=True)

# Concat all to save into one big memory
data = ""
for name in names_quant_verilog:
    with open(memFilesFolder+name, "r") as file:
        data += file.read() + " "
    
with open(memFilesFolder+"GRU.mem", "w") as file:
    file.write(data)

# Rewrite params .mem file as .coe file (Used by Block Memory Generator IP)
data = "memory_initialization_radix=16;\n"
data += "memory_initialization_vector=\n"
with open(memFilesFolder+"GRU.mem") as fparams:
    dataParams = fparams.read().replace(" ", ",\n")
    data += dataParams[:-2] + ";"

with open(memFilesFolder+"GRU.coe", "w") as fparams:
    fparams.write(data)

# Write NNparams.vh
template_dict["{MEMWIDTH}"] = memwidth
template_dict["{GRU_OUT_SIZE}"] = int(np.ceil(GRUkernel_quant_pad.shape[1]/3/m))
template_dict["{INPUT_FEATURE_SIZE}"] = int(np.ceil(input_shape[1]/m))
template_dict["{NUM_INPUT_TIMESTEPS}"] = input_shape[0]

vec_size = model.output.shape[-1]/m
vec_end = int(np.ceil(vec_size))
last_vec_size = m - round((vec_end - vec_size)*m)

template_dict["{ARGMAX_VECEND}"] = vec_end
template_dict["{ARGMAX_LAST_VEC_SIZE}"] = last_vec_size

template_dict["{MEMDEPTH_PARAMS}"] = np.sum([template_dict[x] for x in template_memdepth_placeholders])

## Paths of .mem files
template_dict["{MEM_PARAMS_INIT_FILE}"] = pathlib.Path(memFilesFolder+"GRU.mem").resolve().as_posix()
template_dict["{MEM_INPUT_INIT_FILE_SIM}"] = pathlib.Path(memFilesFolder+"qrandomdata.mem").resolve().as_posix()
template_dict["{MEM_OUTPUT_INIT_FILE_SIM}"] = pathlib.Path(memFilesFolder+"randomOpResult.mem").resolve().as_posix()

template_dict["{MEM_7SEG_LABELS}"] = pathlib.Path(memFilesFolder+"memlabels.mem").resolve().as_posix()
template_dict["{MEM_7SEG_ENCODER}"] = pathlib.Path(memFilesFolder+"memencoder.mem").resolve().as_posix()

with open(verilogFilesFolder+"NNparams.template.vh", "r") as file:
    filestr = file.read()
    for valuePlaceholder, value in template_dict.items():
        filestr = filestr.replace(valuePlaceholder, str(value))

    with open(verilogFilesFolder+"NNparams.vh", "w") as filew:
        filew.write(filestr)