"""Creates random imput data, calculates the output of the model and saves it in a .mem file to compare with the results of the simulation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

import keras
import numpy as np

from aconnect.layers import GRUCell_AConnect
from modules.quant import q, qInteger, quantSigm, quantTanh
from modules.utils import saveForVerilog

modelPath = "models/GRU_L"
integerBits = 3 # Value of m in Qm.n, found with the script findQmn.py
memwidth = 64 # Should be a power of 2 and a multiple of 8



# Load model
custom_objects = {"quantTanh": quantTanh, "quantSigm": quantSigm, "GRUCell_AConnect": GRUCell_AConnect}
model = keras.models.load_model(modelPath, custom_objects=custom_objects)
input_shape = model.input.shape[1:]

# Get model weights

elemwidth = 8
m = int(memwidth/elemwidth)

    # Floating point versions
GRUkernel = model.layers[0].weights[0].numpy()
GRUreckernel = model.layers[0].weights[1].numpy()
GRUbias = model.layers[0].weights[2].numpy()
FCkernel = model.layers[1].weights[0].numpy()
FCbias = model.layers[1].weights[1].numpy()

    # Quantized, float versions
GRUkernel_fquant = q(GRUkernel, integerBits)
GRUreckernel_fquant = q(GRUreckernel, integerBits)
GRUbias_fquant = q(GRUbias, integerBits)
FCkernel_fquant = q(FCkernel, integerBits)
FCbias_fquant = q(FCbias, integerBits)


def qq(x):
    return q(x, integerBits)

sigm = quantSigm
tanh = quantTanh

def inferenceQT(data):
    ## GRU
    units = GRUreckernel_fquant.shape[0]
    htm1 = np.zeros([units])
    
    counter = 0
    for inputs in data[0]:

        matrixx = qq(qq(np.matmul(inputs, GRUkernel_fquant)) + GRUbias_fquant)

        xz, xr, xh = np.split(matrixx, 3)

        matrixinner = qq(np.matmul(htm1, GRUreckernel_fquant[:, :2*units]))

        recz, recr = np.split(matrixinner, 2)
        

        z = qq(sigm(qq(xz + recz)))
        r = qq(sigm(qq(xr + recr)))

        rech = qq(np.matmul(qq(r * htm1), GRUreckernel_fquant[:, 2*units:]))

        hh = qq(tanh(qq(xh + rech)))

        h = qq(qq(z * htm1) + qq(qq(1 - z) * hh))

        htm1 = h # Update last state

        counter = counter + 1        

    out = qq(qq(np.matmul(htm1, FCkernel_fquant)) + FCbias_fquant)

    return out.numpy()

# Generate random input data and save it
randomData = np.random.randn(1, *input_shape).astype(np.float32)*10
qrandomData = qInteger(randomData, integerBits)
saveForVerilog(qrandomData[0], "mem/qrandomdata.mem", m, False, padVariable=True)
qfrandomData = q(randomData, integerBits)

# Calculate output and save to .mem file
out = inferenceQT(qfrandomData) #.numpy()
qout = (out*2**(7-integerBits)).astype(np.int8)
saveForVerilog(qout, "mem/randomOpResult.mem", m, transpose=False, sep="\n", padVariable=True)