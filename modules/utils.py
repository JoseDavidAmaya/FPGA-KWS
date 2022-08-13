import tensorflow as tf
import numpy as np


def pad(variable, m, supressOutput=False):
    paddings = np.int32(np.ceil(np.divide(variable.shape, m))*m) - variable.shape #np.mod(variable.shape, m)
    if (np.sum(paddings)) != 0:
        if not supressOutput:
            print("[Next variable will be padded with zeroes]")
        variable = np.pad(variable, [ [0, x] for x in paddings ])

    return variable

# Note: Data is padded with zeroes at the end if the size isn't a multiple of m
def saveForVerilog(variable, filename, m, transpose=True, sep=" ", padVariable=False, supressOutput=False): # m = how many 8-bit numbers per address (m)
    
    if padVariable:
        # Pad adding zeroes at the end
        variable = pad(variable, m, supressOutput)

        # Pad adding zeroes before the last values
        #for i in range(0, len(variable.shape)):
        #    if paddings[i] != 0:
        #        print("[Variable below is padded]")
        #        pos = variable.shape[i]-(8-paddings[i])
        #        variable = np.insert(variable, [pos,]*paddings[i], 0, axis=i)


    if transpose:
        variable.astype(np.uint8).T.tofile(filename, sep=sep, format="%02X")
    else:
        variable.astype(np.uint8).tofile(filename, sep=sep, format="%02X")

    with open(filename, "r") as f:
        data = f.read()

    dataToWrite = ""
    spacesCounter = 0
    for i in data:
        if i == sep:
            spacesCounter = (spacesCounter+1) % m
            if spacesCounter == 0:
                dataToWrite += i
        
        else:
            dataToWrite += i

    with open(filename, "w") as f:
        f.write(dataToWrite)

    memdepth = int(np.ceil(variable.size/m))
    if not supressOutput:
        print(filename)
        print("NumBytes: ", variable.size)
        print("MemWidth: ", m*8)
        print("MemDepth: ", memdepth)

    return memdepth