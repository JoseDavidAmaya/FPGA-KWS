import numpy as np
from pyftdi.spi import SpiController

showInfo = True

def init(freq):
  spi = SpiController()
  spi.configure('ftdi://ftdi:232h/1')
  slave = spi.get_port(cs=0, freq=freq, mode=0)

  return slave

slave = init(6E6)
resultNumReads = 5 # How many times to read the result to ensure it's the correct one

def writeInData(inData): # inData should be a numpy array of type np.int8/uint8
  slave.write(b'STW')
  parts = np.ceil(inData.size/1024)
  for part in np.split(inData.flatten(), parts): # inData is expected to be 2kB maximum, if we write it all at once we get a timeout error, so we split it in multiple parts
    slave.write(part.tobytes())

import statistics
def getResults():
  result = slave.read(1)
  while result == b'\xFF': # FPGA returns 0xFF when it's busy
    result = slave.read(1)
    
 
  # We expect a value between 0 and 11, if the result is not in the range then the read was usuccessful
  #res = int(result[0])
  #while res > 11:
  #  if showInfo:
  #    print("[FPGA INFO]: Received result [{0}] outside of expected range, retrying read".format(res))
  #  result = slave.read(1)
  #  res = int(result[0])
  
  # To be sure that we read the right value, we read multiple times and choose the most frequent value
  res = int(result[0])
  resultList = [res]
  for _ in range(resultNumReads-1):
    result = slave.read(1)
    res = int(result[0])
    resultList.append(res)
    
  res = statistics.mode(resultList)
    
  return res
  
def inference(inData):
  writeInData(inData)
  return getResults()