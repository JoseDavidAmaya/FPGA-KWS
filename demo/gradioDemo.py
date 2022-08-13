"""Web interface to record audio and perform inference on the FPGA
Built with gradio
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # To disable GPU Usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # To avoid filling the whole memmory

import gradio as gr

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

from modules.kws import model
from modules.kws import dataset
from modules.quant import qInteger
from modules.utils import pad
from modules import fpgaInference as fpga


datasetConfig = model.GRU_L_datasetConfig
integerBits = 3 # Value of m in Qm.n, found with the script findQmn.py
memwidth = 64 # Should be a power of 2 and a multiple of 8
#videoUrl = "http://mega.tplinkdns.com/video"
videoUrl = "http://192.168.10.45:8080/video"

# Some preprocessing
def getImportantAudioSlice(audio): # Supposed to work with 16khz audio
  convWidth = 8000 # 16000
  centerIndex = tf.argmax(np.convolve(tf.cast(tf.abs(audio), tf.float32), tf.ones(convWidth)/tf.cast(convWidth, tf.float32), 'same'))
  
  startIndex = centerIndex - int(16000/2)
  endIndex = centerIndex + int(16000/2)
  
  if startIndex < 0:
    startIndex = 0
    endIndex += abs(startIndex)

  elif endIndex > audio.shape[0]:
    endIndex = audio.shape[0]
    startIndex -= (endIndex - audio.shape[0])
    if startIndex < 0:
      startIndex = 0

  return audio[startIndex:endIndex]

def supressNoise(audio): # Supposed to work with 48khz audio
  filtered = np.convolve(tf.abs(audio), tf.ones(4000)/4000.0, 'same')
  audioNoNoise = audio * (filtered > (np.max(filtered) * 0.02) )
  
  return audioNoNoise

# Inference function
def speech_recognize_fpga(audio):

  print("-"*80)
  print("[DEBUG]:", audio)
  if audio is None:
    return "[ERROR]: No audio, try recording again"
  
  samp_rate, audio = audio
  print("[DEBUG] [Max Audio Val]:", tf.reduce_max(tf.abs(audio)))
  
  ## Supress noise
  #audio = supressNoise(audio)
  
  ## Normalize the audio data
  audioMax = tf.reduce_max(tf.abs(audio))
  multFactor = tf.cast(tf.divide(tf.int32.max, audioMax), tf.int32)
  audio = tf.multiply(audio, multFactor)
  
  # Audio comes in as 32 bit, we convert it to 16 bit
  audio = tf.cast(tf.bitwise.right_shift(audio, 16), tf.int16) 
    
  if samp_rate != 16000:
    audio = tfio.audio.resample(audio, samp_rate, 16000)

  audio = getImportantAudioSlice(audio)

  # pad audio based on memwidth
  elemwidth = 8
  m = int(memwidth/elemwidth)
  audio = pad(audio, m, supressOutput=True)[:audio.shape[0]]
  
  features = dataset.audio_to_features(audio, **datasetConfig)
  
  # Quantize
  features = qInteger(features, integerBits)
  
  label_index = fpga.inference(features)
  
  print("[DEBUG] [label]: ", label_index)
  return dataset.dataset_speech_commands_labels[label_index]


## Gradio

# Webpage formatting

source = "microphone"

title = "Keyword Spotting on FPGA demo"
description = ""
with open("demo/description.html") as descr:
  descriptionFirst = descr.read()
description += "Supported words:<br>"

inRect = lambda x: '<span style="padding: 0.2em 0.4em; margin: 0; font-size: 85%; background-color: rgba(110,118,129,0.4); border-radius: 6px; font-weight: bold;">' + x + '</span>'
description += inRect(dataset.dataset_speech_commands_labels[0])
for l in dataset.dataset_speech_commands_labels[1:10]:
  description += "  " + inRect(l)

article = f'<img style="max-height: 400px;" alt="video" src="{videoUrl}">'

centerHTML = lambda x: "<center>" + x + "</center>"

# start the demo

demo = gr.Interface(fn=speech_recognize_fpga,
                    inputs=gr.Audio(source=source, type="numpy"),
                    outputs="label",
                    title=title,
                    description=(descriptionFirst + centerHTML(description)),
                    article=centerHTML(article))
demo.launch()