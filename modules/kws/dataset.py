"""Obtain the dataset and perform feature extraction
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

# Download background noise samples
from glob import glob

background_noise = []
def prepare_background_noise():
    global background_noise
    # TODO: Do the steps below in python
    #![ ! -d "_background_noise_" ] && wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    #![ ! -d "_background_noise_" ] && tar -xf speech_commands_v0.02.tar.gz ./_background_noise_

    background_noise_files = tf.io.gfile.glob("./_background_noise_/*.wav")
    background_noise = tf.ragged.stack([tf.squeeze(tf.audio.decode_wav(tf.io.read_file(filename)).audio) for filename in background_noise_files])



# Feature extraction

def pad_audio(audio, audio_len_frames=16000): # Pads audio samples so that they are all the same length
  audio = audio[:audio_len_frames]
  audio = tf.pad(audio, [[0, audio_len_frames - tf.shape(audio)[0]]])

  return audio

def log_mel_spectrogram(audio, sample_rate, window_size, stride, mels): # log-Mel spectrogram from waveform
  spectrogram = tfio.audio.spectrogram(audio, nfft=512, window=window_size, stride=stride) # !!nfft
  mel_spectrogram = tfio.audio.melscale(spectrogram, rate=sample_rate, mels=mels, fmin=0, fmax=int(sample_rate/2)) # !!fmin !!fmax
  dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80) # !!top_db

  return dbscale_mel_spectrogram

def mfccs(audio, sample_rate, window_size, stride, dct_coeff_count): # Mel Frequency Cepstrum Coefficients from waveform
  # Based on the code from the github # https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/input_data.py#L377
  spectrogram = tf.raw_ops.AudioSpectrogram(input=tf.expand_dims(audio, 1), stride=stride, window_size=window_size, magnitude_squared=True)
  mfccs = tf.raw_ops.Mfcc(spectrogram=spectrogram, sample_rate=sample_rate, dct_coefficient_count=dct_coeff_count)
  mfccs = tf.squeeze(mfccs)

  return mfccs

def random_time_shift(audio, sample_rate=16000, ms_shift=100e-3): 
  audio_len_frames = tf.shape(audio)[0]
  frames_shift = tf.cast(ms_shift*sample_rate, dtype=tf.int32)
  shift_zero_padding = tf.zeros([frames_shift], dtype=tf.float32)
  
  audio = tf.concat([shift_zero_padding, audio, shift_zero_padding], 0) # add 'ms_shift' ms at the start and at the end (so that we have a shift from in the range [-ms_shift, ms_shift])
  random_shift = tf.random.uniform([], 0, frames_shift*2, dtype=tf.int32)
  audio = audio[random_shift:random_shift+audio_len_frames] # Select a randomly positioned slice with the length of audio
  
  return audio

def random_background_noise(audio, background_frequency=0.8, background_max_volume=0.1):
  if tf.random.uniform([], 0, 1, dtype=tf.float32) < background_frequency:
    audio_len_frames = tf.shape(audio)[0]
    noise_index = tf.random.uniform([], 0, background_noise.shape[0], dtype=tf.int32)
    noise = background_noise[noise_index] # Select a random sample of noise
    noise_start = tf.random.uniform([], 0, tf.shape(noise)[0]-audio_len_frames, dtype=tf.int32) 
    noise = noise[noise_start:noise_start+audio_len_frames] # Select a randomly positioned 1000 ms slice from the sample
    audio = audio + noise * tf.random.uniform([], 0, background_max_volume, dtype=tf.float32) # Scale the noise by a random value and add it to the audio
  
  return audio

def standardization(data, mean=True, std=True):
  if mean:
    data = data - tf.math.reduce_mean(data)
  if std:
    data = data/tf.math.reduce_std(data)

  return data

def audio_to_features(audio, data_augmentation=False, 
                      features_type="mfccs", # "logmel", "mfccs"
                      feature_standardization=[False, False], # [mean_standarization, stdev_standarization]
                      stride_s=20e-3, # Stride in seconds for the framing
                      window_size_s=40e-3, # Window size in seconds for the framing,
                      sample_rate=16000, # Sample rate of input data
                      num_features=10 # Number of features per timestep
                      ): 

  # Pad the waveform with 0's at the end
  audio = pad_audio(audio, audio_len_frames=sample_rate)

  audio = tf.cast(audio, tf.float32) # Cast to float32
  audio = audio/tf.cast(tf.int16.max, dtype=tf.float32) # Scale values to [-1, 1]

  ## Data augmentation
  if data_augmentation:
    # Apply random time shift of 100 ms
    audio = random_time_shift(audio, sample_rate=sample_rate, ms_shift=100e-3)

    # Add background noise to around 80% of samples with a random volume of maximum 10%
    audio = random_background_noise(audio, background_frequency=0.8, background_max_volume=0.1)

  # Features
  stride = int(stride_s * sample_rate)
  window_size = int(window_size_s * sample_rate)

  if features_type == "logmel":
    features = log_mel_spectrogram(audio, sample_rate, window_size, stride, num_features) 

  elif features_type == "mfccs":
    features = mfccs(audio, sample_rate, window_size, stride, num_features)

  else:
    raise Exception('audio_to_features: features_type should be either "logmel" or "mfccs"')

  # Standardization
  features = standardization(features, *feature_standardization)
    
  return features



# Datasets

# Google speech commands dataset

dataset_speech_commands_labels = [ 
           'down',
           'go',
           'left',
           'no',
           'off',
           'on',
           'right',
           'stop',
           'up',
           'yes',
           '_silence_',
           '_unknown_' ]


def get_dataset_speech_commands(datasetConfig, data_augmentation=False):
    train_ds, info = tfds.load('speech_commands', split='train', shuffle_files=True, with_info=True)
    train_ds = train_ds.map(lambda ds_dict: (audio_to_features(ds_dict['audio'], data_augmentation=data_augmentation, **datasetConfig), ds_dict['label']), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = tfds.load('speech_commands', split='validation').map(lambda ds_dict: (audio_to_features(ds_dict['audio'], **datasetConfig), ds_dict['label']), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = tfds.load('speech_commands', split='test').map(lambda ds_dict: (audio_to_features(ds_dict['audio'], **datasetConfig), ds_dict['label']), num_parallel_calls=tf.data.AUTOTUNE)

    train_num_examples = info.splits['train'].num_examples
    val_num_examples = info.splits['validation'].num_examples
    test_num_examples = info.splits['test'].num_examples

    splits = [train_ds, val_ds, test_ds]
    splits_length = [train_num_examples, val_num_examples, test_num_examples]
    for spec, _ in train_ds.take(1):
      features_shape = spec.shape
    num_labels = info.features['label'].num_classes

    return splits, splits_length, features_shape, num_labels

# Spoken digits dataset

def get_dataset_spoken_digits(datasetConfig):
    def sd_get_audio_number(x):
        return tf.strings.to_number(tf.strings.split(tf.strings.split(x['audio/filename'], "_")[-1], ".")[0])

    test_ds_range = [0, 4]
    val_ds_range = [5, 9]
    train_ds_range = [10, 100]

    ds, info = tfds.load('spoken_digit', split='train', shuffle_files=True, with_info=True)

    # Generate splits based on audio number
    test_ds = ds.filter(lambda x: (sd_get_audio_number(x) >= test_ds_range[0]) and (sd_get_audio_number(x) <= test_ds_range[1])).cache()
    val_ds = ds.filter(lambda x: (sd_get_audio_number(x) >= val_ds_range[0]) and (sd_get_audio_number(x) <= val_ds_range[1])).cache()
    train_ds = ds.filter(lambda x: (sd_get_audio_number(x) >= train_ds_range[0]) and (sd_get_audio_number(x) <= train_ds_range[1])).cache()

    train_num_examples = len(list(train_ds.as_numpy_iterator()))
    val_num_examples = len(list(val_ds.as_numpy_iterator()))
    test_num_examples = len(list(test_ds.as_numpy_iterator()))

    train_ds = train_ds.map(lambda ds_dict: (audio_to_features(ds_dict['audio'], data_augmentation=False, **datasetConfig), ds_dict['label']), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda ds_dict: (audio_to_features(ds_dict['audio'], **datasetConfig), ds_dict['label']), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda ds_dict: (audio_to_features(ds_dict['audio'], **datasetConfig), ds_dict['label']), num_parallel_calls=tf.data.AUTOTUNE)

    splits = [train_ds, val_ds, test_ds]
    splits_length = [train_num_examples, val_num_examples, test_num_examples]
    for spec, _ in train_ds.take(1):
      features_shape = spec.shape
    num_labels = info.features['label'].num_classes

    return splits, splits_length, features_shape, num_labels
