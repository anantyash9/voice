###### Voice Comparison Engine ###### 

# imports
from sklearn.manifold import TSNE, MDS
from keras.models import load_model
from IPython.display import SVG, Audio, display
from keras.utils.vis_utils import model_to_dot
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
import soundfile as sf
import os
import sys

# session controll parameters to keep keras from hogging all the GPU memory
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)

set_session(sess)  # set this TensorFlow session as the default session for Keras

# load model
model_path = 'models/siamese_voice.hdf5'
siamese = load_model(model_path)
siamese = load_model(model_path)
siamese._make_predict_function()


def whiten(batch, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude """

    if len(batch.shape) != 3:
        raise(ValueError, 'Input must be a 3D array of shape (n_segments, n_timesteps, 1).')

    # Subtract mean
    sample_wise_mean = batch.mean(axis=1)
    whitened_batch = batch - np.tile(sample_wise_mean, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    # Divide through
    sample_wise_rescaling = rms / np.sqrt(np.power(batch, 2).mean())
    whitened_batch = whitened_batch * np.tile(sample_wise_rescaling, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    return whitened_batch


def preprocessor(downsampling, whitening=True):
    """ Downsample and fix RMS for inference """
    def preprocessor_(batch):
        ([i_1, i_2], labels) = batch
        i_1 = i_1[:, ::downsampling, :]
        i_2 = i_2[:, ::downsampling, :]
        if whitening:
            i_1, i_2 = whiten(i_1), whiten(i_2)

        return [i_1, i_2], labels

    return preprocessor_

downsampling = 4
whiten_downsample = preprocessor(downsampling, whitening=True)

def embedings(file_path_list):
    """ Make embedings from a list of audio files"""
    
    X=[]
    Z=[]
    for file_path in file_path_list:
        instance,samperate=sf.read(file_path)
        instance=instance[:48000]
        Z.append([instance,samperate])
    X_ = np.stack(list(zip(*Z))[0])[:, :, np.newaxis]
    [X_, _], _ = whiten_downsample(([X_, X_], []))

    X.append(X_)    
    X = np.concatenate(X)
    return X

def average_distance(audio_list1,audio_list2):
    """ Calculate distance between list of two audio sample/s """
    
    voice_saved=embedings(audio_list1)
    voice_captured=embedings(audio_list2)
    all_distances=siamese.predict([voice_saved,voice_captured])
    avg_dist=np.average(all_distances)
    return avg_dist
