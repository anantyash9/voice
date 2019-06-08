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

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)

set_session(sess)  # set this TensorFlow session as the default session for Keras

from voicemap.utils import whiten

LIBRISPEECH_SAMPLING_RATE=16000
model_path = 'models/siamese_voice.hdf5'
siamese = load_model(model_path)
downsampling = 4

siamese = load_model(model_path)
siamese._make_predict_function()


def preprocessor(downsampling, whitening=True):
    def preprocessor_(batch):
        ([i_1, i_2], labels) = batch
        i_1 = i_1[:, ::downsampling, :]
        i_2 = i_2[:, ::downsampling, :]
        if whitening:
            i_1, i_2 = whiten(i_1), whiten(i_2)

        return [i_1, i_2], labels

    return preprocessor_


whiten_downsample = preprocessor(downsampling, whitening=True)

def mut_embedings(file_path_list):
    
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
    voice_saved=mut_embedings(audio_list1)
    voice_captured=mut_embedings(audio_list2)
    all_distances=siamese.predict([voice_saved,voice_captured])
    avg_dist=np.average(all_distances)
    return avg_dist
