######## voice client #######

import argparse
import queue
import sys
import time
import azure.cognitiveservices.speech as speechsdk
import _thread
import requests
import helpers
import soundfile as sf
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import matplotlib
#matplotlib.use('TkAgg')


# argument parsing
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-p', '--plot', type=int,
    help=' Plot the audio waveform ? yes:1      No:0 ' ,default=1)    
args = parser.parse_args()



# state managment variables 

can_sst_start = True
can_validate = False
speech_key, service_region = "1e9f1691dcbd43758eadb5f7c2ddbd3f", "centralindia"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
SERVER_URL = 'http://localhost:4500/'
samplerate = 16000
downsample=4
window=200
interval=30
q = queue.Queue()
val = []
i = 0
volume_threshold= 4.5
lines,plotdata=None,None

def voice_pipeine():
    """Starts voice processing pipleine
        """ 
    global can_sst_start, can_validate
    _thread.start_new_thread(record, (4,))
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Speach to text conversion started")

    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        while (not can_validate):
            pass
        validated, emp_id = helpers.validate(result.text)
        if validated:
            requests.get(url=SERVER_URL + 'alternate/' + emp_id + '/' + 'BUILDING_IN/', timeout=5)

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    can_sst_start = True




def audio_callback(indata, frames, time, status):
    """callback for audio queue update"""
    global val, i, can_sst_start
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

    value = (np.average(np.absolute(indata[::4])))
    if len(val) > 50:
        val.pop(0)
    val.append(value)
    i += 1
    if i == 50:
        if sum(val) >= volume_threshold:
            print('Human has spoken ', sum(val))
            if can_sst_start:
                can_sst_start = False
                _thread.start_new_thread(voice_pipeine, ())

        i = 0


def record(sec):
    """Start audio recording for n second"""
    global can_validate
    can_validate = False
    with sf.SoundFile('default.wav', mode='w', samplerate=16000,
                      channels=1) as file:
        t_end = time.time() + sec
        while time.time() < t_end:
            file.write(q.get())
    can_validate = True


def update_plot(frame):
    """Callback for matplotlib on each plot update.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])

    return lines

def sound_loop():
    """ This loops till the program is shut down. All callbacks orginate from this function """
    global plotdata,lines
    try:
        length = int(window * samplerate / (1000 * downsample))
        plotdata = np.zeros((length, 1))
        fig, ax = plt.subplots()
        lines = ax.plot(plotdata)
        ax.axis((0, len(plotdata), -1, 1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
        fig.tight_layout(pad=0)

        stream = sd.InputStream( device=args.device, channels=1,
            samplerate= samplerate, callback=audio_callback)
        if (args.plot==0):
            with stream:
                while True:
                    time.sleep(0.1)
        elif(args.plot==1):
            ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)
            with stream:
                plt.show()

    except Exception as e:
        print("ERROR:", e)

sound_loop()
