"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys
import time
import azure.cognitiveservices.speech as speechsdk
import _thread

import requests

import helpers
import soundfile as sf

can_sst_start = True
can_validate = False
speech_key, service_region = "1e9f1691dcbd43758eadb5f7c2ddbd3f", "centralindia"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

SERVER_URL = 'http://localhost:4500/'


def start_stt():
    global can_sst_start, can_validate
    _thread.start_new_thread(record, (4,))
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something...")
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


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
args = parser.parse_args()
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()

val = []
i = 0


def audio_callback(indata, frames, time, status):
    global val, i, can_sst_start
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata.copy())
    value = (np.average(np.absolute(indata[::args.downsample, mapping])))
    if len(val) > 50:
        val.pop(0)
    val.append(value)
    i += 1
    if i == 50:
        if sum(val) >= 4.5:
            print('Human has spoken ', sum(val))
            if can_sst_start:
                can_sst_start = False
                _thread.start_new_thread(start_stt, ())

        i = 0


def record(sec):
    global can_validate
    can_validate = False
    with sf.SoundFile('default.wav', mode='w', samplerate=16000,
                      channels=1) as file:
        t_end = time.time() + sec
        while time.time() < t_end:
            file.write(q.get())
    can_validate = True


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

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


try:
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    import numpy as np
    import sounddevice as sd
    import matplotlib

    matplotlib.use('TkAgg')
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = 16000

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom='off', top='off', labelbottom='off',
                   right='off', left='off', labelleft='off')
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=1,
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)

    while True:
        time.sleep(0.1)
    # with stream:
    #     update_plot(None)
    # plt.show()

except Exception as e:
    print("ERROR:", e)
    parser.exit(type(e).__name__ + ': ' + str(e))

#  0 HDA Intel PCH: ALC1220 Analog (hw:0,0), ALSA (2 in, 6 out)
#    1 HDA Intel PCH: ALC1220 Digital (hw:0,1), ALSA (0 in, 2 out)
#    2 HDA Intel PCH: ALC1220 Alt Analog (hw:0,2), ALSA (2 in, 0 out)
#    3 HDA NVidia: HDMI 0 (hw:1,3), ALSA (0 in, 8 out)
#    4 HDA NVidia: HDMI 2 (hw:1,8), ALSA (0 in, 8 out)
#    5 HDA NVidia: HDMI 3 (hw:1,9), ALSA (0 in, 8 out)
#    6 HD Pro Webcam C920: USB Audio (hw:2,0), ALSA (2 in, 0 out)
#    7 HD Pro Webcam C920: USB Audio (hw:3,0), ALSA (2 in, 0 out)
#    8 sysdefault, ALSA (128 in, 128 out)
#    9 front, ALSA (0 in, 6 out)
#   10 surround21, ALSA (0 in, 128 out)
#   11 surround40, ALSA (0 in, 6 out)
#   12 surround41, ALSA (0 in, 128 out)
#   13 surround50, ALSA (0 in, 128 out)
#   14 surround51, ALSA (0 in, 6 out)
#   15 surround71, ALSA (0 in, 6 out)
#   16 iec958, ALSA (0 in, 2 out)
#   17 spdif, ALSA (0 in, 2 out)
#   18 pulse, ALSA (32 in, 32 out)
#   19 dmix, ALSA (0 in, 2 out)
# * 20 default, ALSA (32 in, 32 out)
