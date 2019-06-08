import queue
import sys

import sounddevice as sd

q = queue.Queue()


def cb(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)


stream = sd.InputStream(
    device=None, channels=1,
    samplerate=16000, callback=cb)

stream.start()
while stream:
    pass
