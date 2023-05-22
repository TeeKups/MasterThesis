import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import time
import numpy
import queue

import sys
if sys.version_info.major < 3 or sys.version_info.minor < 10:
    exit(f"You're running Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. At least 3.10 is required!")

import logging

logger = logging.getLogger('__main__')

def record(out, signals):
    """Connects to the microphone, reads data, and writes to `out` -queue."""
    def capture(indata, frames, time, status):
        """Callback - writes received data into `out` -queue."""
        # Must import time.perf_counter again as callbacks seem to have their own namespace
        from time import perf_counter
        t = perf_counter()
        out.put({'data': indata.copy(), 'timestamp': t})

    device = sd.query_devices(device='micArray16')
    sd.default.device = device['name']

    fs=44100 # TODO: Read dynamically

    with sd.InputStream(samplerate=fs,\
    device=sd.default.device, channels=16, \
    dtype='float32',  callback=capture) \
    as instream:
        metadata = { 'samplerate': fs, 'num_channels': 16, 'start_time': time.perf_counter() }
        logger.debug('Mic started')
        qsize = signals.qsize()
        signals.put('STARTED')
        while signals.qsize() > qsize:
            pass
        out.put(metadata)
        while signals.empty() or signals.get() != 'STOP':
            pass

    logger.debug('Audio recording stopped')


if __name__ == '__main__':
    """Used for testing."""
    record = queue.Queue()

    def capture(indata, frames, time, status):
        record.put(indata.copy())

    device = sd.query_devices(device='micArray16')
    print(device)
    sd.default.device = device['name']

    fs=44100
    duration=5

    print('recording')
    with sf.SoundFile('output.wav', mode='w', samplerate=fs, channels=16) as file:
        with sd.InputStream(samplerate=fs, device=sd.default.device, channels=16, dtype='float32',  callback=capture) as instream:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 3:
                file.write(record.get(timeout=1))


    print('done')
