import time as pytime
import serial
import serial.tools.list_ports
import numpy
import struct
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
if sys.version_info.major < 3 or sys.version_info.minor < 10:
    exit(f"You're running Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. At least 3.10 is required!")

import logging

logger = logging.getLogger('__main__')

def record(out, signals):
    """Starts the sensor, reads data from it, and writes to `out` -queue"""
    connection = _connect_sensor()
    connection.flush()

    metadata = { 'samplerate(Hz)': 10, 'start_time': pytime.perf_counter() }

    logger.debug("IR started!")
    qsize = signals.qsize()
    signals.put('STARTED')

    while signals.qsize() > qsize:
        pass

    out.put(metadata)
    while signals.empty() or signals.get() != 'STOP':
        frame = _get_frame(connection)
        out.put(frame)


def _connect_sensor():
    """Connects to the correct serial port"""
    ports = serial.tools.list_ports.comports()
    port = next(p.name for p in ports if 'Communication Device Class ASF example' in p.description)

    if port == None:
        logger.error("Panasonic GridEye IR sensor is not connected.")
        exit()

    connection = serial.Serial(port, baudrate=9600)
    return connection


def _get_frame(connection):
    """Reads a frame from the sensor and converts the values to floats. (GridEye communication protocol.pdf)"""
    timestamp = pytime.perf_counter()
    frame = connection.read(135)
    # 3B header, 130B data, 2B tail = 135 B

    data = frame[5:5+128]
    ir_data = struct.unpack('<64h', data)

    # Choose low bytes (T01L, T02L, ... T64L) of Temperature Register for plotting
    ir_data = numpy.array(ir_data)
    ir_data = ir_data/4 # Temperature is given as multiple of 4 -> 0.25 Celsius resolution
    ir_data = numpy.reshape(ir_data, (8, 8))
    ir_data_raw = numpy.reshape(ir_data, (64,))
    ir_data_raw = struct.pack('<64e', *ir_data_raw) # Nicely enough, integer multiples 0.25 are accurate on 16-bit float
    return {'data': ir_data, 'raw_data': ir_data_raw, 'timestamp': timestamp}


if __name__ == '__main__':
    """Used for testing"""
    
    def _update_frame(dummy, img, connection):
        frame = _get_frame(connection)
        img.set_data(frame['data'])
        
    connection = _connect_sensor()
    connection.flush()

    start = pytime.perf_counter()
    num_frames = 1
    N = 10000
    while num_frames <= N:
        frame = _get_frame(connection)
        num_frames += 1
    print(f'FPS: {N/(pytime.perf_counter()-start)}')
    exit()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(frame['data'], vmin=20, vmax=30)

    animation = FuncAnimation(fig, _update_frame, fargs=(img, connection), interval=100)

    plt.show()
