import pyrealsense2 as realsense
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy
import time
import struct

import sys
if sys.version_info.major < 3 or sys.version_info.minor < 10:
    exit(f"You're running Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. At least 3.10 is required!")

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def _start_stream(realsense_config):
    """Create a context object. This object owns the handles to all connected realsense devices."""
    pipe = realsense.pipeline()
    config = realsense.config()
    
    config.enable_stream(realsense.stream.depth, 
            width=realsense_config['resolution'][0], 
            height=realsense_config['resolution'][1], 
            format=realsense.format.z16, 
            framerate=realsense_config['fps']) 

    config.enable_stream(realsense.stream.color,
            width=realsense_config['resolution'][0], 
            height=realsense_config['resolution'][1], 
            format=realsense.format.rgb8, 
            framerate=realsense_config['fps'])
    profile = pipe.start(config)
    return pipe, profile


def _get_frame(pipe, BYTES=True):
    """Gets new RGB and depth frames, and a new timestamp."""
    dataFrame = pipe.wait_for_frames()

    # Convert time reference point from epoch to same as time.perf_counter() and representation from milliseconds to seconds
    timestamp = dataFrame.get_timestamp()*1e-3 + time.perf_counter_ns()*1e-9 - time.time_ns()*1e-9

    rgb_image = numpy.array(dataFrame.get_color_frame().get_data(), dtype='<B')
    depth_image = numpy.array(dataFrame.get_depth_frame().get_data(), dtype='<H')

    #print(f'{timestamp}, {time.perf_counter()}')

    if BYTES:
        return {'RGB': rgb_image.tobytes('C'), 'D': depth_image.tobytes('C'), 'timestamp': timestamp }
    else:
        return {'RGB': rgb_image, 'D': depth_image, 'timestamp': round(dataFrame.get_timestamp()) }


def record(realsense_config, out, signals):
    """Starts the sensor, reads data, and writes into `out` -queue"""
    pipe, profile = _start_stream(realsense_config)
    logger.debug("Camera started!")
    qsize = signals.qsize()
    signals.put('STARTED')
    while signals.qsize() > qsize:
        pass

    first_frame = _get_frame(pipe)
    rgb_res = f"{realsense_config['resolution'][0]}x{realsense_config['resolution'][1]}"
    d_res = f"{realsense_config['resolution'][0]}x{realsense_config['resolution'][1]}"

    metadata = {
            'RGB_resolution': rgb_res,
            'D_resolution': d_res,
            'framerate': realsense_config['fps'] ,
            'start_time': first_frame['timestamp'] }
    out.put(metadata)
    out.put(first_frame)

    while signals.empty() or signals.get() != 'STOP':
        out.put(_get_frame(pipe))


if __name__ == '__main__':
    """Used for testing"""
    def _update_frame(self, pipe, rgb, depth):
        start = time.perf_counter()
        frame = _get_frame(pipe, BYTES=True)
        #print(f'FPS: {1/(time.perf_counter() - start)}')
        rgb.set_data(numpy.frombuffer(frame['RGB'], dtype='<B', count=640*480*3).reshape((480,640,3)))
        depth.set_data(numpy.frombuffer(frame['D'], dtype='<H', count=640*480).reshape((480,640)))
        return rgb, depth

    realsense_config = {
                'resolution' : [640, 480],
                'fps' : 30
            }

    pipe, profile = _start_stream(realsense_config)

    # Make a window to show camera stream in
    fig = plt.figure()
    rgb_ax = fig.add_subplot(121)
    depth_ax = fig.add_subplot(122)

    # Start streaming data from depth camera
    frame = _get_frame(pipe, BYTES=False)

    rgb_im = rgb_ax.imshow(frame['RGB'])
    depth_im = depth_ax.imshow(frame['D'], vmin=0, vmax=10000) # 10 m in millimeters

    cam_animation = FuncAnimation(fig, _update_frame, fargs=(pipe, rgb_im, depth_im), interval= 1e-3/realsense_config['fps'], blit=True)

    plt.show()

