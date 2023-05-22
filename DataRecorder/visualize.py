#import mplcursors
#from capturelib.blur_faces import blur_faces
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy import pi as PI
from scipy import constants
from scipy import interpolate
from scipy import signal
import argparse
import argparse
import csv
import functools
import itertools
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import multiprocessing as mp
import pathlib
import pprint
import queue
import re
import scipy.io as io
import sounddevice
import soundfile
import struct
import threading
import time
import tkinter as tk
import yaml
import concurrent.futures as concurrent

try:
    import cupy as numpy
except ImportError:
    import numpy


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Numpy config
numpy.set_printoptions(edgeitems=3, linewidth=160)

# Get data directory path as argument
parser = argparse.ArgumentParser(description='Multi-Modal Dataset Visualizer')
parser.add_argument('PATH', help='Path to recorded dataset directory')
parser.add_argument('--fov', help='FOV for angular MUSIC, defaults to 60 degrees.')
parser.add_argument('--n-angles', help='Specifies number of angle bins for MUSIC, defaults to 20.')
parser.add_argument('-c', action='store_true', default=False, help='Use processed radar file from cache.')
args = parser.parse_args()

if not pathlib.Path(args.PATH).is_dir():
    raise FileNotFoundError(args.PATH)
    exit("Check the path and try again")

# Some variables
RADAR_FILE = pathlib.Path(args.PATH, 'radar.raw')
DEPTH_FILE = pathlib.Path(args.PATH, 'depth.raw')
RGB_FILE = pathlib.Path(args.PATH, 'rgb.raw')
IR_FILE = pathlib.Path(args.PATH, 'ir.raw')
SOUND_FILE = pathlib.Path(args.PATH, 'audio.wav')
TIMESTAMPS_FILE = pathlib.Path(args.PATH, 'timestamps.csv')
METADATA_FILE = pathlib.Path(args.PATH, 'metadata.yaml')

with open(TIMESTAMPS_FILE, 'rt') as TIMESTAMPS_H, \
     open(METADATA_FILE, 'rt') as METADATA_H:

    TIMESTAMPS_HANDLE = TIMESTAMPS_H.read()
    METADATA = yaml.safe_load(METADATA_H)
    
def get_radar_frame(file, n):
    """Gets Nth radar frame, 0-indexing. Returns None when end is reached."""
    try:
        FILTER = True
        def raw_to_complex(raw):
            return [(complex(raw[idx], raw[idx+2])) for tuple in [(n, n+1) for n in range(0, len(raw), 4)] for idx in tuple]
        
        radar = METADATA['radar']
        iq_samples_per_frame = 2 * radar['num_channels'] * radar['samples_per_chirp'] * radar['chirps_per_frame']
        raw_data = numpy.fromfile(file, dtype='<h', count=iq_samples_per_frame, offset=n*2*iq_samples_per_frame)
        if len(raw_data) < iq_samples_per_frame:
            raise ValueError

        samples = raw_to_complex(raw_data)

        data_ptr = 0
        channel_ptr = 0

        radar_cube = [ [] for n in range(radar['num_channels']) ]
        while data_ptr < len(samples):
            chirp = samples[data_ptr:data_ptr+radar['samples_per_chirp']]
            #chirp.reverse()
            radar_cube[channel_ptr].append(chirp)
            channel_ptr = 0 if (channel_ptr == 3) else (channel_ptr + 1)
            data_ptr += radar['samples_per_chirp']

        if FILTER:
            Fs = radar['samplerate']
            #sos = signal.butter(10, [int(Fs/10), int((Fs/2)-(Fs/10))], 'bandpass', fs=Fs, output='sos')
            sos = signal.butter(10, [int(Fs/25)], 'highpass', fs=Fs, output='sos')
            for ch in range(radar['num_channels']):
                for chirp in range(radar['chirps_per_frame']):
                    radar_ch = radar_cube[ch]
                    chrp = radar_ch[chirp]
                    radar_cube[ch][chirp] = signal.sosfilt(sos, chrp)
                    #radar_cube[ch][chirp] = signal.sosfilt(sos, radar_cube[ch][chirp])

        radar_cube = numpy.array(radar_cube)

        return numpy.flip(radar_cube, 2)
    except ValueError as e:
        return None


def get_ir_frame(file, n):
    """Gets Nth IR frame, 0-indexing. Returns None when end is reached."""
    try:
        samples = numpy.fromfile(file, dtype='<e', count=64, offset=2*64*n)
        if len(samples) < 64:
            raise ValueError
        return numpy.fliplr(numpy.reshape(samples, (8, 8), order='C'))
    except ValueError as e:
        return None

def get_rgb_frame(file, n):
    """Gets Nth RGB frame, 0-indexing. Returns None when end is reached."""
    try:
        raw_data = numpy.fromfile(file, dtype='<B', count=640*480*3, offset=640*480*3*n)
        if len(raw_data) < 640*480*3:
            raise ValueError
        return numpy.reshape(raw_data, (480, 640, 3), order='C')
    except ValueError as e:
        return None

def get_depth_frame(file, n):
    try:
        """Gets Nth Depth frame, 0-indexing. Returns None when end is reached."""
        raw_data = numpy.fromfile(file, dtype='<H', count=640*480, offset=640*480*2*n)
        if len(raw_data) < 640*480:
            raise ValueError
        return numpy.reshape(raw_data, (480, 640), order='C')
    except ValueError as e:
        return None

def MUSIC(frame, seq, FOV):
    radar = METADATA['radar']

    P = []
    wavelength = 1/200
    d = wavelength / 2

    # TODO: estimate number of targets
    L = 12

    #N = radar['num_channels']
    K = radar['chirps_per_frame']
    #M = radar['samples_per_chirp']

    n = frame.shape[0]
    m = frame.shape[1]
    k = frame.shape[2]

    r_max = (radar['samplerate']*constants.c) / (2*radar['slope'])
    n_thetas = 20
    n_ranges = radar['samples_per_chirp']

    Y = numpy.reshape(numpy.reshape(frame, (n, m*k), order='F'), (n*k, m), order='C').T
    Cx = numpy.mean( [numpy.outer(Y[k,:].T, Y[k,:].conj()) for k in range(K)], axis=0)

    eigvals, eigvecs = numpy.linalg.eigh(Cx)
    sort_order = numpy.argsort(eigvals) # argsort -> smallest 1st
    eigvals = eigvals[sort_order]
    eigvecs = eigvecs[:,sort_order]

    Q = eigvecs[:, :len(eigvals)-L]
    
    steering_vector = lambda theta : numpy.exp(1j*((2*PI)/(1/200))*(1/400)*numpy.arange(0, radar['num_channels'])*numpy.sin(theta[:, numpy.newaxis]))
    frequency_manifold = lambda R : numpy.exp(1j*2*PI*((2*R[:, numpy.newaxis]*radar['slope'])/constants.c)*numpy.arange(radar['samples_per_chirp'])*(1/radar['samplerate']))

    a = steering_vector(numpy.linspace(-FOV/2, FOV/2, n_thetas, endpoint=True))
    s = frequency_manifold(numpy.linspace(0, r_max, n_ranges))

    steering_matrix = lambda theta, R : numpy.outer(a[theta, :], s[R, :]).flatten('C').T

    QQ = Q @ Q.conj().T

    def pseudo_power_spectrum(args):
        W = steering_matrix(args[0], args[1])
        return 1/(W.conj().T @ QQ @ W)

    P = numpy.array([*map(pseudo_power_spectrum, itertools.product(numpy.arange(n_thetas), numpy.arange(n_ranges)))]).reshape(n_thetas, n_ranges)

    return (seq, abs(P))

def FFT(frame, seq):
    #frame = frame * numpy.hamming(frame.shape[1]) # Row-wise multiplication https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
    frame = numpy.fft.fft2(frame[0,:,:])
    frame = numpy.absolute(frame)
    #frame = 10*numpy.log10(frame)
    return (seq, frame[ [*range(math.ceil(frame.shape[0]/2), frame.shape[0]), *range(math.ceil(frame.shape[0]/2))], : ])


def process_radar(FOV=PI/3, fft=True, music=True):
    music_frame_queue = queue.Queue()
    fft_frame_queue = queue.Queue()

    music_promises=[]
    fft_promises=[]
    with mp.Pool() as pool:
        n = 0
        while True:
            data_frame = get_radar_frame(RADAR_FILE, n)
            if data_frame is None:
                break

            if music:
                music_promises.append(pool.apply_async(MUSIC, (data_frame, n, FOV), callback=music_frame_queue.put, error_callback=None))
            if fft:
                fft_promises.append(pool.apply_async(FFT, (data_frame, n), callback=fft_frame_queue.put, error_callback=None))

            n += 1

        for item in music_promises:
            item.get(timeout=15)

        for item in fft_promises:
            item.get(timeout=15)


    music_frames = []
    fft_frames = []
    music_processed = False
    fft_processed = False
    while not music_processed or not fft_processed:
        if music:
            if not music_processed and not music_frame_queue.empty():
                music_frames.append(music_frame_queue.get(timeout=5))
            else:
                music_processed = True

        if fft:
            if not fft_processed and not fft_frame_queue.empty():
                fft_frames.append(fft_frame_queue.get(timeout=5))
            else:
                fft_processed = True

    music_frames = [tuple[1] for tuple in sorted(music_frames)] if len(music_frames) > 0 else None
    fft_frames = [tuple[1] for tuple in sorted(fft_frames)] if len(fft_frames) > 0 else None

    return fft_frames, music_frames

def gui():
    global music_frames, fft_frames

    def upscale(frame, factor):
        y = numpy.arange(frame.shape[0])
        x = numpy.arange(frame.shape[1])
        X, Y = numpy.meshgrid(x, y)
        yy = numpy.linspace(0, frame.shape[0], frame.shape[0]*factor)
        xx = numpy.linspace(0, frame.shape[1], frame.shape[1]*factor)
        f = interpolate.interp2d(x, y, frame, kind='linear')
        return f(xx, yy) 

    data, fs = soundfile.read(SOUND_FILE, dtype='float32')
    sound = data[:,0:1]

    # TODO: Get FPS information from metadata or something
    FPS = 30
    ft = 1/FPS

    with open(TIMESTAMPS_FILE, 'r') as ts_f:
        reader = csv.reader(ts_f)
        timestamps = [row for row in reader]

    def init_figure(fig):
        global frame_counter
        frame_counter = 0

        # 2D-MUSIc
        FOV = PI/3

        #radar_data_frame = get_radar_frame(frame_counter)
        fft_frame = fft_frames[frame_counter]
        music_frame = music_frames[frame_counter]

        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.75)

        gs = GridSpec(5, 4, figure=fig)
        ax = fig.add_subplot(gs[2:4, 0])
        ax.set_title('2D-MUSIC')
        r_max = (METADATA['radar']['samplerate']*constants.c) / (2*METADATA['radar']['slope'])
        X_dim = music_frame.shape[1]
        Y_dim = music_frame.shape[0]

        theta_ticks = numpy.linspace(0, Y_dim, 7, endpoint=True)
        theta_labels = [f'{n:.1f}' for n in numpy.linspace(math.degrees(-FOV/2), math.degrees(FOV/2), len(theta_ticks), endpoint=True)]

        ranges = numpy.linspace(0, r_max, X_dim)
        thetas = numpy.linspace(-FOV/2, FOV/2, Y_dim, endpoint=True)

        range_ticks = numpy.linspace(0, X_dim, 12, endpoint=True)
        range_labels = [f'{n:.1f}' for n in numpy.linspace(0, r_max, len(range_ticks), endpoint=True)]

        vmax = (1/200)/(4*METADATA['radar']['chirp_cycle_time'])
        dop_ticks = numpy.linspace(0, fft_frame.shape[0], 7, endpoint=True)
        dop_labels = [f'{n:.2f}' for n in numpy.linspace(-vmax, vmax, len(dop_ticks), endpoint=True)]

        ax.set_xticks(range_ticks)
        ax.set_xticklabels(range_labels)

        ax.set_yticks(theta_ticks)
        ax.set_yticklabels(theta_labels)

        ax.set_ylabel('Angle [deg]')
        ax.set_xlabel('Range [m]')

        music_img = ax.pcolormesh(music_frame)
        music_img.set_clim((numpy.amin(music_frame), numpy.amax(music_frame)))

        # Range projection
        ax = fig.add_subplot(gs[0:2, 0])
        #ax = fig.add_subplot(341)
        range_projection = numpy.mean(music_frame, 0)
        ax.set_xticks(range_ticks)
        ax.set_xticklabels(range_labels)
        ax.set_xlabel('Range [m]')
        range_projection = numpy.log10(range_projection)
        range_plot = ax.plot(range_projection)
        ax.set_yticks([])
        ax.set_title('Range-MUSIC')

        # Angle projection
        ax = fig.add_subplot(gs[2:4, 1])
        #ax = fig.add_subplot(346)
        angle_projection = numpy.max(music_frame, 1)
        angle_projection = numpy.log10(angle_projection)
        angle_plot = ax.plot(angle_projection, range(len(angle_projection)))
        ax.set_xlim((min(angle_projection), max(angle_projection)))
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels(theta_labels)
        ax.set_xticks([])
        ax.set_title('Angle-MUSIC')

        # Range-Doppler
        #fft_frame = FFT(radar_data_frame)
        ax = fig.add_subplot(gs[0:2, 1])
        #ax = fig.add_subplot(342)
        ax.set_xticks(range_ticks)
        ax.set_xticklabels(range_labels)
        ax.set_yticks(dop_ticks)
        ax.set_yticklabels(dop_labels)
        ax.set_xlabel('Range [m]')
        dop_img = ax.pcolormesh(fft_frame)
        ax.set_title('Doppler-Range')
        ax.set_ylabel('Velocity [m/s]')

        # RGB
        rgb_frame = get_rgb_frame(RGB_FILE, frame_counter)
        ax = fig.add_subplot(gs[0:2, 2])
        #ax = fig.add_subplot(343)
        rgb_img = ax.imshow(rgb_frame)
        ax.axis('off')
        ax.set_title('RGB')

        # Depth
        depth_frame = get_depth_frame(DEPTH_FILE, frame_counter)
        ax = fig.add_subplot(gs[2:4, 2])
        #ax = fig.add_subplot(347)
        depth_img = ax.imshow(depth_frame, vmin=0, vmax=6000)
        ax.axis('off')
        ax.set_title('Depth')

        # IR
        ir_frame = get_ir_frame(IR_FILE, frame_counter)
        ax = fig.add_subplot(gs[0:2, 3])
        #ax = fig.add_subplot(344)
        ir_img = ax.imshow(ir_frame)
        ir_img.set_clim((numpy.amin(ir_frame), numpy.amax(ir_frame)))
        ax.axis('off')
        ax.set_title('Infrared')

        # IR upscaled
        ax = fig.add_subplot(gs[2:4, 3])
        #ax = fig.add_subplot(348)
        upscaled_ir_frame = upscale(ir_frame, 4)
        upscaled_ir_img = ax.imshow(upscaled_ir_frame)
        upscaled_ir_img.set_clim((numpy.amin(upscaled_ir_frame), numpy.amax(upscaled_ir_frame)))
        ax.axis('off')
        ax.set_title('Infrared upscaled 4x')

        # Audio
        ax = fig.add_subplot(gs[4, :])
        #ax = fig.add_subplot(3,4,(9,12))
        fs = 44100
        n_samples = int(ft*fs)
        soundframe = sound[0:n_samples,0]
        sound_fft = abs(numpy.fft.fft(soundframe))
        audio_img = ax.plot(sound_fft[:int(len(sound_fft)/2)])
        ax.axis('off')
        ax.set_title('Audio FFT')

        frame_counter += 1
        return music_img, dop_img, range_plot[0], angle_plot[0], rgb_img, depth_img, ir_img, upscaled_ir_img, audio_img[0]

    # Animations
    root = tk.Tk()
    fig = plt.figure(figsize=(19.8, 10.8))
    #fig.subplots_adjust(wspace=0.25)
    music_img, dop_img, range_plot, angle_plot, rgb_img, depth_img, ir_img, upscaled_ir_img, audio_img = init_figure(fig)

    rgb_bytesize = RGB_FILE.stat().st_size
    ir_bytesize = IR_FILE.stat().st_size

    num_radar_frames = len(music_frames)
    num_camera_frames = round(rgb_bytesize/(640*480*3))
    num_ir_frames = round(ir_bytesize/(64*2))
    num_sound_frames = round(sound.shape[0]/(ft*44100))

    duration = METADATA['duration']
    radar_fps = METADATA['radar']['framerate']
    camera_fps = METADATA['camera']['rgb']['framerate']
    ir_fps = METADATA['ir']['framerate']


    canvas = FigureCanvasTkAgg(fig, master=root)

    canvas.draw()
    #sounddevice.play(sound, fs)

    activity_indicator = tk.Label(root, text='Sample Text')

    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    canvas.mpl_connect("key_press_event", lambda event: logger.debug(f"Key press: {event.key}"))
    canvas.mpl_connect("key_press_event", key_press_handler)
    root.bind("q", lambda event : root.quit())

    activity_indicator.pack(side=tk.TOP, fill=tk.BOTH)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


    def advance(dummy, clock_ref):
        global fft_frames, music_frames, frame_counter
        
        while (time_elapsed := ft*frame_counter) < time.perf_counter()-clock_ref:
            frame_counter += 1
        
        radar_frame_number = math.floor((time_elapsed * radar_fps) % num_radar_frames)
        camera_frame_number = math.floor((time_elapsed * camera_fps) % num_camera_frames)
        ir_frame_number = math.floor((time_elapsed * ir_fps) % num_ir_frames)
        sound_frame_number = math.floor((time_elapsed * (1/ft)) % num_sound_frames) - 1
        if sound_frame_number < 0:
            sound_frame_number = 0

        # TODO: improve efficiency
        video_time = time_elapsed % duration
        for row in timestamps:
            timestamp = float(row[0])
            if video_time < timestamp:
                continue
            activity = row[1]

        activity_indicator.config(text=f'Activity: {activity}')
        
        fft_frame = fft_frames[radar_frame_number]
        music_frame = music_frames[radar_frame_number]
        rgb_frame = get_rgb_frame(RGB_FILE, camera_frame_number)
        depth_frame = get_depth_frame(DEPTH_FILE, camera_frame_number)
        ir_frame = get_ir_frame(IR_FILE, ir_frame_number)

        music_img.set_array(music_frame) if music_frame is not None else logger.debug('music_frame is None')
        music_img.set_clim((numpy.amin(music_frame), numpy.amax(music_frame)))

        range_projection = numpy.log10(numpy.mean(music_frame, 0))
        range_plot.set_ydata(range_projection) if music_frame is not None else logger.debug('music_frame is None')
        range_plot.axes.set_ylim((min(range_projection), max(range_projection)+0.01))

        angle_projection = numpy.max(music_frame, 1)
        angle_plot.set_xdata(angle_projection) if music_frame is not None else logger.debug('music_frame is None')
        angle_plot.axes.set_xlim((min(angle_projection), max(angle_projection)+0.0001))

        dop_img.set_array(fft_frame) if fft_frame is not None else logger.debug('fft_frame is None')
        rgb_img.set_array(rgb_frame) if rgb_frame is not None else logger.debug('rgb_frame is None')
        depth_img.set_array(depth_frame) if depth_frame is not None else logger.debug('depth_frame is None')

        fs = 44100
        n_samples = int(ft*fs)
        soundframe = sound[sound_frame_number*n_samples:(sound_frame_number+1)*n_samples,0]
        sound_fft = abs(numpy.fft.fft(soundframe))
        audio_img.set_ydata(sound_fft[:int(len(sound_fft)/2)])
        
        if ir_frame is not None:
            ir_img.set_array(ir_frame) 
            ir_img.set_clim((numpy.amin(ir_frame), numpy.amax(ir_frame)))

            upscaled_ir_frame = upscale(ir_frame, 4)
            upscaled_ir_img.set_array(upscaled_ir_frame)
            #upscaled_ir_img.set_clim((numpy.amin(upscaled_ir_frame), numpy.amax(upscaled_ir_frame)))
        else:
            logger.debug('ir_frame is None')


        frame_counter += 1

        return music_img, dop_img, range_plot, angle_plot, rgb_img, depth_img, ir_img, upscaled_ir_img, audio_img
        
    clock_ref = time.perf_counter()
    advance_animation = FuncAnimation(fig, advance, frames=None, fargs=(clock_ref,), interval=ft*1e3, blit=True)

    root.bind("p", lambda event : advance_animation.event_source.stop() )
    root.bind("o", lambda event : advance_animation.event_source.start() )

    root.protocol("WM_DELETE_WINDOW", root.quit)

    root.mainloop()

"""
def blur(image, n):
    image_data = blur_faces(image).tobytes('C')
    return (n, image_data)
"""

if __name__ == '__main__':

    global fft_frames, music_frames
    if not args.c:
        fft_frames, music_frames = process_radar(FOV=PI/3, fft=True, music=True)
        io.savemat('radar.mat', {'fft': fft_frames, 'music': music_frames}, do_compression=True)
        fft_frames = None
        music_frames = None

    BLUR_FACES = True

    """
    if BLUR_FACES:
        newdata = True
        if newdata:
            logger.debug('Blurring faces')
            blur_start = time.perf_counter()
            n = 0
            rgb_blurred_buf = b''
            with concurrent.ThreadPoolExecutor() as pool:
                futures = []
                while (rgb_image := get_rgb_frame(RGB_FILE, n)) is not None:
                    futures.append(pool.submit(blur_faces, rgb_image))
                    n += 1

                for future in futures:
                    rgb_blurred_buf += future.result().tobytes('C')

            logger.debug(f'Blurring all images took {time.perf_counter() - blur_start:.2f} s')

            with open('blurred.raw', 'wb') as f:
                f.write(rgb_blurred_buf)

        RGB_FILE = pathlib.Path('blurred.raw')
    """

    radar_dict = io.loadmat('radar.mat')
    fft_frames = radar_dict['fft']
    music_frames = radar_dict['music']


    #blur_faces()
    gui()


