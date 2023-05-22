from numpy import pi as PI
from scipy import constants
from scipy import signal
from scipy import interpolate
import scipy.io as io
import argparse
import itertools
import math
import multiprocessing as mp
import concurrent.futures
import pathlib
import queue
import time
import yaml
import logging
import os
import sys
import numpy
from functools import partial
import csv

from capturelib.blur_faces import blur_faces


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Numpy config
numpy.set_printoptions(edgeitems=3, linewidth=160)

    
def get_radar_frame(buffer, metadata, n):
    """Gets Nth radar frame, 0-indexing. Returns None when end is reached."""
    try:
        FILTER = False
        def raw_to_complex(raw):
            return [(complex(raw[idx], raw[idx+2])) for tuple in [(n, n+1) for n in range(0, len(raw), 4)] for idx in tuple]
        
        radar = metadata
        iq_samples_per_frame = 2 * radar['num_channels'] * radar['samples_per_chirp'] * radar['chirps_per_frame']
        #samples = raw_to_complex( struct.unpack_from(f'<{iq_samples_per_frame}h', RADAR_DATA, offset=n*iq_samples_per_frame) )
        samples = raw_to_complex( numpy.frombuffer(buffer, dtype='<h', count=iq_samples_per_frame, offset=n*2*iq_samples_per_frame) )

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
                    radar_cube[ch][chirp] = signal.sosfilt(sos, radar_cube[ch][chirp])

        radar_cube = numpy.array(radar_cube)

        return numpy.flip(radar_cube, 2)
    except ValueError as e:
        return None


def get_ir_frame(buffer, n):
    """Gets Nth IR frame, 0-indexing. Returns None when end is reached."""
    try:
        samples = numpy.frombuffer(buffer, dtype='<e', count=64, offset=2*64*n)
        return numpy.fliplr(numpy.reshape(samples, (8, 8), order='C'))
    except ValueError as e:
        return None

def get_rgb_frame(buffer, resolution, n):
    """Gets Nth RGB frame, 0-indexing. Returns None when end is reached."""
    try:
        return numpy.reshape(numpy.frombuffer(buffer, dtype='<B', count=resolution[1]*resolution[0]*3, offset=resolution[1]*resolution[0]*3*n), (resolution[1], resolution[0], 3), order='C')
    except ValueError as e:
        return None

def get_depth_frame(buffer, resolution, n):
    """Gets Nth Depth frame, 0-indexing. Returns None when end is reached."""
    try:
        return numpy.reshape(numpy.frombuffer(buffer, dtype='<H', count=resolution[1]*resolution[0], offset=resolution[1]*resolution[0]*2*n), (resolution[1], resolution[0]), order='C')
    except ValueError as e:
        return None

def MUSIC(frame, seq, metadata, FOV, n_thetas):
    #radar = METADATA['radar']
    radar = metadata

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
    
def process_depth(outdir, data_file, metadata, timestamps_path, raw: bool, discard_original: bool):
    """
        If `raw`is true, creates a 3D array of depth frames, else stores raw data.
    """
    with open(str(data_file), 'rb') as depth_h:
        DEPTH_DATA = depth_h.read()

    if raw:
        with open(str(outdir.joinpath('depth.raw')), 'wb') as outfile:
            outfile.write(DEPTH_DATA)
    else:
        frames = []
        activities = []
        ft = 1/float(metadata['framerate'])
        with open(timestamps_path, 'rt') as timestamps_h:
            EOF = False
            reader = csv.reader(timestamps_h, delimiter=',')

            row = next(reader)
            time = float(row[0])
            activity = row[1]
            next_row = next(reader)
            next_time = float(next_row[0])
            next_activity = next_row[1]

            n = 0
            while (frame := get_depth_frame(DEPTH_DATA, metadata['resolution'], n)) is not None:
                frames.append(frame)
                activities.append(activity)
                n += 1
                time += ft
                if time >= next_time and not EOF:
                    time = next_time
                    activity = next_activity
                    row = next(reader)
                    next_time = float(row[0])
                    next_activity = row[1]
                    if next_activity.upper() == 'STOP':
                        next_activity = activity
                        EOF = True

        matfile = {'frames': frames, 'labels': activities, 'resolution': metadata['resolution'], 'framerate': metadata['framerate']}
        io.savemat(str(outdir.joinpath('depth.mat')), matfile, do_compression=True)

    if discard_original:
        os.remove(str(data_file)) 

def process_rgb(outdir, data_file, metadata, timestamps_path, blur: bool, raw: bool, discard_original: bool):
    """
        If `raw`is true, creates a 3D array of rgb frames, else stores raw data.
        If `blur`, calls capturelib.blur_faces for each frame and stores the blurred video.
    """
    with open(data_file, 'rb') as rgb_h:
        RGB_DATA = rgb_h.read()

    if blur:
        n = 0
        rgb_buf = b''
        while (rgb_image := get_rgb_frame(RGB_DATA, metadata['resolution'], n)) is not None:
            rgb_buf += blur_faces(rgb_image).tobytes('C')
            n += 1
    else:
        rgb_buf = RGB_DATA

    if raw:
        with open(str(outdir.joinpath('rgb.raw')), 'wb') as outfile:
            outfile.write(rgb_buf)
    else:
        frames = []
        activities = []
        ft = 1/float(metadata['framerate'])
        with open(timestamps_path, 'rt') as timestamps_h:
            EOF = False
            reader = csv.reader(timestamps_h, delimiter=',')

            row = next(reader)
            time = float(row[0])
            activity = row[1]
            next_row = next(reader)
            next_time = float(next_row[0])
            next_activity = next_row[1]

            n = 0
            while (frame := get_rgb_frame(rgb_buf, metadata['resolution'], n)) is not None:
                frames.append(frame)
                activities.append(activity)
                n += 1
                time += ft
                if time >= next_time and not EOF:
                    time = next_time
                    activity = next_activity
                    row = next(reader)
                    next_time = float(row[0])
                    next_activity = row[1]
                    if next_activity.upper() == 'STOP':
                        next_activity = activity
                        EOF = True

        matfile = {'frames': frames, 'labels': activities, 'resolution': metadata['resolution'], 'framerate': metadata['framerate']}
        io.savemat(str(outdir.joinpath('rgb.mat')), matfile, do_compression=True)

    if discard_original:
        os.remove(str(data_file)) 

def process_ir(outdir, data_file, metadata, timestamps_path, raw: bool, discard_original: bool):
    """
        If `raw`is true, creates a 3D array of IR frames, else stores raw data.
    """
    with open(data_file, 'rb') as ir_h:
        IR_DATA = ir_h.read()

    if raw:
        with open(str(outdir.joinpath('ir.raw')), 'wb') as outfile:
            outfile.write(IR_DATA)
    else:
        frames = []
        activities = []
        ft = 1/float(metadata['framerate'])
        with open(timestamps_path, 'rt') as timestamps_h:
            EOF = False
            reader = csv.reader(timestamps_h, delimiter=',')

            row = next(reader)
            time = float(row[0])
            activity = row[1]
            next_row = next(reader)
            next_time = float(next_row[0])
            next_activity = next_row[1]

            n = 0
            while (frame := get_ir_frame(IR_DATA, n)) is not None:
                frames.append(frame)
                activities.append(activity)
                n += 1
                time += ft
                if time >= next_time and not EOF:
                    time = next_time
                    activity = next_activity
                    row = next(reader)
                    next_time = float(row[0])
                    next_activity = row[1]
                    if next_activity.upper() == 'STOP':
                        next_activity = activity
                        EOF = True

        matfile = {'frames': frames, 'labels': activities, 'framerate': metadata['framerate']}

        io.savemat(str(outdir.joinpath('ir.mat')), matfile, do_compression=True)

    if discard_original:
        os.remove(str(data_file)) 

def process_audio(outdir, data_file, metadata, timestamps_path, discard_original: bool):
    """
        Stores audio file.
    """
    with open(str(data_file), 'rb') as audio_h:
        AUDIO_DATA = audio_h.read()
    with open(str(outdir.joinpath('audio.wav')), 'wb') as audio_file:
        audio_file.write(AUDIO_DATA)

    if discard_original:
        os.remove(str(data_file)) 

def process_radar(outdir, data_file, metadata, timestamps_path, fov, n_angles, fft: bool, music: bool, raw: bool, discard_original: bool):
    """
        If `raw` is true, stores the raw radar data.
        Else, applies the MUSIC algorithm and FFT algorithm to radar data,
        based on variables `music` and `fft`,
        and stores them as 3D arrays along some metadata.
    """
    with open(str(data_file), 'rb') as data_h:
        RADAR_DATA = data_h.read()

    raw_frames=[]
    music_futures=[]
    fft_futures=[]
    raw_futures=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        n = 0
        while (data_frame := get_radar_frame(RADAR_DATA, metadata, n)) is not None:
            raw_frames.append(data_frame) if not raw else None
            music_futures.append(executor.submit(MUSIC, data_frame, n, metadata, fov, n_angles)) if music else None
            fft_futures.append(executor.submit(FFT, data_frame, n)) if fft else None
            n += 1

        music_frames = numpy.array([future.result()[1] for future in music_futures]) if music else None
        fft_frames = numpy.array([future.result()[1] for future in fft_futures]) if fft else None

    if not raw or music_frames is not None or fft_frames is not None:
        matfile = {}
        if music_frames is not None:
            matfile['music'] = music_frames
        if fft_frames is not None:
            matfile['fft'] = fft_frames

        if not raw:
            matfile['raw'] = numpy.array(raw_frames)
            matfile['framerate'] = metadata['framerate']
            matfile['chirp_cycle_time'] = metadata['chirp_cycle_time']
            matfile['chirps_per_frame'] = metadata['chirps_per_frame']
            matfile['samplerate'] = metadata['samplerate']
            matfile['samples_per_chirp'] = metadata['samples_per_chirp']
            matfile['slope'] = metadata['slope']

            ft = 1/float(metadata['framerate'])
            activities = []
            with open(str(timestamps_path), 'rt') as timestamps_h:
                EOF = False
                reader = csv.reader(timestamps_h, delimiter=',')
                row = next(reader)
                time = float(row[0])
                activity = row[1]
                next_row = next(reader)
                next_time = float(next_row[0])
                next_activity = next_row[1]

                for frame_num in range(len(raw_frames)):
                    activities.append(activity)
                    time += ft
                    if time >= next_time and not EOF:
                        time = float(next_time)
                        activity = next_activity
                        row = next(reader)
                        next_time = float(row[0])
                        next_activity = row[1]
                        if next_activity.upper() == 'STOP':
                            EOF = True
                            next_activity = activity
            matfile['labels'] = activities

        io.savemat(str(outdir.joinpath('radar.mat')), matfile, do_compression=True)

    if raw:
        with open(str(outdir.joinpath('radar.raw'), 'rb')) as radar_h:
            radar_h.write(RADAR_DATA)

    if discard_original:
        os.remove(str(data_file)) 

def run(args):
    """
        Recursively looks into directories under args.INPUT and processes all
        relevant files/directories under it.
        Writes the processed data under args.OUTPUT, mirroring the directory structure
        from args.INPUT.
    """
    #pool = mp.Pool()
    # Some variables
    radar_f = 'radar.raw'
    depth_f = 'depth.raw'
    rgb_f = 'rgb.raw'
    ir_f = 'ir.raw'
    audio_f = 'audio.wav'
    timestamps_f = 'timestamps.csv'
    metadata_f = 'metadata.yaml'

    in_root = pathlib.Path(args['INPUT']).resolve()
    out_root = pathlib.Path(args['OUTPUT']).resolve()

    get_path = lambda path, file : path.joinpath(file)
    is_file = lambda path, file : path.joinpath(file).is_file()
    not_found_err = lambda path, file : logger.warning(f'File {str(get_path(path, file))} not found! Omitting.')

    stack = [in_root]
    #with mp.Pool() as pool:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        file_futures = []
        while len(stack) > 0:
            in_path = stack.pop()
            
            # Looks for more directories and files
            contents = in_path.iterdir()
            files_found = False
            for child in contents:
                if child.is_dir():
                    stack.append(child.resolve())
                elif child.is_file():
                    files_found = True
            
            # If files are found, process them
            if files_found:
                if is_file(in_path, timestamps_f) and is_file(in_path, metadata_f):
                    with open(str(get_path(in_path, metadata_f)), 'rt') as metadata_h:
                        metadata = yaml.safe_load(metadata_h)
                    timestamps_path = str(in_path.joinpath(timestamps_f))

                    outdir = out_root.joinpath(pathlib.Path(str(in_path)[len(str(in_root))+1:]))
                    os.makedirs(outdir, exist_ok=True)

                    # Everything that uses the process pool has to be ran in the main process
                    # Process radar
                    with open(str(get_path(in_path, radar_f)), 'rb') as data_h:
                        RADAR_DATA = data_h.read()

                    raw_frames=[]
                    music_futures=[]
                    fft_futures=[]
                    raw_futures=[]

                    n = 0
                    while (data_frame := get_radar_frame(RADAR_DATA, metadata['radar'], n)) is not None:
                        raw_frames.append(data_frame) if not args['raw'] else None
                        music_futures.append(executor.submit(MUSIC, data_frame, n, metadata['radar'], args['fov'], args['n_angles'])) if args['music'] else None
                        fft_futures.append(executor.submit(FFT, data_frame, n)) if args['fft'] else None
                        n += 1

                    music_frames = numpy.array([future.result()[1] for future in music_futures]) if args['music'] else None
                    fft_frames = numpy.array([future.result()[1] for future in fft_futures]) if args['fft'] else None

                    if not args['raw'] or music_frames is not None or fft_frames is not None:
                        matfile = {}
                        if music_frames is not None:
                            matfile['music'] = music_frames
                        if fft_frames is not None:
                            matfile['fft'] = fft_frames

                        if not args['raw']:
                            matfile['raw'] = numpy.array(raw_frames)
                            matfile['framerate'] = metadata['radar']['framerate']
                            matfile['chirp_cycle_time'] = metadata['radar']['chirp_cycle_time']
                            matfile['chirps_per_frame'] = metadata['radar']['chirps_per_frame']
                            matfile['samplerate'] = metadata['radar']['samplerate']
                            matfile['samples_per_chirp'] = metadata['radar']['samples_per_chirp']
                            matfile['slope'] = metadata['radar']['slope']

                            ft = 1/float(metadata['radar']['framerate'])
                            activities = []
                            with open(str(timestamps_path), 'rt') as timestamps_h:
                                EOF = False
                                reader = csv.reader(timestamps_h, delimiter=',')
                                row = next(reader)
                                time = float(row[0])
                                activity = row[1]
                                next_row = next(reader)
                                next_time = float(next_row[0])
                                next_activity = next_row[1]

                                for frame_num in range(len(raw_frames)):
                                    activities.append(activity)
                                    time += ft
                                    if time >= next_time and not EOF:
                                        time = float(next_time)
                                        activity = next_activity
                                        row = next(reader)
                                        next_time = float(row[0])
                                        next_activity = row[1]
                                        if next_activity.upper() == 'STOP':
                                            EOF = True
                                            next_activity = activity
                            matfile['labels'] = activities

                        io.savemat(str(outdir.joinpath('radar.mat')), matfile, do_compression=True)

                    if args['raw']:
                        with open(str(outdir.joinpath('radar.raw')), 'wb') as radar_h:
                            radar_h.write(RADAR_DATA)

                    if args['discard_original']:
                        os.remove(str(get_path(in_path_radar_f))) 
                    # End process radar
        
                    # Depth
                    file_futures.append( \
                        executor.submit(process_depth, \
                            outdir, get_path(in_path, depth_f), metadata['camera']['depth'], timestamps_path, args['raw'], args['discard_original'] \
                        ) \
                    ) if is_file(in_path, depth_f) else not_found_err(in_path, depth_f)

                    # RGB
                    file_futures.append( \
                        executor.submit(process_rgb, \
                            outdir, get_path(in_path, rgb_f), metadata['camera']['rgb'], timestamps_path, args['blur'], args['raw'], args['discard_original'] \
                        ) \
                    ) if is_file(in_path, rgb_f) else not_found_err(in_path, rgb_f)

                    # IR
                    file_futures.append( \
                        executor.submit(process_ir, \
                            outdir, get_path(in_path, ir_f), metadata['ir'], timestamps_path, args['raw'], args['discard_original'] \
                        ) \
                    ) if is_file(in_path, ir_f) else not_found_err(in_path, ir_f)

                    # Audio
                    file_futures.append( \
                        executor.submit(process_audio, \
                            outdir, get_path(in_path, audio_f), metadata['audio'], timestamps_path, args['discard_original'] \
                        ) \
                    ) if is_file(in_path, audio_f) else not_found_err(in_path, audio_f)

                else:
                    logger.warning(f'metadata.yaml or timestamps.csv missing from {str(in_path)}, omitting files from directory.')

        for future in file_futures:
            future.result()


if __name__ == '__main__':
    """Gets CLI arguments and calls run(args)"""
    # CLI
    parser = argparse.ArgumentParser(description='Convert recorded data into .mat and optionally perform additional post-processing')
    parser.add_argument('INPUT', help='Path to recorded dataset directory')
    parser.add_argument('OUTPUT', help='Specifies output directory.')
    parser.add_argument('--discard-original', action='store_true', default=False, help='Removes original data after processing (USE WITH CAUTION!)')
    parser.add_argument('--fft', action='store_true', default=False, help='Processes radar data with FFT algorithm (range-doppler) under file fft.mat')
    parser.add_argument('--music', action='store_true', default=False, help='Processes radar data with 2D-MUSIC algorithm (range-algorithm) under file music.mat')
    parser.add_argument('--fov', default=PI/3, help='Specify FOV for angular MUSIC. Defaults to 60 degrees.')
    parser.add_argument('--n-angles', default=20, help='Specify number of angle bins for MUSIC. Defaults to 20 bins.')
    parser.add_argument('--blur', action='store_true', default=False, help='Blurs faces from RGB video.')
    parser.add_argument('--raw', action='store_true', default=False, help='Stores data in raw format (not .mat) for increased compatibility with different languages')

    args = vars(parser.parse_args())

    if not pathlib.Path(args['INPUT']).is_dir():
        logger.error("INPUT doesn't exist or is not a directory. Check the path and try again!")
        raise FileNotFoundError(args['INPUT'])

    start = time.perf_counter()
    run(args)
    logger.info(f'Took: {time.perf_counter() - start : .2f} s')
