import multiprocessing as mp
import time
import argparse
import yaml
import csv
import soundfile as sf
import queue
#from scipy.io.wavfile import write as write_wav
import pathlib
import logging
import os

import sys
if sys.version_info.major < 3 or sys.version_info.minor < 10:
    exit(f"You're running Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. At least 3.10 is required!")

import capturelib.mic
import capturelib.ir
import capturelib.realsense
import capturelib.dca1000evm


logger = logging.getLogger('__main__')
logger.setLevel(logging.DEBUG)


def data_consumer(config, outdir, mic, ir, cam, radar, signals):
    """
        Reads data from queues, synchronizes streams based on received metadata,
        and writes data to output files.
    """
    mic_outfile = 'audio.wav'
    radar_outfile = 'radar.raw'
    ir_outfile = 'ir.raw'
    rgb_outfile = 'rgb.raw'
    depth_outfile = 'depth.raw'

    mic_ft = 1/44100
    ir_ft = 0.1
    cam_ft = 1/config['camera']['fps']
    radar_ft = capturelib.dca1000evm.read_config(config['radar']['config'])['frameCfg'][0][4] * 1e-3

    mic_f = sf.SoundFile( str(pathlib.Path(outdir, mic_outfile   )) , mode='w', samplerate=44100, channels=16 )
    radar_f = open( str(pathlib.Path(outdir, radar_outfile )) , 'wb' , newline=None )
    ir_f    = open( str(pathlib.Path(outdir, ir_outfile    )) , 'wb' , newline=None )
    rgb_f   = open( str(pathlib.Path(outdir, rgb_outfile   )) , 'wb' , newline=None )
    depth_f = open( str(pathlib.Path(outdir, depth_outfile )) , 'wb' , newline=None )

    filehandles = [mic_f, radar_f, ir_f, rgb_f, depth_f]

    producers  = [ mic    , ir    , cam    , radar    ]
    metadatas  = [ None   , None  , None   , None     ] 
    frametimes = [ mic_ft , ir_ft , cam_ft , radar_ft ]
    timers     = [ None   , None  , None   , None     ] 
    frame_counters = [ 0  , 0     , 0      , 0        ] 

    # Wait for 1st signal ('START'), discard everything
    while signals.empty() or signals.get() != 'START':
        #logger.debug('discarding data')
        for idx, producer in enumerate(producers):
            # 1st packet in each queue is metadata. Save it.
            if metadatas[idx] == None:
                metadata = producer.get()
                logger.debug(f'metadata ({idx}) received: {metadata}')
                #print(metadata)
                metadatas[idx] = metadata
                timers[idx] = metadata['start_time']
            while not producer.empty():
                producer.get()

    logger.debug('starting recording')
    
    # 'START' signal received
    #start_time = time.perf_counter()
    start_time = signals.get()
    logger.debug(f'start_time: {start_time}')
    # synchronize streams by discarding until start_time
    # There may be some "old" data coming from buffers that just hasn't been discarded yet

    for idx, producer in enumerate(producers):
        if timers[idx] == None:
            timers[idx] = metadatas[idx]['start_time']

        else:
            while timers[idx] + frametimes[idx]/2 < start_time:
                if not producer.empty():
                    logger.debug(f'Discarding packet from {(idx)}.')
                    packet = producer.get()
                    timers[idx] = packet['timestamp']

                else:
                    logger.debug('Waiting for packets...')
                    while producer.empty():
                        pass

    logger.debug(f'start_time, mic   , ir      , cam     , radar')
    logger.debug(f'{start_time:.3f}, {timers[0]:.3f}, {timers[1]:.3f}, {timers[2]:.3f}, {timers[3]:.3f}')

    max_time = max(timers)
    max_time_idx = timers.index(max_time)

    for idx, producer in enumerate(producers):
        if idx == max_time_idx:
            continue

        while timers[idx] + frametimes[idx]/2 < max_time:
            if not producer.empty():
                logger.debug(f'Discarding packet from {(idx)}.')
                if idx == 0 or idx == 2:
                    # Camera
                    packet = producer.get()
                    timers[idx] = packet['timestamp']
                else:
                    producer.get()
                    timers[idx] += frametimes[idx]
            else:
                logger.debug('Waiting for packets...')
                while producer.empty():
                    pass
    

    logger.info('Streams synchronized')
    logger.debug(f'start_time, mic   , ir      , cam     , radar')
    logger.debug(f'{start_time:.3f}, {timers[0]:.3f}, {timers[1]:.3f}, {timers[2]:.3f}, {timers[3]:.3f}')

    # Start writing data to file
    print("writing")
    c_frame = 0
    last_frame_t = time.perf_counter()
    while signals.empty() or (sig := signals.get()) != 'STOP':
        for idx, producer in enumerate(producers):
            while not producer.empty():
                if idx == 0:
                    # Mic
                    packet = producer.get()
                    mic_f.write(packet['data'])
                    timers[idx] = packet['timestamp']
                    frame_counters[idx] += 44100

                elif idx == 1:
                    # IR
                    packet = producer.get()
                    ir_f.write(packet['raw_data'])
                    timers[idx] = packet['timestamp']
                    frame_counters[idx] +=1 

                elif idx == 2:
                    # Camera
                    packet = producer.get()
                    #print(f'frame: {c_frame}, FPS: {1/(time.perf_counter() - last_frame_t)}')
                    last_frame_t = time.perf_counter()
                    rgb_f.write(packet['RGB'])
                    depth_f.write(packet['D'])
                    timers[idx] = packet['timestamp']
                    frame_counters[idx] +=1 

                elif idx == 3:
                    # Radar
                    packet = producer.get()
                    radar_f.write(packet['data'])
                    timers[idx] = packet['timestamp']
                    frame_counters[idx] +=1 


    print("Consuming remaining data...")
    stop_time = signals.get()

    # Producers are stopped, flush buffers into files
    for producer in producers:
        while not producer.empty():
            if idx == 0:
                # Mic
                packet = producer.get()
                if packet['timestamp'] < stop_time:
                    mic_f.write_wav(packet['data'])
                    frame_counters[idx] += 44100
                    timers[idx] = packet['timestamp']

            elif idx == 1:
                # IR
                packet = producer.get()
                if packet['timestamp'] < stop_time:
                    ir_f.write(packet['raw_data'])
                    frame_counters[idx] += 1 
                    timers[idx] = packet['timestamp']

            elif idx == 2:
                # Camera
                packet = producer.get()
                if packet['timestamp'] < stop_time:
                    rgb_f.write(packet['RGB'])
                    depth_f,write(packet['D'])
                    frame_counters[idx] += 1 
                    timers[idx] = packet['timestamp']

            elif idx == 3:
                # Radar
                packet = producer.get()
                if packet['timestamp'] < stop_time:
                    radar_f.write(packet['data'])
                    frame_counters[idx] += 1 
                    timers[idx] = packet['timestamp']

    print("closing file handles")
    for fh in filehandles:
        fh.close()

    logger.debug(f'start_time, mic   , ir      , cam     , radar')
    logger.debug(f'{start_time:.3f}, {timers[0]:.3f}, {timers[1]:.3f}, {timers[2]:.3f}, {timers[3]:.3f}')
    logger.debug(f'{frame_counters[0]}, {frame_counters[1]}, {frame_counters[2]}, {frame_counters[3]}')


if __name__ == '__main__':
    """
        Initializes queues and subprocesses,
        reads the activities from a file,
        writes markers and labels + metadata,
        and provides an user interface.
    """
    parser = argparse.ArgumentParser(description='MMSDC recorder')
    parser.add_argument('activities', help='CSV file of activities and timestamps')
    parser.add_argument('config', help='YAML configuration file')
    parser.add_argument('outdir', help='Output directory')
    args=parser.parse_args()

    configfile = args.config
    activity_file = args.activities
    outdir = args.outdir

    with open(configfile, 'r') as config_fd:
        config = yaml.safe_load(config_fd)

    # Signaling
    mic_signals = mp.Queue()
    ir_signals = mp.Queue()
    cam_signals = mp.Queue()
    radar_signals = mp.Queue()
    consumer_signals = mp.Queue()
    signals = [mic_signals, ir_signals, cam_signals, radar_signals]

    # Data capture queues
    mic_out = mp.Queue()
    ir_out = mp.Queue()
    cam_out = mp.Queue()
    radar_out = mp.Queue()

    # Start processes
    processes = []
    processes.append( mp.Process(target=capturelib.mic.record, args=(mic_out, mic_signals)) )
    processes.append( mp.Process(target=capturelib.ir.record, args=(ir_out, ir_signals)) )
    processes.append( mp.Process(target=capturelib.realsense.record, args=(config['camera'], cam_out, cam_signals)) )
    processes.append( mp.Process(target=capturelib.dca1000evm.record, args=(config['radar'], radar_out, radar_signals)) )

    for p in processes:
        p.start()

    data_consumer_ = mp.Process(target=data_consumer, args=(config, outdir, \
            mic_out, ir_out, cam_out, radar_out, consumer_signals))
    data_consumer_.start()

    # Read activities and wait until devices are started
    timestamp_f = open(str(pathlib.Path(outdir, 'timestamps.csv')), 'wt', newline='\n')
    with open(activity_file, 'r') as activities:
        device_started = [ False  , False , False  , False    ]
        while False in device_started:
            for idx, signal in enumerate(signals):
                if not device_started[idx] and not signal.empty() and signal.get(timeout=1) == 'STARTED':
                    device_started[idx] = True

        print(" --- Devices started! --- ")
        STARTED = False
        reader = csv.reader(activities, delimiter=',')
        
        # User Interface
        for row in reader:
            activity = row[0]
            print(f'Next activity: {activity}.')
            input_ = input('Press enter to proceed, or q to quit.\n')

            if input_.lower() == 'q' or activity.upper() == 'STOP':
                STARTED=False
                break
            elif not STARTED:
                STARTED=True
                start_time = time.perf_counter()
                consumer_signals.put('START')
                consumer_signals.put(start_time)

            timestamp_f.write(f'{time.perf_counter()-start_time:.6f},{activity}\n')
    try:
        stop_time = time.perf_counter()-start_time
    except NameError:
        stop_time = 0.0
    timestamp_f.write(f'{stop_time:.3f},STOP\n')
    timestamp_f.close()

    # Stop recording
    for signal in signals:
        signal.put('STOP')

    # Synchronize processes
    for p in processes:
        p.join()

    consumer_signals.put('STOP')
    consumer_signals.put(stop_time)
    data_consumer_.join()

    radar_conf = capturelib.dca1000evm.read_config(config['radar']['config'])

    # Write metadata file
    radar_samples_per_chirp = radar_conf['profileCfg'][0][-5]
    radar_chirps_per_frame = radar_conf['frameCfg'][0][2]
    radar_frame_bytesize = radar_samples_per_chirp * radar_chirps_per_frame * 4 * 4 # samples per frame * bytes per sample

    cam_resolution = f"[{config['camera']['resolution'][0]}, {config['camera']['resolution'][1]}]"

    
    metadata = {
        'duration': stop_time,
        'radar': {
            'samplerate': radar_conf['profileCfg'][0][-4]*10**3,
            'samples_per_chirp': radar_conf['profileCfg'][0][-5],
            'chirps_per_frame': radar_conf['frameCfg'][0][2],
            #'framerate': 1/radar_conf['frameCfg'][0][-2],
            'framerate': os.path.getsize(pathlib.Path(outdir, 'radar.raw')) / radar_frame_bytesize / stop_time,
            'slope': radar_conf['profileCfg'][0][7]*10**12,
            'num_channels': 4,
            'chirp_cycle_time': (radar_conf['profileCfg'][0][2]+radar_conf['profileCfg'][0][4])*1e-6,
        },
        'camera': {
            'rgb': {
                'framerate': os.path.getsize(pathlib.Path(outdir, 'rgb.raw')) / (config['camera']['resolution'][0]*config['camera']['resolution'][1]*3) / stop_time,
                'resolution': cam_resolution
            },
            'depth': {
                'framerate': os.path.getsize(pathlib.Path(outdir, 'depth.raw')) / (config['camera']['resolution'][0]*config['camera']['resolution'][1]*2) / stop_time,
                'resolution': cam_resolution
            }
        },
        'ir': {
            'framerate': os.path.getsize(pathlib.Path(outdir, 'ir.raw')) / (2*64) / stop_time
        },
        'audio': {
            'samplerate': 44100
        }
    }

    with open(pathlib.Path(outdir, 'metadata.yaml'), 'tw') as f:
        yaml.safe_dump(metadata, f)

    print('Done!')

