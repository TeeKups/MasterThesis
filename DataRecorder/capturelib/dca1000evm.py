import re
import queue
import struct
import socket
import serial
import serial.tools.list_ports
import time
import queue
import multiprocessing as mp
import logging

import sys
if sys.version_info.major < 3 or sys.version_info.minor < 10:
    exit(f"You're running Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. At least 3.10 is required!")

# User Guide for DCA1000 Data Capture Card.pdf chapter 4
DCA_IP = '192.168.33.180'  
CPORT = 4096
DPORT = 4098
DST = (DCA_IP, CPORT)

logger = logging.getLogger('__main__')

def read_config(radar_config):
    """
        Reads the radar configuration file
        mmwave_sdk_user_guide.pdf page 19-34
    """
    numberfy = lambda n : int(float(n)) if float(n).is_integer() else float(n)
    config = {}
    with open(radar_config, 'rt',) as config_file:
        comment = re.compile(r'(\s*%.*$|^\s*$)')
        for line in ( line.rstrip(' \r\n') for line in config_file.readlines() ):
            if comment.match(line) != None:
                continue
            line = line.split(' ')
            if len(line) < 1:
                continue
            if line[0] == 'sensorStart':
                continue
            if line[0] not in config:
                config[line[0]] = []
            if len(line) > 1:
                config[line[0]].append(list([*map(numberfy, line[1:])]))
            elif len(line) == 1:
                config[line[0]].append([])
    config['lvdsStreamCfg'][0] = [-1,0,1,0]
    return config

def get_samples_per_chirp(config):
    """Reads samples per chirp from config"""
    return config['profileCfg'][-1][-5]

def get_chirps_per_frame(config):
    """Reads chirps per frame from config"""
    return config['frameCfg'][-1][2]*(config['frameCfg'][-1][1]-config['frameCfg'][-1][0]+1)
    
def to_little(bytes_):
    """Takes big-endian hex as argument and returns as little-endian"""
    b = bytearray.fromhex(str(bytes_))
    b.reverse()
    return b

def to_big(bytes_):
    """Takes little-endian hex as argument and returns as big-endian"""
    bytes_ = bytes_.hex()
    b = bytearray.fromhex(bytes_)
    b.reverse()
    return b.hex()

def configure():
    """
        Configures the DCA1000EVM
        User Guide for DCA1000 Data Capture Card.pdf chapter 5
    """
    HDR = to_little('a55a')
    CMD_CODE = to_little('0003')
    data = bytearray.fromhex('01020102031e')
    DATA_SIZE = to_little(f'{str(hex(len(data))[2:]).zfill(4)}')
    FOOTER = to_little('eeaa')
    CONFIG_MSG = HDR + CMD_CODE + DATA_SIZE + data + FOOTER

    TIMEOUT = 10

    RESPONSE_SIZE = 8


    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('', CPORT))
        sock.sendto(CONFIG_MSG, DST)
        received = b''

        start_time = time.perf_counter()
        while len(received) < 8 and start_time - time.perf_counter() < TIMEOUT:
            received += sock.recv(RESPONSE_SIZE-len(received))

        if len(received) < 8:
            logger.debug('FPGA Configuration failed!')
        else:
            header = to_big(received[0:2])
            cmd_code = to_big(received[2:4])
            status = to_big(received[4:6])
            footer = to_big(received[6:8])
            if status != '0000':
                logger.debug('FPGA Configuration failed!')
            else:
                logger.debug('FPGA Configuration successful!')

def get_serial_devices():
    """Looks for the IWR6843 serial ports"""
    ports = serial.tools.list_ports.comports()
    data_port = next(p.name for p in ports if 'XDS110 Class Auxiliary Data Port' in p.description)
    uart_port = next(p.name for p in ports if 'XDS110 Class Application/User UART' in p.description)

    UART = serial.Serial(uart_port, baudrate=115200, timeout=10, parity=serial.PARITY_NONE)
    DATA = serial.Serial(data_port, baudrate=921600, timeout=10, parity=serial.PARITY_NONE)
    return DATA, UART

def config_radar(radar_config):
    """
        Configures the IWR6843, i.e. writes the configuration to UART port
    """
    try:
        config = read_config(radar_config)

        prompt = b'mmwDemo:/>'  # ???
        logger.debug(f'Sending configuration from {radar_config} to radar...')
        for command, param_lists in config.items():
            for params in param_lists:
                cmd=f"{command} {' '.join(map(str, params))}\n"
                UART.write(bytes(cmd.encode('utf-8')))
                echo = UART.readline()
                done = UART.readline()
                prompt = UART.read(len(prompt)) # ???

    except FileNotFoundError as e:
        print(e)
        exit()
    except serial.SerialException as e:
        print(e)
        raise(e)

    return config
    

def start(sock, RADAR_CONFIG):
    """Configures and starts the IWR6843 and DCA1000EVM"""
    config = config_radar(RADAR_CONFIG)
    FRAME_SIZE = 2*2*4*get_samples_per_chirp(config)*get_chirps_per_frame(config)

    UART.write(b'sensorStart\n')
    echo = UART.readline()
    done = UART.readline()
    prompt = b'mmwDemo:/>'
    prompt = UART.read(len(prompt)) # ???

    HDR = to_little('a55a')
    CMD_CODE = to_little('0005')
    DATA_SIZE = to_little('0000')
    FOOTER = to_little('eeaa')
    CONFIG_MSG = HDR + CMD_CODE + DATA_SIZE + FOOTER 

    sock.sendto(CONFIG_MSG, DST)
    received = b''

    TIMEOUT = 10
    RESPONSE_SIZE = 8
    received = sock.recv(RESPONSE_SIZE)

    if len(received) < 8:
        logger.debug('FPGA Configuration failed!')
    else:
        header = to_big(received[0:2])
        cmd_code = to_big(received[2:4])
        status = to_big(received[4:6])
        footer = to_big(received[6:8])
        if status != '0000':
            logger.debug('Start Record failed!')
        else:
            logger.debug('Start Record successful!')

    return FRAME_SIZE

def stop(sock):
    """Stops the IWR6843 and DCA1000EVM"""
    UART.write(b'sensorStop\n')
    echo = UART.readline()
    done = UART.readline()
    prompt = b'mmwDemo:/>'
    prompt = UART.read(len(prompt)) # ???

    HDR = to_little('a55a')
    CMD_CODE = to_little('0006')
    DATA_SIZE = to_little('0000')
    FOOTER = to_little('eeaa')
    CONFIG_MSG = HDR + CMD_CODE + DATA_SIZE + FOOTER 

    sock.sendto(CONFIG_MSG, DST)
    received = b''

    TIMEOUT = 10
    RESPONSE_SIZE = 8
    received = sock.recv(RESPONSE_SIZE)

    if len(received) < 8:
        logger.debug('FPGA Configuration failed!')
    else:
        header = to_big(received[0:2])
        cmd_code = to_big(received[2:4])
        status = to_big(received[4:6])
        footer = to_big(received[6:8])
        if status != '0000':
            logger.debug('Record stop failed!')
        else:
            logger.debug('Record Stop successful!')

def record(config, out, signals):
    """
        Calss configure() to start the IWR6843 and DVA1000EVM,
        and starts reading data from the DCA1000EVM.
        Reads the radar and reorganizes it into packets that contain one radar frame each.
        Writes radar frames into `out` -queue along a timestamp for each packet.
        
        Timestamp is calculated based on the reception time for 1st frame and the number of received frames.
        Clock drift of the IRW6843 clock is +-50ppm (swrs219d_IWR6843_datasheet.pdf chapter 8.10.2.1)
    """
    global DATA, UART
    DATA, UART = get_serial_devices()
    RADAR_CONFIG = config['config']

    configure()
    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    conf_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.settimeout(3)
    data_sock.bind(('', DPORT))
    conf_sock.bind(('', CPORT))

    waiting_area = []
    counter = 1
    buf = bytearray(b'')
    FRAME_SIZE = start(conf_sock, RADAR_CONFIG)
    logger.debug("Radar started")
    qsize = signals.qsize()
    signals.put('STARTED')
    while signals.qsize() > qsize:
        pass

    first_frame = True
    frame_count = 0
    frame_duration = read_config(RADAR_CONFIG)['frameCfg'][0][4] * 1e-3
    try:
        while True:
            if not signals.empty() and signals.get() == 'STOP':
                stop(conf_sock)
            raw = data_sock.recv(1499)

            if first_frame:
                metadata = {'start_time': time.perf_counter()}
                out.put(metadata)
                first_frame = False
            
            start_time = metadata['start_time']

            seq = int(to_big(raw[0:4]), 16)
            byte_count = raw[4:10]
            data = raw[10:]
            #print(len(raw[0:4]), len(byte_count), len(data))
            packet = { 'seq': seq, 'byte_count': byte_count, 'data': data }

            buf += bytearray(packet['data'])
            """
            # Organize packets
            waiting_area.append(packet)
            for idx, pack in enumerate(waiting_area):
                if pack['seq'] == counter:
                    #print(seq)
                    buf += bytearray(pack['data'])
                    counter += 1
                    waiting_area.pop(idx)
                    break
            """
                
            # push to outqueue 1 frame at a time
            if len(buf) >= FRAME_SIZE:
                out.put({'data': buf[:FRAME_SIZE], 'timestamp': start_time + frame_count*frame_duration})
                frame_count += 1
                buf = bytearray(buf[FRAME_SIZE:])
    except socket.timeout as e:
        logger.debug('Radar receive buffer empty (recording stopped and all packets received)')
    finally:
        conf_sock.close()
        data_sock.close()
        out.close()

    DATA.close()
    UART.close()

if __name__ == '__main__':
    """Used for testing"""
    out = mp.Queue()
    signals = mp.Queue()

    p = mp.Process(target=record, args=({'config': 'new3.cfg'}, out, signals))
    p.start()

    time.sleep(2)
    signals.put('STOP')

    frame_size = 4*4*64*64

    buf = bytearray(b'')

    with open('test.bin', 'wb', newline=None) as f:
        while p.is_alive() or not out.empty():
            # This functionality is moved to record
            # TODO: adjust to new behavior
            try:
                packet = out.get(timeout=3)
                buf += bytearray(packet['data'])
                if len(buf) >= frame_size:
                    #print(len(buf))
                    f.write(buf[:frame_size])
                    buf = bytearray(buf[frame_size:])
                    #print(len(buf))
            except queue.Empty:
                print("-------------------------")
    print(len(buf))
    print(2)
    p.join()
    print(3)


