import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
import serial.tools.list_ports
import realsense
import radar


if __name__ == '__main__':
    """Was used for testing"""
    duration = 10
    
    # Realsense
    realsense_config = {
                'depth_resolution' : (640, 480),
                'rgb_resolution' : (640, 480),
                'fps' : 15
            }

    rgbd = realsense.record(realsense_config, duration)

    # Infrared
    ir = infrared.record(duration)

    # Mic
    audio = mic.record(duration)

    # Make a window to show camera stream in
    fig = plt.figure()
    rgb_ax = fig.add_subplot(121)
    depth_ax = fig.add_subplot(122)

    # Start streaming data from depth camera
    rgb_im = rgb_ax.imshow(rgbd['RGB'][0])
    depth_im = depth_ax.imshow(rgbd['D'][0], vmin=0, vmax=10000) # 10 m in millimeters

    def _update_frame(n, rgb_im, depth_im):
        rgb_im.set_data(rgbd['RGB'][n])
        depth_im.set_data(rgbd['D'][n])


    cam_animation = FuncAnimation(fig, _update_frame, frames=range(rgbd['num_frames']), fargs=(rgb_im, depth_im), interval= 1e3/15)

    plt.show()

