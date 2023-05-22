# Recorder

A python program to read data from the sensors, synchronize the data streams, and save the data into a file.
The accompanying `post-process.py` can be used to parse the saved raw data into a easier-to-use `.mat` format.

## Known bugs

* Quite often, the program starts but fails to write any data into files.
	* After 1st activity is started, the CLI should say "writing". If not, the issue has occurred.
		* The program will continue 'normally', but will not exit or write and data into the files.

## TODO

* Improve stability (See known bugs)
* Improve face detection
	* Current version performs **very** poorly
* Make visualize.py work with post-processed data to make it much faster.
* Make sensors more modular
	* Right now the program will crash if some of the sensors is not connected, and it is a pain to remove sensors from the code.
		* It would be nice to easily add and remove sensors.
* Using queues as the highest-level interface for inter-process communication seems to be a bit messy.
	* Object-oriented design for the "capturelib" might be easier to work with
* Increase the performance of the MUSIC algorithm, if possible

## Files and directories

A brief description of the most important files and directories is given here.
There are also many more miscellaneous files, such as all the YAML, CSV, and CFG files.
All of the aforementioned YAML, CSV, and CFG files can be deleted, but the record.py needs at least one of each to run.
The files are therefore given as an example, and it is likely a good idea to have them around as a reference even if not strictly needed.

### Files

* `record.py` is the 'main program'. It starts the sensors and writes received data into files.
* `post-process.py` can be used to process the recorded raw data into a easier to use `.mat` format
* `visualize.py` can be used to visualize the recorded raw data.
	* Calculating the MUSIC algorithm takes ages.
* `requirements.txt` should contain all the pip dependencies

### Directories

* `librealsense` contains files needed to build the librealsense python bindings
* `capturelib` contains device-specific code.

## A brief note on inter-process communication

At the moment, the `record.py` program uses queues to communicate between processes.
For each process, there exists a queue named "signaling" or something similar.
The currently used signals are as follows:
	* "START" starts a sensor.
	* "STOP" stops a sensor.
	* "STARTED" indicates that a sensor is started.
	
Using these would likely be much more straight-forward if the devices were objects that had start and stop functions,
and a check_state or similar function.
