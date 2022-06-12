To generate viewports from heade tracking logs, refer to the file `./get_viewport.py`. The program assumes input head tracking logs have been downloaded and the file paths have been provided in `header.py`. It converts the head-tracking logs from quaternions to pixels on an equirectangular frame. To run the file, follow the usage as shown by `python get_viewport.py --help`
	
	usage: get_viewport.py [-h] -D DATASET -T TOPIC --fps FPS

	Run Viewport Extraction Algorithm

	optional arguments:
	  -h, --help            show this help message and exit
	  -D DATASET, --dataset DATASET
	                        Dataset ID (1 or 2)
	  -T TOPIC, --topic TOPIC
	                        Topic in the particular Dataset (video name)
	  --fps FPS             fps of the video

Here are the possible values for `dataset` and `topic`:  
if `ds=1`, `topic` can be `paris`, `roller`, `venise`, `diving`, `timelapse`  
if `ds=2`, `topic` can be `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`  
