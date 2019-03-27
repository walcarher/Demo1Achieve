#MIT License

# This software has been heavily inspired from: Marvis GitHub code
# Please refer to : https://github.com/marvis/
# Original implementation of YOLOv2 can be found in: https://pjreddie.com/darknet/yolo

# Modifications to include TX2 onboard camera detection, model partitioning and device monitor 
# by: Walther Carballo Hernandez
# Please refer to: https://github.com/walcarher
# Modifications Copyright (c) 2019 Institut Pascal

#Copyright (c) 2015 Preferred Infrastructure, Inc.
#Copyright (c) 2015 Preferred Networks, Inc.

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

import subprocess
import argparse
from utils import *
from darknet import Darknet
import cv2

# Argument configuration
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg",
		 help = "path to the .cfg file",
		 )
parser.add_argument("-w", "--weights",
		 help = "path to the weights file",
		 )
parser.add_argument("-g", "--gpu", type = int, choices=[0, 1],
		 help = "enables gpu mode for inference 0 (default) for CPU mode and 1 for GPU mode",
		 default = 0)
parser.add_argument("-m", "--monitor", type = int, choices=[0, 1],
		 help = "enables the monitoring of usage percentage of available devices",
		 default = 0)
args = parser.parse_args()

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
 
    use_cuda = args.gpu
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    if cap.isOpened():
    	# Window creation and specifications
        windowName = cfgfile
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
	cv2.moveWindow(windowName,1920-1280,0)
        cv2.resizeWindow(windowName,1280,1080)
        cv2.setWindowTitle(windowName,"YOLOv2 Object Detection")
        font = cv2.FONT_HERSHEY_PLAIN
        helpText="'Esc' to Quit"
        showFullScreen = False
	showHelp = True
	start = 0.0
	end = 0.0
    else:
        print("Unable to open camera")
        exit(-1)

    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            #print('------')
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
	    if showHelp == True:
                cv2.putText(img, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(img, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            end = time.time()
            cv2.putText(img, "{0:.0f}fps".format(1/(end-start)), (481,50), font, 3.0, (32,32,32), 8, cv2.LINE_AA)
            cv2.putText(img, "{0:.0f}fps".format(1/(end-start)), (480,50), font, 3.0, (240,240,240), 2, cv2.LINE_AA)
            cv2.imshow(windowName, draw_img)
	    start = time.time()
            key = cv2.waitKey(1)
	    if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break;
            elif key==74: # Toggle fullscreen; This is the F3 key on this particular keyboard
                # Toggle full screen mode
                if showFullScreen == False : 
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL) 
                showFullScreen = not showFullScreen
        else:
             print("Unable to read image")
             exit(-1) 

############################################
if __name__ == '__main__':
	if args.monitor:
		device_monitor_process = subprocess.Popen(["python", "../DeviceMonitor.py"])
		gpu_history_process = subprocess.Popen(["python", "../GPUDeviceHistory.py"])
		cpu_history_process = subprocess.Popen(["python", "../CPUDeviceHistory.py"])
        demo(args.cfg, args.weights)
	if args.monitor:	
		device_monitor_process.terminate()
		gpu_history_process.terminate()
		cpu_history_process.terminate()

