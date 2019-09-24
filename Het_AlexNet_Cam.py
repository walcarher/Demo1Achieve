import subprocess
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time
import cv2

# Argument configuration
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type = int, choices=[0, 1],
		 help = "enables gpu mode for inference 0 (default) for CPU mode and 1 for GPU mode",
		 default = 0)
parser.add_argument("-m", "--monitor", type = int, choices=[0, 1],
		 help = "enables the monitoring of usage percentage of available devices",
		 default = 0)
parser.add_argument("-o", "--onnx", type = int, choices=[0, 1],
		 help = "enables the ONNX model generation for partitioning",
		 default = 0)
args = parser.parse_args()

# Use the Jetson onboard camera
def open_onboard_camera():
    return cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Capture Frame and main inference loop
def read_cam(video_capture):
    if video_capture.isOpened():
	# Window creation and specifications
        windowName = "AlexNetDemo"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
	cv2.moveWindow(windowName,0,0)
        cv2.resizeWindow(windowName,1280,1080)
        cv2.setWindowTitle(windowName,"AlexNet Classification")
        font = cv2.FONT_HERSHEY_PLAIN
        helpText="'Esc' to Quit"
        showFullScreen = False
	showHelp = True
	start = 0.0
	end = 0.0
	# Normalize and Resize input 
	loader = transforms.Compose([transforms.Resize(size=(224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	transPIL = transforms.ToPILImage()
        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            # Frame capture
            ret_val, frame = video_capture.read()
            # Image transformation 
            # to PIL imageBGR to RGB and trasposing channel to ChannelxWidthxHeight
            input = torch.from_numpy(np.transpose(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(2,0,1)))
            #input = loader(transPIL(input))
            input = loader(transPIL(input))
            # Unsqueeze BatchxChannelxWidthxHeight, Batch=1 1xCxWxH
            input = input.unsqueeze(0).float()
            # Forward pass on specific device
            if(torch.cuda.is_available() & args.gpu):
                out = alexNet.forward(input.cuda())
            else:
                out = alexNet.forward(input)
            # Text and info printing  
            if showHelp == True:
                cv2.putText(frame, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(frame, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            cv2.putText(frame, labels[torch.argmax(out[0,:]).item()], (11,450), font, 1.5, (32,32,32), 5, cv2.LINE_AA)
            cv2.putText(frame, labels[torch.argmax(out[0,:]).item()], (10,450), font, 1.5, (240,240,240), 1, cv2.LINE_AA)
            end = time.time()
            cv2.putText(frame, "{0:.0f}fps".format(1/(end-start)), (481,50), font, 3.0, (32,32,32), 8, cv2.LINE_AA)
            cv2.putText(frame, "{0:.0f}fps".format(1/(end-start)), (480,50), font, 3.0, (240,240,240), 2, cv2.LINE_AA)
            cv2.imshow(windowName,frame)
            start = time.time()
            key=cv2.waitKey(1)
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
     print ("camera open failed")


# Using pretained model for Forward pass evaluation in platform
alexNet = models.alexnet(pretrained = True)
alexNet.eval()

# Check for CUDA availability
if(torch.cuda.is_available() & args.gpu):
	if(not torch.cuda.is_available()): 
		print('Error : No GPU available on the system')
		sys.exit()
	device = torch.device(torch.cuda.current_device())
	torch.cuda.init()
	alexNet.cuda()
	dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
	print('Running inference on GPU mode')
else:
	dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
	device = torch.device("cpu")
	print('Running inference on CPU mode')

# Label file
f = open("labels.txt")
labels = f.readlines()
# Open device usage monitor if selected 
if args.monitor:
	device_monitor_process = subprocess.Popen(["python", "DeviceMonitor.py"])
	device_monitor_processCPU = subprocess.Popen(["python", "CPUDeviceHistory.py"])
	device_monitor_processGPU = subprocess.Popen(["python", "GPUDeviceHistory.py"])
# Open Image Stream and plots
video_capture=open_onboard_camera()
read_cam(video_capture)
video_capture.release()
cv2.destroyAllWindows()
if args.monitor:
	device_monitor_process.terminate()
	device_monitor_processCPU.terminate()
	device_monitor_processGPU.terminate()
if args.onnx:
	input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
	output_names = [ "output1" ]
	torch.onnx.export(alexNet, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


