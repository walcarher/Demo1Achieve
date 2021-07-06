from utils import *
import cv2

def test():
    #cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1")
    if cap.isOpened():
    	# Window creation and specifications
        windowName = "Camera Test"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(windowName,1920-1280,0)
        cv2.resizeWindow(windowName,1920,1080)
        cv2.setWindowTitle(windowName,"Camera Test")
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
            #sized = cv2.resize(img, (1280, 720))
            if showHelp == True:
                cv2.putText(img, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(img, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            end = time.time()
            cv2.putText(img, "{0:.0f}fps".format(1/(end-start)), (531,50), font, 3.0, (32,32,32), 8, cv2.LINE_AA)
            cv2.putText(img, "{0:.0f}fps".format(1/(end-start)), (530,50), font, 3.0, (240,240,240), 2, cv2.LINE_AA)
            cv2.imshow(windowName, img)
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
    test()
