from utils import *
from darknet import Darknet
import cv2

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
 
    use_cuda = 1
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    if cap.isOpened():
    	# Window creation and specifications
        windowName = cfgfile
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
	cv2.moveWindow(windowName,0,0)
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
            print('------')
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
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')
