# -*- coding: utf-8 -*-
"""
Este script permite que se rastreie um objeto que se queira treinar o seu reconhecimento, para que ao longo do video, sejam gerados samples das imagens, bem como dos
arquivos xml(Pascal Voc) ou txt(Yolo Darknet), que sao usados pelo tensorflow para o treinamento
@author: Dudu
"""
# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt
# python opencv_object_tracking.py --video videos/Becks.mp4 --tracker csrt --label becks --vocxml true --desting train --imagesdirectory train
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import cv2
# pip install opencv-contrib-python==4.2.0.34
import os
import time
from datetime import datetime
import xml.etree.cElementTree as ET
import tkinter as tk

# from read_video import MainWindow

def rescale_images(percentsizeimage,image):
    #for root, dirs, files in os.walk(directory):
    #    for filename in files:
        scale_percent = percentsizeimage /100
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        dsize = (width, height)
        output = cv2.resize(image, dsize,cv2.INTER_NEAREST)
        return output
            
def save_images(directory,filename,image):
    path = os.getcwd()+os.path.sep+directory
    if (os.path.isdir(path)==False):
        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
    
    now = datetime.now()
    formatnow = now.strftime('%H%M%S%f')
    filename = filename+'_'+formatnow+'.jpg'
    full_path = path+os.path.sep+filename
    if(not cropImages):
        cv2.imwrite(full_path,image)#Save the full image
    else:
        (x, y, w, h) = [int(v) for v in box]
        if(x > 0 and y > 0 and w > 0 and h > 0 ):
            #cv2.imshow('cropped', crop_img)
            crop_img = image[y:y+h, x:x+w]
            cv2.namedWindow('cropped', cv2.WINDOW_FREERATIO)
            cv2.imshow("cropped", crop_img)
            cv2.resizeWindow('cropped', w + x ,h + y )
            cv2.imwrite(full_path,crop_img)#Save track portion
        
    return filename

def save_xml_coordinates(imagesdirectory,folder,filename,image,box,label):
    pathfull = os.getcwd()+os.path.sep+imagesdirectory
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = filename
    ET.SubElement(root, 'path').text = pathfull
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'unknown'

    size = ET.SubElement(root, 'size')
    ET.SubElement(size , 'width').text  = str(int(image.shape[1]))
    ET.SubElement(size , 'height').text = str(int(image.shape[0]))
    ET.SubElement(size , 'depth').text  = str(3)
    
    ET.SubElement(root, 'segmented').text = str(0)
    
    object = ET.SubElement(root, 'object')
    ET.SubElement(object , 'name').text       = label
    ET.SubElement(object , 'pose').text       = 'Unspecified'
    ET.SubElement(object , 'truncated').text  = str(0)
    ET.SubElement(object , 'difficult').text  = str(0)
    
    bndbox = ET.SubElement(object, 'bndbox')
    (x, y, w, h) = [int(v) for v in box]
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    ET.SubElement(bndbox , 'xmin').text  = str(xmin)
    ET.SubElement(bndbox , 'ymin').text  = str(ymin)
    ET.SubElement(bndbox , 'xmax').text  = str(xmax)
    ET.SubElement(bndbox , 'ymax').text  = str(ymax)
    
    tree = ET.ElementTree(root)
    tree.write(pathfull+os.path.sep+filename.replace('.jpg', '.xml'))                
            
def save_yolo_format(imagesdirectory,filename,point_1, point_2, width, height):
    _ = os.path.sep
    pathfull = os.getcwd()+_+imagesdirectory 
    filename = pathfull +_+filename.replace('jpg','txt')
    f= open(filename,"w+")
    x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    items = map(str, [indexYoloLabel, x_center, y_center, x_width, y_height])    
    f.write(' '.join(items))
    f.close()
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')     


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v'  , '--video'  , type=str,help='path to input video file')
ap.add_argument('-t'  , '--tracker', type=str, default='mil',help='OpenCV object tracker type (csrt,kcf,boosting,mil,tld,medianflow,mosse,goturn)')
ap.add_argument('-l'  , '--label'  , type=str, required=True, help='Label to image')
ap.add_argument('-p'  , '--percentsizeimage', type=int, default = 100,help='Percent of original image´s size')
ap.add_argument('-a'  , '--augmentation'    , type=int, default='0',help='Augmentation images samples. 0 = true    1 = false')
ap.add_argument('-al' ,'--augmentlevel'    , type=int, default='0',help='Augmentation level. 0 = soft  1 = medium 2 = full ')
ap.add_argument('-d'  , '--imagesdirectory' , type=str, default = 'imagens', help='Folder name that the outputs will be saved')
ap.add_argument('-dbg','--debug' , type=str2bool, default = False, help='Show boundbox, coordinates and labels in images. This is not required')
ap.add_argument('-voc'  ,'--vocxml' , type=str2bool, default = False, help='Generate pascal voc xml file to training')
ap.add_argument('-y'  ,'--yolotxt' , type=str2bool, default = False, help='Generate yolo txt file to training')
ap.add_argument('-ily'  ,'--indexlabelyolo' , type=int, default = 0, help='Define the index to label for multiclass detection training')
ap.add_argument('-c'  ,'--cropimages' , type=str2bool, default = False, help='Generate crop images in image directory')
ap.add_argument('-dest'  ,'--desting' , type=str)
args = vars(ap.parse_args())

video               = args['video']
tracker             = args['tracker']
label               = args['label']
percentsizeimage    = args['percentsizeimage']
augmentation        = args['augmentation']
augmentlevel        = args['augmentlevel']
imagesdirectory     = args['imagesdirectory']
isdebugmode         = args['debug']
isgeneratevocformat = args['vocxml'] #gera o Pascal Voc
isgenerateyoloformat= args['yolotxt'] #gera o Yolo txt
indexYoloLabel      = args['indexlabelyolo'] #já gera o arquivo do yolo com o índice daquele label(objeto) que deseja treinar
cropImages          = args['cropimages']
desting             = args['desting']
# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(tracker.upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt":       cv2.TrackerCSRT_create,
        "kcf":        cv2.TrackerKCF_create,
        "boosting":   cv2.TrackerBoosting_create,
        "mil":        cv2.TrackerMIL_create,
        "tld":        cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse":      cv2.TrackerMOSSE_create,
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[tracker]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    print('[INFO] starting video file '+args['video'])
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
qtdSamples = 0

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    #frame = rescale_images(percentsizeimage,frame)
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            frameclone = frame.copy()

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),(0, 255, 0), 2)
            cv2.putText(frame, label,(x-10, y-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, ("P1({},{})".format(x,y)),(x-60, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, ("P2({},{})".format(x,ymax)),(x-60, y+h+30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, ("P3({},{})".format(xmax,y)),(x+w+10, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, ("P4({},{})".format(xmax,ymax)),(x+w-10, y+h+30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
            if isdebugmode:
                filename = save_images(imagesdirectory,label,frame)
                if isgeneratevocformat:
                    save_xml_coordinates(imagesdirectory,desting,filename,frame,box,label)
                if isgenerateyoloformat:
                    save_yolo_format(imagesdirectory,filename,(xmin, ymin), (xmax, ymax), W, H)
            else: 
                filename = save_images(imagesdirectory,label,frameclone)
                if isgeneratevocformat:
                    save_xml_coordinates(imagesdirectory,desting,filename,frameclone,box,label)
                if isgenerateyoloformat:
                    save_yolo_format(imagesdirectory,filename,(xmin, ymin), (xmax, ymax), W, H)
            
            qtdSamples +=1
            
        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
            ("Recording", "Yes" if success else "No"),
            ("Debugging mode", "Yes" if isdebugmode else "No"),
            ("Qtd samples", "{}".format(qtdSamples)),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            #cv2.rectangle(frame, (0, H-100), (W, H),(0, 255, 0), -1)
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # show the output frame
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
    
    frame = rescale_images(percentsizeimage,frame)
    cv2.imshow("Frame", frame)


# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

#os.system("createTrainngandTestImages")
# close all windows
cv2.destroyAllWindows()