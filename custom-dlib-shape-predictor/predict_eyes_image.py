# USAGE
# python predict_eyes.py --shape-predictor eye_predictor.dat

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# # initialize the video stream and allow the cammera sensor to warmup
# print("[INFO] camera sensor warming up...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)


image = cv2.imread('/home/hunglv/Desktop/Face-Side.jpg')

frame = image

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
xmin,ymin,xmax,ymax = 0,0,gray.shape[1],gray.shape[0]
rect = dlib.rectangle(left=xmin, top=ymin, right=xmax, bottom=ymax)
# rect = [(xmin,ymin),(xmax,ymax)]
shape = predictor(gray,rect)
shape = face_utils.shape_to_np(shape)

# loop over the (x, y)-coordinates from our dlib shape
# predictor model draw them on the image
for (sX, sY) in shape:
	cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)



# show the frame
cv2.imshow("Frame", frame)
key = cv2.waitKey(0) & 0xFF

# if the `q` key was pressed, break from the loop
if key == ord("q"):
	quit()
