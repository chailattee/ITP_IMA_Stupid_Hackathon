# import the necessary packages
from imutils.video import VideoStream
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from face_utils import MOUTH_AR_THRESH, draw_mouth, get_mouth_loc_with_height, mouth_aspect_ratio


orb = cv2.ORB_create(nfeatures=1000, fastThreshold=5, edgeThreshold=10)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
# time.sleep(2.0)

def checkKey(dict, key):
    return key in dict.keys()

i = 0 # frame counter
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to greyscale

    frame = vs.read()
    if frame is None:
        break

    i += 1

    enhanced = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
    frame = imutils.resize(frame, width=500)

    result = get_mouth_loc_with_height(enhanced)
    tongue_is_out = False

    if "error" not in result:
        shape = result['shape']
        frame = draw_mouth(frame, shape)
        len_kp = 0
        
        # mouth aspect ratio to determine if mouth is open
        mouthMAR = mouth_aspect_ratio(shape) 

        if (mouthMAR > MOUTH_AR_THRESH) :
            mX, mY, mW, mH = result['mouth_x'], result['mouth_y'], result['mouth_w'], result['mouth_h']
            iMY = result['inner_mouth_y']
            roi = enhanced[iMY:mY+mH, mX:mX + mW]

            # resize the mouth region to a standard size
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # Detect keypoints (tongue has texture, empty mouth doesn't)
            kp = orb.detect(roi,None)
            len_kp = len(kp)
            ''' Debugging: show keypoints in mouth ROI '''
            # print(f"Keypoints found: {len(kp)}") 

            # boolean: tongue is out if mouth is open + keypoints found
            if len(kp) > 30: # adjust threshold as needed
                tongue_is_out = True

        status_text = "TONGUE OUT" if tongue_is_out else "NO TONGUE"
        color = (0, 0, 255) if tongue_is_out else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "mouth aspect ratio: {:.2f}".format(mouthMAR), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "keypoints: {}".format(len_kp), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Tongue Detection", frame)
           
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"): 
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()