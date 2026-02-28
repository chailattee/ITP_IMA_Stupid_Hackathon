# import the necessary packages
from imutils.video import VideoStream
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from face_utils import MOUTH_AR_THRESH, draw_mouth, get_mouth_loc_with_height, mouth_aspect_ratio

orb = cv2.ORB_create(nfeatures=1000, fastThreshold=5, edgeThreshold=10)

def check_tongue_for_player(shape, frame, mouth_data):
    """Helper: Detect if tongue is out for a specific player given their shape and mouth data"""
    try:
        mouthMAR = mouth_aspect_ratio(shape)
        
        if mouthMAR > MOUTH_AR_THRESH:
            mX = mouth_data['mouth_x']
            mY = mouth_data['mouth_y']
            mW = mouth_data['mouth_w']
            mH = mouth_data['mouth_h']
            iMY = mouth_data['inner_mouth_y']
            
            # Extract mouth region
            roi = frame[iMY:mY+mH, mX:mX + mW]
            
            if roi.size == 0:
                return False
            
            # Detect keypoints (tongue has texture)
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            kp = orb.detect(roi, None)
            
            # Tongue out if enough keypoints found
            return len(kp) > 50
        return False
    except:
        return False

# Function to detect if tongue is out, True = out, False = not out
def is_tongue_out(frame):
    if frame is None:
        return False
    
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
            print(f"Keypoints found: {len(kp)}") 

            # boolean: tongue is out if mouth is open + keypoints found
            if len(kp) > 50: # ADJUST THRESHOLD ONCE OVALS ARE DRAWN
                tongue_is_out = True
        return tongue_is_out
    else:
        print("Error in mouth detection: ", result["error"])
        return False

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(0).start()

# loop over the frames from the video stream
while True:
    frame = vs.read()
    if frame is None:
        break

    tongue_is_out = is_tongue_out(frame)
    status_text = "TONGUE OUT" if tongue_is_out else "NO TONGUE"
    color = (0, 0, 255) if tongue_is_out else (0, 255, 0)
    
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # cv2.putText(frame, "keypoints: {}".format(len_kp), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Tongue Detection", frame)
           
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"): 
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()