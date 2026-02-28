# import the necessary packages
from imutils.video import VideoStream
import imutils
import cv2
import numpy as np
# from matplotlib import pyplot as plt
import math
from face_utils import MOUTH_AR_THRESH, draw_mouth, get_mouth_loc_with_height, mouth_aspect_ratio

orb = cv2.ORB_create(nfeatures=1000, fastThreshold=5, edgeThreshold=10)
len_kp=0

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
def is_tongue_out(shape, frame, mouth_data):
    try: 
        tongue_is_out = False
        if frame is None: return False
        
        enhanced = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        
        # calculate tilt angle 
        dx = shape[8][0] - shape[27][0]
        dy = shape[8][1] - shape[27][1]
        tilt_angle = math.degrees(math.atan2(dx, dy))

        (h, w) = enhanced.shape[:2]
        center = (float(shape[27][0]), float(shape[27][1]))
        M = cv2.getRotationMatrix2D(center, tilt_angle, 1.0)
       
        leveled_frame = cv2.warpAffine(enhanced, M, (w, h))

        leveled_result = get_mouth_loc_with_height(leveled_frame)
        if "error" in leveled_result: return False

        # mouth aspect ratio to determine if mouth is open
        mouthMAR = mouth_aspect_ratio(shape) 
        if (mouthMAR > MOUTH_AR_THRESH) :
            mX = leveled_result['mouth_x']
            mY = leveled_result['mouth_y']
            mW = leveled_result['mouth_w']
            mH = leveled_result['mouth_h']
            iMY = leveled_result['inner_mouth_y']

            roi = leveled_frame[iMY:mY+mH, mX :mX + mW ]

            if roi.size == 0: return False

            roi_final = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            kp = orb.detect(roi_final, None)

            # Detect keypoints (tongue has texture, empty mouth doesn't)
            # kp = orb.detect(roi,None)
            ''' Debugging: show keypoints in mouth ROI '''
            print(f"Keypoints found: {len(kp)}") 

            # boolean: tongue is out if mouth is open + keypoints found
            if len(kp) > 5: # ADJUST THRESHOLD ONCE OVALS ARE DRAWN
                tongue_is_out = True
        return tongue_is_out
    except Exception as e:
        print(f"Error in is_tongue_out: {e}")
        return False

'''
# loop over the frames from the video stream
vs = VideoStream(0).start()

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
'''