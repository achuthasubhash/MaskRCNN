import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names

capture = cv2.VideoCapture('videofile.mp4') #read video
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), #get width, height
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked.avi', codec, 60.0, size) #write video,fps, frame size

while(capture.isOpened()):
    ret, frame = capture.read()  
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0) #frame send to model and get result
        r = results[0]
        frame = display_instances( 
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)  #output writing to file
        cv2.imshow('frame', frame) #display frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
