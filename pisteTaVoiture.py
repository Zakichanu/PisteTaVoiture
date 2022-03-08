import cv2
import numpy as np

# Capture video frames
snippet = cv2.VideoCapture('VideoSnippet/carv2.mp4')

# Using cascade classifier technique to capture if the object is a car
cascadeAlgorithm = cv2.CascadeClassifier('classfierCascadeCar.xml')

# Detecting cars
while True:
    ret, frames = snippet.read()
    #gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    # Applying algorith to detect cars, using B&W filter with 1.1 as ScaleFactor and 9 neighbors minimum
    cars = cascadeAlgorithm.detectMultiScale(frames, 1.05, 13)

    # Drawing rectangle to around cars
    for(coord_x, coord_y, width, height) in cars:
        cv2.rectangle(frames, (coord_x, coord_y), (coord_x+width, coord_y+height), color=(0, 0, 255), thickness=2)
        cv2.putText(frames, 'Car', (coord_x, coord_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Car Detection System', frames)
    k = cv2.waitKey(33)
    if k == 27:
        break
    cv2.destroyAllWindows()