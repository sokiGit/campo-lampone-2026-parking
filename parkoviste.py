
from collections import deque

import cv2
import numpy as np
import requests





cap = cv2.VideoCapture("http://gateson.lan/stream")

while(True):
    _, image = cap.read()


    cv2.imshow("Output", image)

    # cv2.imshow("Output", true_red_only)


    _, image1 = cv2.threshold(image[:,:,0],127,253,cv2.THRESH_BINARY)
    _, image2 = cv2.threshold(image[:,:,1],127,253,cv2.THRESH_BINARY)
    _, image3 = cv2.threshold(image[:,:,2],127,253,cv2.THRESH_BINARY)

    image_return = cv2.bitwise_or(image2,image3)
    image_not  = cv2.bitwise_not(image_return)
    image_end = cv2.bitwise_and(image_return, image1)


    print (image.shape)

    cv2.imshow("output", image_return)

    if cv2.waitKey(1) == ord('q'):
        break

    from collections import deque
    dq = deque([1, 2])

    print(dq[0])
    print(dq[-1])
