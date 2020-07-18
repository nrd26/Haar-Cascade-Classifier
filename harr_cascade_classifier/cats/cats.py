import numpy as np
import cv2 as cv


cat_cascade = cv.CascadeClassifier('cats.xml')
cap = cv.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    cats = cat_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in cats:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = gray[y:y + h, x:x + w]


    cv.imshow('img', img)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()


