import numpy as np
import cv2


cat_cascade = cv2.CascadeClassifier('cats.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cats = cat_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in cats:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = gray[y:y + h, x:x + w]


    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


