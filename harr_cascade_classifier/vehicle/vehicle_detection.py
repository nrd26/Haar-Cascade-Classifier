import cv2 as cv

cascade_src = 'cars.xml'
# video_src = 'video1.avi'
video_src = 'video2.avi'

cap = cv.VideoCapture(video_src)
car_cascade = cv.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)      
    
    cv.imshow('video', img)
    
    if cv.waitKey(25) == 27:
        break

cv.destroyAllWindows()
