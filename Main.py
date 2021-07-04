import cv2
import face_recognition
import numpy as nm

imgMullai = face_recognition.load_image_file("TestImages/test1.jpg")
imgMullai = cv2.cvtColor(imgMullai,cv2.COLOR_RGB2GRAY)
cv2.imshow("Mullai Rajan",imgMullai)
imgHari = face_recognition.load_image_file("TestImages/test.jpg")
imgHari = cv2.cvtColor(imgHari,cv2.COLOR_RGB2GRAY)
cv2.imshow("Hari",imgHari)

cv2.waitKey(0)