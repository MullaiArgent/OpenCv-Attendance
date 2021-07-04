import cv2
import face_recognition

imgMullai = face_recognition.load_image_file("TestImages/test1.jpg")
imgMullai = cv2.cvtColor(imgMullai, cv2.COLOR_BGR2RGB)


facelocMullai = face_recognition.face_locations(imgMullai)[0]
encodMullai = face_recognition.face_encodings(imgMullai)[0]
cv2.rectangle(imgMullai, (facelocMullai[3], facelocMullai[0]), (facelocMullai[1], facelocMullai[2]), (0, 0, 0, 255), 3)
cv2.imshow("Mullai Rajan", imgMullai)
imgHari = face_recognition.load_image_file("TestImages/test.jpg")
imgHari = cv2.cvtColor(imgHari, cv2.COLOR_BGR2RGB)


facelocHari = face_recognition.face_locations(imgHari)[0]
encodHari = face_recognition.face_encodings(imgHari)[0]
cv2.rectangle(imgHari, (facelocHari[3], facelocHari[0]), (facelocHari[1], facelocHari[2]), (0, 0, 0, 255), 3)
cv2.imshow("Hari", imgHari)
cv2.waitKey(0)
