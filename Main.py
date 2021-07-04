import cv2
import face_recognition

imgMullai_f = face_recognition.load_image_file("TestImages/mullai_f.jpg")
imgMullai_f = cv2.cvtColor(imgMullai_f, cv2.COLOR_BGR2RGB)

facelocMullai = face_recognition.face_locations(imgMullai_f)[0]
encodMullai = face_recognition.face_encodings(imgMullai_f)[0]
cv2.rectangle(imgMullai_f, (facelocMullai[3], facelocMullai[0]), (facelocMullai[1], facelocMullai[2]), (0, 0, 0, 255), 3)

imgMullai_t = face_recognition.load_image_file("TestImages/mullai_t.jpg")
imgMullai_t = cv2.cvtColor(imgMullai_t, cv2.COLOR_BGR2RGB)

facelocmullai_t = face_recognition.face_locations(imgMullai_t)[0]
encodHari = face_recognition.face_encodings(imgMullai_t)[0]
cv2.rectangle(imgMullai_t, (facelocmullai_t[3], facelocmullai_t[0]), (facelocmullai_t[1], facelocmullai_t[2]), (0, 0, 0, 255), 3)









cv2.imshow("Mullai Rajan", imgMullai_f)
cv2.imshow("Hari", imgMullai_t)
cv2.waitKey(0)
