import cv2
import face_recognition
import numpy
import os
import time

path = "Images"
Imgs = []
ImgNames = []
print("Path Added...")
mylist = os.listdir(path)
counter = 0
for a in mylist:
    curimg = cv2.imread(f'{path}/{a}')
    Imgs.append(curimg)
    ImgNames.append(os.path.splitext(a)[0])
    counter += 1
    print('\r', end="")
    time.sleep(0.5)
    print(counter, "file(s) founded...", end="")
print("Beginning Face Encoding")

counter = 0


def find_encoding(a):
    encode_list = []
    for img in Imgs:
        img = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
        globals()['counter'] += 1
        print('\r', end="")
        time.sleep(0.5)
        print(counter, "file(s) founded...", end="")
    return encode_list


encode_list = find_encoding(Imgs)
print("Initializing Cam...")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurloc = face_recognition.face_locations(imgS)
    encodeCur = face_recognition.face_encodings(imgS, faceCurloc)

    for encodeface, faceloc in zip(encodeCur, faceCurloc):
        matchs = face_recognition.compare_faces(encode_list, encodeface)
        facedis = face_recognition.face_distance(encode_list, encodeface)

        matchindex = numpy.argmin(facedis)

        if matchs[matchindex]:
            name = ImgNames[matchindex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
    cv2.imshow("wc", img)
    cv2.waitKey(1)