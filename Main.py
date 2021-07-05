import cv2
import face_recognition
import os
import numpy

path = 'Images'
images = []
images_name = []

mylist = os.listdir(path)
print("Path added.. Successfully")
counter = 0
for a in mylist:
    curimg = cv2.imread(f'{path}/{a}')
    images.append(curimg)
    images_name.append(os.path.splitext(a)[0])
    counter += 1
    print(counter,"files added...")
print("Totaly ,"counter,"Files found")

def findencodings(a):
    encodelist = []
    for img in a:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    print("Encoding Completed")
    return encodelist


encodelist = findencodings(images)
print("Initiating WebCam...")
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    faceCurframe = face_recognition.face_locations(imgs)
    encodeCurframe = face_recognition.face_encodings(imgs, faceCurframe)

    for encodeface, faceloc in zip(encodeCurframe, faceCurframe):
        matchs = face_recognition.compare_faces(encodelist, encodeface)
        facedis = face_recognition.face_distance(encodelist, encodeface)

        matchIndex = numpy.argmin(facedis)

        if matchs[matchIndex]:
            name = images_name[matchIndex].upper()
            print(name)

