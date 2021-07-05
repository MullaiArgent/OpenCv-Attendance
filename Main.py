import cv2
import face_recognition
import os


path = 'Images'
images = []
images_name = []

mylist = os.listdir(path)

for a in mylist:
    curimg = cv2.imread(f'{path}/{a}')
    images.append(curimg)
    images_name.append(os.path.splitext(a)[0])

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img) [0]
        encodelist.append(encode)
    print("Encoding Completed")
    return encodelist


encodelist = findencodings(images)


cap  = cv2.VideoCapture(0)

while(True):
    success, img = cap.read()


