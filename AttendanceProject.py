import csv
import cv2
import numpy as np
import face_recognition
import os

from datetime import datetime

path = "C:/Users/Anom/PycharmProjects/Face-Recognition-Project/ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#finding encodings
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def pasname(name):
    #csv sheet path given and appended names in it.
    with open('C:\\Users\\Anom\\PycharmProjects\\Face-Recognition-Project\\Attendance.csv','a',newline='') as f:

        writer= csv.writer(f)
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding completed')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
        success, img = cap.read()
        #because its real time capture, we wld reduce the size of image to speed up the process
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        #realtime image size has been divided by 4 using 0.25
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        #finding matches
        for encodeFace,faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches =face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)


            matchIndex = np.argmin(faceDis)
            print('matchIndex', matchIndex)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                pasname(name)

                cv2.imshow('Webcam', img)

        #press'esc' to close program
        if cv2.waitKey(1) :
            break
#release camera
cap.release()
cv2.destroyAllWindows()


