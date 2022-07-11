import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
import eel
from math import sqrt

eel.init("static")


@eel.expose
def chkcoord(ax, ay):
    test_point = {"lat": ax, "lng": ay}
    radius = 5
    center_point = {"lat": 8.00877, "lng": 70.8888}
    a = center_point["lat"] - test_point["lat"]
    b = center_point["lng"] - test_point["lng"]
    c = sqrt(a * a + b * b)
    print(test_point)
    global ll
    if c < radius:
        ll = 1
    else:

        ll = 0


eel.start("index.html", block=False)
eel.sleep((10))
time.sleep(10)
path = "Training"
l = []
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f"\n{name},{dtString}")
def removedups():
    # with open("Attendance.csv",'r+') as f:
    #     myDataList                
    import pandas as pd
    file_name = "Attendance.csv"
    file_name_output = "Updated_Attendance.csv"

    df = pd.read_csv(file_name)
    df.drop_duplicates(subset=None, inplace=True)
    df.to_csv(file_name_output, index=False)

encodeListKnown = findEncodings(images)
print("Encoding Complete")
markAttendance("Kannan Ramu")

cap = cv2.VideoCapture(1)
removedups()

# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#     cv2.imshow("BGR2RGB", imgS)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             # print(name)
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(
#                 img,
#                 name,
#                 (x1 + 6, y2 - 6),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1,
#                 (255, 255, 255),
#                 2,
#             )
#             if name not in l:
#                 l.append(name)
#                 markAttendance(name)

#     cv2.imshow("Webcam", img)
#     cv2.waitKey(1)
