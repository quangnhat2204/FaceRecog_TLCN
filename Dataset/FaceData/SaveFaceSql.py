import cv2
import numpy as np
import pyodbc
import os

def insertOrUpdate(id, name):
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                      'Server=DESKTOP-9E4M4BN\QUANGNHAT;'
                      'Database=Arc_Face;'
                      'Trusted_Connection=yes;')

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM People')

    isRecordExist = 0

    for row in cursor:
        isRecordExist = 1

    if(isRecordExist == 0):
        query = "INSERT INTO People(ID,Name) VALUES ("+ str(id) + ",'"+ str(name)+ "')"

    else:
        query = "UPDATE People SET Name='"+ str(name)+"'WHERE ID="+str(id)

    conn.execute(query)
    conn.commit()
    conn.close()

#load tv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

#insert to db
id = input("Enter your ID: ")
name = input("Enter your name: ")

insertOrUpdate(id,name)

sampleNum = 0

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,225),2)

        if not os.path.exists('raw/'+str(name) + '_' +str(id)):
            os.makedirs('raw/'+str(name) + '_' +str(id))
        sampleNum += 1
        cv2.imwrite('raw/'+str(name) + '_' +str(id)+'/User.'+str(id)+'.'+str(sampleNum)+ '.jpg', frame[y:y+h, x: x+w])
      #  cv2.imwrite('dataSet/User.'+str(id)+'.'+str(sampleNum)+ '.jpg', frame[y:y+h, x: x+w])
        

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if sampleNum > 50 :
        break;

cap.release()
cv2.destroyAllWindows()

