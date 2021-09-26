
from logging import exception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2 
import os
from imutils.video import VideoStream
import imutils
import time
import mysql.connector
import random
# import dlib


database = mysql.connector.connect(host="covid-assist-db.cdbjavxo0vob.us-east-2.rds.amazonaws.com", username="admin", password="admin1234", database="covidAssist")
mycursor = database.cursor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_and_predict_mask(frame,faceNet,maskNet):
    #grab the dimensions of the frame and then construct a blob
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    faceNet.setInput(blob)
    detections=faceNet.forward()

    #initialize our list of faces, their corresponding locations and list of predictions
    
    faces=[]
    locs=[]
    preds=[]
    
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
    
        if confidence>0.5:
        #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
        
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
        
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        #only make a predictions if atleast one face was detected
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=12)
        
        return (locs,preds)



prototxtPath=os.path.sep.join([r'D:\Projects\Face Mask Detector_3-Final_2\face_detector','deploy.prototxt'])
weightsPath=os.path.sep.join([r'D:\Projects\Face Mask Detector_3-Final_2\face_detector','res10_300x300_ssd_iter_140000.caffemodel'])


faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)


maskNet=load_model(r'D:\Projects\Face Mask Detector_3-Final_3\mobileNet_v2.model')


vs=VideoStream(src=0).start()

start_time= time.time()
seconds = 15
count =0

c=0
while True :
    #grab the frame from the threaded video stream and resize it
    #to have a maximum width of 400 pixels
    frame=vs.read()
    frame=imutils.resize(frame,width=800)

    gray = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    
    #detect faces in the frame and preict if they are waring masks or not
    (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
    
    if(len(preds)):
        c = c+1
    else:
        c=0
    
    #loop over the detected face locations and their corrosponding loactions
    # zip means two together
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY)=box
        (mask,withoutMask)=pred
        
        #determine the class label and color we will use to draw the bounding box and text
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        if(c==8):
            place_id = random.randint(1,10)
            if(label=='No Mask'):
                status = 0
            elif (label=='Mask'):
                status = 1
            try:
                sql = "INSERT INTO facemask( place_id, date_time, facemask_status) values ( %s, now(), %s)"
                value = (place_id, status)
                mycursor.execute(sql, value)
                database.commit()
            except exception as e:
                print(e)
            print(label)
    


        #display the label and bounding boxes
        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        
    #show the output frame
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    
    if key==ord('q'):
        break
        
cv2.destroyAllWindows()
vs.stop()
