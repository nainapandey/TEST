# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:49:13 2020

@author: lenovo
"""


from keras.models import load_model
import cv2
import numpy as np
import tkinter 
from tkinter import messagebox
import smtplib


# Initialize Tkinter
root= tkinter.Tk()
root.withdraw()
"""connection=pymysql.connect(host="localhost",port=3306, user='root', password="praku", db="register")
try:
    with connection.cursor() as cursor:
         query= "SELECT * FROM users WHERE Email= '%s';" % (x)
         print("SQL Test")
         data_1= cursor.execute(query)
         records= cursor.fetchone()
finally:
           connection.close()"""

#Load trained deep learning model
model = load_model(r'C:\Users\lenovo\seriouslymridula.h5')

#classifier to detect face
face_det_classifier= cv2.CascadeClassifier(r'C:\Users\lenovo\haarcascade_frontalface_default.xml')

#Capture Video
vid_source = cv2.VideoCapture(0)

# Dictionaries containing details of Wearing Mask and Color of rectangle around face. If wearing mask then green and 
# if not wearing mask then color of rectangle around face would be red
text_dict={0: 'Mask ON',1:'No Mask'}
rect_color_dict={0:(0,255,0), 1:(0,0,255)}

SUBJECT = "Subject"
TEXT = "One Visitor violated Face Mask Policy. See in the camera to recognize user. A Person has been detected without a face mask."

#While Loop to continuously detect camera feed
while(True):
    ret, img = vid_source.read()
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_det_classifier.detectMultiScale(grayscale_img, 1.3,5)
    
    for (x,y,w,h) in faces:
        
        face_img = grayscale_img[y:y+w, x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,112,112,1))
        result=model.predict(reshaped_img)
        
        label=np.argmax(result, axis=1)[0]
        
        cv2.rectangle(img,(x,y), (x+w, y+h), rect_color_dict[label], 2)
        cv2.rectangle(img, (x,y-40), (x+w,y), rect_color_dict[label],-1)
        cv2.putText(img, text_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        
#If label=1 then it means wearing no  mask and 0 means wearing mask
        if (label==1):
    #throw a warning message to tell a user to wear a mask if not wearing one this will stay open and no access will be given he/she wears the mask  
            messagebox.showwarning("Warning","Access Denied. Please wear a face mask.")
    #send an email to the administrator if access denied/user not wearing face mask
            message='Subject:{}\n\n{}'.format(SUBJECT,TEXT)
            mail=smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login('projectminor20@gmail.com','nishi123!@#' )
            to=['projectminor20@gmail.com', 'dashoreprakrati@gmail.com']
            #to=['projectmionr20@gmail.com', x]
            mail.sendmail('projectminor20@gmail.com', to,message)
            mail.close
            #Model=load_model(r'E:\study material\Minor\train_dump.json')
        
        else:
            pass
            break
    cv2.imshow('Live Video Feed',img)
    key=cv2.waitKey(1)
    if(key==27):
        break
    
cv2.destroyAllWindows()
vid_source.release()

    