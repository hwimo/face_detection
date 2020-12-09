import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import winsound

model = load_model('C:\\Users\\hmku1\\객체지향\\일단 해보는거\\mask_data\\mask_detection_model.h5')
xml = 'C:\\Users\\hmku1\\Desktop\\haarcascades\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)

cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
cap.set(3,680) # 너비
cap.set(4,480) # 높이

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.05, 5)
    #print("Number of faces detected: " + str(len(faces)))

    if len(faces):
        for (x,y,w,h) in faces:
            fa = frame[y:y+h, x:x+w]
            fa_input = cv2.resize(fa,dsize = (224,224))
            fa_input = cv2.cvtColor(fa_input,cv2.COLOR_BGR2RGB)
            fa_input = preprocess_input(fa_input)
            fa_input = np.expand_dims(fa_input, axis = 0)

            mask, nomask = model.predict(fa_input).squeeze()

            if mask > nomask:
                color = (0,255,0)
                label = 'mask %d%%' % (mask * 100)
            else:
                color = (0,0,255)
                label = 'no mask %d%%' % (nomask * 100) 
                frequency = 2500
                duration = 1000
                #winsound.Beep(frequency,duration)

            


        cv2.rectangle(frame, pt1 = (x,y), pt2 = (x+w,y+h),color=color,thickness = 2)
        cv2.putText(frame, text = label, org = (x,y-10),fontScale = 0.8,fontFace = cv2.FONT_HERSHEY_SIMPLEX,color = color,thickness = 2)
        
    cv2.imshow('result', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()