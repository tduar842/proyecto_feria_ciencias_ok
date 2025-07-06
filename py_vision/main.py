import cv2
from obj_det import *
from rec_facial import *

while True:
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    ret, frame_input, = cap.read()
    if ret == False:
        break

    #detección de objetos
    classes, detections, tupla_dimensiones = obj_det.check(frame=frame_input)
    width = tupla_dimensiones[0]
    height = tupla_dimensiones[1]
    for detection in detections[0][0]:
        #print(detection)
        if detection[2] > 0.45:
            label = classes[detection[1]]
            #print("Label:", label)
            box = detection[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_input, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame_input, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
            cv2.putText(frame_input, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)

    #detección facial

    auxFrame, face_recognizer, faces = rec_facial.check(frame=frame_input)
    for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame_input,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        
            # LBPHFace
            if result[1] < 70:
                cv2.putText(frame_input, "godines", (x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame_input, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame_input,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame_input, (x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow("Frame", frame_input)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    past_cap = cap #detestable, absolutamente repugnante, hambreadisimo el interpretador, no deberia ser vital estar en cache.



cap.release()
cv2.destroyAllWindows()
