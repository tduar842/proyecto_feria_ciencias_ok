import cv2
# ----------- lectura modelo dnn -----------
# arquitectura
prototxt = "model/model_1/MobileNetSSD_deploy.prototxt.txt"
# pesos
model = "model/model_1/MobileNetSSD_deploy.caffemodel"
# etiquetas
classes = {0:"background", 1:"aeroplane", 2:"bicycle",
          3:"bird", 4:"boat",
          5:"bottle", 6:"bus",
          7:"car", 8:"cat",
          9:"chair", 10:"cow",
          11:"diningtable", 12:"dog",
          13:"horse", 14:"motorbike",
          15:"person", 16:"pottedplant",
          17:"sheep", 18:"sofa",
          19:"train", 20:"tvmonitor"}
# carga de modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)
# ----------- lectura de la imagen y preprocesado -----------
cap = cv2.VideoCapture("media/9dejulio_video.mp4")
while True:
     ret, frame, = cap.read()
     if ret == False:
          break
     
     height, width, _ = frame.shape
     frame_resized = cv2.resize(frame, (300, 300))

     # crear un blob
     blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

     #print("blob.shape:", blob.shape)
     # ----------- detección y predicción -----------
     net.setInput(blob)
     detections = net.forward()
     for detection in detections[0][0]:
          #print(detection)
          if detection[2] > 0.45:
               label = classes[detection[1]]
               #print("Label:", label)
               box = detection[3:7] * [width, height, width, height]
               x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
               cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
               cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
               cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)
     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) & 0xFF == 27:
          break
cap.release()
cv2.destroyAllWindows()