import cv2
# ------------ leer modelo dnn ------------
# arquitectura modelo
prototxt = "model/model_2/deploy.prototxt"
# pesos
model = "model/model_2/res10_300x300_ssd_iter_140000.caffemodel"
# carga modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)
# ------- sabes que flaco ya me canse de hacer esto chau -------
cap = cv2.VideoCapture("media/9dejulio_video.mp4")
while True:
     ret, frame = cap.read()
     if ret == False:
          break
     height, width, _ = frame.shape
     frame_resized = cv2.resize(frame, (300, 300))
     # Create a blob
     blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104, 117, 123))
     # ------- DETECTIONS AND PREDICTIONS ----------
     net.setInput(blob)
     detections = net.forward()
     #print("detections.shape:", detections.shape)
     for detection in detections[0][0]:
          #print("detection:", detection)
          if detection[2] > 0.5:
               box = detection[3:7] * [width, height, width, height]
               x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
               cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
               cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)
     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) & 0xFF == 27:
          break
cap.release()
cv2.destroyAllWindows()