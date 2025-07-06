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

class obj_det:
    def __init__(self):  
        pass

    def check(frame):
        # ----------- lectura de la imagen y preprocesado -----------
        #cap = cv2.VideoCapture("media/9dejulio_video.mp4")
        height, width, _ = frame.shape
        frame_resized = cv2.resize(frame, (300, 300))

        # crear un blob
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

            #print("blob.shape:", blob.shape)
            # ----------- detección y predicción -----------
        net.setInput(blob)
        detections = net.forward()
        return classes, detections, (width, height)