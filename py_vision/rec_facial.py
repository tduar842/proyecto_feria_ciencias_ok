import cv2

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Leyendo el modelo
face_recognizer.read('modeloLBPHFace.xml')

class rec_facial:
    def __init__(self):
        pass
	
    def check(frame):
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        return auxFrame, face_recognizer, faces
	