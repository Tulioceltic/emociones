import cv2
import os
import numpy as np

def emotionImage(emotion):
	# Emojis
	if emotion == 'angry': image = cv2.imread('Emojis/enojo.jpeg')
	if emotion == 'happy': image = cv2.imread('Emojis/felicidad.jpeg')
	if emotion == 'neutral': image = cv2.imread('Emojis/tristeza.jpeg')
	if emotion == 'sad': image = cv2.imread('Emojis/tristeza.jpeg')
	if emotion == 'surprise': image = cv2.imread('Emojis/sorpresa.jpeg')

	return image





def traer_imagen():
	emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	emotion_recognizer.read('modeloFisherFaces.xml')


	dataPath = 'train' #Cambia a la ruta donde hayas almacenado Data
	imagePaths = os.listdir(dataPath)

	cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

	faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(48,48),interpolation= cv2.INTER_CUBIC)
		result = emotion_recognizer.predict(rostro)
		
		if result[1] < 500:
			image = emotionImage(imagePaths[result[0]])
			nFrame = cv2.hconcat([frame,image])
		#else:
