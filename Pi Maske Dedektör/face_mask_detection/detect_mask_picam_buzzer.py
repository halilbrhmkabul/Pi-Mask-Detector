
import RPi.GPIO as GPIO
 
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
rly = 18
V = 0

GPIO.setup(rly, GPIO.OUT)
GPIO.output(rly, GPIO.LOW)

from gpiozero import Buzzer, LED

buzzer = Buzzer(21)
red = LED(14)
green = LED(15)




from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os



def detect_and_predict_mask(frame, faceNet, maskNet):
	# cerceve boyutları ayarlama
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
    #yüzleri algıla 
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# yüz listesi ve ağdaki tahmin listesi
	faces = []
	locs = []
	preds = []

	# algıları denetle
	for i in range(0, detections.shape[2]):
		# algılama ile olasılık belirtme
		confidence = detections[0, 0, i, 2]

		# düşük seviyedeki algılamaların tespiti
		if confidence > args["confidence"]:
			# x,y koordinatları belirler
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# çerçeve boyutu belirleme
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# boyutlandırma
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# yüz sınırlama
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# en az 1 yüz algılandığında yapılacak işlem
	if len(faces) > 0:
		# yüzleri aynı anda tahmin etme
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# yüz konumlarına karşılık gelen yer
	return (locs, preds)

# değişken ayrıştırıcı oluşturup, ayrıştırma işlemi
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="yuz dedektoru dizinine giden yol")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="egitilmis yuz modeli dedektörü dizinine giden yol")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="düsük seviye algılamaları filtrelemek")
args = vars(ap.parse_args())

print("[BILGI] maske dedektoru modeli yukleniyor...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


print("[BILGI] maske dedektoru modeli yukleniyor...")
maskNet = load_model(args["model"])

print("[BILGI] canli yayin baslatiliyor...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)



#canlı video uzerindeki karelerin dongusu
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
        
		if mask > withoutMask:
            
			label = "Tesekkurler. Maske Takili"
			color = (0, 255, 0)
			buzzer.off()
			red.off()
			green.on()
			GPIO.output(rly, GPIO.HIGH) # kapi kilit acik
			
		else:
            
			label = "Maske Takili Degil, Lutfen Takin"
			color = (0, 0, 255)
			buzzer.on()
			green.off()
			red.on()
			GPIO.output(rly, GPIO.LOW) # kapi kilit kapalı
				
		
		
		cv2.putText(frame, label, (startX-50, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	
	cv2.imshow("Yuz Maskesi Dedektoru", frame)
	key = cv2.waitKey(1) & 0xFF
	
	#cikis icin q harfine basılır
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
