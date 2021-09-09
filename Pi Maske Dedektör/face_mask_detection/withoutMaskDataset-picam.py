import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

name = 'without_mask' #maskesiz klasörüne ekle

cam = PiCamera()
cam.resolution = (512, 304)
cam.framerate = 10
rawCapture = PiRGBArray(cam, size=(512, 304))
    
img_counter = 0

while True:
    for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Fotograf cekmek icin space tusuna basin", image)
        rawCapture.truncate(0)
    
        k = cv2.waitKey(1)
        rawCapture.truncate(0)
        if k%256 == 27:
            break
        elif k%256 == 32:
            img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, image)
            print("{} written!".format(img_name))
            img_counter += 1
            
    if k%256 == 27:
        print("kapanis...")
        break

cv2.destroyAllWindows()
