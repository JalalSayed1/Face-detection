#pip install opencv-python
import cv2

# Load the cacade:
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# To capture video:
cap = cv2.VideoCapture(0) # 0 means the default camera
# To use a video file as input:
# cap = cv2.VideoCapture("fileName.mp4")

while True:
    # read the frame:
    ret, frame = cap.read() # ret is the return value. returns False if the camera did not work 
    
    # concert to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the faces:
    # scaleFactor specifies how much the image size is reduced at each time (1.05 = not a lot, detects more faces, 1.4 = a lot and could miss some faces)
    # minNeighbors specifies how many neighbors each candidate rectangle should have to retain it (higher num means fewer detections)
    # minSize specifies minimum possible object size. Objects smaller than that are ignored.
    # maxSize specifies Maximum possible object size. Objects bigger than this are ignored.
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.05, minNeighbors=5, minSize=(50,50))
    print(faces)

    # draw the regtangle around each face:
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x ,y), (x+w, y+h), (0, 255, 255), 2)

        # eyes are in the face (obviously) so we use our frame to search for thee eyes. roi means region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.04, 10)
        for (ex, ey, ew, eh) in eyes:
            # to draw a regrangle around the eyes. we use roi_colour because we detected the eyes relative to the face area not the whole frame
            cv2.rectangle(roi_colour, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)


    # display:
    cv2.imshow("frame", frame)

    #* stop if escape key is pressed:
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# release the videoCapture object:
cap.release()



