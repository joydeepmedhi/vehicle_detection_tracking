import cv2 as cv

#change based on source
cap = cv.VideoCapture(1)

face_front_cascade = cv.CascadeClassifier("models/haarcascade_frontalface_alt.xml")    
tracker = cv.TrackerKCF_create()
bbox = ()
initTracker = False  # Flag to indicate if tracker has been initialized

while True:
    ret, frame = cap.read()

    # Press 's' to capture the face
    
    if cv.waitKey(20) & 0xFF == ord("s"):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_front_cascade.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=3)
        if len(faces) > 0:  # Check if at least one face was detected
            x, y, w, h = faces[0]  # Use the first detected face
            colour = (0, 0, 255)
            stroke = 2
            cv.rectangle(frame, (x, y), (x+w, y+h), colour, stroke)
            bbox = (x, y, w, h)
            tracker = cv.TrackerKCF_create()  # Overwrite old tracker
            initTracker = True
            tracker.init(frame, bbox)  # Initialize the tracker


    # Trace face and draw box around it
    if initTracker:
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # Show result
    cv.imshow("frame", frame)

    # Press ESC to exit
    if cv.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
