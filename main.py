import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Détection de Visages', img)

    if cv2.waitKey(1) == ord('q'):  
        break  


    if cv2.getWindowProperty('Détection de Visages', cv2.WND_PROP_VISIBLE) < 1:
        break  


cap.release()
cv2.destroyAllWindows()



