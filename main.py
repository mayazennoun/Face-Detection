import cv2

# Charger le classificateur de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ouvrir la caméra (0 = webcam par défaut)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Lire une frame vidéo
    ret, img = cap.read()
    if not ret:
        break  # Arrêter si la capture échoue

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Afficher l'image
    cv2.imshow('Détection de Visages', img)

    # Ajout de cv2.waitKey(1) pour éviter que le programme se bloque
    if cv2.waitKey(1) == ord('q'):  
        break  # Optionnel : permet de quitter avec la touche 'q'

    # Vérifier si la fenêtre a été fermée
    if cv2.getWindowProperty('Détection de Visages', cv2.WND_PROP_VISIBLE) < 1:
        break  # Sortir de la boucle si la fenêtre est fermée

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()



