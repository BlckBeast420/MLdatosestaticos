import cv2
import mediapipe as mp
import csv
import os

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Archivo CSV
archivo_csv = 'gestos.csv'
existe = os.path.isfile(archivo_csv)

# Crear archivo CSV si no existe
if not existe:
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(1, 22)] + [f"y{i}" for i in range(1, 22)] + ["letra"]
        writer.writerow(header)

# Captura de cámara
cap = cv2.VideoCapture(0)
print("Presiona una letra (A-Z) para guardar el gesto actual, o 'ESC' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append(lm.x)
            for lm in hand_landmarks.landmark:
                coords.append(lm.y)

            # Esperar entrada del usuario
            key = cv2.waitKey(1) & 0xFF
            if 65 <= key <= 90 or 97 <= key <= 122:  # Letras A-Z o a-z
                letra = chr(key).upper()
                with open(archivo_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(coords + [letra])
                print(f"Gesto guardado como letra: {letra}")
            elif key == 27:  # ESC para salir
                cap.release()
                cv2.destroyAllWindows()
                print("Recolección finalizada.")
                exit()

    cv2.imshow("Captura de gestos", frame)
