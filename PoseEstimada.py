import cv2
import mediapipe as mp
import time

# Criar os Models para detectar a pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Carregar e exibir vídeo
capt = cv2.VideoCapture('VideosPose/GymnasticsTest.mp4')
tAnterior = 0

while True:
    success, img = capt.read()

    # Converter de BGR para RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Enviar imagem para o Model
    resultado = pose.process(imgRGB)
    # print(resultado.pose_landmarks)

    # Checar se o landmarks foi detectado, criar as marcações e linhas
    if resultado.pose_landmarks:
        mpDraw.draw_landmarks(img, resultado.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # Extrair informação de dentro do objeto através do ID
        for id, landmark in enumerate(resultado.pose_landmarks.landmark):
            altura, largura, channel = img.shape
            print(id, landmark)
            # Extrair o valor do pixel
            cx, cy = int(landmark.x * largura), int(landmark.y * altura)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Checar e corrigir FrameRate
    tAtual = time.time()
    fps = 1 / (tAtual - tAnterior)
    tAnterior = tAtual

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Imagem", img)
    cv2.waitKey(1)
