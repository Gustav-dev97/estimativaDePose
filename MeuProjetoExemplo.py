import cv2
import time
# Realizar o import para usar o Módulo (Módulo deverá estar na mesma pasta do projeto):
import PoseModulo as pm

capt = cv2.VideoCapture('VideosPose/BalletTest.mp4')
tAnterior = 0
detector = pm.DetectorPose()
while True:
    success, img = capt.read()
    img = detector.acharPose(img)
    lmLista = detector.acharPosicao(img)

    # Apontar o indice para rastrear o ponto desejado, ex: print(lmLista[14])
    if len(lmLista) != 0:
        print(lmLista[14])
        # Exemplo de demarcação do ponto:
        cv2.circle(img, (lmLista[14][1], lmLista[14][2]), 15, (0, 0, 255), cv2.FILLED)

    tAtual = time.time()
    fps = 1 / (tAtual - tAnterior)
    tAnterior = tAtual

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Imagem", img)
    cv2.waitKey(1)
