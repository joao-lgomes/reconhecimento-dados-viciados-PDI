import cv2
import numpy as np
from collections import deque
from scipy.stats import chisquare

min_threshold = 10
max_threshold = 200
min_area = 60
max_area = 135
min_circularity = 0.2
min_inertia_ratio = 0.55
video = cv2.VideoCapture('videoDices.mp4')
video.set(15, -4)
i=1

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.filterByInertia = True
params.minThreshold = min_threshold
params.maxThreshold = max_threshold
params.minArea = min_area
params.minCircularity = min_circularity
params.minInertiaRatio = min_inertia_ratio
params.filterByColor = False
detector = cv2.SimpleBlobDetector_create(params)
contador = 0
realTimeReadings = deque([0, 0, 0, 0, 0, 0, 0, 0, 0,], maxlen=10)
display = deque([0, 0], maxlen=10)


leituras={
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0
}


while True:
    ret, im = video.read()
    if not ret:
        break
    if ret == True:

        keypoints = detector.detect(im)
        if len(keypoints):
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("VIDEO", 540, 960)
            cv2.imshow("VIDEO", im_with_keypoints)
        if contador % 2== 0:
            reading = len(keypoints)
            realTimeReadings.append(reading)

            if realTimeReadings[-1] == realTimeReadings[-2] == realTimeReadings[-3] == realTimeReadings[-4]== realTimeReadings[-5]== realTimeReadings[-6]== realTimeReadings[-7]== realTimeReadings[-8]== realTimeReadings[-9]:
                display.append(realTimeReadings[-1])


            if display[-1] != display[-2] and display[-1] != 0:
                text = f"jogada n°{i} --> LADO {display[-1]}"
                leituras[display[-1]] = leituras[display[-1]] + 1
                i = i + 1
                print(text)

        contador += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
values=list(leituras.values())
# defining the table
# interpret p-value
print("\nA QUANTIDADE VEZES QUE CADA FACE SAIU, RESPECTIVAMENTE, É DE:")
print(values)
print("")
expected_values = np.full(6,(i-1)/6).tolist()
data = [values,expected_values]
chisq, p = chisquare(values, f_exp=expected_values)

print("Para verificar se os dados são viciados, foi utilizado o teste qui-quadrado unidirecional")
print("Esse teste verifica a hipótese nula de que os dados categóricos têm as frequências dadas. Ou seja:")
print("O teste é usado para comparar a distribuição de amostra observada com a distribuição de probabilidade esperada \n")

alpha = 0.05
print("Alpha representa o valor divisor de limites para verificar se o dado é viciado ou não.")
print("Após pesquisas sobre testes qui-quadrado, o melhor valor a ser utilizado é 0,05 \n")


print("O VALOR DO TESTE QUI-QUADRADO UNIDIRECIONAL FOI DE " + str(p))
if p <= alpha:
    print('LOGO, O DADO E VICIADO, pois o valor do teste deu inferior à 0,05')
else:
    print('LOGO, O DADO NãO E VICIADO, pois o valor do teste deu superior à 0,05')

cv2.destroyAllWindows()