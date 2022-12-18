import cv2
import numpy as np
from collections import deque
from scipy.stats import chisquare

video = cv2.VideoCapture('videoDices.mp4')
video.set(15, -4)
contador = 0
realTimeReadings = deque([0, 0, 0, 0, 0, 0, 0, 0, 0,], maxlen=10)
display = deque([0, 0], maxlen=10)

min_threshold = 10
max_threshold = 200
min_area = 60
max_area = 135
min_circularity = 0.2
min_inertia_ratio = 0.55

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

i = 1

leituras = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0
}


while True:
    #Reading the frame of video
    ret, im = video.read()

    #Verify if video already ends
    if not ret:
        break

    #Verify if video is runing
    if ret == True:

        #Detects a blob that corresponds to all params set up there 
        keypoints = detector.detect(im)
        if len(keypoints):

            #Draw a circle around the blob detected, to make it easier for the viewer to see what is going on
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #Create a window to show the video, resize to be in the right size, and show it
            cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("VIDEO", 540, 960)
            cv2.imshow("VIDEO", im_with_keypoints)

        #read only even frames, as it is not necessary to read all frames as this slows down the video
        if contador % 2 == 0:
            #Read the quantity of blobs detected
            reading = len(keypoints)

            #insert the quantity inside the array realTimeReadings
            realTimeReadings.append(reading)

            #disregards consecutive readings that are the same, as they are the result of a reading error, and places only one
            if realTimeReadings[-1] == realTimeReadings[-2] == realTimeReadings[-3] == realTimeReadings[-4] == realTimeReadings[-5] == realTimeReadings[-6] == realTimeReadings[-7] == realTimeReadings[-8] == realTimeReadings[-9]:
                display.append(realTimeReadings[-1])

            #Verify if the reading is correct
            if display[-1] != display[-2] and display[-1] != 0:
                #Print the text in console to show the number of the move, and its result
                text = f"jogada n°{i} --> LADO {display[-1]}"
                leituras[display[-1]] = leituras[display[-1]] + 1
                i = i + 1
                print(text)

        contador += 1

    #If the user press 'c', we close and quit the video 
    if cv2.waitKey(25) & 0xFF == ord('c'):
        break

#Here, we already finished to read the video, and we will analise the values obtained

values = list(leituras.values())

#Print the quantity of times each face was obtained 
print("\nA QUANTIDADE VEZES QUE CADA FACE SAIU, RESPECTIVAMENTE, É DE:")
print(values)
print("")

#here, we create an array with the expected times that each face should be obtained, in an ideal and perfect scenario
expected_values = np.full(6, (i-1)/6).tolist()

#here we calculate the chi-square test comparing the number of times each face was obtained in the video, with the ideal number it should be obtained in a perfect scenario
chisq, p = chisquare(values, f_exp=expected_values)

#Now, we print and explain the method to the user
print("Para verificar se os dados são viciados, foi utilizado o teste qui-quadrado unidirecional")
print("Esse teste verifica a hipótese nula de que os dados categóricos têm as frequências dadas. Ou seja:")
print("O teste é usado para comparar a distribuição de amostra observada com a distribuição de probabilidade esperada \n")

print("Alpha representa o valor divisor de limites para verificar se o dado é viciado ou não.")
print("Após pesquisas sobre testes qui-quadrado, o melhor valor a ser utilizado é 0,05 \n")

#Now we set the value of alpha, that represents the point of limit to determinate if the dice is addicted or not
alpha = 0.05


print("O VALOR DO TESTE QUI-QUADRADO UNIDIRECIONAL FOI DE " + str(p))
#Now we compare the value of the qui-square test with alpha, and print to user the result, if the dice is addicted
if p <= alpha:
    print('LOGO, O DADO E VICIADO, pois o valor do teste deu inferior à 0,05')
else:
    print('LOGO, O DADO NãO E VICIADO, pois o valor do teste deu superior à 0,05')

cv2.destroyAllWindows()
