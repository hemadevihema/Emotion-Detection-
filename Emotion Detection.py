from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1= cv2.imread("C:\\Users\\hemad\\Pictures\\elonmusk.jpg")
plt.imshow(img1[:,:,::-1])
plt.show()

#analyze emotion
result = DeepFace.analyze(img1 , actions=['emotion'])

print("Dominant emotion:" ,result[0]['dominant_emotion'])


# after closing the image generated only we can see the output
# in print statement we can also give print(result) it will show the all results which it predicted


