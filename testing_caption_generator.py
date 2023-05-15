import argparse
import cv2
from gtts import gTTS 
import os
import pyttsx3  
from modu_img import *


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']


max_length = 32

tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_0.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)

img = cv2.imread(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description[6:-3])

"""  #ispolizovanie golosovoro ozvucivanie teksta cerez modul GTTS
speech = gTTS(text = description[6:-3], lang = 'en', slow = False)
speech.save("text.mp3")
os.system("start text.mp3")  """


cv2.imshow(description[6:-3], img)


#ispolizovanie golosovoro ozvucivanie teksta cerez modul pyttsx3
s = pyttsx3.init()  
s.say(description[6:-3])  
s.runAndWait()

cv2.waitKey(0)
cv2.destroyAllWindows()


