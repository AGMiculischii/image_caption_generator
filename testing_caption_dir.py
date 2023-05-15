import argparse
import cv2
from gtts import gTTS 
import os
import pyttsx3
import pathlib  
from modu_img import *


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', required=True, help="Directory Path")
args = vars(ap.parse_args())

max_length = 32

tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_0.h5')
xception_model = Xception(include_top=False, pooling="avg")

# определение пути
currentDirectory = pathlib.Path(args['dir'])
print(currentDirectory)

# определение шаблона
currentPattern = "*.jpg"
print("\n\n")
f = open('spisok.txt','w')

for currentFile in currentDirectory.glob(currentPattern):  
    img_path = '.\\'+str(currentFile)

    photo = extract_features(img_path, xception_model)

    img = cv2.imread(img_path)

    description = generate_desc(model, tokenizer, photo, max_length)
    text_foto = str(img_path[len(args['dir'])+1:]+"  "+description[6:-3])
    print(text_foto)
    f.write(text_foto+"\n")

f.close()
