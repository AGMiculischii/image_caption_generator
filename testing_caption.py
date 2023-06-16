import argparse
import cv2
from gtts import gTTS 
import os
import pyttsx3
import sys  
import pathlib
from modu_img import *

param = sys.argv[1]
ap = argparse.ArgumentParser()
if param == '-i':
    ap.add_argument('-i', '--image', required=True, help="Image Path")
    args = vars(ap.parse_args())
    img_path = args['image']
    print("Image")
if param == '-d':
    ap.add_argument('-d', '--dir', required=True, help="Directory Path")
    args = vars(ap.parse_args())
    print("direct")

max_length = 32

tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_0.h5')
xception_model = Xception(include_top=False, pooling="avg")

if param == '-i':
    photo = extract_features(img_path, xception_model)
    img = cv2.imread(img_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    print(description[6:-3])
    cv2.imshow(description[6:-3], img)

    """  #ispolizovanie golosovoro ozvucivanie teksta cerez modul GTTS
    speech = gTTS(text = description[6:-3], lang = 'en', slow = False)
    speech.save("text.mp3")
    os.system("start text.mp3")  """

    #ispolizovanie golosovoro ozvucivanie teksta cerez modul pyttsx3
    s = pyttsx3.init()  
    s.say(description[6:-3])  
    s.runAndWait()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
if param == '-d':
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
        #img = cv2.imread(img_path)
        description = generate_desc(model, tokenizer, photo, max_length)
        text_foto = str(img_path[len(args['dir'])+3:]+"  "+description[6:-3])
        print(text_foto)
        f.write(text_foto+"\n")
    f.close()

