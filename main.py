import cv2
# import awscam
import os
import json
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

from websocket import create_connection
import base64
import face_recognition

from multiplethreading import ThreadWithReturnValue
import face_rec
import people
import functions

# Local Recognition
def localEmotionRecognition(img):
    img = cv2.resize(img, (740, 560))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
         # crop detected face
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]
        detected_face = cv2.cvtColor(
            detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(
            detected_face, (48, 48))  # resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # pixels are in scale of [0, 255]. normalize all pixels in scale of
        # [0, 1]
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        # connect face and expressions
        cv2.line(img, (int((x + x + w) / 2), y + 15),
                 (x + w, y - 20), (255, 255, 255), 1)
        cv2.line(img, (x + w, y - 20),
                 (x + w + 10, y - 20), (255, 255, 255), 1)
        all_emotions_numbers = []
        for i in range(len(predictions[0])):
            returned_emotions['reactions'][emotions[i]] = round(
                predictions[0][i]*100, 2)
            all_emotions_numbers.append(round(predictions[0][i]*100, 2))
        if max(all_emotions_numbers) > 75:
            returned_emotions['mood'] = emotions[all_emotions_numbers.index(
                max(all_emotions_numbers))]
        else:
            returned_emotions['mood'] = 'Unknown'
    return returned_emotions

# Open a file
known_people_name = []
known_people_encodings = []
face_locations = []
face_encodings = []
process_this_frame = True
temp_person = 'unknown'
temp_emotion = 'unknown'

functions.getAllEncodingsFromS3()

path = "data/"
dirs = os.listdir(path)
for file in dirs:
    fname, fext = file.split('.')
    known_people_name.append(fname)
    temp_encoding = np.load('data/' + file)
    known_people_encodings.append(temp_encoding)

# Local Recognition Dep
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

model = model_from_json(open(
    "facial_expression_model_structure.json", "r").read())

model.load_weights(
        'facial_expression_model_weights.h5')


emotions = ('angry', 'disgusted', 'confused',
            'happy', 'sad', 'surprised', 'calm')

returned_emotions = {
    'mood': '',
    'reactions': {
        'happy': '0',
        'sad': '0',
        'angry': '0',
        'calm': '0',
                'disgusted': '0',
                'confused': '0',
                'surprised': '0'
    }
}

video_capture = cv2.VideoCapture(0)
ws = create_connection("ws://gitex.ahla.io:5555")
if ws is None :
    ws = create_connection("ws://gitex.ahla.io:5555")
else:
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        person = people.people()
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        if not face_locations == []:
            facial_recognition = ThreadWithReturnValue(target=face_rec.personRecognition, args=(
                face_encodings, known_people_encodings, known_people_name, small_frame, rgb_small_frame))
            facial_recognition.start()
            personData , personEncoding = facial_recognition.join()
            emotionData = localEmotionRecognition(frame)
            if not personEncoding == None:
                known_people_name.append(personData['name'])
                known_people_encodings.append(personEncoding[0])
                personData['personEncoding'] = ''
            person.setAllInfo(personData, emotionData)

        # cv2.imshow('Video', small_frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        temp_Data= person.getPersonData()
        if temp_person != temp_Data['personInfo']['name'] or temp_emotion != temp_Data['personEmotion']['mood']:
            ws.send(json.dumps(temp_Data))
            temp_person = temp_Data['personInfo']['name']
            temp_emotion = temp_Data['personEmotion']['mood']

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
