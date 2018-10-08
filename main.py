import cv2
# import awscam
import os
import json
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import websocket
import base64
import face_recognition

from multiplethreading import ThreadWithReturnValue
import face_rec
import people
import functions

# Function that takes the captrued frame and return the detected emotions with percentage
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

# Local Recognition model and weights
face_cascade = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml')

model = model_from_json(open(
    "models/facial_expression_model_structure.json", "r").read())

model.load_weights(
        'models/facial_expression_model_weights.h5')


emotions = ('angry', 'disgusted', 'confused',
            'happy', 'sad', 'surprised', 'calm')
# object of the returned emotions
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
# saved people names array
known_people_name = []
# saved people encodings array
known_people_encodings = []
# faces location in the captured frame
face_locations = []
# face encodings
face_encodings = []
# boolean variable for switching the processing functionality
process_this_frame = True
# temporary person name and emotion to make sure that either one of them has changed before sending the data to the server
temp_person = 'unknown'
temp_emotion = 'unknown'
# calling the getAllEncodingsFromS3() function to get the saved encodings from the bucket
functions.getAllEncodingsFromS3()
# the data saved locally in the data folder so we need to import the encodings and names
path = "data/"
dirs = os.listdir(path)
for file in dirs:
    fname, fext = file.split('.')
    known_people_name.append(fname)
    temp_encoding = np.load('data/' + file)
    known_people_encodings.append(temp_encoding)
# start capturing the live stream from webcam/deeplens
video_capture = cv2.VideoCapture(0)
while True:
    # try and except to handle the broken pipe with the server
    try:
        # creating connection with the server
        ws = websocket.create_connection("ws://gitex.ahla.io:5555")
        while True:
            # getting the captured frame 
            ret, frame = video_capture.read()
            # resizing the frame
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # converting the resized frame to rgb 
            rgb_small_frame = small_frame[:, :, ::-1]
            # creating an instance from the people blueprint
            person = people.people()
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            # getting encodings from the captured faces
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)
            # check if there's a face before starting the emotion and facial recognition
            if not face_locations == []:
                # creating a thread for the face recognition
                facial_recognition = ThreadWithReturnValue(target=face_rec.personRecognition, args=(
                    face_encodings, known_people_encodings, known_people_name, small_frame, rgb_small_frame))
                # starting the face recognition thread
                facial_recognition.start()
                # starting the emotion recognition / Now both the emotion and face recognition working together
                emotionData = localEmotionRecognition(frame)
                # joining the face recognition before proceding
                personData , personEncoding = facial_recognition.join()
                # check if there's a data in the returned personEncoding or not
                # if personEncoding isn't Null
                if not personEncoding == None:
                    # append the returned name to the known people names array
                    known_people_name.append(personData['name'])
                    # append the encoding to the known people encodings array
                    known_people_encodings.append(personEncoding[0])
                    # set the personEncoding to ''
                    personData['personEncoding'] = ''
                person.setAllInfo(personData, emotionData)       
            # setting the temp_Data (Which is the data that will be sent to the server ) to the current person's data
            temp_Data= person.getPersonData()
            # check if the person or the mood have changed or not , If any of them is changed then will send 
            # to the server the new data, If now then won't send anything
            if temp_person != temp_Data['personInfo']['name'] or temp_emotion != temp_Data['personEmotion']['mood']:
                # parsing the data to JSON and send them to the server
                ws.send(json.dumps(temp_Data))
                # updating the temp_person and temp_emotion to make sure not to send the data again
                temp_person = temp_Data['personInfo']['name']
                temp_emotion = temp_Data['personEmotion']['mood']
    # handeling the broken pipe exception to make sure that the script won't crash
    except websocket.WebSocketConnectionClosedException:
        print('Connection lost and trying to reconnect ...')
video_capture.release()
cv2.destroyAllWindows()
