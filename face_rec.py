
import cv2
import face_recognition
import functions

returned_person = {
    'face_found': False,
    'name': '',
    'imageName': '',
    'personEncoding': ''
}
face_encodings = []
known_people_encodings = []
known_people_name = []
personEncoding =  []

def personRecognition(face_encodings, known_people_encodings, known_people_name, small_frame, rgb_small_frame):
    face_encodings = face_encodings
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_people_encodings, face_encoding, tolerance=0.5)
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            returned_person['name'] = known_people_name[first_match_index]
            returned_person['imageName'] = 'none'
            returned_person['face_found'] = True
            return returned_person , None
        elif not True in matches:
            personName, personEncoding = functions.uploadToS3(
            small_frame, rgb_small_frame)
            returned_person['name'] = personName
            returned_person['imageName'] = personName+'.jpg'
            returned_person['personEncoding'] = personEncoding
            return returned_person , personEncoding
        else:
            print('Error')
