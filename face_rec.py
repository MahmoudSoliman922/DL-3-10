
import cv2
import face_recognition
import functions
# the return of personRecognition function
returned_person = {
    'face_found': False,
    'name': '',
    'imageName': '',
    'personEncoding': ''
}
# face encodings of the captured frames
face_encodings = []
# saved people face encodings
known_people_encodings = []
# saved people names
known_people_name = []
# encoding of a new face
personEncoding =  []

# takes the captured frame's encoding, saved people encodings, names of the saved faces, captured frame, rgb frame
def personRecognition(face_encodings, known_people_encodings, known_people_name, small_frame, rgb_small_frame):
    face_encodings = face_encodings
    # loop through all the passed face encodings
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_people_encodings, face_encoding, tolerance=0.5)
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            # get the index of the matched face
            first_match_index = matches.index(True)
            # get the name of the matched face
            returned_person['name'] = known_people_name[first_match_index]
            # set the imageName to none because we already have the person's image
            returned_person['imageName'] = 'none'
            # set the face_found object to True 
            returned_person['face_found'] = True
            # return the returned_person object after modifying it and None for the encoding
            return returned_person , None
        # If no mach found with the dataset
        elif not True in matches:
            # passing the frame to uploadToS3 function to upload the image to the bucket and get the person
            # name and the encoding from it
            personName, personEncoding = functions.uploadToS3(
            small_frame, rgb_small_frame)
            # set the name = the returned person name from the past function
            returned_person['name'] = personName
            # set the imageName to the name+.jpg
            returned_person['imageName'] = personName+'.jpg'
            # set the personEncoding to the returned encoding
            returned_person['personEncoding'] = personEncoding
            # finally return the returned_person object and the personencoding
            return returned_person , personEncoding
        else:
            # just to check if any unhandled cases will rise
            print('Error')
