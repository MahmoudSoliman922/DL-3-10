import random
import string
import boto3
import cv2
import face_recognition
import numpy as np
from multiplethreading import ThreadWithReturnValue

s3Client = boto3.client('s3')
s3 = boto3.resource('s3')

picturesBucket = "face-rec-final"
encodingsBucket = "face-rec-final-enc"

known_people_name = []
known_people_encodings = []


def getAllEncodingsFromS3():
    list = s3Client.list_objects(Bucket='face-rec-final-enc')['Contents']
    for key in list:
        s3Client.download_file('face-rec-final-enc',
                               key['Key'], 'data/'+key['Key'])

# picks a randomword as an ID
def randomword():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))

# make sure that the face is suitable for further recognitions
def suitableFace(rgb_small_frame):
    face_locations = face_recognition.face_locations(
        rgb_small_frame, model='cnn')
    if not face_locations == []:
        return True
    else:
        return False

# Saves the caputred image locally
def saveImageLocally(passed_small_frame):
    temp_image_word = randomword()
    imageName = temp_image_word+'.jpg'
    personName = temp_image_word
    cv2.imwrite(filename='Faces/' +
                imageName, img=passed_small_frame)
    imageEncoding = saveEncodingLocally(personName)
    return personName, imageEncoding

# Saves the saved image's Encoding locally
def saveEncodingLocally(personName):
    temp_image = face_recognition.load_image_file(
        'Faces/'+personName+'.jpg')
    temp_encoding = face_recognition.face_encodings(temp_image)
    np.save('ImagesEncodings/'+personName+'.npy', temp_encoding[0])
    return temp_encoding

# Upload both the Encoding and the image to S3 buckets
def uploadToS3(passed_small_frame, rgb_small_frame):
    upload_or_not = suitableFace(rgb_small_frame)
    if upload_or_not == True:
        personName, imageEncoding = saveImageLocally(passed_small_frame)
        s3.Bucket(picturesBucket).upload_file("Faces/" +
                                              personName+'.jpg', personName+'.jpg', ExtraArgs={'ACL': 'public-read'})
        s3.Bucket(encodingsBucket).upload_file("ImagesEncodings/" +
                                               personName+'.npy', personName+'.npy', ExtraArgs={'ACL': 'public-read'})
        return personName, imageEncoding
    else:
        print('Not Suitable!')
        return '', ''
