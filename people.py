
# this class presents a blueprint for the returned data of 1 person
class people():
    def __init__(self):
        # person information
        self.personInfo = {
            'name': 'Unknown',
            'imageName': 'none'
        }
        # Server Information
        self.serverInfo = {
            'cam' : 'one',
            'eventName': 'profile'
        }
        # person's emotions information
        self.personEmotion = {
            'mood': 'Unknown',
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
        # combining all the information in 1 object
        self.allInfo = {
            'personInfo':self.personInfo,
            'serverInfo':self.serverInfo,
            'personEmotion':self.personEmotion
        }
    # Set the gathered information
    def setAllInfo(self, personInfo, personEmotion):
        self.allInfo['personInfo'] = personInfo
        self.allInfo['personEmotion'] = personEmotion
        self.allInfo['serverInfo'] = self.serverInfo

    # Get the gatherd information
    def getPersonData(self):
        return self.allInfo
