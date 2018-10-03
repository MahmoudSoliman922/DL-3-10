
class people():
    def __init__(self):
        self.personInfo = {
            'name': 'Unknown',
            'imageName': 'none'
        }

        self.serverInfo = {
            'cam' : 'one',
            'eventName': 'profile'
        }

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
        self.allInfo = {
            'personInfo':self.personInfo,
            'serverInfo':self.serverInfo,
            'personEmotion':self.personEmotion
        }
    # Setters
    def setAllInfo(self, personInfo, personEmotion):
        self.allInfo['personInfo'] = personInfo
        self.allInfo['personEmotion'] = personEmotion
        self.allInfo['serverInfo'] = self.serverInfo

    # Getters
    def getPersonData(self):
        return self.allInfo
