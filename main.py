import json
import os
from ffmpy import FFmpeg
from sklearn import tree as tr

labelThreshold = 5
usefulCount = {}
zeroCount = 5
encoder = None

def split(path, framerate):
        for filename in os.listdir(path):
                if (filename.endswith(".mov") or filename.endswith(".wmv") or filename.endswith(".avi")):                        
                        try:
                                os.mkdir('internal\\movies\\'+filename[:-4])
                        except:
                                print('Error')
                        os.chdir(path)
                        ff = FFmpeg(inputs={filename: None}, outputs={'..\\internal\\movies\\'+filename[:-4] + '\img%' + str(zeroCount) + 'd.png': ['-vf', 'fps='+str(framerate)]})
                        ff.run()
                        os.chdir('..')

i = 'input movies'



#send to google

#recieve data
allTags = []
movieLabels = {}
trainingLabels = {}
tree = tr.DecisionTreeClassifier()



def extractDataFromJson(name, testing=None):
        try:
                file = open('internal\\responses\\' + name[:-4] + '.txt')
        except:
                try:
                    file = open('internal\\responses\\' + name[:-4] + '.json')
                except:
                    return

        jsonData = json.load(file)
        if testing is not None:
                if testing not in trainingLabels:
                        trainingLabels[testing] = {}
                else:
                trainingLabels[testing][name] = []
        else:
                movieLabels[name] = []
        count = 0
        for i in jsonData['labelAnnotations']:
                if count < labelThreshold:
                        if testing is not None:
                                trainingLabels[testing][name].append(i['description'])
                        else:
                                movieLabels[name].append(i['description'])
                        count += 1
                else: break       



def encode(tagsInImage):
        encoded = []
        for tag in allTags:
                if tag in tagsInImage:
                        encoded.append(1)
                else:
                        encoded.append(0)
        return encoded



def moveUsefulFrames(name):
        if name in usefulCount:
                usefulCount[name] += 1
        else:
                usefulCount[name] = 1
        try:
                os.mkdir('Useful frames')
        except:
                
        movieName = re.split('(\d+)',name)[0]
        try:
                os.mkdir('Useful frames\\'+movieName)
        except:
        os.rename("internal\\movies\\"+movieName+'\\'+name, "Useful frames\\"+movieName+'\\'+name)



#def sendToGoggle():
        


def getTags():
        for imageSet in trainingLabels:
                for image in trainingLabels[imageSet]:
                        for tag in trainingLabels[imageSet][image]:
                                if tag not in allTags:
                                        allTags.append(tag)

        for image in movieLabels:
                 for tag in movieLabels[image]:
                       if tag not in allTags:
                                allTags.append(tag)

def trainTree():
        usefulNames = []
        notUsefulNames = []
        
        for i in os.listdir('training images\\useful images'):
                usefulNames.append(i)
        for i in os.listdir('training images\\not useful images'):
                notUsefulNames.append(i)

        #sendToGoogle()
        
        for i in notUsefulNames:
                extractDataFromJson(i, 0)

        for i in usefulNames:
                extractDataFromJson(i, 1)
        
        getTags()

        trainingSet = []
        trainingSetLabels = []
        for imageSet in trainingLabels:
                for image in trainingLabels[imageSet]:
                        trainingSet.append(trainingLabels[imageSet][image])
                        trainingSetLabels.append(imageSet)

        
        for i in range(len(trainingSet)):
                trainingSet[i] = encode(trainingSet[i])
        tree.fit(trainingSet, trainingSetLabels)




def evaluateImages():

        imageNames = []
        
        for movieSet in os.listdir('internal\\movies'):
                try:
                        os.chdir('internal\\movies\\'+movieSet)
                        for image in os.listdir():
                                imageNames.append(image)
                                #send to google()
                        os.chdir('..\\..\\..')
                except:
                        continue

        
        for image in imageNames:
                extractDataFromJson(image)

        getTags()

        movieSet = []
        movieOrder = []
        for image in movieLabels:
                movieSet.append(encode(movieLabels[image]))
                movieOrder.append(image)

        results = tree.predict(movieSet)

        for index in range(len(results)):
                if results[index] == 1:
                        moveUsefulFrames(movieOrder[index])
