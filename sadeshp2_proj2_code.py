import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

#Loading the data
def load_data(path):
    dirs = os.listdir(path)
    dataset = []
    for item in tqdm(dirs):
        if os.path.isfile(path+item):
            inputImage = cv2.imread(path + item,0)
            inputImage = cv2.resize(inputImage,(16,16),interpolation = cv2.INTER_LINEAR)/255
            dataset.append(inputImage)
    return dataset
#Integral Image
def getIntegralImage(image):
    r, c = image.shape
    summedMatrix = np.zeros([r, c])
    for x in range(r):
        for y in range(c):
            if x == 0:
                s[x, y] = s[x, y - 1] + i[x, y]
            elif y == 0:
                s[x, y] = s[x - 1, y] + i[x, y]
            else:
                s[x, y] = s[x - 1, y] + s[x, y - 1] - s[x - 1, y - 1] + i[x, y]
    return summedMatrix

#Checking if feature extraction is possible
def isFeaturePossible(feature_type,i,j,finalX,finalY):
    if((i+finalX <= 16) and (j+finalY <= 16)):
        featurePossible = 1
    else:
        featurePossible = 0
    return featurePossible

#Extracting haar features
def getFeatures(image,totalFeatures):
    iRows, iCols = image.shape
    
    features = []
    feature_scales = np.array(((1,2),(2,1), (1,3), (3,1),(2,2)));

    count = 0
    for feature_type in range(totalFeatures):
        for xScale in range(1,16+1):
            for yScale in range(1,16+1):
                finalX = feature_scales[feature_type][0] * xScale
                finalY = feature_scales[feature_type][1] * yScale
                for i in range(iRows):
                    for j in range(iCols):
                        if(isFeaturePossible(feature_type, i, j, finalX, finalY)):
                            if(feature_type == 0):
                                plusList = image[i:i+finalX, j:j +(int)(finalY/2)]
                                plusTerm = 5*np.sum(plusList)
                                minusList = image[i:i+ finalX,j +(int)(finalY/2):j + finalY]
                                minusTerm = np.sum(minusList)
                                appendList = [plusTerm - minusTerm,i,j, feature_type,finalX, finalY, count]
                                count += 1
                                features.append(appendList)
                            
                            if(feature_type == 1):
                                plusList = image[i:i+(int)(finalX/2), j:j + finalY]
                                plusTerm = 5*np.sum(plusList)
                                minusList = image[i+(int)(finalX/2):i+ finalX,j:j + finalY]
                                minusTerm = np.sum(minusList)
                                appendList = [plusTerm - minusTerm,i,j, feature_type,finalX, finalY, count]
                                count += 1
                                features.append(appendList)
                            
                            if(feature_type == 2):
                                minusList1 = image[i:i+finalX , j:j +(int)(finalY/3)]
                                minusTerm1 = np.sum(minusList1)
                                plusList = image[i:i+finalX , j +(int)(finalY/3):j + 2 * (int)(finalY/3)]
                                plusTerm = 5*np.sum(plusList)
                                minusList2 = image[i:i+finalX,j + 2 * (int)(finalY/3):j + finalY ]
                                minusTerm2 = np.sum(minusList2)
                                appendList = [plusTerm - minusTerm1 - minusTerm2,i,j, feature_type,finalX, finalY, count]
                                count += 1
                                features.append(appendList)
                            
                            if(feature_type == 3):
                                minusList1 = image[i:i+(int)(finalX/3) , j:j +finalY]
                                minusTerm1 = np.sum(minusList1)
                                plusList = image[i+(int)(finalX/3):i + 2 * (int)(finalX/3) , j :j + finalY]
                                plusTerm = 5*np.sum(plusList)
                                minusList2 = image[i + 2 * (int)(finalY/3) :i+finalX,j :j + finalY]
                                minusTerm2 = np.sum(minusList2)
                                appendList = [plusTerm - minusTerm1 - minusTerm2,i,j, feature_type,finalX, finalY, count]
                                count += 1
                                features.append(appendList)
                            
                            if(feature_type == 4):
                                minusList1 = image[i:i+(int)(finalX/2) , j:j +(int)(finalY/2)]
                                minusTerm1 = np.sum(minusList1)
                                plusList1 = image[i:i+(int)(finalX/2), j +(int)(finalY/2) :j + finalY]
                                plusTerm1 = 5*np.sum(plusList1)
                                plusList2 = image[i+(int)(finalX/2):i + finalX , j:j +(int)(finalY/2)]
                                plusTerm2 = 5*np.sum(plusList2)
                                minusList2 = image[i+(int)(finalX/2):i + finalX, j +(int)(finalY/2) :j + finalY]
                                minusTerm2 = np.sum(minusList2)
                                appendList = [plusTerm1 + plusTerm2 - minusTerm1 - minusTerm2,i,j, feature_type,finalX, finalY, count]
                                count += 1
                                features.append(appendList)

    return features        
#Finding Thresholds
def findThresholds(fData, nfData):
    thList = np.concatenate((fData, nfData))
    minTh = min(thList)
    maxTh = max(thList)
    thr = 0
    tCorrect_old = 0

    for i in np.arange(minTh, maxTh, ((maxTh - minTh)/50)):
        fCorrect = len(np.where(fData < i)[0])
        nfCorrect = len(np.where(nfData > i)[0])
        tCorrect_new = fCorrect + nfCorrect

        if(tCorrect_new >= tCorrect_old):
            thr = i
            tCorrect_old = tCorrect_new
    tCorrect_old = tCorrect_old/len(thList)

    return thr, tCorrect_old

#Error Calculation
def CalculateError(hi, labels, weights):
    hi[hi == labels] = 0
    hi[hi != 0] = 1
    
    error = np.sum(weights * hi)
    return error

#Function to visualize features
def visualizeFeature(image, i, j, finalX, finalY, feature_type):
    if (feature_type == 0):
        image[i:i+finalX, j:j +(int)(finalY/2)] = 0
        image[i:i+ finalX,j +(int)(finalY/2):j + finalY] = 255
    
    elif (feature_type == 1):
        image[i:i+(int)(finalX/2), j:j + finalY] = 0
        image[i+(int)(finalX/2):i+ finalX,j:j + finalY] = 255
    
    elif (feature_type == 2):
        image[i:i+finalX , j:j +(int)(finalY/3)] = 0
        image[i:i+finalX , j +(int)(finalY/3):j + 2 * (int)(finalY/3)] = 255
        image[i:i+finalX,j + 2 * (int)(finalY/3):j + finalY] = 0
    
    elif (feature_type == 3):
        image[i:i+(int)(finalX/3) , j:j +finalY] = 0
        image[i+(int)(finalX/3):i + 2 * (int)(finalX/3) , j :j + finalY] = 255
        image[i + 2 * (int)(finalY/3) :i+finalX,j :j + finalY] = 0
    
    elif (feature_type == 4):
        image[i:i+(int)(finalX/2) , j:j +(int)(finalY/2)] = 0
        image[i:i+(int)(finalX/2), j +(int)(finalY/2) :j + finalY] = 255
        image[i+(int)(finalX/2):i + finalX , j:j +(int)(finalY/2)] = 255
        image[i+(int)(finalX/2):i + finalX, j +(int)(finalY/2) :j + finalY] = 0

    image = cv2.resize(image,(150,150))
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


#Driver
if __name__ == '__main__':
    
#Loading the data
    facePath = "Data/face_16/"
    nonFacePath = "Data/nonface_16/"
    print("Loading Face Data..")
    faceData = load_data(facePath)
    print("Loading Non Face Data..")
    nonFaceData = load_data(nonFacePath)
    
    fTrain = faceData[0:400]
    nfTrain = nonFaceData[0:400]
    
    fTest = faceData[800:900]
    nfTest = nonFaceData[800:900]

#Extracting the features
    fFeatures = []
    nfFeatures = []
    print("Extracting Features..")
    
    for i in tqdm(range(len(fTrain))):
        fImageFeatures = getFeatures(fTrain[i],5)
        fFeatures.append(fImageFeatures)
    
    for i in tqdm(range(len(nfTrain))):
        nfImageFeatures = getFeatures(nfTrain[i],5)
        nfFeatures.append(nfImageFeatures)
    fFeatures = np.array(fFeatures)
    nfFeatures = np.array(nfFeatures)

#Finding the thresholds
    thresholds = []
    print("Finding Thresholds..")
    for i in tqdm(range(len(fFeatures[0]))):
        fImageData = fFeatures[:,i,0]
        nfImageData = nfFeatures[:,i,0]
        val, performance = findThresholds(fImageData, nfImageData)
        thresholds.append([val,performance,int(fFeatures[0,i,1]),int(fFeatures[0,i,2]),int(fFeatures[0,i,3]),int(fFeatures[0,i,4]),int(fFeatures[0,i,5]),int(fFeatures[0,i,6])])
    thresholds = np.array(thresholds)

#Top 10 features before AdaBoost
    sortedThr = thresholds[thresholds[:,1].argsort()]
    bestInitialFeatures = sortedThr[-10:len(sortedThr)][::-1]
    for i in range(len(bestInitialFeatures)):
        image = visualizeFeature(fTrain[2],int(bestInitialFeatures[i][2]),int(bestInitialFeatures[i][3]),int(bestInitialFeatures[i][5]),int(bestInitialFeatures[i][6]),int(bestInitialFeatures[i][4]))

#AdaBoost Algorithm

# Weight Initialization
    weights = np.ones(2*len(fFeatures))/(2*len(fFeatures))   
    labels = np.concatenate((np.ones(len(fFeatures)),-np.ones(len(nfFeatures))))
    
    
    numberOfIterations = 10
    alphaBoost = []
    indexBoost = []
    
    hTrainBoost = []
    for iterator in range(numberOfIterations):
        error_i = []
        h_i = []
        weights = weights/np.sum(weights)

# Calculate error of each classifier using indicator function    
        for i in range(len(thresholds)):

            thresh_value = thresholds[i][0]
            feature_index = int(thresholds[i][7])
        
            fData = fFeatures[:,feature_index,0]
            nfData = nfFeatures[:,feature_index,0]
        
            fData = [1 if x < thresh_value else -1 for x in fData]
            nfData = [-1 if x > thresh_value else 1 for x in nfData]
        
            hi = np.concatenate((np.array(fData),np.array(nfData)))
            h_i.append(np.concatenate((np.array(fData),np.array(nfData))))
            
            
            error = CalculateError(hi, labels, weights)
            error_i.append(error)

# Choose classifier with lowest error
        hT = np.argmin(error_i)
        indexBoost.append(hT)

# Calculate modular weight and update data weights        
        alphaT = 0.5 * np.log((1-error_i[hT])/(error_i[hT]))
        weights = weights * np.exp(- labels * alphaT * h_i[hT])
    
        alphaBoost.append(alphaT)
        hTrainBoost.append(h_i[hT])

# Calculate training accuracy        
        acc = np.sum([hTrainBoost[i] * alphaBoost[i] for i in range(len(alphaBoost))],0)
        accuracy = np.sum(np.sign(acc) == labels)/(2*len(fFeatures))
        print("Iteration: ", iterator, "  Training Accuracy: ",accuracy)
        

#Visualizing Features
    count = 0
    for i in indexBoost:
        count += 1
        image = visualizeFeature(fTrain[2],int(thresholds[i][2]),int(thresholds[i][3]),int(thresholds[i][5]),int(thresholds[i][6]),int(thresholds[i][4]))
        if(count == 10):
            break

#Extracting features from test data
    t_fFeatures = []
    t_nfFeatures = []
    print("Extracting Test Data Features..")
    
    for i in tqdm(range(len(fTest))):
        t_fImageFeatures = getFeatures(fTest[i],5)
        t_fFeatures.append(t_fImageFeatures)
    
    for i in tqdm(range(len(nfTest))):
        t_nfImageFeatures = getFeatures(nfTest[i],5)
        t_nfFeatures.append(t_nfImageFeatures)
    
    t_fFeatures = np.array(t_fFeatures)
    t_nfFeatures = np.array(t_nfFeatures)

#Finding Test Accuracy
    t_h_i = []
    t_labels = np.concatenate((np.ones(len(t_fFeatures)),-np.ones(len(t_nfFeatures))))
    for i in indexBoost:
        
        t_fData = t_fFeatures[:,i,0]
        t_nfData = t_nfFeatures[:,i,0]
        
        thresh_value = thresholds[i][0]
        
        t_fData = [1 if x < thresh_value else -1 for x in t_fData]
        t_nfData = [-1 if x > thresh_value else 1 for x in t_nfData]
        
        t_hi = np.concatenate((np.array(t_fData),np.array(t_nfData)))
        t_h_i.append(t_hi)
    
    t_acc = np.sum([alphaBoost[i] * t_h_i[i]  for i in range(len(alphaBoost))],0)
    t_accuracy = np.sum(np.sign(t_acc) == t_labels)/(2*len(t_fFeatures))
    print("Test Accuracy: ",t_accuracy)

#ROC Curve
    roc_acc = np.sum([(t_h_i[i]) * alphaBoost[i] for i in range(len(alphaBoost))],0)
    
    min_roc = min(roc_acc)
    max_roc = max(roc_acc)
    
    t_fpr = []
    t_tpr = []

    for thresh in np.arange(min_roc, max_roc, ((max_roc - min_roc)/200)):

        roc_vector = [1 if x > thresh else -1 for x in roc_acc]
        tn, fp, fn, tp = confusion_matrix(roc_vector, t_labels).ravel()
     
        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)   
        
        t_fpr.append(fpr)
        t_tpr.append(tpr)
       
    plt.plot(t_fpr, t_tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.show()


