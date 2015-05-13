__author__ = 'zephyrYin'

import math
import copy
import time
from random import randint
from tree import Node
from scipy import stats

class DecisionTree:
    def __init__(self, featurePath, trainFeaturePath, trainLabelPath, testFeaturePath, testLabelPath, criterion):
        self.featurePath = featurePath
        self.trainFeaturePath = trainFeaturePath
        self.trainLabelPath = trainLabelPath
        self.testFeaturePath = testFeaturePath
        self.testLabelPath = testLabelPath
        self.trainFeatures = []
        self.testFeatures = []
        self.featureNames = []
        self.featureValue = []
        self.contrastResult = []
        self.dTreeRoot = None
        self.chiCriterion = criterion
        self.nodeCnt = 0
        self.postiveRatio = 0.5

    def buildTree(self):
        self.readFeatureNames()
        self.readTrainFeatures()
        print('building decision tree ...')
        start = time.clock()
        self.dTreeRoot = self.ID3(self.trainFeatures, [i for i in range(len(self.featureNames))])
        elapsed = time.clock() - start
        print('tree built using ' + str(elapsed) +' seconds')

    def ID3(self, features, attributes):
        self.nodeCnt += 1
        wholeCnt = len(features)
        positiveCnt = self.countPositive(features)
        if positiveCnt == wholeCnt:
            return Node(-1, 1, {})
        elif positiveCnt == 0:
            return Node(-1, 0, {})
        elif len(attributes) == 0:                  # return major label
            return Node(-1, 1 if positiveCnt/float(wholeCnt) >= self.postiveRatio else 0, {})
        else:
            candidates = []
            maxGain = 0
            cnt = 0
            for attribute in attributes:
                gain, childs = self.computeGain(features, attribute)
                print(str(cnt) + ' : ' + str(gain))
                cnt += 1
                #gain /= self.splitRatio(features, childs)
                if gain > maxGain:
                    maxGain = gain
                    candidates = []
                    candidates.append([attribute, childs])
                elif gain == maxGain:
                    candidates.append([attribute, childs])
            bestPick = candidates[randint(0, len(candidates)-1)]                # [attribute, childs]
            print('choose: ' + str(bestPick[0]) + ' ' + self.featureNames[bestPick[0]])
            chi = self.computeChiSquaredCriterion(features, bestPick[1])
            pValue = 1 - stats.chi2.cdf(chi, len(bestPick[1]) - 1)
            print('chi: ' + str(chi) + '  pvalue: ' + str(pValue) )
            if pValue > self.chiCriterion:                                         # split stop
                return Node(-1, 1 if positiveCnt/float(wholeCnt) >= self.postiveRatio else 0, {})
            currentNode = Node(bestPick[0], '', {})
            newAttributes = copy.deepcopy(attributes)
            newAttributes.remove(bestPick[0])
            for i in range(len(self.featureValue[bestPick[0]])):
                value = self.featureValue[bestPick[0]][i]
                currentNode.children[value] = self.ID3(bestPick[1][i], newAttributes)
            return currentNode

    def splitRatio(self, features, childs):
        wholeCnt = len(features)
        ratio = 0
        for child in childs:
            r = len(child)/float(wholeCnt)
            ratio += r*self.safeLog(r, 2)
        if ratio == 0.0:
            return 1
        return -ratio

    def readTrainFeatures(self):
        print('reading train set ...')
        self.trainFeatures = []

        file = open(self.trainFeaturePath)
        lines = file.readlines()                        # read train feature
        file.close()
        for line in lines:
            line = [int(i) for i in line.split(' ')]
            self.trainFeatures.append(line)
            for i in range(len(line)):
                if line[i] not in self.featureValue[i]:
                    self.featureValue[i].append(line[i])
        for fea in self.featureValue:
            fea.sort()

        file = open(self.trainLabelPath)
        lines = file.readlines()                        # combine train label
        file.close()
        cnt = 0
        for i in range(len(lines)):
            if cnt > 99:
                break
            cnt += 1
            self.trainFeatures[i].append(int(lines[i]))
        posCnt = self.countPositive(self.trainFeatures)
        self.postiveRatio = posCnt/float(len(self.trainFeatures))
        print('train set loaded')

    def readTestFeatures(self):
        print('reading test set ...')
        self.testFeatures = []
        file = open(self.testFeaturePath)
        lines = file.readlines()
        file.close()
        for line in lines:
            line = [int(i) for i in line.split(' ')]
            self.testFeatures.append(line)
        file = open(self.testLabelPath)
        lines = file.readlines()                        # combine test label
        file.close()
        for i in range(len(lines)):
            self.testFeatures[i].append(int(lines[i]))
        print('test set loaded')

    def readFeatureNames(self):
        self.featureNames = []
        file = open(self.featurePath)
        lines = file.readlines()
        file.close()
        for line in lines:
            self.featureNames.append(line.strip('\n').strip('\"'))
            self.featureValue.append([])

    def splitByAttribute(self, features, attributeIndex):
        attributeDimen = self.featureValue[attributeIndex]
        childs = [[] for i in attributeDimen]
        dict = {}
        for i in range(len(attributeDimen)):
            dict[attributeDimen[i]] = i

        for feature in features:
            fea = feature[attributeIndex]
            childs[dict[fea]].append(feature)
        return childs

    def countPositive(self, features):
        cnt = 0
        for fea in features:
            if fea[-1] == 1:
                cnt += 1
        return cnt

    def computeEntropy(self, features):
        allCnt = len(features)
        if allCnt == 0:
            return 0
        postiveCnt = self.countPositive(features)
        posProbility = postiveCnt/float(allCnt)
        negProbility = 1 - posProbility
        return round(-posProbility * self.safeLog(posProbility, 2) - negProbility * self.safeLog(negProbility, 2), 8)

    def computeGain(self, features, attributeIndex):
        childs = self.splitByAttribute(features, attributeIndex)
        dimension = len(childs)
        assert dimension == len(self.featureValue[attributeIndex])
        wholeEntropy = self.computeEntropy(features)
        wholeNumber = len(features)
        entropy = 0
        for child in childs:
            entropy += self.computeEntropy(child) * len(child)/float(wholeNumber)
        return wholeEntropy - entropy, childs

    def predict(self, feature):
        p = self.dTreeRoot
        if p == None:
            return 'none'
        while(p.attribute >= 0):
            p = p.children[feature[p.attribute]]
        return p.label

    def predictTestSet(self):
        print('start prediction on test set ...')
        start = time.clock()
        self.readTestFeatures()
        self.contrastResult = []
        self.contrastResult.append([feature[-1] for feature in self.testFeatures])
        predictions = []
        for feature in self.testFeatures:
            predictions.append(self.predict(feature))
        self.contrastResult.append(predictions)
        elapsed = time.clock() - start
        print('prediction done using ' + str(elapsed) +' seconds')

    def computeChiSquaredCriterion(self, features, childs):
        wholeCnt = len(features)
        posCnt = self.countPositive(features)
        negCnt = wholeCnt - posCnt
        chi = 0
        for child in childs:
            if len(child) == 0:
                continue
            propotion = len(child) / float(wholeCnt)
            posEstimate = posCnt * propotion
            negEstimate = negCnt * propotion
            posActual = self.countPositive(child)
            negActual = len(child) - posActual
            chi += (pow((posActual - posEstimate), 2)/float(posEstimate) + pow((negActual - negEstimate), 2)/float(negEstimate))
        return chi

    def evaluate(self, labels, predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(predictions)):
            if predictions[i] == 1:
                if labels[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if labels[i] == 1:
                    FN += 1
                else:
                    TN += 1
        if TP == 0:
            precision = 0
            recall = 0
        else:
            precision = float(TP)/(TP + FP)
            recall = float(TP)/(TP + FN)
        if precision == 0 and recall == 0:
            F1 = 0
        else:
            F1 = 2*recall*precision/(recall + precision)

        accuracy = (TP + TN) / float(len(predictions))

        return [accuracy, precision, recall, F1]

    def safeLog(self, x, base):
        if x == 0:
            return 0
        if base == 0:
            print('base can not be 0')
            return
        return math.log(x, base)

    def getTreeHeight(self, node):
        if node.label == -1:
            return 0
        max = -1
        for child in node.children:
            h = self.getTreeHeight(node.children[child])
            if h > max:
                max = h
        return max + 1