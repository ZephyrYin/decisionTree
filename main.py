__author__ = 'zephyrYin'

from decisionTree import DecisionTree

dT = DecisionTree('data/featnames.csv', 'data/trainfeat.csv', 'data/trainlabs.csv', 'data/testfeat.csv', 'data/testlabs.csv', 0.01)

#dT = DecisionTree('data/weatherFeatureName.csv', 'data/weatherTrainFeature.csv', 'data/weatherTrainLabel.csv', 'data/weatherTrainFeature.csv', 'data/weatherTrainLabel.csv', 1)
dT.buildTree()
dT.predictTestSet()

result = dT.evaluate(dT.contrastResult[0], dT.contrastResult[1])
print(result)
print(str(dT.nodeCnt) + ' nodes')

# dT.readTrainFeatures()
# wholeCnt = len(dT.testFeatures)
# posCnt = dT.countPositive(dT.testFeatures)
# print(wholeCnt)
# print(posCnt)
# print(posCnt/float(wholeCnt))

# dT.readFeatureNames()
# dT.readTrainFeatures()
# dT.readTestFeatures()
#
# for testFea in dT.testFeatures:
#     for i in range(len(testFea)-1):
#         if testFea[i] not in dT.featureValue[i]:
#             print i
#             print(str(testFea[i]) + ' not in ' + str(dT.featureValue[i]))
#
# dT.buildTree()
# print('tree height')
# print(dT.getTreeHeight(dT.dTreeRoot))