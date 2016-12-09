# Spark ML Decision Tree

# Input Data
training=sqlContext.read.load("hdfs://**:**/labdata/trainfinal",format='parquet',header="false",inferSchema="true")
validation = sqlContext.read.load("hdfs://**:**/labdata/valfinal",format="parquet",header="false",inferSchema="true")
test = sqlContext.read.load("hdfs://**:**/labdata/testfinal",format="parquet",header="false",inferSchema="true")

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np

def gridBuildDT(maxDepthList, maxBinsList):
	gridList=[]	
	for x in maxDepthList:
		for y in maxBinsList:
			paramDict = {"maxDepth":x,"maxBins":y}
			gridList.append(paramDict)
	return gridList

def DFGenerator(gridList):
	modelList = []
	for x in gridList:
		modelList.append([DecisionTreeClassifier(maxDepth=x['maxDepth'],maxBins=x['maxBins'],maxMemoryInMB=1024, seed=4321),x])
	return modelList

def gridSearch(modelList, traindf, valdf):
	AUCList=[]
	for x in modelList:
		model = x[0].fit(traindf)
		predsLabels = model.transform(valdf)
		evaluator = BinaryClassificationEvaluator()
		AUCList.append(evaluator.evaluate(predsLabels))
	print "Max (Validation) AUC: "+str(max(AUCList))+".\nOptimal maxDepth: "+str(modelList[np.argmax(AUCList)][1]["maxDepth"])+".\nOptimal maxBins: "+str(modelList[np.argmax(AUCList)][1]["maxBins"])
	return [modelList[np.argmax(AUCList)][1]["maxDepth"], modelList[np.argmax(AUCList)][1]["maxBins"]]

# Optimal parameters:
# maxDepth of 4, maxBin of 16

def DTTrainer(paramList,traindf):
	model = DecisionTreeClassifier(maxDepth=paramList[0],maxBins=paramList[1])
	return model.fit(traindf)

def modelSummary(model):
	testPL = model.transform(test)
	evaluator = BinaryClassificationEvaluator()
	testAUC = evaluator.evaluate(testPL)
	testAcc = testPL.filter(testPL.prediction==testPL.label).count()/float(testPL.count())
	testPercImp = round((float(testAcc)/(testPL.filter(testPL.label==0).count()/float(testPL.count()))-1)*100,5)
	testSens = round(float(testPL.filter(testPL.prediction==1).filter(testPL.label==1).count())/(testPL.filter(testPL.prediction==1).filter(testPL.label==1).count()+testPL.filter(testPL.prediction==0).filter(testPL.label==1).count()),5)
	testSpec = round(float(testPL.filter(testPL.prediction==0).filter(testPL.label==0).count())/(testPL.filter(testPL.prediction==0).filter(testPL.label==0).count()+testPL.filter(testPL.prediction==1).filter(testPL.label==0).count()), 5)
	print "Test AUC: {0}.".format(testAUC)
	print "Test Accuracy: {0}.".format(testAcc)
	print "Percent improvement from test baseline accuracy: {0}%.".format(testPercImp)
	print "Test Sensitivity: {0}.".format(testSens)
	print "Test Specificity: {0}.".format(testSpec)
	return

# Using [4, 16], we find the following
# Test AUC: 0.966313849974.
# Test Accuracy: 0.989400584795.
# Percent improvement from test baseline accuracy: 0.29641%.
# Test Sensitivity: 0.35135.
# Test Specificity: 0.99815.
