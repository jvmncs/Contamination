# Spark ML Gradient-Boosted Trees

# Input data
training=sqlContext.read.load("hdfs://**:**/labdata/trainfinal",format='parquet',header="false",inferSchema="true")
validation = sqlContext.read.load("hdfs://**:**/labdata/valfinal",format="parquet",header="false",inferSchema="true")
test = sqlContext.read.load("hdfs://**:**/labdata/testfinal",format="parquet",header="false",inferSchema="true")

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np

def gridBuildGBT(maxDepthList,maxBinsList,maxIterList,stepSizeList):
	gridList = []
	for x in maxDepthList:
		for y in maxBinsList:
			for z in maxIterList:
				for a in stepSizeList:
					paramDict = {"maxDepth":x,"maxBins":y, "maxIter":z, "stepSize":a}
					gridList.append(paramDict)
	return gridList

def GBTGenerator(gridList):
	modelList = []
	for x in gridList:
		modelList.append([GBTClassifier(maxDepth = x["maxDepth"], maxBins = x["maxBins"], maxIter = x["maxIter"], stepSize = x['stepSize'], seed = 4321),x])
	return modelList

def gridSearchGBT(modelList, traindf,valdf):
	AUCList=[]
	for x in modelList:
		model = x[0].fit(traindf)
		predsLabels = model.transform(valdf)
		evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
		AUCList.append(evaluator.evaluate(predsLabels))
	print "Max (Validation) AUC: "+str(max(AUCList))+".\nOptimal maxDepth: "+str(modelList[np.argmax(AUCList)][1]["maxDepth"])+".\nOptimal maxBins: "+str(modelList[np.argmax(AUCList)][1]["maxBins"])+".\nOptimal maxIter: "+str(modelList[np.argmax(AUCList)][1]["maxIter"])+".\nOptimal stepSize: "+str(modelList[np.argmax(AUCList)][1]["stepSize"])
	return [modelList[np.argmax(AUCList)][1]["maxDepth"], modelList[np.argmax(AUCList)][1]["maxBins"], modelList[np.argmax(AUCList)][1]["maxIter"], modelList[np.argmax(AUCList)][1]["stepSize"]]

# Final paramList = [6,40,25,.1]

def GBTTrainer(paramList,traindf):
	# paramList = [maxDepth, maxBins, maxIter, stepSize]
	model = GBTClassifier(maxDepth = paramList[0], maxBins = paramList[1], maxIter = paramList[2], stepSize = paramList[3], seed = 4321)
	return model.fit(traindf)

def modelSummary(model):
	testPL = model.transform(test)
	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
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

# Test AUC: 0.91725163474.
# Test Accuracy: 0.99451754386.
# Percent improvement from test baseline accuracy: 0.81512%.
# Test Sensitivity: 0.83784.
# Test Specificity: 0.99667.
