# Spark ML Random Forest Classifier

# Input data
training=sqlContext.read.load("hdfs://**:**/labdata/trainfinal",format='parquet',header="false",inferSchema="true")
validation = sqlContext.read.load("hdfs://**:**/labdata/valfinal",format="parquet",header="false",inferSchema="true")
test = sqlContext.read.load("hdfs://**:**/labdata/testfinal",format="parquet",header="false",inferSchema="true")

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np

def RFGenerator(nTreeList):
	modelList=[]
	if (type(nTreeList)==list) & (len(nTreeList)>0):
		for x in nTreeList:
			if type(x)==int:
				modelList.append([RandomForestClassifier(numTrees=x),x])
			else:
				return "nTreeList must contain all integers."
		return modelList
	else:
		return "nTreeList must be a nonempty list."

def RFAUCGather(modelList,traindf,valdf):
	AUCList=[]
	for x in modelList:
		model = x[0].fit(traindf)
		predsLabels = model.transform(valdf)
		evaluator = BinaryClassificationEvaluator()
		AUCList.append(evaluator.evaluate(predsLabels))
	print "Max (Validation) AUC: "+str(max(AUCList))
	print "Optimal number of trees: "+str(modelList[np.argmax(AUCList)][1])
	return modelList[np.argmax(AUCList)][1]

def RFTrainer(numTrees,traindf):
	RF = RandomForestClassifier(numTrees=numTrees)
	model = RF.fit(traindf)
	predsLabels = model.transform(traindf)
	evaluator = BinaryClassificationEvaluator()
	print "Training AUC: "+str(evaluator.evaluate(predsLabels))
	print "Training Accuracy: "+str(predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))
	print "Percent improvement from baseline accuracy: "+str(round(((predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))/(predsLabels.filter(predsLabels.label==0).count()/float(predsLabels.count()))-1)*100,5))+"%."
	return model

def RFModel(nTreeList):
	modelList = RFGenerator(nTreeList)
	numTrees = RFAUCGather(modelList,validation)
	return RFTrainer(numTrees,training)

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
