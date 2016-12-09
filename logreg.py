# Spark ML Logistic Regression

# Input data
training=sqlContext.read.load("hdfs://**:**/labdata/trainfinal",format='parquet',header="false",inferSchema="true")
validation = sqlContext.read.load("hdfs://**:**/labdata/valfinal",format="parquet",header="false",inferSchema="true")
test = sqlContext.read.load("hdfs://**:**/labdata/testfinal",format="parquet",header="false",inferSchema="true")

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Normalizer
import numpy as np

def normalizeFeats(df):
	# Discarded after I realized LogisticRegression has a parameter that automatically standardizes data
	normalizer = Normalizer(inputCol="features", outputCol="newFeats")
	return normalizer.transform(df).select("label","newFeats").withColumnRenamed("newFeats","features")

def gridBuildLR(maxIterList,regParamList,elasticNetList):
	gridList = []
	if len(maxIterList)>0:
		if len(regParamList)>0:
			if len(elasticNetList)>0:
				for x in maxIterList:
					for y in regParamList:
						for z in elasticNetList:
							dictParam = {"maxIter":x, "regParam":y, "elasticNetParam":z}
							gridList.append(dictParam)
				return gridList
	else:
		return "Each parameter list must be nonempty."

def LRGenerator(gridList,tol):
	modelList=[]
	if (type(gridList)==list) & (len(gridList)>0):
		for x in gridList:
			modelList.append([LogisticRegression(maxIter=x["maxIter"], regParam=x["regParam"], elasticNetParam=x["elasticNetParam"],tol=tol),{"maxIter":x["maxIter"],"regParam":x["regParam"],"elasticNetParam":x["elasticNetParam"]}])
		return modelList
	else:
		return "gridList must be a nonempty list."

def AUCGather(modelList, valdf, traindf):
	if (type(modelList)==list) & (len(modelList)>0):
		AUCList=[]
		for x in modelList:
			model = x[0].fit(traindf)
			predsLabels = model.transform(validation)
			evaluator = BinaryClassificationEvaluator()
			AUCList.append(evaluator.evaluate(predsLabels))
		print "Max (Validation) AUC: "+str(max(AUCList))+".\nOptimal maxIter: "+str(modelList[np.argmax(AUCList)][1]["maxIter"])+".\nOptimal regParam: "+str(modelList[np.argmax(AUCList)][1]["regParam"])+".\nOptimal elasticNetParam: "+str(modelList[np.argmax(AUCList)][1]["elasticNetParam"])
		return [modelList[np.argmax(AUCList)][1]["maxIter"], modelList[np.argmax(AUCList)][1]["regParam"], modelList[np.argmax(AUCList)][1]["elasticNetParam"]]
	else:
		return "modelList must be a nonempty list"

def trainer(paramList, traindf,tol):
	lr = LogisticRegression(maxIter=paramList[0], regParam=paramList[1], elasticNetParam=paramList[2],tol=tol)
	model = lr.fit(traindf)
	predsLabels = model.summary.predictions
	print "Training AUC: "+str(model.summary.areaUnderROC)
	print "Training Accuracy: "+str(predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))
	print "Percent improvement from baseline accuracy: "+str(round(((predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))/(predsLabels.filter(predsLabels.label==0).count()/float(predsLabels.count()))-1)*100,5))+"%."
	return model

def LRModel(maxIterList,regParamList,elasticNetList,tol):
	gridList = gridBuildLR(maxIterList,regParamList,elasticNetList)
	modelList = LRGenerator(gridList,tol)
	paramList = AUCGather(modelList, validation, training)
	return trainer(paramList, training,tol)

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

# Note: do not use tol less than .0001

# Using paramList=[200,.001,.03], we find:
# Test AUC: 0.970409460962.
# Test Accuracy: 0.99634502924.
# Percent improvement from test baseline accuracy: 1.00037%.
# Test Sensitivity: 0.81081.
# Test Specificity: 0.99889.
