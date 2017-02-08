# Spark ML Random Forest Classifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np

# Input data
training=sqlContext.read.load("hdfs://lebron:9000/labdata/trainfinal",format='parquet',header="false",inferSchema="true")
validation = sqlContext.read.load("hdfs://lebron:9000/labdata/valfinal",format="parquet",header="false",inferSchema="true")
test = sqlContext.read.load("hdfs://lebron:9000/labdata/testfinal",format="parquet",header="false",inferSchema="true")

def measurableRF(numTrees=1, traindf=training, valdf=validation):
	'''The function to be optimized with Bayesian Optimization, 
	trained on traindf and evaluated on valdf.'''
	RF = RandomForestClassifier(numTrees=round(numTrees), seed = 4321)
	model = RF.fit(traindf)
	predsLabels = model.transform(traindf)
	evaluator = BinaryClassificationEvaluator()
	print "Training AUC: "+str(evaluator.evaluate(predsLabels))
	validationPL = model.transform(valdf)
	evaluator = BinaryClassificationEvaluator()
	return 	evaluator.evaluate(validationPL)

params={'numTrees':(5,1000)}
RFBO = BayesianOptimization(measurableRF,params)
RFBO.maximize(init_points=5,n_iter=50,acq='ei')

def RFTrainer(numTrees=1,traindf=training):
	'''Trains a random forest model on traindf with specified number of trees.'''
	RF = RandomForestClassifier(numTrees=numTrees, seed=4321)
	model = RF.fit(traindf)
	predsLabels = model.transform(traindf)
	evaluator = BinaryClassificationEvaluator()
	print "Training AUC: "+str(evaluator.evaluate(predsLabels))
	print "Training Accuracy: "+str(predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))
	print "Percent improvement from baseline accuracy: "+str(round(((predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))/(predsLabels.filter(predsLabels.label==0).count()/float(predsLabels.count()))-1)*100,5))+"%."
	return model

def modelSummary(model,testdf=test):
	'''Prints out test metrics for model as evaluated on testdf.'''
	testPL = model.transform(testdf)
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
	pass

#Max Validation AUC: 0.99706426161945494
#>>> modelSummary(RFTrainer(144))
#Training AUC: 0.998993859919
#Training Accuracy: 0.997452652564
#Percent improvement from baseline accuracy: 1.13419%.
#Test AUC: 0.931876671039.
#Test Accuracy: 0.997076023392.
#Percent improvement from test baseline accuracy: 1.07447%.
#Test Sensitivity: 0.86486.
#Test Specificity: 0.99889.
