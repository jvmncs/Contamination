# Spark ML Logistic Regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Input data
training=sqlContext.read.load("hdfs://lebron:9000/labdata/trainfinal",format='parquet',header="false",inferSchema="true")
validation = sqlContext.read.load("hdfs://lebron:9000/labdata/valfinal",format="parquet",header="false",inferSchema="true")
test = sqlContext.read.load("hdfs://lebron:9000/labdata/testfinal",format="parquet",header="false",inferSchema="true")

def measurableLR(maxIter=500, regParam=.1,elasticNetParam=1, traindf=training, valdf=validation):
	'''The function to be optimized with Bayesian Optimization, 
	trained on traindf and evaluated on valdf.'''
	lr = LogisticRegression(maxIter=round(maxIter), regParam=regParam, elasticNetParam=elasticNetParam)
	model = lr.fit(traindf)
	predsLabels = model.summary.predictions
	print "Training AUC: "+str(model.summary.areaUnderROC)
	validationPL = model.transform(valdf)
	evaluator = BinaryClassificationEvaluator()
	return 	evaluator.evaluate(validationPL)

params = {'regParam':(.00001,1), 'elasticNetParam':(.00001,1)}
BO = BayesianOptimization(measurableLR,params)
BO.maximize(init_points=5, n_iter=50, acq='ei')

def trainer(maxIter=500, regParam=0,elasticNetParam=0, traindf=training):
	'''Trains a logistic regression model on traindf with specified parameters.'''
	lr = LogisticRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
	model = lr.fit(traindf)
	predsLabels = model.summary.predictions
	print "Training AUC: "+str(model.summary.areaUnderROC)
	print "Training Accuracy: "+str(predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))
	print "Percent improvement from baseline accuracy: "+str(round(((predsLabels.filter(predsLabels.prediction==predsLabels.label).count()/float(predsLabels.count()))/(predsLabels.filter(predsLabels.label==0).count()/float(predsLabels.count()))-1)*100,5))+"%."
	return model

def modelSummary(model, testdf=test):
	'''Prints out test metrics for model as evaluated on testdf.'''
	testPL = model.transform(testdf)
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
	pass

#Max validation AUC: 0.99802402224386388
#>>> modelSummary(trainer(maxIter=500,regParam=.00001,elasticNetParam=0.11513282874291846))
#Training AUC: 0.983963340639
#Training Accuracy: 0.997120389855
#Percent improvement from baseline accuracy: 1.10051%.
#Test AUC: 0.974765428637.
#Test Accuracy: 0.99634502924.
#Percent improvement from test baseline accuracy: 1.00037%.
#Test Sensitivity: 0.81081.
#Test Specificity: 0.99889.
