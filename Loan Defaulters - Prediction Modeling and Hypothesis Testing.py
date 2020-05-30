#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############################################################################################
## Loan Defaulters - Prediction Modeling and Hypothesis Testing
##
## Preprocess the data and perform hypothesis testing and classification ML Models
##
## Team Name: A Walk in the Spark
## Team Members : Aman Agarwal, Mayank Tiwari, Abhinay Reddy Madunanthu, Ujjwal Arora 
##
## Data pipeline - HDFS and Spark
## Data concepts - Machine Learning(Classification and Regression) and Hypothesis Testing
## Cluster - Google Cloud DataProc (5 node cluster) + Debian(1.4.27-debian9)
#############################################################################################

#General Spark and python library imports
import pyspark
import sys
import scipy
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import LabelEncoder
from handyspark import *
import string
import seaborn as sns

#Spark SQL imports
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import SQLContext

#Spark ML imports
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer,StandardScaler,Imputer
from pyspark.ml.regression import LinearRegression,GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.stat._statistics import Statistics


# In[2]:


#the rows in the data are pipe seperated, thus splitting by pipe
def parseDataForCorr(x):
    x = x.split("|")
    return x


# In[3]:


def getMonthYear(record):
    """
    Get month and year from the origination date and first payment date attributes
    """
    columnData = record.split("|")
    originationDate = columnData[acqColList.value['origination_date']]
    firstPaymentDate = columnData[acqColList.value['first_payment_date']]
    originationMonth = int(originationDate.split('/')[0])
    originationYear = int(originationDate.split('/')[1])
    firstPaymentMonth = int(firstPaymentDate.split('/')[0])
    firstPaymentYear = int(firstPaymentDate.split('/')[1])
    dateYearList = [originationMonth,originationYear,firstPaymentMonth,firstPaymentYear]
    return ((originationYear,(columnData[acqColList.value['loan_id']:acqColList.value['original_loan_term']+1] + dateYearList +
    columnData[acqColList.value['ltv']:acqColList.value['relocation_mortgage_indicator']+1])))
    


# In[4]:


def dropColumns(record):
    """
    Dropping column zip and insurance Type - redundant column
    We already have information whether insurance was taken or not
    """
    #dropping zip, insuranceType
    loanYear = record[0]
    loanAcqDetails = record[1]
    list1 = loanAcqDetails[acqColListFE.value['loan_id']:acqColListFE.value['property_state']+1]
    list2 = loanAcqDetails[acqColListFE.value['mi']:acqColListFE.value['product_type']+1]
    list3 = [loanAcqDetails[acqColListFE.value['relocation_mortgage_indicator']]]
#     return ((loanYear),(list1+list2+list3))
    return list1+list2+list3


# In[5]:


def getDummies(record,defualterDf):
    """
    Function to do label encoding, filling nan values and joining defaulter's data
    not used in the final code iteration as it was not distributed
    """
    loanYear = record[0]
    acqYearDetails = record[1]
    defaultersLoanId = list(defualterDf['default'])
    acqYearDetailsList = list(acqYearDetails)
    df = pd.DataFrame(acqYearDetailsList, columns =list(acqColListDC.value))
    cat = df.select_dtypes(include=['object'])
    num = df.drop(cat.columns, axis=1)
    le = LabelEncoder()
    data = df[cat.columns].apply(le.fit_transform)
    df = pd.concat([num,data], axis=1).reset_index(drop=True)
    df.dropna(subset=['cltv', 'borrower_count', 'dti','min_credit_score'], inplace=True)
    df.fillna(df.mean(),inplace = True)
    df['default'] = df['loan_id'].apply(lambda x: 1 if x in defaultersLoanId else 0)
    return (loanYear,df)


# In[6]:


def fillNaAndLabelEncode(record,defualterDf):
    """
    Function to fillna row by row
    not used in the final code iteration as it was not distributed
    """
    loanYear = record[0]
    acqYearDetails = record[1]
    defaultersLoanId = list(defualterDf['default'])
    for i in range(len(acqYearDetails)):
        if not acqYearDetails[i]:
            acqYearDetails[i] = 0
            
    loanId = acqYearDetails[acqColListDC.value['loan_id']]
    
    if loanId in defaultersLoanId:
        return((loanYear),(acqYearDetails+[1]))
    else:
        return((loanYear),(acqYearDetails+[0]))


# In[7]:


def filterOnYearAndCredit(record):
    """
    Function to filter out rows which had acquisition year before 2000
    Also filter out rows where there was no credit score for both borrower and co-borrower (if present)
    """
    year = record[0]
    acqDetails = record[1]
    borrower_credit_score = acqDetails[acqColListMY.value['borrower_credit_score']]
    coborrower_credit_score = acqDetails[acqColListMY.value['co_borrower_credit_score']]
    if(year>=2000 and (borrower_credit_score !='' or coborrower_credit_score!='')):
        return True
    return False


# In[8]:


def featureEngineering(record):
    """
    Function to introduce 2 new engineered features:
    MI: Mortgage Insurance - whether the borrower has taken insurance on loan or not
    Min_credit_score: minimum credit score of borrower and co-borrower
    """
    loanYear = record[0]
    loanAcqDetails = record[1]
    ins_perc = loanAcqDetails[acqColListMY.value['insurance_percentage']]
    borrower_credit_score = loanAcqDetails[acqColListMY.value['borrower_credit_score']]
    coborrower_credit_score = loanAcqDetails[acqColListMY.value['co_borrower_credit_score']]
    mi = 0
    if(ins_perc !='' and int(ins_perc)>0):
        mi = 1
    if(borrower_credit_score ==''):
        borrower_credit_score = '1000'
    if(coborrower_credit_score ==''):
        coborrower_credit_score = '1000'
    borrower_credit_score = int(borrower_credit_score)
    coborrower_credit_score = int(coborrower_credit_score)
    min_credit_score = 1000
    if(borrower_credit_score<coborrower_credit_score):
        min_credit_score = borrower_credit_score
    else:
        min_credit_score = coborrower_credit_score
    
    list1 = loanAcqDetails[acqColListMY.value['loan_id']:acqColListMY.value['dti']+1]
    list3 = loanAcqDetails[acqColListMY.value['first_time_homebuyer']:acqColListMY.value['zip']+1]
    list5 = [loanAcqDetails[acqColListMY.value['product_type']]]
    list6 = loanAcqDetails[acqColListMY.value['mortgage_insurance_type']:acqColListMY.value['relocation_mortgage_indicator']+1]
    return ((loanYear),(list1+[min_credit_score]+list3+[mi]+list5+list6))


# In[9]:


def fillNullValuesSparkImputer(out_array,inp_array, df):
    """
    Filling missing values through mean
    Using spark's imputer which imputes missing values, either using the mean or the median 
     of the columns in which the missing values are located.
    Mean/median value is computed after filtering out missing values. 
    """
    imputer = Imputer(inputCols=inp_array, outputCols=out_array).setStrategy("mean")
    model = imputer.fit(df)

    df_nonNull = model.transform(df)
    return df_nonNull


# In[10]:


def standardizeData(df):
    """
    Using spark standardScaler to standardize the columns
    Output's a vector of scaled features 
    """
    assembler = VectorAssembler(inputCols=out_array,outputCol="features")
    output = assembler.transform(df_nonNull)

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(output)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(output)
    return scaledData


# In[11]:


def hypothesisTesting(data):
    """
    Performing the hypothesis testing on the scaled features
    Uses spark's Generalized linear regression
    Output's Standard Errors, T value and P Value
    """
#     data.printSchema()
    df = data.select(['scaledFeatures', 'is_foreclosed'])

    lr = GeneralizedLinearRegression(family="gaussian", link="identity", 
                                     maxIter=10, regParam=0.3,featuresCol = 'scaledFeatures',labelCol ='is_foreclosed')
    lrModel = lr.fit(df)
    print(f'Intercept: {lrModel.intercept}\nCoefficient: {lrModel.coefficients.values}')
    print()
    summary = lrModel.summary
    print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    print()
    print("T Values: " + str(summary.tValues))
    print()
    print("P Values: " + str(summary.pValues))
    print()
    return summary
    # print("Dispersion: " + str(summary.dispersion))
    # print("Null Deviance: " + str(summary.nullDeviance))
    # print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    # print("Deviance: " + str(summary.deviance))
    # print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    # # print("AIC: " + str(summary.aic))
    # print("Deviance Residuals: ")
    # summary.residuals().show()


# In[12]:


def logisticRegression(trainingData,testData):
    """
    Performing logistic regression binary classification for loan default prediction
    Output is a vector of predicted values and Area under ROC
    """
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))
    return predictions


# In[13]:


def randomForestClassifier(trainingData,testData):
    """
    Performing random forest binary classification for loan default prediction
    Output is a vector of predicted values and Area under ROC
    """
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    rfModel = rf.fit(trainingData)
    predictions = rfModel.transform(testData)
    evaluator = BinaryClassificationEvaluator()
#     auprc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
#     print("Area under PR Curve: {:.4f}".format(auprc))
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    return predictions


# In[14]:


def gradientBoost(trainingData,testData):
    """
    Performing gradient boost binary classification for loan default prediction
    Output is a vector of predicted values and Area under ROC
    """
    gbt = GBTClassifier(maxIter=10)
    gbtModel = gbt.fit(trainingData)
    predictions = gbtModel.transform(testData)
    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    return predictions


# In[15]:


def applyBonferroniCorrection(pValList, hypothesisCount,colName):
    return [('Column Name: ', colName[index], 'P-Value: ', pVal, 'Corrected P-Value: ', pVal * hypothesisCount) for index, pVal in enumerate(pValList)]


# In[16]:


#Spark Context, Spark Session and Spark SQL context
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
sqlContext = SQLContext(sc)


# In[17]:


#Dictionary of column names present in the acquistion file and generated during preprocessing

acqCols = {'loan_id':0,'channel':1,'seller_name':2,'original_interest_rate':3,'original_unpaid_principal_balance':4,'original_loan_term':5,
           'origination_date':6,'first_payment_date':7,'ltv':8,'cltv':9,'borrower_count':10,'dti':11,'borrower_credit_score':12,
          'first_time_homebuyer':13,'loan_purpose':14,'property_type':15,'unit_count':16,'occupancy_status':17,'property_state':18,'zip':19,
          'insurance_percentage':20,'product_type':21,'co_borrower_credit_score':22,'mortgage_insurance_type':23,
          'relocation_mortgage_indicator':24,'unknownCol':25}

acqColsAfterMonthYear = {'loan_id':0,'channel':1,'seller_name':2,'original_interest_rate':3,'original_unpaid_principal_balance':4,'original_loan_term':5,
           'origination_month':6,'origination_year':7,'first_payment_month':8,'first_payment_year':9,'ltv':10,'cltv':11,'borrower_count':12,'dti':13,'borrower_credit_score':14,
          'first_time_homebuyer':15,'loan_purpose':16,'property_type':17,'unit_count':18,'occupancy_status':19,'property_state':20,'zip':21,
          'insurance_percentage':22,'product_type':23,'co_borrower_credit_score':24,'mortgage_insurance_type':25,
          'relocation_mortgage_indicator':26}

acqColsAfterFeatureEngineering = {'loan_id':0,'channel':1,'seller_name':2,'original_interest_rate':3,'original_unpaid_principal_balance':4,'original_loan_term':5,
           'origination_month':6,'origination_year':7,'first_payment_month':8,'first_payment_year':9,'ltv':10,'cltv':11,'borrower_count':12,'dti':13,'min_credit_score':14,
          'first_time_homebuyer':15,'loan_purpose':16,'property_type':17,'unit_count':18,'occupancy_status':19,'property_state':20,'zip':21,
          'mi':22,'product_type':23,'mortgage_insurance_type':24,'relocation_mortgage_indicator':25}

acqColsAfterDropColumns = {'loan_id':0,'channel':1,'seller_name':2,'original_interest_rate':3,'original_unpaid_principal_balance':4,'original_loan_term':5,
           'origination_month':6,'origination_year':7,'first_payment_month':8,'first_payment_year':9,'ltv':10,'cltv':11,'borrower_count':12,'dti':13,'min_credit_score':14,
          'first_time_homebuyer':15,'loan_purpose':16,'property_type':17,'unit_count':18,'occupancy_status':19,'property_state':20,'mi':21,
            'product_type':22,'relocation_mortgage_indicator':23}


# In[18]:


#broadcast variable for the dictionaries of column names
acqColList = sc.broadcast(acqCols)
acqColListMY = sc.broadcast(acqColsAfterMonthYear)
acqColListFE = sc.broadcast(acqColsAfterFeatureEngineering)
acqColListDC = sc.broadcast(acqColsAfterDropColumns)


# In[19]:


###############################################################################################
##
## Data Path is the hdfs path of the data file of a year on which we are predicting defaulters
## Defaulter data path is the preprocessed list of defaulters over all years stored in the hdfs
##
###############################################################################################
# data_path = 'hdfs:/final/acq/Acquisition_2005*'
data_path = sys.argv[1]
defaulter_data_path = 'hdfs:/final/defaulterData/part-*'


# In[20]:


#reading the contents of the file in a rdd
rdd = sc.textFile(data_path)
rdd.take(5)


# In[21]:


rdd.count()


# In[22]:


"""
Preprcoessing the acquistion file data
Converting origination and first Payment date to month and year
Introducing 2 new features- min credit score and mortgage insurance (binary)
Dropping some non relevant columns
"""
processedRDD_W_O_Dummies = rdd.map(getMonthYear).filter(filterOnYearAndCredit).map(featureEngineering).map(dropColumns)
processedRDD_W_O_Dummies.count()


# In[23]:


"""
Reading the defaulters id from hdfs and converting it to a spark dataframe
"""
row = Row("default")
defaulterData = sc.textFile(defaulter_data_path).map(lambda x: int(eval(x)[0])).map(row).toDF()
defaulterData.printSchema()


# In[24]:


"""
Converting rdd of acquistion data to spark dataframe
"""
# processedRDD_W_O_DummiesDF = processedRDD_W_O_Dummies.toDF(['loan_id', 'channel', 'seller_name', 'original_interest_rate', 'original_unpaid_principal_balance', 'original_loan_term', 'origination_month', 'origination_year', 'first_payment_month', 'first_payment_year', 'ltv', 'cltv', 'borrower_count', 'dti', 'min_credit_score', 'first_time_homebuyer', 'loan_purpose', 'property_type', 'unit_count', 'occupancy_status', 'property_state', 'mi', 'product_type', 'relocation_mortgage_indicator'])
processedRDD_W_O_DummiesDF = processedRDD_W_O_Dummies.toDF(list(acqColsAfterDropColumns))
# processedRDD_W_O_DummiesDF.printSchema()
processedRDD_W_O_DummiesDF_Final = processedRDD_W_O_DummiesDF                                                           


# In[25]:


"""
Converting categorical non numerical columns to numerical values
Using Spark's spring indexer for generating indexes for the column values
New ColName = oldColName+"_index"
Preserves old columns also
"""
catCols = [
    "channel", "seller_name", "first_time_homebuyer", "loan_purpose", 
    "product_type", "property_type", "occupancy_status", "relocation_mortgage_indicator"]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(processedRDD_W_O_DummiesDF_Final) for column in catCols]
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(processedRDD_W_O_DummiesDF_Final).transform(processedRDD_W_O_DummiesDF_Final)


# In[26]:


"""
Peforming a left outer join on loan id and defaulter id to join the acquistion and defaulter's data
If loan id in acqusition file is also in defaulter list then defualt will be loan id else null
"""
# sqlContext.sql("select o.loan_id, o.channel, CASE WHEN d.default IS NULL then 0 ELSE 1 END from df_r o LEFT JOIN defaulterData d ON o.loan_id = d.default").head()
full_data = df_r.join(defaulterData, df_r.loan_id == defaulterData.default, "left_outer")


# In[27]:


"""
Creating a temproray view for performing Spark SQL queries
"""
full_data.createOrReplaceTempView("loan_data_default")


# In[28]:


"""
Query to get the count of number of defaulters in joined dataframe
"""
spark.sql("SELECT count(*) FROM loan_data_default where default IS NOT NULL").show()


# In[29]:


"""
Converting default column to 1(default) or 0(not default)
If default column and loan id column in joined dataframe are equal then default is 1 else 0
"""

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import when

full_data_1 = full_data.withColumn('is_foreclosed', when(full_data['default'] == full_data['loan_id'], 1).otherwise(0))


# In[30]:


full_data_1.createOrReplaceTempView("loan_data_default_joined")
spark.sql("SELECT is_foreclosed, count(*) FROM loan_data_default_joined group by is_foreclosed").show()


# In[31]:


"""
Casting the data type of columns to double
"""
full_data_casted = full_data_1.withColumn('original_interest_rate',full_data_1["original_interest_rate"].cast("double").alias("original_interest_rate")).withColumn('original_unpaid_principal_balance',full_data_1["original_unpaid_principal_balance"].cast("double").alias("original_unpaid_principal_balance")).withColumn('original_loan_term',full_data_1["original_loan_term"].cast("double").alias("original_loan_term")).withColumn('ltv',full_data_1["ltv"].cast("double").alias("ltv")).withColumn('cltv',full_data_1["cltv"].cast("double").alias("cltv")).withColumn('borrower_count',full_data_1["borrower_count"].cast("double").alias("borrower_count")).withColumn('dti',full_data_1["dti"].cast("double").alias("dti")).withColumn('unit_count',full_data_1["unit_count"].cast("double").alias("unit_count")).withColumn('origination_month',full_data_1["origination_month"].cast("double").alias("origination_month")).withColumn('origination_year',full_data_1["origination_year"].cast("double").alias("origination_year")).withColumn('first_payment_month',full_data_1["first_payment_month"].cast("double").alias("first_payment_month")).withColumn('first_payment_year',full_data_1["first_payment_year"].cast("double").alias("first_payment_year")).withColumn('min_credit_score',full_data_1["min_credit_score"].cast("double").alias("min_credit_score")).withColumn('mi',full_data_1["mi"].cast("double").alias("mi"))


# In[32]:


full_data_casted.printSchema()


# # Filling Null Values
# 

# In[33]:


out_array = []
inp_array =["original_interest_rate", "original_unpaid_principal_balance", "original_loan_term","origination_month", 
    "origination_year","first_payment_month", "first_payment_year", "ltv", "cltv","borrower_count","dti","min_credit_score",
    "unit_count", "mi", "channel_index","seller_name_index","first_time_homebuyer_index","loan_purpose_index", 
    "product_type_index", "property_type_index", "occupancy_status_index","relocation_mortgage_indicator_index"]
for val in inp_array:
    out_array.append("out_"+val)
df_nonNull = fillNullValuesSparkImputer(out_array, inp_array, full_data_casted)


# In[34]:


df_nonNull.show(1)


# # Standardization

# In[35]:


scaledData = standardizeData(df_nonNull)
scaledData.show(2)


# # Correlation Plot

# In[36]:


def makeColumnsReadable(columns):
    """
    Function to transform column to make it more readable
    """
    outputColumns = []
    for col in columns:
        col = col.replace("out_", "").replace("_index", "").replace('_', ' ')
        outputColumns.append(string.capwords(col))
    return outputColumns


def plotCorrelationMatirx(corr, columns):
    """
    Function to plot the correlatiom plot of the given input
    """
    corrDf = pd.DataFrame(corr, columns)
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    return sns.heatmap(
        corrDf, 
        mask=mask, 
        cmap=cmap, 
        vmax=.3, center=0, square=True, linewidths=.5, 
        cbar_kws={"shrink": .5}
    )


def computeCorrelation(inputDataFrame):
    """
    Function to compute correlation for a dataframe 
    """
    inputRDD = inputDataFrame.rdd.map(tuple)
    return Statistics.corr(inputRDD, method="pearson")


corr = computeCorrelation(scaledData.select(out_array))

f, ax = plt.subplots(figsize=(11, 9))

columns = makeColumnsReadable(out_array)
plotCorrelationMatirx(corr, columns);


# # Hypothesis Testing

# In[37]:


# Getting p values for the hypothesis testing column
numHypothesis = len(out_array)
colsName = makeColumnsReadable(out_array)
colsName = colsName + ['biasedCol']
summary = hypothesisTesting(scaledData)
pValCorrectedPVal = applyBonferroniCorrection(summary.pValues,numHypothesis,colsName)
for val in pValCorrectedPVal:
    print(val)
sc.parallelize(summary.pValues).saveAsTextFile(sys.argv[2]+'/pval/')


# # Test train split and aliasing col Names
# 
# 

# In[38]:


"""
Changing the input col name to feature and ouptut col name to label
Features is a vector of input features and label is a vector of output col
"""
data = scaledData.select(col("scaledFeatures").alias("features"), col("is_foreclosed").alias("label"))
data.printSchema()


# In[39]:


#test and train split
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=11)
evaluator = BinaryClassificationEvaluator()


# # LOGISTIC REGRESSION MODEL

# In[50]:


predictionsLR = logisticRegression(trainingData,testData)
lrROCVal = evaluator.evaluate(predictionsLR)
dataLR = [lrROCVal]
sc.parallelize(dataLR).saveAsTextFile(sys.argv[2]+'/logisticRegression/')


# # LOGISTIC REGRESSION PLOT

# In[ ]:


bcm = BinaryClassificationMetrics(predictionsLR, scoreCol='probability', labelCol='label')

print("Area under ROC Curve: {:.4f}".format(bcm.areaUnderROC))
print("Area under PR Curve: {:.4f}".format(bcm.areaUnderPR))

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
bcm.plot_roc_curve(ax=axs[0])
bcm.plot_pr_curve(ax=axs[1]);


# In[ ]:


bcm.print_confusion_matrix(.415856);


# # Random Forest

# In[ ]:


predictionsRF = randomForestClassifier(trainingData,testData)
rfROCVal = evaluator.evaluate(predictionsRF)
dataRF = [rfROCVal]
sc.parallelize(dataRF).saveAsTextFile(sys.argv[2]+'/randomForest/')


# # Gradient Boost

# In[ ]:


predictionsGB = gradientBoost(trainingData,testData)
gbROCVal = evaluator.evaluate(predictionsGB)
dataGB = [gbROCVal]
sc.parallelize(dataGB).saveAsTextFile(sys.argv[2]+'/gradientBoost/')


# In[ ]:


# # GRADIENT BOOST PLOT

# In[ ]:


bcmGB = BinaryClassificationMetrics(predictionsGB, scoreCol='probability', labelCol='label')

print("Area under ROC Curve: {:.4f}".format(bcmGB.areaUnderROC))
print("Area under PR Curve: {:.4f}".format(bcmGB.areaUnderPR))

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
bcmGB.plot_roc_curve(ax=axs[0])
bcmGB.plot_pr_curve(ax=axs[1]);


# In[ ]:


bcmGB.print_confusion_matrix(.415856);

