#!/usr/bin/env python
# coding: utf-8

# In[16]:


#############################################################################################
## Defaulter Data and Visualization
##
## Preprocess the data and perform hypothesis testing and classification ML Models
##
## Team Name: A Walk in the Spark
## Team Members : Aman Agarwal, Mayank Tiwari, Abhinay Reddy Madunanthu, Ujjwal Arora 
##
## Data pipeline - HDFS and Spark
## Data concepts - Machine Learning(Classification and Regression) and Hypothesis Testing
#############################################################################################

#General Spark and python library imports
import pyspark
import scipy
import datetime
import pandas
import seaborn as sns
from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext


# # Defaulters Processing

# In[8]:


def makeDefaulterRdd(file):
"""
Function to filter out all the unique Loan Identifiers which went into foreclosure.
Input: File name
Output: Text file on HDFS of Loan Identifier and foreclosure date
"""
    def parseData(x):
        columns = x.split("|")
        return columns[0], columns[15]
    
    rdd = sc.textFile(file)
    inputRDD = rdd.map(parseData)
    foreclosureRDD = inputRDD.filter(lambda x: x[1] is not None and len(x[1]) > 0 ).map(lambda x: (x[0], x[1]))
    foreclosureRDD.saveAsTextFile("hdfs:/final/defaulterData",compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")


# In[114]:


"""
This function takes the entire list of Performance files as input and
adds the output RDD to HDFS.
"""
makeDefaulterRdd("hdfs:/final/perf/*.bz2")


# # Last Entry for Every Loan

# In[12]:


def makeCleanedRdd(file):
"""
Function to clean the RDD to obtain the last entries for every Loan Identifier
Input: File Name
Output: RDD containing the latest entry for every Loan Identifier added to HDFS
"""
    def parseData1(x):
        columns = x.split("|")
        return (columns[0],columns)
    
    def returnLastEntry(x):
        newList = sorted(x[1],key=lambda t: datetime.datetime.strptime(t[1], '%m/%d/%Y'))
        return newList[-1]
    def compareDate(x,y):
        return x if (datetime.datetime.strptime(x[1], '%m/%d/%Y') >= datetime.datetime.strptime(y[1], '%m/%d/%Y')) else y

    rdd = sc.textFile(file)
    rdd = rdd.map(parseData1)
    rdd = rdd.reduceByKey(lambda x,y:compareDate(x,y))
    rdd.saveAsTextFile("hdfs:/final/cleanData",compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")


# In[13]:


"""
This function takes the entire list of Performance files as input and
adds the output RDD to HDFS.
"""
makeCleanedRdd("hdfs:/final/perf/*.bz2")


# # Visualization Plot

# In[ ]:


"""
Preprocessing the data to create a dataframe
"""
def parseDataForCorr(x):
    """
    Function to parse the input RDD record of length 1 to a record of length 12
    which includes all numeric columns and one categorical column from the input record.
    """
    x = x.split("|")
    return (x[3], x[4], x[5], x[8], x[9], x[10], x[11], x[12], x[14], x[20], x[22], x[23])

conf = SparkConf() .setAppName("Defaulter Data and Visualization")

sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession(sc)
rawRdd = sc.textFile("hdfs:/final/acq/*2005*").map(parseDataForCorr)
rddForNumeric = rawRdd.toDF(['OriginalInterestRate','OriginalUnpaiPrincipalBalance','OriginalLoanTerm','OriginalLoanToValue','OriginalCombinedLoanToValue','NumberOfBorrowers','DebtToIncomeRatio','BorrowerCreditScore','LoanPurpose','MortgageInsurancePercentage','CoBorrowerCreditScore','MortgageInsuranceType'])

rddForNumeric = rddForNumeric.withColumn("OriginalInterestRate",rddForNumeric['OriginalInterestRate'].cast("double")).withColumn("OriginalUnpaiPrincipalBalance",rddForNumeric['OriginalUnpaiPrincipalBalance'].cast("double")).withColumn("OriginalLoanTerm",rddForNumeric['OriginalLoanTerm'].cast("double")).withColumn("OriginalLoanToValue",rddForNumeric['OriginalLoanToValue'].cast("double")).withColumn("OriginalCombinedLoanToValue",rddForNumeric['OriginalCombinedLoanToValue'].cast("double")).withColumn("NumberOfBorrowers",rddForNumeric['NumberOfBorrowers'].cast("double")).withColumn("DebtToIncomeRatio",rddForNumeric['DebtToIncomeRatio'].cast("double")).withColumn("BorrowerCreditScore",rddForNumeric['BorrowerCreditScore'].cast("double")).withColumn("MortgageInsurancePercentage",rddForNumeric['MortgageInsurancePercentage'].cast("double")).withColumn("CoBorrowerCreditScore",rddForNumeric['CoBorrowerCreditScore'].cast("double")).withColumn("MortgageInsuranceType",rddForNumeric['MortgageInsuranceType'].cast("double")).withColumn("LoanPurpose",rddForNumeric['LoanPurpose'])

columns = rddForNumeric.columns
columns.remove('LoanPurpose')
imputer = Imputer(inputCols = columns, outputCols=["{}_imputed".format(c) for c in columns])
df_imputed = imputer.fit(rddForNumeric).transform(rddForNumeric)
df_imputed = df_imputed.select(df_imputed['OriginalInterestRate_imputed'],df_imputed['OriginalUnpaiPrincipalBalance_imputed'],df_imputed['OriginalLoanTerm_imputed'],df_imputed['OriginalLoanToValue_imputed'],df_imputed['OriginalCombinedLoanToValue_imputed'],df_imputed['NumberOfBorrowers_imputed'],df_imputed['DebtToIncomeRatio_imputed'],df_imputed['BorrowerCreditScore_imputed'],df_imputed['MortgageInsurancePercentage_imputed'],df_imputed['CoBorrowerCreditScore_imputed'],df_imputed['MortgageInsuranceType_imputed'],df_imputed['LoanPurpose'])

df_forPlot = df_imputed.select(['OriginalUnpaiPrincipalBalance_imputed','BorrowerCreditScore_imputed','LoanPurpose'])
df_forPlot = df_forPlot.toPandas()

"""
Code to plot a scatter plot with 3 columns, BorrowerCreditScore on x-axis, OriginalUnpaiPrincipalBalance on y-axis with hue as LoanPurpose
"""
sns.scatterplot(x=df_forPlot['BorrowerCreditScore_imputed'],y=df_forPlot['OriginalUnpaiPrincipalBalance_imputed'],hue = df_forPlot['LoanPurpose'])

