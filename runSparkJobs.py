# Import to make subprocess calls from python.
import subprocess
import sys
# sys.argv[1] contains the corresponding python frile name which needs to be submitted to Spark.
str1 = "/usr/bin/spark-submit" + str(sys.argv[1]) +"\'hdfs:/final/acq/Acquisition"
l1=[]
# We make a list of all the commands which need to be run in sequence.
for i in range(2000,2017):
    str2 = str1 + str(i) + "*.bz2\' \'hdfs:/final/"+str(i) + "\'"
    procs_list = subprocess.Popen(str2,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
