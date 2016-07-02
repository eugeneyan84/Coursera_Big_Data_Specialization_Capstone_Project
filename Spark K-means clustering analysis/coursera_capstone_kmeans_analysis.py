# This python script performs k-means cluster analysis via spark-submit. The example command is as follows:
#
# spark-submit coursera_capstone_kmeans_analysis.py combined-data-userId-avg.csv 2 avgPurchaseCount avgExpPerSession avgAdClicks
#
# Arguments are as follows:
# 1st: filename of this python script
# 2nd: filename of target CSV file, location is defaulted to '/home/cloudera/flamingo-data/'
# 3rd: non-negative integer as target cluster size. If argument is 0, script would attempt to analyze from k=1 to k=10 and print 
#      results to a log file, otherwise it would only perform analysis of the indicated size, and display a simple matplotlib 
#      scatterplot of the data-points and centers
#  
# 4th parameter and beyond would constitute the targeted attributes to be used for the k-means cluster analysis. At least 2 
# attributes are expected to be specified by user.
#
# For purpose of this assignment, location of targeted CSV file is defaulted to location is defaulted to 
# '/home/cloudera/flamingo-data/'. If location is different, please update the dataFilesDir variable in the script below
#

import pandas as pd
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from datetime import datetime
import sys
import os.path as ospath
import matplotlib.pyplot as plt

if len(sys.argv) < 5:
	print('Required parameters not present. Use \'spark-submit coursera_capstone_kmeans_analysis.py <csv-file-located-in-flamingo-data-folder>\' <target-cluster-size> <attr1> <attr2>...')
	sys.exit()
else:
	targetCSV = sys.argv[1]

#Absolute path to location of required data files
dataFilesDir = '/home/cloudera/flamingo-data/'

if not ospath.exists(dataFilesDir+targetCSV):
	print('Error: '+ dataFilesDir + targetCSV + ' does not exist')
	sys.exit()

targetSize = 0

if sys.argv[2].isdigit():
	if int(sys.argv[2]) > 0:
		targetSize = int(sys.argv[2])
		print('\nExecution of k-means clustering would end with basic matplotlib image for ' +str(targetSize)+ ' center(s).\n')
else:
	print('Error: 2nd parameter (target cluster size) must be a non-negative integer')
	sys.exit()

attrList = sys.argv[3:]

#Context instantiation
conf = (SparkConf()
         .setMaster("local")
         .setAppName("SparkMLlib Clustering Example")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

#Loading CSV and cleaning headers
combinedDataDF = pd.read_csv(dataFilesDir + targetCSV)
combinedDataDF = combinedDataDF.rename(columns=lambda x: x.strip())

#Isolating columns for analysis
trainingDF = combinedDataDF[attrList]

#Transform to required format for k-means training
pDF = sqlContext.createDataFrame(trainingDF)
parsedData = pDF.rdd.map(lambda l: array([l[0], l[1], l[2]])) # 0-> 'avgPurchaseCount', 1-> 'avgHits', 2-> 'avgAdClicks'


if targetSize == 0:
	#Redirect console output to print to file
	stdout_placeholder = sys.stdout
	f = open(dataFilesDir + 'cousera_kmeans_cluster_analysis_[' + str(datetime.now()) + '].txt', 'w')
	sys.stdout = f

	print('\n[SAMPLE DATA]')
	print(parsedData.take(10))
	print('[DATA DIMENSION]')
	print(trainingDF.shape)
	print('\n')

	for k in range(1,11):
		#Train the model
		clusters = KMeans.train(parsedData, k, maxIterations=15, runs=10, initializationMode="random")

		#Capture centers, cost and cluster sizes
		print('[K]='+str(clusters.k))
		print('[CENTER(S)]')
		print(clusters.centers)
		print('[COST]='+ str(clusters.computeCost(parsedData)))

		cluster_ind = clusters.predict(parsedData)
		cluster_sizes = cluster_ind.countByValue().items()
		print('[CLUSTER SIZES]')
		print(cluster_sizes)
		print('\n')

	#Redirect back the stdout
	sys.stdout = stdout_placeholder
	f.close()

else:
	clusters = KMeans.train(parsedData, targetSize, maxIterations=15, runs=10, initializationMode="random")

	x = parsedData.map(lambda r: r[0]).collect()
	y = parsedData.map(lambda r: r[1]).collect()
	plt.scatter(x,y,s=10)
	plt.scatter([x[0] for x in clusters.centers], [x[1] for x in clusters.centers], c='yellow', s=125, marker='*')
	plt.show()
