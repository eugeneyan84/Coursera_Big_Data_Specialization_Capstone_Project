# This python script performs tranformation and aggregation of CSV data from Catch the Pink Flamingo. The example command is as 
# follows:
#
# spark-submit --packages com.databricks:spark-csv_2.10:1.4.0 coursera_data_transformation.py local /home/cloudera/flamingo-data/test-agg3.csv
#
# spark-submit --packages com.databricks:spark-csv_2.10:1.4.0 coursera_data_transformation.py hdfs /flamingo-data/aggregated_data
#
# Arguments are as follows:
# 1st: filename of this python script
# 2nd: method of saving the final dataset. Use 'local' if saving to a local folder, otherwise use 'hdfs' to save to a folder in Hadoop FS
# 3rd: filename of target CSV file with full path for 'local' option, or location in HDFS for 'hdfs' option
#  
# IMPORTANT:
# For package flag, use spark-csv_2.10:1.4.0, not spark-csv_2.11:1.4.0
# Version 2.11 would run into save method issues for Dataframe, 2.10 is working.
# Error for 2.11 -> NoSuchMethodError: scala.Predef$.$conforms()Lscala/Predef$$less$colon$less
# Tip gotten from here: http://stackoverflow.com/questions/28487305/nosuchmethoderror-while-running-spark-streaming-job-on-hdp-2-2
# 
#

from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.sql import functions as f
from pyspark.sql.types import *
import pandas
import sys

if len(sys.argv) < 3:
	print('Required parameters not present. Use spark-submit coursera_data_transformation.py <save-method> <csv-filename-with-full-path>')
	sys.exit()
else:
	saveMethod = sys.argv[1]
	csvFile = sys.argv[2]
	if not (saveMethod == 'local' or saveMethod=='hdfs'):
		print('Invalid save method. Use either local or hdfs')
		sys.exit()

conf = (SparkConf()
         .setMaster("local")
         .setAppName("Coursera data transformation")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

# load user-session, as well as all the click data
us_df = sqlContext.load(path='file:///home/cloudera/flamingo-data/user-session.csv', source='com.databricks.spark.csv', header='true', inferSchema='true')
ac_df = sqlContext.load(path='file:///home/cloudera/flamingo-data/ad-clicks.csv', source='com.databricks.spark.csv', header='true', inferSchema='true')
bc_df = sqlContext.load(path='file:///home/cloudera/flamingo-data/buy-clicks.csv', source='com.databricks.spark.csv', header='true', inferSchema='true')
gc_df = sqlContext.load(path='file:///home/cloudera/flamingo-data/game-clicks.csv', source='com.databricks.spark.csv', header='true', inferSchema='true')

# filter row and columns from user-session
usf = us_df.filter(us_df.sessionType=='end')
usf = usf[['userSessionId','userId']]
usf = usf.withColumn("sessionCount", f.lit(1).cast('int'))

# filter columns from ad-clicks
acf = ac_df[['userSessionId']]
acf = acf.withColumn("adClickCount", f.lit(1).cast('int'))
acf_agg = acf.groupBy('userSessionId').agg(f.sum(acf.adClickCount).alias('adClickCountPerSession'))

# filter columns from buy-clicks
bcf = bc_df[['userSessionId','price']]
bcf = bcf.withColumn("buyClickCount", f.lit(1).cast('int'))
bcf_agg = bcf.groupBy('userSessionId').agg(f.sum(bcf.buyClickCount).alias('buyClickCountPerSession'),f.sum(bcf.price).alias('pricePerSession'))

# filter columns from game-clicks
gcf = gc_df[['userSessionId','isHit']]
gcf = gcf.withColumn("gameClickCount", f.lit(1).cast('int'))
gcf_agg = gcf.groupBy('userSessionId').agg(f.sum(gcf.gameClickCount).alias('gameClickCountPerSession'),f.sum(gcf.isHit).alias('hitPerSession'))

# join all click-data to user-session on userSessionId as key
combinedRawData = usf.join(acf_agg, on="userSessionId", how="left_outer").join(bcf_agg, on="userSessionId", how="left_outer").join(gcf_agg, on="userSessionId", how="left_outer")

# replace all None values with zero
combinedData = combinedRawData.na.fill({'adClickCountPerSession':0,'buyClickCountPerSession':0,'pricePerSession':0.0})

# aggregate total values by userId
combinedAggData = combinedData.groupBy('userId').agg(f.sum(combinedData.sessionCount).alias('totalSessions'),
							f.sum(combinedData.gameClickCountPerSession).alias('totalGameClicks'),
							f.sum(combinedData.hitPerSession).alias('totalHits'),
							f.sum(combinedData.buyClickCountPerSession).alias('totalPurchaseCount'),
							f.sum(combinedData.pricePerSession).alias('totalExpenditure'),
							f.sum(combinedData.adClickCountPerSession).alias('totalAdClicks'))

# create simple averaging function
def getAverage(total,count):
	return total * 1.0 / count

# aggregate all the mean data from the total values, with totalSessions as a divisor
combinedAggData = combinedAggData.withColumn('avgGameClicks', getAverage(combinedAggData['totalGameClicks'], combinedAggData['totalSessions'])).withColumn('avghits', getAverage(combinedAggData['totalHits'], combinedAggData['totalSessions'])).withColumn('avgPurchaseCount', getAverage(combinedAggData['totalPurchaseCount'], combinedAggData['totalSessions'])).withColumn('avgExpPerSession', getAverage(combinedAggData['totalExpenditure'], combinedAggData['totalSessions'])).withColumn('avgAdClicks', getAverage(combinedAggData['totalAdClicks'], combinedAggData['totalSessions']))

# save data using preferred method
if saveMethod == 'local':
	combinedAggData.toPandas().to_csv(csvFile, index=False)
else:
	combinedAggData.select('userId',
				'totalSessions',
				'totalGameClicks',
				'totalHits',
				'totalPurchaseCount',
				'totalExpenditure',
				'totalAdClicks',
				'avgGameClicks',
				'avghits',
				'avgPurchaseCount',
				'avgExpPerSession',
				'avgAdClicks').save(csvFile, 'com.databricks.spark.csv')
