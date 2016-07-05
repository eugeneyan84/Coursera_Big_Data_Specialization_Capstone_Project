# This script demonstrates how to convert a column with yyyy-MM-dd string
# to an integer age as a new column using Pandas API. Age would be calculated
# based on the user's exact birthday, not calendar year.
#
# users.csv from Coursera Big Data Capstone Project is used here.
#
# Run script with the following command:
#
# spark-submit users-age-extraction.py <new csv-file>
#
# Solution created in response to original discussion thread:
# https://www.coursera.org/learn/big-data-capstone/discussions/all/threads/fM9rTkIlEeaPHQrkCWo3rw
#

import pandas as pd
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext

# create a Pandas dataframe for users.csv data
usersDF = pd.read_csv('/home/cloudera/flamingo-data/users.csv')

# convert the object type of 'dob' column into a pandas datetime structure
usersDF['dob'] = usersDF['dob'].apply(lambda x: pd.to_datetime(x))

# create an instance to hold today's datetime value
today = pd.datetime.now()

# 1) create an 'age' column
# 2) perform subtraction
# 3) set result of subtraction as number of days (timedelta64[D] for days, ...[ns] for nanoseconds)
# 4) divide by 365.25 (leap year accounted for)
# 5) cast to integer, i.e. decimal values discarded, as if like a floor function (int64 type)
usersDF['age'] = ((today - usersDF['dob']).astype('timedelta64[D]') / 365.25).astype(int)

# save updated table as csv file
usersDF.to_csv('/home/cloudera/flamingo-data/users-with-age.csv', index=False)
