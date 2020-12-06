import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from datetime import date

sc = SparkContext()
spark = SQLContext(sc)

current_date = date.today()

filename = "football_results"

bucket_name = "gs://naaru-etl"

schema = StructType(
    [
        StructField("date", StringType(), True),
        StructField("home_team", StringType(), True),
        StructField("away_team", StringType(), True),
        StructField("home_score", IntegerType(), True),
        StructField("away_score", IntegerType(), True),
        StructField("tournament", StringType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True),
        StructField("neutral", BooleanType(), True), 
    ])

def to_date_(col, formats=("MM/dd/yyyy", "yyyy-MM-dd", "dd-MM-yyyy", "dd/MM/yyyy")):
    return coalesce(*[to_date(col, f) for f in formats])

football_stats = spark.read.csv(bucket_name+"/"+filename+".csv", header=True, dateFormat = 'yyyy-MM-dd', schema = schema) #read csv 
#by accepting the following options

football_stats.registerTempTable("football_stats") #Registers this DataFrame as a temporary table using the given name.

qry = """
        select * from football_stats
      """

raw_data = spark.sql(qry) #Raw data
raw_data = raw_data.withColumn("date_id", monotonically_increasing_id()+1)
raw_data = raw_data.withColumn("game_id", monotonically_increasing_id()+5)
raw_data = raw_data.select("date_id", "date","game_id", "home_team","away_team", "home_score", "away_score", "tournament", "city", "country", "neutral")

FIFA = raw_data.filter(raw_data.tournament == 'FIFA World Cup')

columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'city', 'country']
# columns = FIFA.columns
for column in columns:
    FIFA = FIFA.withColumn(column,when(isnan(col(column)),None).otherwise(col(column))) #converting nan to null

columns = FIFA.columns
for column in columns:
    if column == 'neutral':
        FIFA.na.fill(False)
    else:
        FIFA.na.fill(-1) #handling null values


columns = ['home_team', 'away_team', 'tournament', 'city', 'country', 'neutral']
for column in columns:
    FIFA.withColumn(column, lower(col(column)))
    FIFA.withColumn(column, initcap(col(column))) #All strings to capitalize

results = FIFA.withColumn("date", to_date_("date")) #converting string to actual date fomrat yyyy-MM-dd

results = results.withColumn('neutral',col('neutral').cast("boolean")).withColumn('country', col('country').cast('string')).withColumn('tournament', col('tournament').cast('string')).withColumn('home_team', col('home_team').cast('string')).withColumn('away_team', col('away_team').cast('string')).withColumn('home_score', col('home_score').cast('integer')).withColumn('away_score', col('away_score').cast('integer')).withColumn('date_id', col('date_id').cast('integer')).withColumn('game_id', col('game_id').cast('integer'))

date_dim = results.select('date_id', 'date')

team_loc_dim = results.select('game_id', 'home_team', 'away_team', 'tournament', 'city', 'country', 'neutral')

fifa_fact = results.select("date_id", "game_id", "home_score", "away_score")

FIFA_results = bucket_name+"/transformed-FIFA-data/"+str(current_date)+"_results"

FIFA_fact = bucket_name+"/transformed-FIFA-data/star-schema/fact/"+str(current_date)+"_results"

FIFA_dim1 = bucket_name+"/transformed-FIFA-data/star-schema/dim1/"+str(current_date)+"_results"

FIFA_dim2 = bucket_name+"/transformed-FIFA-data/star-schema/dim2/"+str(current_date)+"_results"

results.coalesce(1).write.format("csv").save(FIFA_results, header='true')

fifa_fact.coalesce(1).write.format("csv").save(FIFA_fact, header='true')

date_dim.coalesce(1).write.format("csv").save(FIFA_dim1, header='true')

team_loc_dim.coalesce(1).write.format("csv").save(FIFA_dim2, header='true')

