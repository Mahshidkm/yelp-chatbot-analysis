from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
import json
import pandas as pd

class RestaurantDataService:
    def __init__(self, businesses_path, reviews_path, spark_master='local', app_name='Restaurant Analysis'):
        self.spark_master = spark_master
        self.app_name = app_name
        self.businesses_path = businesses_path
        self.reviews_path = reviews_path
        
        self.spark_context = self._initialize_spark()
        self._load_data()
        
    def _initialize_spark(self):
        """Initialize and configure Spark context"""
        try:
            SparkContext.getOrCreate().stop()
        except Exception as e:
            print(f"Error stopping existing Spark context: {e}")
            
        return SparkContext(
            self.spark_master,
            self.app_name
        )

    def _load_data(self):
        """Load and prepare datasets"""
        # Load businesses data
        self.businesses_rdd = self.spark_context.textFile(self.businesses_path)\
            .map(lambda x: json.loads(x))
        
        # Load reviews data
        self.reviews_df = pd.read_csv(self.reviews_path)

    def get_restaurant_data(self):
        """Process and return restaurant information"""
        # Create business key-value pairs
        businesses_kv = self.businesses_rdd.map(
            lambda x: (x['business_id'], (x['name'], x.get('stars'), x['review_count']))
        )

        # Create reviews key-value pairs
        reviews_kv = self.spark_context.parallelize(
            self.reviews_df.to_dict(orient='records')
        ).map(lambda x: (x['business_id'], x))

        # Join datasets and process results
        joined_rdd = reviews_kv.join(businesses_kv)
        
        restaurant_info = joined_rdd.map(
            lambda x: (x[1][1][0], x[1][1][1], x[1][1][2], x[0])  #name, stars, review_count, business_id
        ).distinct().collect()

        return sorted(restaurant_info, key=lambda x: x[2], reverse=True)

    def __del__(self):
        """Clean up resources"""
        try:
            if self.spark_context:
                self.spark_context.stop()
        except Exception as e:
            print(f"Error stopping Spark context: {e}")
####################################################################
class AverageScoreAnalysisService:
    def __init__(self, reviews_path, spark_master,app_name):
        self.reviews_path = reviews_path
        self.spark_master = spark_master
        self.app_name = app_name
        
        self.spark = self._initialize_spark()
        self._load_data()
        
    def _initialize_spark(self) -> SparkSession:
        """Initialize and configure Spark session"""
        return SparkSession.builder \
            .appName(self.app_name) \
            .master(self.spark_master) \
            .getOrCreate()
        
    def _load_data(self):
       self.reviews = self.spark.read.csv(self.reviews_path, header=True, inferSchema=True) 
        
    def get_average_scores(self, business_id):
        """Calculate average scores for a specific business"""
        result = self.reviews.filter(F.col("business_id") == business_id) \
        .groupBy("date") \
        .agg(
            F.avg("`food quality`").alias("food_quality"),
            F.avg("`staff behavior`").alias("staff_behavior"),
            F.avg("cleanliness").alias("cleanliness"),
            F.avg("`value for money`").alias("value_for_money"),
            F.avg("convenience").alias("convenience"),
            F.avg("`wait/delivery time`").alias("wait_time"),
            F.avg("`range of choices`").alias("range_of_choices"),
            F.avg("atmosphere").alias("atmosphere"),
            F.avg("service").alias("service")
        ) \
        .orderBy("date") \
        .na.fill(0)  # Replace nulls with 0
        # Convert to pandas DataFrame for easier manipulation
        df_result = result.toPandas()
        if df_result.empty:
            return {"years": [], "avg_scores": {}}
    
        # Structure data for Chart.js
        return {
            "years": df_result["date"].tolist(),
            "avg_scores": {
                col: df_result[col].tolist()
                for col in df_result.columns
                if col != "date"
                }
        }
    def __del__(self):
        """Clean up resources"""
        try:
            if self.spark:
                self.spark.stop()
        except Exception as e:
            print(f"Error stopping Spark context: {e}")

            