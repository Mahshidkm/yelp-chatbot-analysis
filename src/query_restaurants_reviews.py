# src/query_restaurants.py

import findspark
findspark.init()

from pyspark.sql import SparkSession

class YelpDataProcessor:
    def __init__(self, review_path, business_path):
        self.spark = SparkSession.builder.appName("Query Restaurants using SparkSQL").getOrCreate()
        self.review_df = self.spark.read.json(review_path)
        self.business_df = self.spark.read.json(business_path)

    def create_temp_views(self):
        self.review_df.createOrReplaceTempView("review")
        self.business_df.createOrReplaceTempView("business")

    def query_restaurants(self, state="IL"):
        return self.spark.sql(f"""
            SELECT review.*
            FROM review
            JOIN business ON review.business_id = business.business_id
            WHERE categories LIKE '%Restaurants%' AND state='{state}'
        """)

    def save_to_json(self, dataframe, output_path):
        dataframe.write.json(output_path, mode='overwrite')
        
    def num_reviews(self, dataframe):
        return yelp_review_rest_df.count()

# Usage example
if __name__ == "__main__":
    review_path = 'my_project/data/raw/yelp_academic_dataset_review.json'
    business_path = 'my_project/data/raw/yelp_academic_dataset_business.json'
    output_path = 'my_project/data/processed/yelp_restaurants_reviews.json'

    processor = YelpDataProcessor(review_path, business_path)
    processor.create_temp_views()
    yelp_review_rest_df = processor.query_restaurants()
    processor.save_to_json(yelp_review_rest_df, output_path)
    print(processor.num_reviews(yelp_review_rest_df))