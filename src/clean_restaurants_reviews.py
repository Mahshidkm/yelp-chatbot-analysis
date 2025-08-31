#!/usr/bin/env python
# coding: utf-8
from pyspark import SparkContext, SparkConf
import json
import re
from datetime import datetime

class YelpReviewCleaner:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    @staticmethod
    def clean_text(text):
        """Clean the input text by removing unwanted characters."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s.,?!]', '', text.lower())).strip()

    @staticmethod
    def process_record(record):
        """Process a single JSON record to clean text and extract year."""
        try:
            data = json.loads(record)
            cleaned_text = YelpReviewCleaner.clean_text(data.get('text', ''))
            if cleaned_text == "":
                return None
            date_str = data.get('date', '')
            year = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').year if date_str else None
            data['text'] = cleaned_text
            data['date'] = year
            return json.dumps(data)
        except Exception as e:
            print(f"Error processing record: {e}")
            return None

    def run(self):
        """Execute the cleaning process using Spark RDD."""
        conf = SparkConf().setAppName("CleanRestaurantsReviews")
        sc = SparkContext.getOrCreate(conf)
        
        try:
            print("Running RDD processing...")
            raw_rdd = sc.textFile(self.input_path)
            processed_rdd = raw_rdd.map(self.process_record).filter(lambda x: x is not None)
            processed_rdd.saveAsTextFile(self.output_path)
        finally:
            sc.stop()

if __name__ == "__main__":
    INPUT_PATH = "my_project/data/processed/yelp_restaurants_reviews.json"
    OUTPUT_PATH = "my_project/data/cleaned/"

    cleaner = YelpReviewCleaner(INPUT_PATH, OUTPUT_PATH)
    cleaner.run()