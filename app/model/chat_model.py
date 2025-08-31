from transformers import pipeline, AutoTokenizer
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
import json
import pandas as pd
import torch
from functools import reduce

class ChatBot:
    def __init__(self, reviews_path, businesses_path, spark_master='local', app_name='Chat-Bot Analysis'):
        self.chat_model = None
        self.tokenizer = None
        self.reviews_path = reviews_path
        self.businesses_path = businesses_path
        self.spark_master = spark_master
        self.app_name = app_name
        self.spark = self._initialize_spark()
        self._load_data()
        
    def _initialize_spark(self):
        """Initialize and configure Spark session"""
        return SparkSession.builder \
            .appName(self.app_name) \
            .master(self.spark_master) \
            .getOrCreate()
        
    def _load_data(self):
        self.reviews = self.spark.read.csv(self.reviews_path , header=True, inferSchema=True)
        self.businesses = self.spark.read.json(self.businesses_path)
        
    def load_chat_model(self):
        if self.chat_model is None:
            print("Initializing chatbot model...")
            model_name = "HuggingFaceH4/zephyr-7b-beta"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.chat_model = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("Chatbot model loaded!")
        except Exception as e:
            print("Error loading model:", e)
            
    def process_user_input(self, user_input):       
        system_prompt = """Convert user requests to this EXACT format:
        food quality:YES/NO
        staff behavior:YES/NO
        cleanliness:YES/NO
        value for money:YES/NO
        atmosphere:YES/NO
        range of choices:YES/NO
        service:YES/NO
        wait/delivery time:YES/NO
        convenience:YES/NO
        noise level:YES/NO
        parking:YES/NO
        outdoor seating:YES/NO
        wifi:YES/NO
        good for kids:YES/NO
        good for groups:YES/NO

        RULES:
        1.If a criteria not mentioned, say only this format: criteria : No
        2.If a criteria is optional, say only this format: criteria:NO
        3. NO parentheses, explanations, or special cases
        4. ALL 15 lines MUST be present
        5. Follow this exact sequence
        Your response must strictly adhere to this format. Do not include any additional commentary or variations.
        6. Example conversion:
        Input: "Find places with good food and parking"
        Output:
        food quality:YES
        staff behavior:NO
        cleanliness:NO
        value for money:NO
        atmosphere:NO
        range of choices:NO
        service:NO
        wait/delivery time:NO
        convenience:NO
        noise level:NO
        parking:YES
        outdoor seating:NO
        wifi:NO
        good for kids:NO
        good for groups:NO
       """
    
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
        ]
    
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
        outputs = self.chat_model(
            prompt,
            max_new_tokens=179,
            do_sample=False,
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.2
        )
        response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
        return response

    def post_processed_response(self, response):
        required_criteria = [
            "food quality", "staff behavior", "cleanliness",
            "value for money", "atmosphere", "range of choices",
            "service", "wait/delivery time", "convenience",
            "noise level", "parking", "outdoor seating",
            "wifi", "good for kids", "good for groups"
        ]
    
        processed_dict = {}
    
        # First parse the response text into a dictionary
        parsed_response = {}
        for line in response.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                parsed_response[key.strip().lower()] = value.strip().upper()
    
        # Now map to required criteria with defaults
        for criterion in required_criteria:
            processed_dict[criterion] = parsed_response.get(criterion.lower(), 'NO')
    
        return processed_dict


    def analize_response(self, response):

        joined_df = self.businesses.join(self.reviews, on='business_id', how='inner')
        grouped_df = joined_df.groupBy('business_id')\
        .agg(
                F.avg("`food quality`").alias("food_quality"),
                F.avg("`staff behavior`").alias("staff_behavior"),
                F.avg("cleanliness").alias("clean"),
                F.avg("`value for money`").alias("value_for_money"),
                F.avg("convenience").alias("conv"),
                F.avg("`wait/delivery time`").alias("wait_time"),
                F.avg("`range of choices`").alias("range_of_choices"),
                F.avg("atmosphere").alias("atmosphere"),
                F.avg("service").alias("service")
            )
        final_df = self.businesses.join(grouped_df, on='business_id', how='inner')
        conditions =[]
        if response['food quality'].lower() == 'yes':
            conditions.append(final_df.food_quality >= 4)
        if response['staff behavior'].lower() == 'yes':
            conditions.append(final_df.staff_behavior >= 4)
        if response['cleanliness'].lower() == 'yes':
            conditions.append(final_df.clean >= 4)
        if response['value for money'].lower() == 'yes':
            conditions.append(final_df.value_for_money >= 4)  
        if response['convenience'].lower() == 'yes':
            conditions.append(final_df.conv >= 4)
        if response['wait/delivery time'].lower() == 'yes':
            conditions.append(final_df.wait_time >= 4)
        if response['range of choices'].lower() == 'yes':
            conditions.append(final_df.range_of_choices >= 4)
        if response['atmosphere'].lower() == 'yes':
            conditions.append(final_df.atmosphere >= 4)
        if response['service'].lower() == 'yes':
            conditions.append(final_df.service >= 4)
        if response['noise level'].lower() == 'yes':
            conditions.append(
                (final_df.attributes.NoiseLevel != "loud") & 
                (final_df.attributes.NoiseLevel != "very_loud") & 
                (final_df.attributes.NoiseLevel != "u'loud'") & 
                (final_df.attributes.NoiseLevel != "u'very_loud'")
            )
        if response['parking'].lower() == 'yes':
            conditions.append(
                (final_df.attributes.BusinessParking.garage == True) |
                (final_df.attributes.BusinessParking.street == True) |
                (final_df.attributes.BusinessParking.lot == True) |
                (final_df.attributes.BusinessParking.validated == True) |
                (final_df.attributes.BusinessParking.valet == True)
            ) 
        if response['outdoor seating'].lower() == 'yes':
            conditions.append(final_df.attributes.OutdoorSeating == True)
        if response['wifi'].lower() == 'yes':
            conditions.append(
                (final_df.attributes.WiFi == "free") | 
                (final_df.attributes.WiFi == "u'free'")
        )
        if response['good for kids'].lower() == 'yes':
            conditions.append(final_df.attributes.GoodForKids == True)
        if response['good for groups'].lower() == 'yes':
            conditions.append(final_df.attributes.RestaurantsGoodForGroups == True)    

        if conditions:
            combined_condition = reduce(lambda x, y: x & y, conditions)
            filtered_df = final_df.filter(combined_condition)
        else:
            filtered_df = final_df 
        
        top_restaurants = (filtered_df
                       .select("name", "stars")
                       .orderBy(F.desc("stars"))
                       .limit(10)
                       .rdd
                       .collect())  
        #restaurant_names = filtered_df['name'].tolist()
        # Create a formatted string from the dictionary
        formatted_dict = "\n".join([f"{key}: {value}" for key, value in response.items()])
        if not top_restaurants:
            answer = "There was not such a restaurant."
        else:
        # Create the formatted list of restaurants
            restaurant_list = "\n".join([f"{name}: {stars} stars" for name, stars in top_restaurants])
        
            answer = (f"You are looking for a restaurant with these criteria:\n{formatted_dict}\n\n\n"  # Added an extra \n
                  f"Here are restaurants that you can go and enjoy:\n{restaurant_list}")

        return answer 

    def __del__(self):
        """Clean up resources"""
        try:
            if self.spark:
                self.spark.stop()
        except Exception as e:
            print(f"Error stopping Spark context: {e}") 
    