#__init__.py
from flask import Flask
from app.routes import main_bp
from app.services import RestaurantDataService, AverageScoreAnalysisService
from app.model.chat_model import ChatBot

def create_app():
    app = Flask(__name__)
    # Initialize Spark service with direct parameters
    app.services = RestaurantDataService(
        businesses_path='my_project/data/raw/yelp_academic_dataset_business.json',
        reviews_path='my_project/data/processed/yelp_extarct_aspect_and_sentiment_scores_results.csv',
        spark_master='local',
        app_name='Restaurant Data Analysis'
    )

    app.services2 = AverageScoreAnalysisService(
        reviews_path = 'my_project/data/processed/yelp_extarct_aspect_and_sentiment_scores_results.csv',
        spark_master ='local',
        app_name='Average Score Analysis'
    )
    app.chat_model = ChatBot(
        reviews_path = 'my_project/data/processed/yelp_extarct_aspect_and_sentiment_scores_results.csv',
        businesses_path = 'my_project/data/raw/yelp_academic_dataset_business.json'
    )
#    # Register blueprints
    app.register_blueprint(main_bp)
    
    return app