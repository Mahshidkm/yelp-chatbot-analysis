# Restaurant Insight & Recommendation ChatBot

## Project Overview

This project is a full-stack data science application that transforms raw Yelp reviews into actionable business intelligence and a personalized customer recommendation system. It leverages state-of-the-art NLP models to perform granular **Aspect-Based Sentiment Analysis (ABSA)**, providing deep insights into what customers truly think about specific aspects (food quality, service, atmosphere, etc.) of a restaurant. These insights are presented through two intuitive dashboards: one for **restaurant owners** to track their performance and one for **customers** to find their perfect meal through a conversational chatbot.

## 🚀 Video Demo

**Video Demo:** [Demo Link on Vercel/Railway]()  

---

## ✨ Key Features

### **For Restaurant Owners: Analytics Dashboard**
*   **Trend Analysis:** Visualize sentiment trends for key aspects (food quality, service, atmpsphere, value for money) over multiple years.
*   **Performance Benchmarking:** Identify strengths and weaknesses based on actual customer feedback.
*   **Data-Driven Decisions:** Pinpoint exactly what to improve to boost ratings and customer satisfaction.

### **For Customers: Conversational Chatbot**
*   **Natural Language Search:** Find restaurants using intuitive queries like "Find a romantic Italian place with great ambiance but isn't too expensive."
*   **Aspect-Filtering:** The chatbot understands sentiment towards specific aspects, going beyond simple star ratings.
*   **Personalized Recommendations:** Get tailored suggestions based on nuanced customer preferences.

### **Under the Hood**
*   **Advanced NLP Pipeline:** Utilizes multiple Hugging Face Transformer models for accurate aspect extraction and sentiment scoring.
*   **Large-Scale Data Processing:** Apache Spark ensures efficient processing of the massive Yelp dataset.
*   **GPU-Accelerated Model Inference:** Slurm job scheduling manages computationally intensive NLP tasks on GPU clusters.
*   **Modern Web Application:** A responsive Flask app with a clean interface serves both dashboards seamlessly.

---

## 🛠️ Tech Stack

*   **Backend Framework:** Python, Flask
*   **Frontend:** HTML, CSS
*   **Natural Language Processing (NLP):**
    *   Hugging Face `Transformers` Library
    *   Custom pipeline for Aspect-Based Sentiment Analysis
    *   Conversational AI model for the chatbot
*   **Big Data Processing:** Apache Spark (SparkSQL)
*   **High-Performance Computing:** Slurm Workload Manager for GPU-based model inference
  

### **Hugging Face Models Used**
This project implements a multi-model NLP pipeline:
1.  **Aspect Extraction & Sentiment Scoring:** We used facebook/bart-large-mnli for Aspect Extraction 
and yangheng/deberta-v3-base-absa-v1.1 for Sentiment Scoring

2.  **Conversational Chatbot:** We used HuggingFaceH4/zephyr-7b-beta 

---

## Project Structure

The project is organized as follows:

```
my_project/
├── data/                 # Raw and processed data files (raw not included)
│   ├── raw/       
│   ├── processed/           
│   ├── cleaned/
├── job/                  # Slurm job scripts for cluster execution
├── app/                  # Flask application code
│   ├── templates/        # HTML templates for web pages
│   ├── model/            # Chatbot model implementation
│   ├── routes.py         # Flask route definitions
│   └── services.py       # Data processing services
├── src/                  # Additional source code
└── run.py                # Application entry point
```     

## How It Works / Methodology

1.  **Data Acquisition:** The pipeline begins with the [Yelp Open Dataset](https://www.yelp.com/dataset). Due to its size, it is not stored in this repository.
2.  **Aspect-Based Sentiment Analysis (ABSA):**

    a. **Submit the ABSA Slurm Job:** Submit the `extract_aspect_and_sentiment_job.sh` script to perform aspect-based sentiment analysis on the cleaned restaurant reviews. This script will likely train or load an ABSA model and apply it to the reviews to extract aspects and determine sentiment.

    ```bash
    sbatch job/extract_aspect_and_sentiment_job.sh
    ```

    *Important:* The `extract_aspect_and_sentiment_job.sh` script should:

    *   Load the cleaned restaurant reviews data.
    *   Initialize the ABSA model.
    *   Apply the ABSA model to the reviews.
    *   Save the processed data (with aspect and sentiment information) to `data/processed/yelp.csv`.

3.  **Chatbot and Flask Application:**

    a. **Submit the Chatbot Slurm Job:** Submit the `chat_bot_job.sh` script to run the Flask application with the chatbot interface. This script will likely load the processed data (`data/processed/yelp.csv`) and start the Flask web server.

    ```bash
    sbatch job/chat_bot_job.sh
    ```

    *Important:* The `chat_bot_job.sh` script should:

    *   Load the processed data (`data/processed/yelp.csv`).
    *   Initialize the chatbot.
    *   Start the Flask web server.

4.  **Accessing the Web Application:**

    *   Once the `chat_bot_job.sh` Slurm job is running, the Flask application will be accessible at the address specified in the `chat_bot_job.sh` script (e.g., `http://your_server_address:5000`).
    *   Open a web browser and navigate to the specified address to interact with the chatbot and view the dashboards.

