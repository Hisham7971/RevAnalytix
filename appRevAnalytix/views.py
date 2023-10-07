from django.shortcuts import render
from django.http import HttpResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline
import numpy as np
from scipy.special import softmax
from google_play_scraper import Sort, reviews
import logging
import os
import csv
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import sentiment analysis library

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Load sentiment analysis model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize the sentiment analysis pipeline
sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)

def index(request):
    app_reviews = []  # Initialize an empty list for reviews
    if request.method == 'POST':
        app_id = request.POST.get('app_id')
        if app_id:
            try:
                result, _ = reviews(
                    app_id,
                    lang='en',
                    country='us',
                    sort=Sort.NEWEST,
                    count=10,  # You can adjust the number of reviews to fetch
                )
                app_reviews = result  # Assign the reviews to the app_reviews variable

                # Save the reviews to a CSV file in the 'data' folder
                data_folder = os.path.join(os.path.dirname(__file__), 'data')
                os.makedirs(data_folder, exist_ok=True)
                csv_file_path = os.path.join(data_folder, 'app_reviews.csv')

                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                    fieldnames = [
                        'reviewId', 'userName', 'userImage', 'content', 'score',
                        'thumbsUpCount', 'reviewCreatedVersion', 'at',
                        'replyContent', 'repliedAt', 'appVersion'
                    ]
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                    # Write the column names as the first row
                    writer.writeheader()

                    for review in app_reviews:
                        writer.writerow({
                            'reviewId': review['reviewId'],
                            'userName': review['userName'],
                            'userImage': review['userImage'],
                            'content': review['content'],
                            'score': review['score'],
                            'thumbsUpCount': review['thumbsUpCount'],
                            'reviewCreatedVersion': review['reviewCreatedVersion'],
                            'at': review['at'].strftime('%Y-%m-%d %H:%M:%S'),  # Format datetime
                            'replyContent': review['replyContent'],
                            'repliedAt': (
                                review['repliedAt'].strftime('%Y-%m-%d %H:%M:%S')
                                if review['repliedAt'] is not None
                                else None
                            ),
                            'appVersion': review['appVersion'],
                        })

            except Exception as e:
                error_message = str(e)
    
    return render(request, 'index.html', {'app_reviews': app_reviews})

def analyze_sentiment_and_priority(request):
    priority_results = {'P1': [], 'P2': [], 'P3': []}  # Initialize dictionaries for different priority levels

    try:
        # Read data from the CSV file
        csv_file_path = 'appRevAnalytix/data/app_reviews.csv'

        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for review in csv_reader:
                text = review['content']

                # Perform sentiment analysis
                sentiment_analyzer = SentimentIntensityAnalyzer()
                sentiment_scores = sentiment_analyzer.polarity_scores(text)
                sentiment_score = sentiment_scores['compound']

                # Categorize by sentiment
                if sentiment_score >= 0.05:
                    sentiment_category = 'Positive'
                elif sentiment_score <= -0.05:
                    sentiment_category = 'Negative'
                else:
                    sentiment_category = 'Neutral'

                # Determine priority based on your criteria
                # Replace these conditions with your logic for prioritization
                if sentiment_category == 'Negative':
                    if sentiment_score >= -0.5:
                        priority = 'P1'
                    elif sentiment_score >= -0.7:
                        priority = 'P2'
                    else:
                        priority = 'P3'
                else:
                    priority = 'N/A'  # Not applicable for non-negative reviews

                # Store in respective priority category
                priority_results[priority].append((text, sentiment_category, sentiment_score))

        # Render the results in the basic-table.html template
        return render(request, 'basic-table.html', {'priority_results': priority_results})

    except Exception as e:
        error_message = str(e)
        logging.error(f"An error occurred: {error_message}")

    # If there's an issue, return a response (you can customize this)
    return HttpResponse("An error occurred.", status=500)



def chat_page(request):
    # Your logic here if needed
    return render(request, 'chat.html')