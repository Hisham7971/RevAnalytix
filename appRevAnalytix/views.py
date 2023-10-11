from django.shortcuts import render
import os
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import re
import emoji
import plotly.express as px
from google_play_scraper import Sort, reviews_all


# Define the sentiment analysis model and tokenizer
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# ------------------------------Home Page------------------------------ #
def index(request):
    # Your logic here if needed
    return render(request, 'index.html')

# ------------------------------Search & Display------------------------------ #
def search_and_display_reviews(request):
    if request.method == "POST":
        app_id = request.POST.get("app_id")

        if not app_id:
            # If no app ID is provided, send a message to the user
            message = "Please enter an App ID."
            context = {
                'message': message,
            }
            return render(request, 'index.html', context)

        try:
            # Check if the CSV file exists
            directory = 'data/'
            csv_filename = os.path.join(directory, f'{app_id}.csv')

            if os.path.exists(csv_filename):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_filename)
                data_from_csv = df.to_dict(orient='records')
            else:
                # Fetch app reviews if the CSV file doesn't exist
                result = reviews_all(
                    app_id,
                    sleep_milliseconds=0,
                    lang='en',
                    country='us',
                    sort=Sort.NEWEST
                )

                # Create a DataFrame from the result
                df = pd.DataFrame(result)

                # Save the DataFrame to a CSV file
                if not os.path.exists(directory):
                    os.makedirs(directory)
                df.to_csv(csv_filename, index=False)

                data_from_csv = df.to_dict(orient='records')

            context = {
                'data_from_csv': data_from_csv
            }
        except Exception as e:
            # Handle any exceptions that may occur during the process
            error_message = f"An error occurred: {str(e)}"
            context = {
                'error_message': error_message,
            }

        return render(request, 'index.html', context)
    else:
        return render(request, 'index.html')

# ------------------------------Sentiment & Priority------------------------------ #

# Preprocess function
def preprocess(text):
    # Convert emojis to their text representation
    text = emoji.demojize(text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return ' '.join(text.split())

# Sentiment analysis and priority categorization function
def sentiment_analysis_and_priority(row):
    # Preprocess the text
    preprocessed_text = preprocess(row['content'])
    
    # Split the text into chunks
    chunk_size = 256
    chunks = [preprocessed_text[i:i+chunk_size] for i in range(0, len(preprocessed_text), chunk_size)]
    
    chunk_scores = []
    
    # Initialize scores with a default value
    scores = np.array([0.0, 0.0, 0.0])
    
    for chunk in chunks:
        if chunk:
            encoded_input = tokenizer(chunk, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
        
            # Get the top sentiment label and score for the chunk
            top_label = config.id2label[ranking[0]]
            top_score = np.round(float(scores[ranking[0]]), 4)
        
            chunk_scores.append(top_score)
    
    # Calculate the average score for all chunks if there are scores
    if chunk_scores:
        average_score = np.mean(chunk_scores)
        sentiment_score = average_score
    else:
        sentiment_score = np.nan
    
    # The order of labels in this model is ['negative', 'neutral', 'positive']
    sentiment = ['negative', 'neutral', 'positive'][np.argmax(scores)]
    
    # Categorize priority based on sentiment and rating
    if sentiment == 'negative':
        # Use the probability score of the negative sentiment and the rating to categorize priority
        negative_score = scores[0]
        rating = row['score']
        if negative_score > 0.7:
            priority = 'P1'
        elif negative_score > 0.5 or rating <= 2:
            priority = 'P2'
        else:
            priority = 'P3'
    else:
        priority = None
    
    return sentiment, sentiment_score, priority


def sentiment_analysis_view(request):
    try:
        # Load the CSV data
        df = pd.read_csv('data/in.evolve.android.csv')

        # Apply sentiment analysis and priority categorization to the data
        df[['sentiment', 'sentiment_score', 'priority']] = df.apply(sentiment_analysis_and_priority, axis=1, result_type='expand')

        # Filter rows with 'negative' sentiment and non-null priority
        negative_reviews = df[(df['sentiment'] == 'negative') & ~df['priority'].isnull()]

        # Separate data into P1, P2, and P3 categories
        p1_reviews = negative_reviews[negative_reviews['priority'] == 'P1']
        p2_reviews = negative_reviews[negative_reviews['priority'] == 'P2']
        p3_reviews = negative_reviews[negative_reviews['priority'] == 'P3']

        context = {
            'p1_reviews': p1_reviews,
            'p2_reviews': p2_reviews,
            'p3_reviews': p3_reviews,
        }

        return render(request, 'basic-table.html', context)

    except Exception as e:
        error_message = str(e)
        return render(request, '400.html', {'error_message': error_message})








def chat_page(request):
    # Your logic here if needed
    return render(request, 'chat.html')

def basic_table(request):
    # Your logic here if needed
    return render(request, 'basic-table.html')

def datatable(request):
    # Your logic here if needed
    return render(request, 'datatable.html')

def highchart(request):
    # Your logic here if needed
    return render(request, 'highchart.html')

def knob_chart(request):
    # Your logic here if needed
    return render(request, 'knob-chart.html')

def apexcharts(request):
    # Your logic here if needed
    return render(request, 'apexcharts.html')