from django.shortcuts import render
from django.shortcuts import redirect
import os
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import re
import emoji
import plotly.express as px
import plotly.offline as opy
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
            # Fetch app reviews
            directory = 'data/'
            csv_filename = os.path.join(directory, 'input.csv')
            
            result = reviews_all(
                app_id,
                sleep_milliseconds=0,
                lang='en',
                country='us',
                sort=Sort.NEWEST
            )

            # Create a DataFrame from the result
            df = pd.DataFrame(result)

            # Save the DataFrame to a CSV file, overwriting the existing one
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

# ------------------------------Text Preprocess------------------------------ #

# Preprocess function
def preprocess(text):
    # Convert emojis to their text representation
    text = emoji.demojize(text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return ' '.join(text.split())

# ------------------------------Sentiment & Priority------------------------------ #

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
        if negative_score > 0.7 or rating <= 2:
            priority = 'P1'
        elif negative_score > 0.5 or rating == 3:
            priority = 'P2'
        else:
            priority = 'P3'
    else:
        priority = None
    
    return sentiment, sentiment_score, priority
# ------------------------------Sentiment View------------------------------ #

def sentiment_analysis_view(request):
    try:
        # Check if the CSV file exists
        directory = 'data/'
        csv_filename = os.path.join(directory, 'input.csv')

        # Load the CSV data
        df = pd.read_csv(csv_filename)

        # Apply sentiment analysis and priority categorization to the data
        df[['sentiment', 'sentiment_score', 'priority']] = df.apply(sentiment_analysis_and_priority, axis=1, result_type='expand')  # Add 'axis=1' to apply to rows

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
        # Create a new DataFrame with all the columns including 'sentiment', 'sentiment_score', and 'priority'
        new_df = df[['reviewId', 'userName', 'userImage', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'appVersion', 'sentiment', 'sentiment_score', 'priority']]

        # Define the path for the new CSV file
        new_csv_filename = os.path.join(directory, 'output.csv')

        # Save the new DataFrame to the new CSV file
        new_df.to_csv(new_csv_filename, index=False)

        return render(request, 'basic-table.html', context)

    except Exception as e:
        error_message = str(e)
        return render(request, '400.html', {'error_message': error_message})


def generate_charts(request):
    # Load your data (replace with your actual data source)
    data = pd.read_csv('data/output.csv')

        # Data Analysis 1: Histogram of 'score'
    score_histogram = px.histogram(data, x='score', title='Review Score Distribution')

    # Data Analysis 2: Box Plot of 'score' by 'sentiment'
    box_plot = px.box(data, x='sentiment', y='score', title='Review Scores by Sentiment')

    # Data Analysis 3: Scatter Matrix of selected columns
    scatter_matrix = px.scatter_matrix(data, dimensions=['score', 'thumbsUpCount', 'sentiment_score'], title='Scatter Matrix')

    # Data Analysis 4: Heatmap of correlation between columns
    heatmap = px.imshow(data.corr(), title='Correlation Heatmap')

    # Data Analysis 5: Bar Chart of reviews by app version
    app_version_bar_chart = px.bar(data['appVersion'].value_counts(), x=data['appVersion'].value_counts().index, y=data['appVersion'].value_counts().values, title='Reviews by App Version')

    # Data Analysis 6: Time Series Line Chart of review counts over time
    data['at'] = pd.to_datetime(data['at'])
    # Aggregate data to count reviews by date
    review_counts = data.groupby(data['at'].dt.date)['reviewId'].count().reset_index()
    
    time_series_line_chart = px.line(review_counts, x='at', y='reviewId', title='Review Count Over Time')


    # Convert the Plotly figures to HTML
    score_histogram_html = opy.plot(score_histogram, auto_open=False, output_type='div')
    box_plot_html = opy.plot(box_plot, auto_open=False, output_type='div')
    scatter_matrix_html = opy.plot(scatter_matrix, auto_open=False, output_type='div')
    heatmap_html = opy.plot(heatmap, auto_open=False, output_type='div')
    app_version_bar_chart_html = opy.plot(app_version_bar_chart, auto_open=False, output_type='div')
    time_series_line_chart_html = opy.plot(time_series_line_chart, auto_open=False, output_type='div')

    # Pass the HTML content to the template
    context = {
        'score_histogram_html': score_histogram_html,
        'box_plot_html': box_plot_html,
        'scatter_matrix_html': scatter_matrix_html,
        'heatmap_html': heatmap_html,
        'app_version_bar_chart_html': app_version_bar_chart_html,
        'time_series_line_chart_html': time_series_line_chart_html,
    }

    return render(request, 'highchart.html', context)





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