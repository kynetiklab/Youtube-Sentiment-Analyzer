import csv
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

def get_youtube_comments(api_key, video_id):
    # Create a YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Extract comments from the specified video
    comments = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in response:
            next_page_token = response['nextPageToken']
        else:
            break

    return comments

def analyze_sentiment(comments):
    # Perform sentiment analysis on the comments
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for comment in comments:
        sentiment_score = sid.polarity_scores(comment)
        sentiment_scores.append(sentiment_score)

    return sentiment_scores

def save_results(comments, sentiment_scores, csv_file):
    # Save the comments and sentiment scores in a CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment', 'Sentiment Score'])
        for comment, score in zip(comments, sentiment_scores):
            writer.writerow([comment, score])
def save_sentiment_visualization(sentiment_scores, output_file):
    # Visualize the sentiment distribution using a bar chart
    neg_count = sum(1 for score in sentiment_scores if score['compound'] < -0.05)
    neu_count = sum(1 for score in sentiment_scores if -0.05 <= score['compound'] <= 0.05)
    pos_count = sum(1 for score in sentiment_scores if score['compound'] > 0.05)

    labels = ['Negative', 'Neutral', 'Positive']
    counts = [neg_count, neu_count, pos_count]

    plt.bar(labels, counts, color=['red', 'gray', 'green'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis')

    # Save the visualization as a PNG image
    plt.savefig(output_file, format='png')

def main():
    # Define the video ID for which you want to extract comments
    video_id = '_fqxMZi7P7U'

    # Set up YouTube API credentials
    api_key = 'YOUR API KEY'

    # Define the path to save the CSV file
    csv_file = 'comments_sentiment.csv'

    # Get YouTube comments
    comments = get_youtube_comments(api_key, video_id)

    # Perform sentiment analysis
    sentiment_scores = analyze_sentiment(comments)

    # Save the results
    save_results(comments, sentiment_scores, csv_file)
    # Define the path to save the PNG visualization
    visualization_output_file = 'sentiment_visualization.png'

    # Visualize the sentiment distribution and save as PNG
    save_sentiment_visualization(sentiment_scores, visualization_output_file)
if __name__ == '__main__':
    main()
