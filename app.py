import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PredictionPipeline import predict
import googleapiclient.discovery
import re
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv

load_dotenv()

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = os.environ.get('DEVELOPER_KEY')

def clean_comment(text):
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove links
    text = re.sub(r'http\S+', '', text)
    
    # Remove <br> tags
    text = text.replace('<br>', '')
    
    return text

def fetch_comments(link, maxResults):
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)
    
    try:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=link,
            order='relevance',
            maxResults=maxResults
        ).execute()

        comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
        cleaned_comments = [clean_comment(re.sub(r'http\S+', '', comment)) for comment in comments]
        return cleaned_comments

    except HttpError as e:
        print('An error occurred:', e)
        return None

def Predict(text):

    try:
        output = predict(text)
        return output
    except Exception as e:
        raise e

def get_count(text):
    positive_count = 0
    negative_count = 0
    for i in text:
        if i['prediction'] == 'Postive':
            positive_count += 1
        else:
            negative_count += 1
    return positive_count, negative_count
    
def main():
    st.title("Youtube Sentiment Analysis")
    url = st.text_input("Enter the link of the video")
    video_id = url.split('=')[-1]
    number_of_comments = st.number_input("Enter the number of comments", min_value = 1, max_value = 200, value = 1)

    # START YOUTUBE API CALL FETCH DATA 
    if st.button("Fetch Comments"):
        with st.spinner("Fetching Comments..."):
            text = fetch_comments(video_id, number_of_comments)
            predictions = Predict(text)
        st.success("Comments Fetched")
    # END YOUTUBE API CALL FETCH DATA 

        positive_count, negative_count = get_count(predictions)
        
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#0E1117')

        categories = ['Positive', 'Negative']
        counts = [positive_count, negative_count]

        # Create histogram
        ax.bar(categories, counts, color = ['#00c878', '#fe0366'], 
               width = 0.7, align = 'center', edgecolor = 'white', linewidth = 2, 
               capsize = 10, label = 'Toxicity Percentage')
        fig.set_size_inches(10, 6)
        ax.set_facecolor('#0E1117')
        # change the color of x and y axis
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis = 'x', colors = 'white')
        ax.tick_params(axis = 'y', colors = 'white')
        # change color of y axis label
        ax.yaxis.label.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set labels and title
        ax.set_ylabel('Count')

        # Display plot
        st.pyplot(fig)
        
        df = pd.DataFrame(predictions)
        df.index += 1

        st.table(df)
        
    st.text("Made by: Gaurav Jha")


if __name__ == "__main__":
    main()