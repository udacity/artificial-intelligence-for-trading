from bs4 import BeautifulSoup
import requests

def get_tweets():
    
    # Get HTML data
    html_data = requests.get('https://twitter.com/AIForTrading1', params = {'count':'28'}).text

    # Create a BeautifulSoup Object
    page_content = BeautifulSoup(html_data, 'lxml')

    # Find all the <div> tags that have a class="js-tweet-text-container" attribute
    tweets = page_content.find_all('div', class_='js-tweet-text-container')
    
    # Create empty list to hold all out tweets
    all_tweets = []

    # Add each tweet to all_tweets. Use the .strip() method rto eturn a copy of
    # the string with the leading and trailing characters removed.
    for tweet in tweets:
        all_tweets.append(tweet.p.get_text().strip())

    return all_tweets