import requests
import time
import json
#its bad practice to place your bearer token directly into the script (this is just done for illustration purposes)
#BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAANm6dgEAAAAACVUVEBLtjd%2F1tnYAk82Y8zmfGoQ%3DZHCdTe4nnTOfVpeE1z8aedujyCajSTsd0pN1maGm2DedP7CxUv"
             ##"AAAAAAAAAAAAAAAAAAAAANm6dgEAAAAACVUVEBLtjd%2F1tnYAk82Y8zmfGoQ%3DZHCdTe4nnTOfVpeE1z8aedujyCajSTsd0pN1maGm2DedP7CxUv"
#define search twitter function


BEARER_TOKEN =  "AAAAAAAAAAAAAAAAAAAAAImjjQEAAAAA%2BGiS6KuwY7hup%2BpAWOIOnREqG%2BE%3DGiKwMvyUSx441NercYxxHwoxafwQcLGrywSqWDixCplrxC4ghr"
def search_twitter(query, tweet_fields, max_results, bearer_token = BEARER_TOKEN, next_token=None):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}


    if next_token is not None:
        url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}&max_results={}&next_token={}".format(
            query, tweet_fields, max_results, next_token
        )
    else:
        url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}&max_results={}".format(
            query, tweet_fields, max_results
        )
    response = requests.request("GET", url, headers=headers)

    print('status:{}'.format(response.status_code))

    if response.status_code != 200:
        return None
    return response.json()

all_results = []
all_tweets_time = []
all_tweets = []
cnt = -1
next_token = None
total_cnt = 0

## follow instruction for pagination here: https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/paginate
## use next_token or since_id parameter to get next batch of tweets, loop until exhaust all tweets for the given query
## there is a daily rate limit for free api: https://developer.twitter.com/en/docs/twitter-api/rate-limits
## you can also add start_time and end_time to ==
while total_cnt < 30000:

    results = search_twitter('ukraine war biden', 'tweet.fields=text,author_id,created_at', 10, BEARER_TOKEN, next_token)
    if results == None:
        time.sleep(900)
        results = search_twitter('ukraine war biden', 'tweet.fields=text,author_id,created_at', 10, BEARER_TOKEN,  next_token)
        if results == None:
            break
    cnt = results['meta']['result_count']
    next_token =  results['meta']['next_token']
    all_results.append(results)
    for tweet in results['data']:
        all_tweets_time.append(tweet['created_at'])
        all_tweets.append(tweet['text'])
    total_cnt += cnt


from urlextract import URLExtract
from textblob import TextBlob

def sentimentAnalysis(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    return polarity, subjectivity

with open('/Users/sherylchen/PycharmProjects/pythonProject/tweepy/examples/API_v2/test1.txt', 'w') as f:
    for i in range(len(all_tweets)):
        time = all_tweets_time[i]
        text = all_tweets[i]
        p,s = sentimentAnalysis(text)
        output = "time: {}, text: {}, polarity:{} , sujectivity:{}\n".format(time, text, p, s)
        f.write(output)


print('{} tweets collected from {} response!'.format(total_cnt, len(all_results)))