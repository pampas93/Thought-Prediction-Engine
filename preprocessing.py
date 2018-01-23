import re
#from Collect_Tweets import DOWNLOADED_TWEETS_PATH
from guess_language import guess_language
import sys

Processed_TWEETS_PATH = 'processedTweets.txt'
store = open(Processed_TWEETS_PATH, 'a')
DOWNLOADED_TWEETS_PATH = "tweets.txt"

HASHTAGS_REGEX = re.compile('#')
MENTIONS_REGEX = re.compile('@[^\s]+')
EMOJI_NAME_REGEX = re.compile(':[a-z_-]+:')
LINK_REGEX = re.compile('https?://[^\s]+')
EXTRA_SPACES_REGEX = re.compile('\s{2,}')
HAYSTACK_REGEX = re.compile('(RT)')
ASCII_REGEX = re.compile('[[:ascii:]]')

def preprocess_tweet(tweet, pipeline):
    for pipe in pipeline:
        tweet = pipe(tweet)
    return tweet

def preprocess_hashtags(tweet):
    return HASHTAGS_REGEX.sub('', tweet)    #Remove Hastags

def preprocess_mentions(tweet):
    return MENTIONS_REGEX.sub('', tweet)    #Remove Mentions

def remove_extra_spaces(tweet):
    return EXTRA_SPACES_REGEX.sub(' ', tweet).strip()   #Remove Extra spaces

def remove_hyperlinks(tweet):
    return LINK_REGEX.sub('', tweet)        #Remoce Hyperlinks

def remove_haystack(tweet):
    return HAYSTACK_REGEX.sub('', tweet)

def remove_unicode(tweet):
    return ASCII_REGEX.sub('', tweet)       #Remove Unicodes

# LINK_REGEX.sub('', tweet)        #Remoce Hyperlinks
# ASCII_REGEX.sub('', tweet)       #Remove Unicodes
# EXTRA_SPACES_REGEX.sub(' ', tweet).strip()   #Remove Extra spaces
# MENTIONS_REGEX.sub('', tweet)    #Remove Mentions
# HASHTAGS_REGEX.sub('', tweet)    #Remove Hastags

def extract_emoji(tweet):
    emojis = EMOJI_NAME_REGEX.findall(tweet)
    tweet = EMOJI_NAME_REGEX.sub('', tweet)
    return [tweet, emojis]

def is_valid_training_data(tweet, emoji):
    if len(emoji) > 0 and len(tweet) > 2 and guess_language(tweet) == 'en':
        return True
    return False

preprocessing_pipeline = [
    preprocess_hashtags,
    preprocess_mentions,
    remove_hyperlinks,
    remove_unicode,
    remove_haystack,
]

def get_tweets():
    with open(DOWNLOADED_TWEETS_PATH, 'r') as raw_dataset:
        for raw_tweet in raw_dataset:
            processed_tweet = preprocess_tweet(raw_tweet, preprocessing_pipeline)
            tweet, emojis = extract_emoji(processed_tweet)
            print(guess_language((tweet)))
            if is_valid_training_data(tweet, emojis):
                store.write(processed_tweet)
                yield [remove_extra_spaces(tweet), emojis, raw_tweet]

def get_processed_tweets():
    with open(Processed_TWEETS_PATH, 'r') as dataset:
        for processed_tweet in dataset:
            tweet, emojis = extract_emoji(processed_tweet)
            yield [remove_extra_spaces(tweet), emojis, processed_tweet]


if __name__ == '__main__':
    with open(DOWNLOADED_TWEETS_PATH, 'r') as raw_dataset:
        for raw_tweet in raw_dataset:
            processed_tweet = preprocess_tweet(raw_tweet, preprocessing_pipeline)
            tweet, emojis = extract_emoji(processed_tweet)
            if is_valid_training_data(tweet, emojis):
                store.write(processed_tweet)