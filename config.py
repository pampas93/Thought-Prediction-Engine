from emoji.unicode_codes import UNICODE_EMOJI

class TwitterAuth:
    CONSUMER_KEY = 'A8NnbW2i2QIXk5Nn0fxsLqP4a'
    CONSUMER_SECRET = 'yTEyXbsqhbO97tclJjoEZySfnWApEfb03BRAVsHLW0C2dVYnsD'
    ACCESS_TOKEN = '479701832-RJbhnRGhjrxMGsafoMORvOvbUXDOa0XEoHzrPt4A'
    ACCESS_TOKEN_SECRET = 'I0pzaFpAxaqgbX3EJrSTUFdo9uZfUuFXOYIjAaiMGXeND'

# max is 400
raw_emojis = '😀😂😅😆😇😘😍😜😎🤓😶😏🤗😐😡😟😞🙄☹😔😮😴💤💩😭😈👿👌👸🎅👅👀👍💪👻🤖😺🐟🐠🐷🐌🐼🐺🐯🐅🦃🐕🐇🌾🎍🍀🐾🌏🌚🌝🌞🌦🔥💥☃✨❄💧🍏🍊🍌🌽🍔🌮☕🍧⚽🏐🎖🎵💬🎣🏓🚵🎮🎬🚗🚓🚨🚋🚠🛥🚀🚢🎠🚧🚧🚧✈🏥📱⌨💻📠📞🔦💴💸🔮💊🔬🔭📫📈📉🖇✂🔒🔓📒💛❤💙💔💞💕💝💘🚾⚠♻🕐'

def is_valid(e):
    try:
        UNICODE_EMOJI[e]
        return e
    except KeyError:
        pass

LANGUAGE = 'en'
# filter out emojis not in our library
EMOJIS = list(filter(None, [is_valid(e) for e in raw_emojis]))
DOWNLOADED_TWEETS_PATH = 'emoji_twitter_data.txt'
SENTRY_DSN = ''
