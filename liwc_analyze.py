import pandas as pd
import numpy as np
import preprocessor as p
import re
from langdetect import detect, detect_langs
import csv
from liwc import Liwc
import datetime

WORD_RE = re.compile(r"[\w']+")
liwc = Liwc('LIWC2007_English100131.dic')
pd.set_option('mode.chained_assignment', None)


def detect_language(text):
    result ='non-en'
    try:
        lang_list = detect_langs(text)
        if (len(lang_list) == 1) and (lang_list[0].lang == 'en') and (float(lang_list[0].prob) > 0.99):
            result = 'en'
    except:
        return 'no_id_text'
    return result

def preprocess(text):
    if detect_language(text) != 'en':
        return []
    else:
        text = p.clean(text.lower())
        words = WORD_RE.findall(text)
        return words
    
def extract_emotion(l):
    d = {'sad':0, 'anger':0, 'anx':0}
    for w in l:
        for emo in liwc.search(w):
            if emo in d.keys():
                d[emo] += 1
    return d

def analyze_text(df):
    df['tokens'] = df.text.apply(preprocess)
    df['emo_dict'] = df.tokens.apply(extract_emotion)
    df['sadness'] = df.emo_dict.apply(lambda x: x['sad'])
    df['anger'] = df.emo_dict.apply(lambda x: x['anger'])
    df['anxiety'] = df.emo_dict.apply(lambda x: x['anx'])
    
def aggregate(df):
    d = {}
    d['N'] = df.shape[0]
    d['sadness_count'] = df.anger[df.sadness > 0].shape[0]
    d['sadness_retweet_sum'] =df.retweets[df.sadness > 0].sum()
    d['anxiety_count'] =df.anger[df.anxiety > 0].shape[0]
    d['anxiety_retweet_sum'] =df.retweets[df.anxiety > 0].sum()
    d['anger_count'] =df.anger[df.anger > 0].shape[0]
    d['anger_retweet_sum'] =df.retweets[df.anger > 0].sum()
    
    return d

def analyze_one_day(FILEPATH):
    line = [FILEPATH[-14:-4]]
    
    df = pd.read_csv(FILEPATH, delimiter=';', error_bad_lines=False)
    analyze_text(df)
    df.to_csv('analyzed_{}.csv'.format(FILEPATH[-14:-4]), index=False)

    d = aggregate(df)
    line = line + list(d.values())

    print(line)

    with open('liwc_result.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(line)

def main():
    date_range = []
    for d in np.arange(11, 32):
        date_range.append('2020-03-{}'.format(d))
    for d in np.arange(1,10):
        date_range.append('2020-04-0{}'.format(d))
    for d in np.arange(10,31):
        date_range.append('2020-04-{}'.format(d))
    for d in np.arange(1,10):
        date_range.append('2020-05-0{}'.format(d))
    for d in np.arange(10,26):
        date_range.append('2020-05-{}'.format(d))

    for d in date_range:
        FILEPATH = '../GetOldTweets-python/covid_tweets_{}.csv'.format(d)
        analyze_one_day(FILEPATH)
        print(str(d) + ' tweets analyzed.')

if __name__=='__main__':
    main()


