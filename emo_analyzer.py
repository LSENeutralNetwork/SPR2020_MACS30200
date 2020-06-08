import numpy as np
import pandas as pd
import time
import re
from langdetect import detect, detect_langs
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, EmotionOptions 
import csv
from datetime import datetime
import preprocessor as p
pd.set_option('mode.chained_assignment', None)

authenticator = IAMAuthenticator("SM7MmQAk6Hl7s7PIFZaxpHu80YEEO278MUG2s4r_cpGX")
natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12', authenticator=authenticator)

service_url = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/c13b0f4c-72e2-4d06-8e83-4513aa9c5c3b"

natural_language_understanding.set_service_url(service_url)

### Text Processing Functions
def rid_url(text):
    return  re.sub(r'http[s]*\:\/\/\w*|\S+\.co\w*\S+', '', text)

def rid_non_word(text):
    return re.sub(r'[^\w\s]|\xa0|[0-9]{4,}|\w{15,}', '', text)

def clean(text):
    return rid_non_word(rid_url(p.clean(str(text))))

def detect_language(text):
    result ='non-en'
    try:
        lang_list = detect_langs(text)
        if (len(lang_list) == 1) and (lang_list[0].lang == 'en') and (float(lang_list[0].prob) > 0.99):
            result = 'en'
    except:
        return 'no_id_text'
    return result

def word_count(text):
    l = text.split()
    return len(l)
    
def preprocess_df(df):
    df['clean_text'] = df.text.apply(clean)
    df['lang'] = df.clean_text.apply(detect_language) 
    df1 = df[df.lang == 'en']  
    df2 = df1[df1.clean_text.apply(word_count) > 3]
    df2.reset_index(drop=True, inplace=True)
    return df2

### Emotion Analysis Method
def text_emotion(text):
    result = None
    try:
        response = natural_language_understanding.analyze(text=text,
                     features=Features(emotion=EmotionOptions())).get_result()
        result = response['emotion']['document']['emotion']
    except:
        print(text)

        return None
    
    return result

def dominant_emotion(d):
    return max(d)

def second_dominant_emotion(d):
    max_key = max(d)
    max2 = 0
    max2_key = None
    
    for k, v in d.items():
        if float(v) > max2 and k != max_key:
            max2 = float(v)
            max2_key = k
            
    return max2_key

def classify_emotion(df):
    assert('clean_text' in df.columns), "Require a column of cleaned text!"
    df['emo_dict'] = df.clean_text.apply(text_emotion)
    df.dropna(subset=['emo_dict'], inplace=True)
    df['emo_1'] = df.emo_dict.apply(dominant_emotion)
    df['emo_2'] = df.emo_dict.apply(second_dominant_emotion)
    return df

def agg_emotion(s):
    agg_emo = {'sadness': 0, 'joy': 0, 'fear': 0, 'anger': 0, 'disgust': 0}

    for d in s:
        agg_emo['sadness'] += d['sadness']
        agg_emo['joy'] += d['joy']
        agg_emo['fear'] += d['fear']
        agg_emo['anger'] += d['anger']
        agg_emo['disgust'] += d['disgust']

    return agg_emo

def agg_retweet(retweet, emo_1, emo_2):
    d = {'sadness': 0,'joy': 0, 'fear': 0, 'anger': 0, 'disgust': 0}
    
    for i in range(retweet.shape[0]):
        if emo_1[i] in d.keys() and emo_2[i] in d.keys():
            d[emo_1[i]] += float(retweet[i])
            d[emo_2[i]] += float(retweet[i]) * 0.5
        
    return d

def analyze_one_day(FILEPATH):
    line = [datetime.strptime(FILEPATH[-14:-4], "%Y-%m-%d")]
    
    df = pd.read_csv(FILEPATH, delimiter=';', error_bad_lines=False)
    dfc = preprocess_df(df)
    if dfc.shape[0] > 500:
        dfc = dfc.sample(n=500)
    print('finished preprocessing')
    analyzed_df = classify_emotion(dfc)
    analyzed_df.reset_index(drop=True, inplace=True)
    analyzed_df.to_csv('analyzed_{}.csv'.format(FILEPATH[-14:-4]))
    N = analyzed_df.shape[0]
    line.append(N)

    assert('emo_1' in analyzed_df.columns), "No emo_1 col!"
    assert('emo_2' in analyzed_df.columns), "No emo_2 col!"
    line.append(analyzed_df.emo_1.value_counts().index[0])
    line.append(analyzed_df.emo_2.value_counts().index[0])

    assert('emo_dict' in analyzed_df.columns), "No column for emo dict!"
    emo_dict = agg_emotion(analyzed_df.emo_dict)
    retweet_dict = agg_retweet(analyzed_df.retweets, analyzed_df.emo_1, analyzed_df.emo_2)
    line.append(emo_dict['sadness'] / N)
    line.append(retweet_dict['sadness'])
    line.append(emo_dict['joy'] / N)
    line.append(retweet_dict['joy'])
    line.append(emo_dict['fear'] / N)
    line.append(retweet_dict['fear'])
    line.append(emo_dict['anger'] / N)
    line.append(retweet_dict['anger'])
    line.append(emo_dict['disgust'] / N)
    line.append(retweet_dict['disgust'])

    print(line)

    with open('agg_result.csv', 'a+', newline='') as f:
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


