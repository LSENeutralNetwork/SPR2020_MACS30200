import sys, getopt, datetime, codecs, got3 as got, numpy as np, pandas as pd, time

def export(since_date, until_date):

    tweetCriteria = got.manager.TweetCriteria()
    outputFileName = "covid_tweets_{}.csv".format(since_date)

    keywords = "corona OR coronavirus OR covid OR covid19 OR sarscov2"

    tweetCriteria.setSince(since_date)
    tweetCriteria.setUntil(until_date) 
    tweetCriteria.setQuerySearch(keywords)
    tweetCriteria.setLang('en')
    tweetCriteria.setMaxTweets(20000)

    outputFile = codecs.open(outputFileName, "w+", "utf-8")
    outputFile.write('username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink')
    
    print('Searching...\n')

    def receiveBuffer(tweets):
        for t in tweets:
            outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;"%s";%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
        outputFile.flush()
        #print('More %d saved on file...\n' % len(tweets))

    got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

    outputFile.close()
    print('Done. Output file generated "%s".' % outputFileName)


def main():

    date_range = []
    for d in np.arange(17, 18):
        date_range.append('2020-03-{}'.format(d))
    for d in np.arange(1,31):
        date_range.append('2020-04-{}'.format(d))
    for d in np.arange(1,27):
        date_range.append('2020-05-{}'.format(d))

    for i in range(len(date_range)-1):
        try:
            export(date_range[i], date_range[i+1])
        except:
            print(date_range[i] + " limit reached")
            pass
        time.sleep(500)


if __name__=="__main__":
    main()








