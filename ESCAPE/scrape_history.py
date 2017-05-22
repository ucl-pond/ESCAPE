import os
import re
import json
import time
# Written in Python 2.7 use pickle for 3
import cPickle as pkl
import sys
import argparse


def get_cookie(curl):
    curl = curl.split('-H')
    cookie = [x.strip() for x in curl if 'Cookie' in x][0]
    cookie = cookie[cookie.find(':')+2:cookie.rfind('\"')+1]
    return cookie


def get_interactions(echo_id, cred):
    curl = ('curl -s \'https://layla.amazon.co.uk/api/'
            'activities?startTime=&size=50&offset=-1\''
            ' -H \'Cookie: {}\' > .res{}.json'.format(cred, echo_id))
    os.system(curl)
    f = open('.res{}.json'.format(echo_id), 'rb')
    interactions = []
    res = json.loads(f.read())
    start_date = res['startDate']
    interactions += res['activities']
    counter = 0
    while(start_date is not None):
        time.sleep(1)
        curl = ('curl -s \'https://layla.amazon.co.uk/api/activities?'
                'startTime={}&size=50&offset=-1\''
                ' -H \'Cookie: {}\' > .res{}.json'.format(start_date,
                                                          cred,
                                                          counter))
        os.system(curl)
        f = open('.res{}.json'.format(counter), 'rb')
        counter += 1
        res = json.loads(f.read())
        start_date = res['startDate']
        interactions += res['activities']
    with open('{}_Interactions.pkl'.format(echo_id), 'wb') as f:
        pkl.dump(interactions, f)
    pattern = '\.*res\S+\.json'
    for f in os.listdir('.'):
        if re.search(pattern, f):
            os.remove(os.path.join('.', f))


def get_audio(echo_id, cred):
    directory = '{}_Audio'.format(echo_id)
    if not os.path.exists(directory):
        os.mkdir(directory)
    interactions = pkl.load(open('{}_Interactions.pkl'.format(echo_id)))
    saveCount = 0
    skipCount = 0

    n_items = len(interactions)
    for activity in interactions:
        sys.stdout.flush()
        audioId = activity['utteranceId']
        if(audioId is None or len(audioId) == 0):
            skipCount += 1
            continue
        fname = audioId.replace('.', '').replace('/', '')
        saveCount += 1
        if(os.path.exists('./{}_Audio/{}.wav'.format(echo_id, fname))):
            continue
        curl = ('curl -s \'https://layla.amazon.co.uk/api/utterance/audio/'
                'data?id={uter}\' -H \'Cookie: {cook}\' > ./{pid}_Audio/'
                '{fname}.wav'.format(uter=audioId, cook=cred,
                                     pid=echo_id, fname=fname))
        sys.stdout.write(('\r{} files saved, '
                          '{} files skipped, '
                          '{} files total'.format(saveCount,
                                                  skipCount,
                                                  n_items)))
        os.system(curl)
        time.sleep(1)
    sys.stdout.write(('\r{} files saved, '
                      '{} files skipped, '
                      '{} files total\n'.format(saveCount,
                                                skipCount,
                                                n_items)))


def main(args):
    echo_id = args.echo_id
    curl = open('.credentials').read()
    cookie = get_cookie(curl)
    # get_interactions(echo_id, cookie)
    if(args.audio):
        get_audio(echo_id, cookie)


if __name__ == '__main__':
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                     epilog='')
    parser.add_argument('-a', action='store_true', dest='audio',
                        help='Scrape audio data')
    parser.add_argument('-p', action='store', dest='echo_id', type=str,
                        help='An ID used to identify the Echo being scraped',
                        default='ECHO')
    args = parser.parse_args()
    main(args)
