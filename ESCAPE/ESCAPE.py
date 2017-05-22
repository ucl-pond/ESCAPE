#!/usr/local/bin/python
from __future__ import print_function
from __future__ import division
import scrape_history as scraper
import kl_labelling as kl
import hmm_generate_features as hmm_gen
import argparse
import os
import scipy.io.wavfile
import cPickle as pkl
from sklearn import decomposition
from sklearn import preprocessing
from matplotlib import pyplot as plt

plt.style.use('seaborn')
colors = plt.style.library['seaborn']['axes.prop_cycle'].by_key()['color']


def main(args):
    echo_id = args.echo_id
    curl = open(args.cred).read()
    cookie = scraper.get_cookie(curl)
    scraper.get_interactions(echo_id, cookie)
    # scraper.get_audio(echo_id, cookie)
    exit()
    audio_dir = '{}_Audio/'.format(echo_id)

    fnames = [x for x in os.listdir(audio_dir) if x[-3:] == 'wav']
    tag_fname = '{}_{}.pkl'.format(echo_id, args.tag)
    tagged_data = kl.read_tags(tag_fname)
    for fname in fnames:
        fname = audio_dir+fname
        if(os.stat(fname).st_size == 0):
            continue
        rate, sig = scipy.io.wavfile.read(fname)
        tag, corr = kl.match_audio(sig, tagged_data)
        if(tag is None):
            tag = kl.tag_audio(fname)
            if(tag > 0):
                tagged_data = kl.update_tags(fname, sig, tag,
                                             tagged_data, tag_fname,
                                             True)
            if(tag == 0):
                print('Audio not tagged')
            elif(tag == -1):
                print('{} audio files tagged.'
                      'Quit early.'.format(len(tagged_data)))
                break
        else:
            print('Closest tag is {} with a distance of {}'.format(tag,
                                                                   corr))
    pkl.dump(tagged_data, open(tag_fname, 'wb'))
    tagged_data = hmm_gen.build_hmms(tagged_data)
    pkl.dump(tagged_data, open(tag_fname, 'wb'))

    sorted_names = sorted(tagged_data.keys())
    models = [tagged_data[x]['model'] for x in sorted_names]
    mfccs = [tagged_data[x]['mfcc'] for x in sorted_names]

    tagged_likes = hmm_gen.compute_sim_mat(mfccs, models)
    audio_tags = [tagged_data[x]['tag'] for x in sorted_names]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(audio_tags)

    pca = decomposition.PCA()
    X = pca.fit_transform(tagged_likes)

    plt_ax = [(0, 1), (0, 2), (1, 2)]
    fig, ax = plt.subplots(3)
    lines = dict((x, None) for x in xrange(le.classes_.shape[0]))
    for i in xrange(3):
        x_ax, y_ax = plt_ax[i]
        ax[i].set_xlabel('PC{}'.format(x_ax+1))
        ax[i].set_ylabel('PC{}'.format(y_ax+1))
        for j in xrange(le.classes_.shape[0]):
            line = ax[i].scatter(X[y == j, x_ax], X[y == j, y_ax],
                                 color=colors[j])
            lines[j] = line
    fig.legend(lines.values(), le.classes_,
               loc='lower right')
    fig.tight_layout()
    fig.show()
    plt.show()


if __name__ == '__main__':
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                     epilog='')
    parser.add_argument('-p', action='store', dest='echo_id', type=str,
                        help='An ID used to identify the Echo being scraped.',
                        default='ECHO')
    parser.add_argument('-c', action='store', dest='cred', type=str,
                        help='The file containing the cURL for this account.',
                        default='.cred')
    parser.add_argument('-t', action='store', dest='tag', type=str,
                        help='Suffix of the tagged data file. If it already '
                        'exists then this file will be read and used.',
                        default='tagged')
    parser.add_argument('-s', action='store_true', dest='save',
                        help='Save after every tag. Slower, but safer.')
    args = parser.parse_args()
    main(args)
