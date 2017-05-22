from hmmlearn import hmm
import os
import cPickle as pkl
from python_speech_features import mfcc
import numpy as np
import warnings
import scipy.io.wavfile
import cPickle as pickle
import sys
import argparse


def get_mfcc(sig, rate=16000, cutoff=1.5):
    max_win = int(cutoff*rate)
    sig1_red = sig[:max_win]
    mf = mfcc(sig1_red, rate)
    return mf


def read_tags(fname):
    if not os.path.exists(fname):
        sys.exit('Input file not found, starting from no tags.')
    return pkl.load(open(fname, 'rb'))


def build_hmms(tagged_data, n_components=5):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    for key, value in tagged_data.iteritems():
        value['mfcc'] = get_mfcc(value['sig'])
        model = hmm.GaussianHMM(n_components=n_components)
        model.fit(value['mfcc'])
        value['model'] = model
    return tagged_data


def compute_sim_mat(X, models):
    warnings.catch_warnings()
    warnings.simplefilter('ignore')
    n_models = len(models)
    n_audio = len(X)
    likelihoods = np.empty((n_audio, n_models))
    for i in xrange(n_audio):
        for j in xrange(n_models):
            likelihoods[i, j] = models[j].score(X[i])
    return likelihoods


def main(args):
    warnings.catch_warnings()
    warnings.simplefilter('ignore')

    directory = args.directory
    in_file = args.input
    tagged_data = read_tags(in_file)

    feats = []
    models = []
    lengths = []
    y = []
    feature_order = []
    for key, value in tagged_data.iteritems():
        feats.append(get_mfcc(value['sig']))
        lengths.append(feats[-1].shape[0])
        model = hmm.GaussianHMM(n_components=5)
        model.fit(feats[-1])
        models.append(model)
        y.append(value['tag'])
        feature_order.append(key)
    y = np.array(y)

    likes = []
    for i in xrange(len(feats)):
        likes.append([])
        for j in xrange(len(feats)):
            likes[-1].append(models[j].score(feats[i]))
    likes = np.array(likes)
    with open('{}_tagged_features.pkl'.format(args.echo_id), 'wb') as f:
        pkl.dump([y, feature_order, likes], f)

    if(directory is not None):
        fnames = [x for x in os.listdir(directory) if x[-3:] == 'wav']
        all_features = {}
        for fname in fnames:
            all_features[fname] = []
            full_fname = directory+fname
            rate, sig = scipy.io.wavfile.read(full_fname)
            for i in xrange(len(feats)):
                all_features[fname].append(models[i].score(get_mfcc(sig)))
        with open('{}_wav_features.pkl'.format(args.echo_id), 'wb') as f:
            pickle.dump(all_features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='')
    parser.add_argument('-i', action='store', dest='input', type=str,
                        help='Audio files already tagged to be used.')
    parser.add_argument('-d', action='store', dest='directory', type=str,
                        help='Directory containing files to be tagged.',
                        default=None)
    parser.add_argument('-p', action='store', dest='echo_id', type=str,
                        help='An ID used to identify the data being tagged',
                        default='ECHO')
    args = parser.parse_args()
    main(args)
