from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io.wavfile
import os
import cPickle as pkl
import wave
import pyaudio
import argparse
from python_speech_features import mfcc
import sys


def read_tags(fname):
    if not os.path.exists(fname):
        print('Input file not found, starting from no tags.')
        return {}
    return pkl.load(open(fname, 'rb'))


def get_correlation(sig1, sig2, rate=16000, cutoff=1.5):
    max_win = int(cutoff*rate)
    sig1_red = sig1[:max_win]
    sig2_red = sig2[:max_win]

    mf1 = mfcc(sig1_red, rate)
    mf2 = mfcc(sig2_red, rate)

    m1 = np.mean(mf1, axis=0)
    c1 = np.cov(mf1.T)
    m2 = np.mean(mf2, axis=0)
    c2 = np.cov(mf2.T)
    return kl_distance(m1, m2, c1, c2)


def kullback_liebler(m1, m2, c1, c2):
    k = m1.shape[0]
    c2_inv_term = np.matmul(np.linalg.inv(c2), c1)
    kl = np.matmul(np.linalg.inv(c2), (m2-m1))
    kl = np.matmul((m2-m1), kl)
    kl = np.matrix.trace(c2_inv_term)+kl-k-np.log(np.linalg.det(c2_inv_term))
    return 0.5*kl


def kl_distance(m1, m2, c1, c2):
    dist = kullback_liebler(m1, m2, c1, c2)
    dist += kullback_liebler(m2, m1, c2, c1)
    return dist


def match_audio(sig, tagged_data, rate=16000, kl_cutoff=50):
    best_tag = None
    best_corr = 9e10
    for key, value in tagged_data.iteritems():
        taged_sig = value['sig']
        corr = get_correlation(sig, taged_sig, rate=rate)
        if(corr < best_corr):
            best_tag = value['tag']
            best_corr = corr
    if(best_corr < kl_cutoff):
        return best_tag, best_corr
    return None, None


def tag_audio(fname):
    wf = wave.open(fname, 'rb')
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True, stream_callback=callback)
    stream.start_stream()

    tag = raw_input('Tag this audio (q! for early quitting):')
    if(tag == 'q!'):
        return -1

    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()
    if(tag == ''):
        return 0
    return tag


def update_tags(fname, sig, tag, tagged_data, save_name, save=True):
    tagged_data[fname] = {}
    tagged_data[fname]['sig'] = sig
    tagged_data[fname]['tag'] = tag
    if(save):
        pkl.dump(tagged_data, open(save_name, 'wb'))
    return tagged_data


def main(args):
    if(args.directory is None):
        sys.exit('Must provide a directory with wav files')
    directory = args.directory
    fnames = [x for x in os.listdir(directory) if x[-3:] == 'wav']
    tagged_data = read_tags(args.input)
    for fname in fnames:
        print(fname)
        fname = directory+fname
        rate, sig = scipy.io.wavfile.read(fname)
        tag, corr = match_audio(sig, tagged_data)
        if(tag is None):
            tag = tag_audio(fname)
            if(tag is not None):
                tagged_data = update_tags(fname, sig, tag,
                                          tagged_data, args.output,
                                          args.save)
            print('\n\n')
        else:
            print(tag, '\n\n')
    pkl.dump(tagged_data, open(args.output, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='')
    parser.add_argument('-i', action='store', dest='input', type=str,
                        help='Audio files already tagged to be used.',
                        default=None)
    parser.add_argument('-o', action='store', dest='output', type=str,
                        help='Output file name.',
                        default='ECHO_tags.pkl')
    parser.add_argument('-d', action='store', dest='directory', type=str,
                        help='Directory containing files to be tagged.',
                        default=None)
    parser.add_argument('-s', action='store_true', dest='save',
                        help='Save after every tag. Enables early quitting.')
    args = parser.parse_args()
    main(args)
