import numpy
import time
import sys
import subprocess
import os
import random
import argparse
import pdb
import re

sys.path.append(os.path.abspath('..'))

from is13.data import load
from is13.data import mpqa_load
from is13.rnn.deep_sentence_lstm import model
from is13.metrics.accuracy import conlleval, accuracy
from is13.utils.tools import shuffle, minibatch, contextwin
from gensim.models import Word2Vec

def load_embeddings(embfile, idx2word, vocsize):
    storage_path = os.path.join('./embeddings', embfile + '.npy')
    try:
        embeddings = numpy.load(storage_path)
    except IOError:
        print "Embeddings not found on disk, loading them from model binary"
        mikolov_model = Word2Vec.load_word2vec_format(embfile, binary=True)
        embeddings = numpy.zeros((vocsize, 300))
        for i in xrange(vocsize):
            word = re.sub('",', '', idx2word[i].lower())
            try:
                embeddings[i] = mikolov_model[word]
            except:
                print "%s not found in model" % word
        print "Done loading embeddings"
        numpy.save(storage_path, embeddings)

    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Adjust level of verbosity.')
    parser.add_argument('-nh', '--num-hidden', dest='num_hidden', type=int, default=20,
                        help='Set dimension of hidden units.')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='Set number of stacked layers')
    parser.add_argument('-l', '--lambda', dest='lam', type=float, default=0.00000001,
                        help='Set lambda value used for L2-regularization')
    parser.add_argument('--seed', type=int, default=345,
                        help='Set PRNG seed')
    parser.add_argument('--emb-file', dest='emb_file', type=str,
                        help='Location of file containing word embeddings')
    parser.add_argument('-e', '--num-epochs', dest='num_epochs', type=int, default=200,
                        help='Set number of epochs to train')
    parser.add_argument('-a', '--alpha', dest='alpha', type=float, default=0.01,
                        help='Set the initial learning rate')
    parser.add_argument('--ex', dest='examples_file', type=str, default='./mpqa2data.pkl',
                    help='Path to file containing the pkled complete dataset')
    parser.add_argument('--adagrad', dest='adagrad', type=bool, default=True,
                    help='Enable adagrad')

    args = parser.parse_args()

    s = {'lr': args.alpha,
         'verbose': args.verbose,
         'decay': True, # decay on the learning rate if improvement stops
         'nhidden': args.num_hidden, # number of hidden units
         'depth': args.depth, # number of layers in space
         'seed': args.seed,
         'nepochs': args.num_epochs}

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = mpqa_load.mpqa(args.examples_file)
    idx2label = dic['idx2label']
    idx2word  = dic['idx2word']

    print idx2label

    train_lex, train_y = train_set
    valid_lex, valid_y = valid_set
    test_lex,  test_y  = test_set

    vocsize = max(set([item for sublist in train_lex+valid_lex+test_lex for item in sublist])) + 1

    nclasses = len(set([item for sublist in train_y+valid_y+test_y for item in sublist]))
    
    nsentences = len(train_lex)

    # instantiate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    depth = s['depth'],
                    embeddings = load_embeddings(args.emb_file, idx2word, vocsize),
                    lam=args.lam,
                    adagrad=args.adagrad )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    s['be'] = 0
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            words = numpy.asarray(train_lex[i]).astype('int32').reshape((len(train_lex[i]), 1))
            labels = numpy.asarray(train_y[i]).astype('int32').reshape(len(train_y[i]))
            if len(words) == 0: continue

            cost, _s = rnn.train(words, labels, s['clr'])
            
            if args.verbose > 0 and i % nsentences/4 == 0:
                for idx in xrange(len(words)):
                    print [round(item, 3) for item in _s[idx,0,:].tolist()], labels[idx], numpy.argmax(_s[idx,0,:]), idx2word[words[idx]]
                print '[learning] epoch %i >> %2.2f%%' % (e, (i+1)*100./nsentences), '\tCurrent cost: %.3f' % cost
                sys.stdout.flush()
        
        #pdb.set_trace()
        # evaluation // back into the real world : idx -> words
        predictions_test = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(x).astype('int32').reshape((len(x), 1)))) \
                             for x in test_lex if len(x) > 0 ]
        groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y if len(y) > 0 ]
        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex if len(w) > 0 ]

        predictions_valid = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(x).astype('int32').reshape((len(x), 1)))) \
                             for x in valid_lex if len(x) > 0 ]
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y if len(y) > 0 ]
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex if len(w) > 0 ]

        #pdb.set_trace()

        predictions_test_labels = [item for sublist in predictions_test for item in sublist]
        zero_predictions = [item for item in predictions_test_labels if item == '0']
        nonzero_predictions = [item for item in predictions_test_labels if item != '0'] 

        print "Num zero_predictions: %d" % len(zero_predictions)
        print "Num nonzero_predictions: %d" % len(nonzero_predictions)

        errors = 0
        for idx in xrange(len(groundtruth_test)):
            cur_gt_labels = groundtruth_test[idx]
            cur_pred_labels = predictions_test[idx]
            for subidx in xrange(len(cur_gt_labels)):
                if cur_gt_labels[subidx] != cur_pred_labels[subidx]:
                    errors += 1

        accuracy = (1.0 - errors / float(len(predictions_test_labels)))

        # evaluation // compute the accuracy using conlleval.pl
        # error_rate = accuracy([item for sublist in predictions_test for item in sublist], 
        #                                                     [item for sublist in groundtruth_test for item in sublist])
        if accuracy > best_f1:
            best_f1 = accuracy
            s['be'] = e

        #print "Nonzero prediction count: %d" % len([item for item ])
        print "accuracy after %d epochs: %g" % (e, accuracy)
        
        #res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        #res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        # if res_valid['f1'] > best_f1:
        #     rnn.save(folder)
        #     best_f1 = res_valid['f1']
        #     if s['verbose']:
        #         print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
        #     s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
        #     s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
        #     s['be'] = e
        #     subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
        #     subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        # else:
        #     print ''

    #print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

if __name__ == '__main__':
    main()
