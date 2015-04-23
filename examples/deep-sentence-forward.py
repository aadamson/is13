import numpy
import time
import sys
import subprocess
import os
import random
import argparse
import pdb

sys.path.append(os.path.abspath('..'))

from is13.data import load
from is13.data import mpqa_load
from is13.rnn.deep_sentence import model
from is13.metrics.accuracy import conlleval, accuracy
from is13.utils.tools import shuffle, minibatch, contextwin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Adjust level of verbosity.')
    parser.add_argument('-nh', '--num-hidden', dest='num_hidden', type=int, default=100,
                        help='Set dimension of hidden units.')
    parser.add_argument('-w', '--window', type=int, default=5,
                        help='Set size of context window (in words).')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='Set number of stacked layers')
    parser.add_argument('--seed', type=int, default=345,
                        help='Set PRNG seed')
    parser.add_argument('--emb-dim', dest='emb_dimension', type=int, default=100,
                        help='Set size of word embeddings')
    parser.add_argument('-e', '--num-epochs', dest='num_epochs', type=int, default=50,
                        help='Set number of epochs to train')

    args = parser.parse_args()

    s = {'fold': 3, # 5 folds 0,1,2,3,4
         'lr': 0.15,
         'verbose': args.verbose,
         'decay': False, # decay on the learning rate if improvement stops
         'win': args.window, # number of words in the context window
         'bs': 9, # number of backprop through time steps
         'nhidden': args.num_hidden, # number of hidden units
         'depth': args.depth, # number of layers in space
         'seed': args.seed,
         'emb_dimension': args.emb_dimension, # dimension of word embedding
         'nepochs': args.num_epochs}

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = mpqa_load.mpqa('mpqa2data.pkl')
    idx2label = dic['idx2label']
    idx2word  = dic['idx2word']

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
                    de = s['emb_dimension'],
                    cs = s['win'],
                    depth = s['depth'] )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            # try:
                words = numpy.asarray(train_lex[i]).astype('int32').reshape((len(train_lex[i]), 1))
                labels = numpy.asarray(train_y[i]).astype('int32').reshape(len(train_y[i]))
                #print words
                if len(words) == 0: continue
                rnn.train(words, labels, s['clr'])
                rnn.normalize()
           #except:
                # 1
                #pdb.set_trace()
            #try:
                # cwords = contextwin(train_lex[i], s['win'])
                # words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                #              minibatch(cwords, s['bs']))

                # clabels = contextwin(train_y[i], s['win'])
                # labels = map(lambda x: numpy.asarray(x).astype('int32'),\
                #              minibatch(clabels, s['bs']))
                # for word_batch, label_batch in zip(words, labels):
                #     print "Word batch: ", word_batch
                #     print "label_batch: ", label_batch
                #     rnn.train(word_batch, label_batch, s['clr'])
                #     rnn.normalize()
                if s['verbose']:
                    print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                    sys.stdout.flush()
            #except:
                #pdb.set_trace()
        
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

        # evaluation // compute the accuracy using conlleval.pl

        print "Accuracy after %d epochs: %g" % (e, accuracy(predictions_test, groundtruth_test))
        
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
        
        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5 
        if s['clr'] < 1e-5: break

    #print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

if __name__ == '__main__':
    main()
