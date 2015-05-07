import gzip
import cPickle
import urllib
import os

from os.path import isfile
from math import floor
from is13.utils.tools import shuffle

PREFIX = os.getenv('ATISDATA', '')

def download(origin):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print 'Downloading data from %s' % origin
    name = origin.split('/')[-1]
    urllib.urlretrieve(origin, name)

def load(filename):
    if not isfile(filename):
        download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
    f = gzip.open(filename,'rb')
    return f

def mpqa(filename):
    f = open(filename, 'rb')
    data, dicts = cPickle.load(f)
    sentences, labels = data
    shuffle([sentences, labels], 69)

    n = len(sentences)
    training_size = int(floor(n * 0.8))
    training_end = training_size-1
    train_set = (sentences[0:training_end], labels[0:training_end])

    valid_size = int(floor(n * 0.1))
    valid_end = training_size + valid_size - 1
    valid_set = (sentences[training_size:valid_end], labels[training_size:valid_end])

    test_size = int(floor(n * 0.1))
    test_begin = valid_end + 1
    test_end = valid_end + test_size - 1
    test_set = (sentences[test_begin:test_end], labels[test_begin:test_end])

    f.close()

    return (train_set, valid_set, test_set, dicts)



def atisfull():
    f = load(PREFIX + 'atis.pkl.gz')
    train_set, test_set, dicts = cPickle.load(f)
    return train_set, test_set, dicts

def atisfold(fold):
    assert fold in range(5)
    f = load(PREFIX + 'atis.fold'+str(fold)+'.pkl.gz')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts
 
if __name__ == '__main__':
    
    ''' visualize a few sentences '''

    import pdb
    data = atisfull()

    w2ne, w2la = {}, {}
    train, test, dic = data
    
    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']
    
    idx2w  = dict((v,k) for k,v in w2idx.iteritems())
    idx2ne = dict((v,k) for k,v in ne2idx.iteritems())
    idx2la = dict((v,k) for k,v in labels2idx.iteritems())

    test_x,  test_ne,  test_label  = test
    train_x, train_ne, train_label = train
    wlength = 35

    for e in ['train','test']:
      for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
        print 'WORD'.rjust(wlength), 'LABEL'.rjust(wlength)
        for wx, la in zip(sw, sl): print idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength)
        print '\n'+'**'*30+'\n'
        pdb.set_trace()
