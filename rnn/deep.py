import theano
import numpy as np
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
    
    def __init__(self, nh, nc, ne, de, cs, depth):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        depth :: height of network
        '''
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * np.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))

        # Inner layer paramters
        self.forward_v = theano.shared(name='forward_v', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_v    = theano.shared(name='back_v', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.forward_w = theano.shared(name='forward_w', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_w    = theano.shared(name='back_w', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.forward_b = theano.shared(name='forward_b',
                                       value=np.zeros((depth, nh),
                                       dtype=theano.config.floatX))
        self.back_b    = theano.shared(name='back_b',
                                       value=np.zeros((depth, nh),
                                       dtype=theano.config.floatX))

        # First layer parameters
        self.forward_w1 = theano.shared(name='forward_w1',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de * cs, nh)))
        self.back_w1    = theano.shared(name='back_w1',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de * cs, nh)))

        # Output layer parameters
        self.c         = theano.shared(name='c',
                                       value=np.zeros(nc,
                                       dtype=theano.config.floatX))
        self.forward_U = theano.shared(name='forward_U',
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (nh, nc))
                                       .astype(theano.config.floatX))
        self.back_U    = theano.shared(name='back_U',
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (nh, nc))
                                       .astype(theano.config.floatX))

        self.forward_h1_0 = theano.shared(name='forward_h1_0',
                                          value=np.zeros(nh,
                                          dtype=theano.config.floatX))
        self.back_h1_tp1 = theano.shared(name='back_h1_tp1',
                                         value=np.zeros(nh,
                                         dtype=theano.config.floatX))


        # bundle
        self.params = [self.emb, 
                       self.forward_w, self.back_w, 
                       self.forward_v, self.back_v, 
                       self.forward_b, self.back_b, 
                       self.c, self.forward_U, self.back_U,
                       self.forward_w1, self.back_w1,
                       self.forward_h1_0, self.back_h1_tp1]

        self.names  = ['embeddings', 
                       'forward_w', 'back_w', 
                       'forward_v', 'back_v', 
                       'forward_b', 'back_b', 
                       'c', 'forward_U', 'back_U',
                       'forward_w1', 'back_w1',
                       'forward_h1_0', 'back_h1_tp1']

        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y = T.iscalar('y') # label

        def s_t(forward_h_t_i, back_h_t_i):
            s = T.nnet.softmax(T.dot(forward_h_t_i, self.forward_U)
                               + T.dot(back_h_t_i, self.back_U) 
                               + self.c)

            return s

        def forward_recurrence_igt1(forward_h_t_im1, back_h_t_im1, forward_h_tm1_i, i):
            forward_h_t_i = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.forward_w[i])
                                           + T.dot(back_h_t_im1, self.forward_w[i])
                                           + T.dot(forward_h_tm1_i, self.forward_v[i]) 
                                           + self.forward_b[i])
            return forward_h_t_i

        def back_recurrence_igt1(forward_h_t_im1, back_h_t_im1, back_h_tp1_i, i):
            back_h_t_i    = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.back_w[i])
                                           + T.dot(back_h_t_im1, self.back_w[i])
                                           + T.dot(back_h_tp1_i, self.back_v[i]) 
                                           + self.back_b[i])
            return back_h_t_i

        def forward_recurrence_ieq1(x_t, forward_h_tm1_i):
            forward_h_t_i = T.nnet.sigmoid(T.dot(x_t, self.forward_w1)
                                           + T.dot(forward_h_tm1_i, self.forward_v[0])
                                           + self.forward_b[0])

            return forward_h_t_i

        def back_recurrence_ieq1(x_t, back_h_tp1_i):
            back_h_t_i    = T.nnet.sigmoid(T.dot(x_t, self.back_w1)
                                           + T.dot(back_h_tp1_i, self.back_v[0])
                                           + self.back_b[0])

            return back_h_t_i

        # Construct forward hidden nodes for first layer
        forward_h_1, _ = theano.scan(fn=forward_recurrence_ieq1,
                                     sequences=x,
                                     outputs_info=self.forward_h1_0,
                                     n_steps=x.shape[0])
        back_h_1, _    = theano.scan(fn=back_recurrence_ieq1,
                                     sequences=x[::-1],
                                     outputs_info=self.back_h1_tp1,
                                     n_steps=x.shape[0])

        forward_h_im1 = forward_h_1
        back_h_im1    = back_h_1[::-1]

        for i in range(1, depth):
            # Do forward recurrence for layer i+1
            forward_h_i, _ = theano.scan(fn=forward_recurrence_igt1,
                                         sequences=[forward_h_im1, back_h_im1],
                                         outputs_info=self.forward_h1_0,
                                         non_sequences=i,
                                         n_steps=x.shape[0])
  
            # Do backward recurrence for layer i+1
            back_h_i, _ = theano.scan(fn=back_recurrence_igt1,
                                      sequences=[forward_h_im1[::-1], back_h_im1[::-1]],
                                      outputs_info=self.back_h1_tp1,
                                      non_sequences=i,
                                      n_steps=x.shape[0])
  
            forward_h_im1 = forward_h_i
            back_h_im1    = back_h_i[::-1]

        s, _ = theano.scan(fn=s_t, sequences=[forward_h_im1, back_h_im1])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
