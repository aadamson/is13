import theano
import numpy as np
import os

from theano import tensor as T
from collections import OrderedDict

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

class model(object):
    
    def __init__(self, nh, nc, ne, depth, embeddings, lam=0.03, adagrad=False):
        f = relu
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        depth :: height of network
        lam :: lambda value used for L2 regularization
        '''
        # Hyperparameters
        self.lam = theano.shared(name='lambda',
                                 value=lam)
        de = embeddings.shape[1]
        # parameters of the model

        self.emb = theano.shared(name='embeddings',
                                 value=embeddings.astype(theano.config.floatX))

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
                                        (de, nh)))
        self.back_w1    = theano.shared(name='back_w1',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))

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
        self.back_h1_tp1  = theano.shared(name='back_h1_tp1',
                                          value=np.zeros(nh,
                                          dtype=theano.config.floatX))


        # bundle
        self.params = [self.forward_w, self.back_w, 
                       self.forward_v, self.back_v, 
                       self.forward_b, self.back_b, 
                       self.c, self.forward_U, self.back_U,
                       self.forward_w1, self.back_w1,
                       self.forward_h1_0, self.back_h1_tp1]
        if adagrad:
          grad_cache = {}
          for param in self.params:
              grad_cache[param] = theano.shared(name=param.name+'_grad_cache',
                                                value=np.zeros(param.get_value().shape, dtype=theano.config.floatX))

        self.regularized_params = [self.forward_w, self.back_w, 
                                   self.forward_v, self.back_v,
                                   self.forward_w1, self.back_w1]

        self.names  = ['forward_w', 'back_w', 
                       'forward_v', 'back_v', 
                       'forward_b', 'back_b', 
                       'c', 'forward_U', 'back_U',
                       'forward_w1', 'back_w1',
                       'forward_h1_0', 'back_h1_tp1']
        
        # Inputs               
        idxs = T.imatrix()
        x    = self.emb[idxs].reshape((idxs.shape[0], embeddings.shape[1]))
        y    = T.ivector('y') # labels

        # Notation taken from Irsoy and Cardie
        def s_t(forward_h_t_i, back_h_t_i):
            s = T.nnet.softmax(T.dot(forward_h_t_i, self.forward_U)
                               + T.dot(back_h_t_i, self.back_U) 
                               + self.c)

            return s

        def forward_recurrence_igt1(forward_h_t_im1, back_h_t_im1, forward_h_tm1_i, i):
            forward_h_t_i = f(T.dot(forward_h_t_im1, self.forward_w[i])
                              + T.dot(back_h_t_im1, self.forward_w[i])
                              + T.dot(forward_h_tm1_i, self.forward_v[i]) 
                              + self.forward_b[i])
            return forward_h_t_i

        def back_recurrence_igt1(forward_h_t_im1, back_h_t_im1, back_h_tp1_i, i):
            back_h_t_i    = f(T.dot(forward_h_t_im1, self.back_w[i])
                              + T.dot(back_h_t_im1, self.back_w[i])
                              + T.dot(back_h_tp1_i, self.back_v[i]) 
                              + self.back_b[i])
            return back_h_t_i

        def forward_recurrence_ieq1(x_t, forward_h_tm1_i):
            forward_h_t_i = f(T.dot(x_t, self.forward_w1)
                              + T.dot(forward_h_tm1_i, self.forward_v[0])
                              + self.forward_b[0])

            return forward_h_t_i

        def back_recurrence_ieq1(x_t, back_h_tp1_i):
            back_h_t_i    = f(T.dot(x_t, self.back_w1)
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

        # Probability and prediction
        p_y_given_x = s[:,0,:]
        y_pred = T.argmax(p_y_given_x, axis=1)

        alpha          = T.scalar('alpha')
        log_likelihood = T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y])

        l2_norm   = sum([(param ** 2).sum() for param in self.regularized_params])
        cost      = -log_likelihood + self.lam*l2_norm
        gradients = T.grad(theano.gradient.grad_clip(cost, -1, 1), self.params)

        updates = []
        if adagrad:
            # Use 1e-6 as fudge factor for numerical stability
            updates += [(grad_cache[param], grad_cache[param] + gradient**2) for param, gradient in zip(self.params, gradients)]
            updates += [(param, param - alpha/T.sqrt(1e-6 + grad_cache[param])*gradient) for param, gradient in zip(self.params, gradients)]
        else:
            updates += [(param, param - alpha*gradient) for param, gradient in zip(self.params, gradients)]
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function(inputs  = [idxs, y, alpha],
                                     outputs = [cost, s],
                                     updates = OrderedDict(updates))

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())
