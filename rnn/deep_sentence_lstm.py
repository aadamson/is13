import theano
import numpy as np
import os

from theano import tensor as T
from collections import OrderedDict

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

class model(object):
    
    def __init__(self, nh, nc, ne, depth, embeddings, lam=0.03):
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
        self.forward_v_in = theano.shared(name='forward_v_in', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_v_in    = theano.shared(name='back_v_in', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.forward_w_in = theano.shared(name='forward_w_in', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_w_in    = theano.shared(name='back_w_in', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))

        self.forward_v_forget = theano.shared(name='forward_v_forget', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_v_forget    = theano.shared(name='back_v_forget', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.forward_w_forget = theano.shared(name='forward_w_forget', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_w_forget    = theano.shared(name='back_w_forget', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))

        self.forward_v_out = theano.shared(name='forward_v_out', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_v_out    = theano.shared(name='back_v_out', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.forward_w_out = theano.shared(name='forward_w_out', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_w_out    = theano.shared(name='back_w_out', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))

        self.forward_v_cell  = theano.shared(name='forward_v_cell ', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_v_cell     = theano.shared(name='back_v_cell ', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.forward_w_cell  = theano.shared(name='forward_w_cell ', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))
        self.back_w_cell     = theano.shared(name='back_w_cell ', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                       (depth, nh, nh))
                                       .astype(theano.config.floatX))

        # First layer parameters
        self.forward_w1_in = theano.shared(name='forward_w1_in',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))
        self.back_w1_in    = theano.shared(name='back_w1_in',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))

        self.forward_w1_forget = theano.shared(name='forward_w1_forget',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))
        self.back_w1_forget    = theano.shared(name='back_w1_forget',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))

        self.forward_w1_out = theano.shared(name='forward_w1_out',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))
        self.back_w1_out    = theano.shared(name='back_w1_out',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))

        self.forward_w1_cell = theano.shared(name='forward_w1_cell',
                                        value=0.2 * np.random.uniform(-1.0, 1.0,
                                        (de, nh)))
        self.back_w1_cell    = theano.shared(name='back_w1_cell',
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



        self.forward_c1_0 = theano.shared(name='forward_c1_0',
                                          value=np.zeros(nh,
                                          dtype=theano.config.floatX))
        self.back_c1_tp1 = theano.shared(name='back_c1_tp1',
                                         value=np.zeros(nh,
                                         dtype=theano.config.floatX))
        self.forward_h1_0 = theano.shared(name='forward_h1_0',
                                          value=np.zeros(nh,
                                          dtype=theano.config.floatX))
        self.back_h1_tp1 = theano.shared(name='back_h1_tp1',
                                         value=np.zeros(nh,
                                         dtype=theano.config.floatX))


        # bundle
        self.params = [self.forward_w_in, self.back_w_in, 
                       self.forward_v_in, self.back_v_in,
                       self.forward_w_forget, self.back_w_forget, 
                       self.forward_v_forget, self.back_v_forget,
                       self.forward_w_out, self.back_w_out, 
                       self.forward_v_out, self.back_v_out,
                       self.forward_w_cell, self.back_w_cell, 
                       self.forward_v_cell, self.back_v_cell, 
                       self.c, self.forward_U, self.back_U,
                       self.forward_w1_in, self.back_w1_in,
                       self.forward_w1_forget, self.back_w1_forget,
                       self.forward_w1_out, self.back_w1_out,
                       self.forward_w1_cell, self.back_w1_cell,
                       self.forward_h1_0, self.back_h1_tp1,
                       self.forward_c1_0, self.back_c1_tp1]

        self.regularized_params = [self.forward_w_in, self.back_w_in, 
                                   self.forward_v_in, self.back_v_in,
                                   self.forward_w_forget, self.back_w_forget, 
                                   self.forward_v_forget, self.back_v_forget,
                                   self.forward_w_out, self.back_w_out, 
                                   self.forward_v_out, self.back_v_out,
                                   self.forward_w_cell, self.back_w_cell, 
                                   self.forward_v_cell, self.back_v_cell,
                                   self.forward_w1_in, self.back_w1_in,
                                   self.forward_w1_forget, self.back_w1_forget,
                                   self.forward_w1_out, self.back_w1_out,
                                   self.forward_w1_cell, self.back_w1_cell]

        self.names  = ['forward_w_in', 'back_w_in', 
                       'forward_v_in', 'back_v_in',
                       'forward_w_forget', 'back_w_forget', 
                       'forward_v_forget', 'back_v_forget',
                       'forward_w_out', 'back_w_out', 
                       'forward_v_out', 'back_v_out',
                       'forward_w_cell', 'back_w_cell', 
                       'forward_v_cell', 'back_v_cell', 
                       'c', 'forward_U', 'back_U',
                       'forward_w1_in', 'back_w1_in',
                       'forward_w1_forget', 'back_w1_forget',
                       'forward_w1_out', 'back_w1_out',
                       'forward_w1_cell', 'back_w1_cell',
                       'forward_h1_0', 'back_h1_tp1',
                       'forward_c1_0', 'back_c1_tp1']
        
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

        def forward_recurrence_igt1(forward_h_t_im1, back_h_t_im1, forward_c_tm1_i, forward_h_tm1_i, i):
            input_gate = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.forward_w_in[i])
                                        + T.dot(back_h_t_im1, self.forward_w_in[i])
                                        + T.dot(forward_h_tm1_i, self.forward_v_in[i]))

            forget_gate = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.forward_w_forget[i])
                                         + T.dot(back_h_t_im1, self.forward_w_forget[i])
                                         + T.dot(forward_h_tm1_i, self.forward_v_forget[i]))

            output_gate = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.forward_w_out[i])
                                         + T.dot(back_h_t_im1, self.forward_w_out[i])
                                         + T.dot(forward_h_tm1_i, self.forward_v_out[i]))

            c_tilde = T.tanh(T.dot(forward_h_t_im1, self.forward_w_cell[i])
                             + T.dot(back_h_t_im1, self.forward_w_cell[i])
                             + T.dot(forward_h_tm1_i, self.forward_v_cell[i]))

            forward_c_t_i = forget_gate * forward_c_tm1_i + input_gate * c_tilde

            forward_h_t_i = output_gate * T.tanh(forward_c_t_i)

            return forward_c_t_i, forward_h_t_i

        def back_recurrence_igt1(forward_h_t_im1, back_h_t_im1, back_c_tp1_i, back_h_tp1_i, i):
            input_gate = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.back_w_in[i])
                                        + T.dot(back_h_t_im1, self.back_w_in[i])
                                        + T.dot(back_h_tp1_i, self.back_v_in[i]))

            forget_gate = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.back_w_forget[i])
                                         + T.dot(back_h_t_im1, self.back_w_forget[i])
                                         + T.dot(back_h_tp1_i, self.back_v_forget[i]))

            output_gate = T.nnet.sigmoid(T.dot(forward_h_t_im1, self.back_w_out[i])
                                         + T.dot(back_h_t_im1, self.back_w_out[i])
                                         + T.dot(back_h_tp1_i, self.back_v_out[i]))

            c_tilde = T.tanh(T.dot(forward_h_t_im1, self.back_w_cell[i])
                             + T.dot(back_h_t_im1, self.back_w_cell[i])
                             + T.dot(back_h_tp1_i, self.back_v_cell[i]))

            back_c_t_i = forget_gate * back_c_tp1_i + input_gate * c_tilde

            back_h_t_i = output_gate * T.tanh(back_c_t_i)

            return back_c_t_i, back_h_t_i

        def forward_recurrence_ieq1(x_t, forward_c_tm1_i, forward_h_tm1_i):
            input_gate = T.nnet.sigmoid(T.dot(x_t, self.forward_w1_in)
                                        + T.dot(forward_h_tm1_i, self.forward_v_in[0]))

            forget_gate = T.nnet.sigmoid(T.dot(x_t, self.forward_w1_forget)
                                         + T.dot(forward_h_tm1_i, self.forward_v_forget[0]))

            output_gate = T.nnet.sigmoid(T.dot(x_t, self.forward_w1_out)
                                         + T.dot(forward_h_tm1_i, self.forward_v_out[0]))

            c_tilde = T.tanh(T.dot(x_t, self.forward_w1_cell)
                             + T.dot(forward_h_tm1_i, self.forward_v_cell[0]))

            forward_c_t_i = forget_gate * forward_c_tm1_i + input_gate * c_tilde

            forward_h_t_i = output_gate * T.tanh(forward_c_t_i)

            return forward_c_t_i, forward_h_t_i

        def back_recurrence_ieq1(x_t, back_c_tp1_i, back_h_tp1_i):
            input_gate = T.nnet.sigmoid(T.dot(x_t, self.back_w1_in)
                                        + T.dot(back_h_tp1_i, self.back_v_in[0]))

            forget_gate = T.nnet.sigmoid(T.dot(x_t, self.back_w1_forget)
                                         + T.dot(back_h_tp1_i, self.back_v_forget[0]))

            output_gate = T.nnet.sigmoid(T.dot(x_t, self.back_w1_out)
                                         + T.dot(back_h_tp1_i, self.back_v_out[0]))

            c_tilde = T.tanh(T.dot(x_t, self.back_w1_cell)
                             + T.dot(back_h_tp1_i, self.back_v_cell[0]))

            back_c_t_i = forget_gate * back_c_tp1_i + input_gate * c_tilde

            back_h_t_i = output_gate * T.tanh(back_c_t_i)

            return back_c_t_i, back_h_t_i

        # Construct forward hidden nodes for first layer
        [forward_c_1, forward_h_1], _ = theano.scan(fn=forward_recurrence_ieq1,
                                     sequences=x,
                                     outputs_info=[self.forward_c1_0, self.forward_h1_0],
                                     n_steps=x.shape[0])
        [back_c_1, back_h_1], _    = theano.scan(fn=back_recurrence_ieq1,
                                     sequences=x[::-1],
                                     outputs_info=[self.back_c1_tp1, self.back_h1_tp1],
                                     n_steps=x.shape[0])

        forward_h_im1 = forward_h_1
        back_h_im1    = back_h_1[::-1]

        for i in range(1, depth):
            # Do forward recurrence for layer i+1
            [forward_c_i, forward_h_i], _ = theano.scan(fn=forward_recurrence_igt1,
                                         sequences=[forward_h_im1, back_h_im1],
                                         outputs_info=[self.forward_c1_0, self.forward_h1_0],
                                         non_sequences=i,
                                         n_steps=x.shape[0])
  
            # Do backward recurrence for layer i+1
            [back_c_i, back_h_i], _ = theano.scan(fn=back_recurrence_igt1,
                                      sequences=[forward_h_im1[::-1], back_h_im1[::-1]],
                                      outputs_info=[self.back_c1_tp1, self.back_h1_tp1],
                                      non_sequences=i,
                                      n_steps=x.shape[0])
  
            forward_h_im1 = forward_h_i
            back_h_im1    = back_h_i[::-1]

        s, _ = theano.scan(fn=s_t, sequences=[forward_h_im1, back_h_im1])

        # Probability and prediction
        p_y_given_x = s[:,0,:]
        y_pred = T.argmax(p_y_given_x, axis=1)

        alpha = T.scalar('alpha')
        log_likelihood = T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y])

        l2_norm = sum([(param ** 2).sum() for param in self.regularized_params])
        
        cost = -log_likelihood + self.lam*l2_norm
        gradients = T.grad(cost, self.params)
        updates = OrderedDict((param, param - alpha*gradient) for param, gradient in zip(self.params, gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function(inputs  = [idxs, y, alpha],
                                     outputs = [cost, s],
                                     updates = updates)

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())
