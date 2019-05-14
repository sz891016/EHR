from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, GRU, Dropout, Embedding
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep
from mimic3models.keras_utils import ExtendMask
from mimic3models.AttentionLayer import AttentionLayer
from keras import regularizers

class Network(Model):

    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=69, **kwargs):

        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth

        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")

        # Input layers and masking
        X = Input(shape=(48, input_dim), name='X')
        inputs = [X]
        mX = Masking()(X)
        

        # Configurations
        is_bidirectional = True

        # Main part of the network
        for i in range(depth - 1):
            num_units = dim
            if is_bidirectional:
                num_units = num_units // 2

            gru = GRU(units=num_units,
                        activation='tanh',
                        return_sequences=True,
                        recurrent_dropout=rec_dropout,
                        #kernel_regularizer=regularizers.l2(0.01),
                        dropout=dropout)

            if is_bidirectional:
                mX = Bidirectional(gru)(mX)
            else:
                mX = gru(mX)

        # Output module of the network
        return_sequences = (target_repl or deep_supervision)
        G = GRU(units=dim,
                 activation='tanh',
                 return_sequences=True,
                 dropout=dropout,
                 #kernel_regularizer=regularizers.l2(0.01),
                 recurrent_dropout=True)(mX)
                 
        if dropout > 0:
            G = Dropout(dropout)(G)
        A_L = AttentionLayer()(G)
        
       
        y = Dense(num_classes, activation=final_activation)(A_L)
        outputs = [y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}{}.dep{}".format('k_AttBigru',
                                           self.dim,
                                           ".bn" if self.batch_norm else "",
                                           ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                           ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                           self.depth)
	
