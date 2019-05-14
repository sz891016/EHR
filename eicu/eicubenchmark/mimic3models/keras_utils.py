from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from mimic3models import metrics

import keras
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

from keras.layers import Layer


# ===================== METRICS ===================== #

class InHospitalMortalityMetrics(keras.callbacks.Callback):
    def __init__(self, train_data, val_data, target_repl, batch_size=32, early_stopping=True, verbose=2):
        super(InHospitalMortalityMetrics, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.target_repl = target_repl
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data, history, dataset, logs):
        y_true = []
        predictions = []
        B = self.batch_size
        for i in range(0, len(data[0]), B):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, len(data[0])), end='\r')
            if self.target_repl:
                (x, y, y_repl) = (data[0][i:i + B], data[1][0][i:i + B], data[1][1][i:i + B])
            else:
                (x, y) = (data[0][i:i + B], data[1][i:i + B])
            outputs = self.model.predict(x, batch_size=B)
            if self.target_repl:
                predictions += list(np.array(outputs[0]).flatten())
            else:
                predictions += list(np.array(outputs).flatten())
            y_true += list(np.array(y).flatten())
        print('\n')
        predictions = np.array(predictions)
        predictions = np.stack([1 - predictions, predictions], axis=1)
        ret = metrics.print_metrics_binary(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data, self.val_history, 'val', logs)

        if self.early_stopping:
            max_auc = np.max([x["auroc"] for x in self.val_history])
            cur_auc = self.val_history[-1]["auroc"]
            if max_auc > 0.85 and cur_auc < 0.83:
                self.model.stop_training = True


class LengthOfStayMetrics(keras.callbacks.Callback):
    def __init__(self, train_data_gen, val_data_gen, partition, batch_size=32,
                 early_stopping=True, verbose=2):
        super(LengthOfStayMetrics, self).__init__()
        self.train_data_gen = train_data_gen
        self.val_data_gen = val_data_gen
        self.batch_size = batch_size
        self.partition = partition
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_gen, history, dataset, logs):
        y_true = []
        predictions = []
        for i in range(data_gen.steps):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, data_gen.steps), end='\r')
            (x, y_processed, y) = data_gen.next(return_y_true=True)
            pred = self.model.predict(x, batch_size=self.batch_size)
            if isinstance(x, list) and len(x) == 2:  # deep supervision
                if pred.shape[-1] == 1:  # regression
                    pred_flatten = pred.flatten()
                else:  # classification
                    pred_flatten = pred.reshape((-1, 10))
                for m, t, p in zip(x[1].flatten(), y.flatten(), pred_flatten):
                    if np.equal(m, 1):
                        y_true.append(t)
                        predictions.append(p)
            else:
                if pred.shape[-1] == 1:
                    y_true += list(y.flatten())
                    predictions += list(pred.flatten())
                else:
                    y_true += list(y)
                    predictions += list(pred)
        print('\n')
        if self.partition == 'log':
            predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
            ret = metrics.print_metrics_log_bins(y_true, predictions)
        if self.partition == 'custom':
            predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
            ret = metrics.print_metrics_custom_bins(y_true, predictions)
        if self.partition == 'none':
            ret = metrics.print_metrics_regression(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_gen, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_gen, self.val_history, 'val', logs)

        if self.early_stopping:
            max_kappa = np.max([x["kappa"] for x in self.val_history])
            cur_kappa = self.val_history[-1]["kappa"]
            max_train_kappa = np.max([x["kappa"] for x in self.train_history])
            if max_kappa > 0.38 and cur_kappa < 0.35 and max_train_kappa > 0.47:
                self.model.stop_training = True


# ===================== LAYERS ===================== #


def softmax(x, axis, mask=None):
    if mask is None:
        mask = K.constant(True)
    mask = K.cast(mask, K.floatx())
    if K.ndim(x) is K.ndim(mask) + 1:
        mask = K.expand_dims(mask)

    m = K.max(x, axis=axis, keepdims=True)
    e = K.exp(x - m) * mask
    s = K.sum(e, axis=axis, keepdims=True)
    s += K.cast(K.cast(s < K.epsilon(), K.floatx()) * K.epsilon(), K.floatx())
    return e / s


def _collect_attention(x, a, mask):
    """
    x is (B, T, D)
    a is (B, T, 1) or (B, T)
    mask is (B, T)
    """
    if K.ndim(a) == 2:
        a = K.expand_dims(a)
    a = softmax(a, axis=1, mask=mask)  # (B, T, 1)
    return K.sum(x * a, axis=1)  # (B, D)


class CollectAttetion(Layer):
    """ Collect attention on 3D tensor with softmax and summation
        Masking is disabled after this layer
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(CollectAttetion, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        x = inputs[0]
        a = inputs[1]
        # mask has 2 components, both are the same
        return _collect_attention(x, a, mask[0])

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][2]

    def compute_mask(self, input, input_mask=None):
        return None


class Slice(Layer):
    """ Slice 3D tensor by taking x[:, :, indices]
    """

    def __init__(self, indices, **kwargs):
        self.supports_masking = True
        self.indices = indices
        super(Slice, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if K.backend() == 'tensorflow':
            xt = tf.transpose(x, perm=(2, 0, 1))
            gt = tf.gather(xt, self.indices)
            return tf.transpose(gt, perm=(1, 2, 0))
        return x[:, :, self.indices]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.indices))

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def get_config(self):
        return {'indices': self.indices}


class GetTimestep(Layer):
    """ Takes 3D tensor and returns x[:, pos, :]
    """

    def __init__(self, pos=-1, **kwargs):
        self.pos = pos
        self.supports_masking = True
        super(GetTimestep, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, self.pos, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        return {'pos': self.pos}


LastTimestep = GetTimestep


class ExtendMask(Layer):
    """ Inputs:      [X, M]
        Output:      X
        Output_mask: M
    """

    def __init__(self, add_epsilon=False, **kwargs):
        self.supports_masking = True
        self.add_epsilon = add_epsilon
        super(ExtendMask, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, input, input_mask=None):
        if self.add_epsilon:
            return input[1] + K.epsilon()
        return input[1]

    def get_config(self):
        return {'add_epsilon': self.add_epsilon}
