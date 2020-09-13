import os
from abc import ABC, abstractmethod
import tensorflow as tf


class Net(ABC):

    @abstractmethod
    def get_output(*args, **kwargs):
        pass

    @abstractmethod
    def load_param(*args, **kwargs):
        pass
    
    @abstractmethod
    def save_param(*args, **kwargs):
        pass


class BaseNet(Net):

    @abstractmethod
    def _get_output(self, *args, **kwargs):
        pass
    
    def __init__(self, scope):
        self.scope = scope
        self._template = tf.make_template(
            self.scope,
            self._get_output,
        )
        return
    
    def get_output(self, *args, **kwargs):
        return self._template(*args, **kwargs)
    
    def _get_saver(self):
        if not hasattr(self, '_saver'):
            self._saver = tf.train.Saver(list(filter(lambda a: 'Adam' not in a.name, tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=self.scope,
            ))), max_to_keep=None)
        return self._saver
    
    def load_param(self, sess, pretrain):
        if os.path.isdir(pretrain):
            pretrain = tf.train.latest_checkpoint(os.path.join(pretrain, self.scope))
        if pretrain:
            self._get_saver().restore(sess, pretrain)
        return
    
    def save_param(self, sess, save_dir, it):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, self.scope, 'log-%d' % it)
        self._get_saver().save(sess, save_path)
        return


class AggNet(Net):

    @abstractmethod
    def _get_output(self, *args, **kwargs):
        pass

    def __init__(self, sub_net_list):
        self.sub_net_list = sub_net_list
        return

    def get_output(self, *args, **kwargs):
        return self._get_output(*args, **kwargs)
    
    def load_param(self, sess, pretrain):
        for sub_net in self.sub_net_list:
            sub_net.load_param(sess, pretrain)
        return
    
    def save_param(self, sess, save_dir, it):
        for sub_net in self.sub_net_list:
            sub_net.save_param(sess, save_dir, it)
        return
    