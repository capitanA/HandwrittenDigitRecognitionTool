import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(logs.get('accuracy'))
        if (logs.get('accuracy') >= 0.992):
            print("\nReached 99.2% accuracy so cancelling training!")
            self.model.stop_training = True

