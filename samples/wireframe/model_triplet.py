import numpy as np
import keras
import keras.backend as K
import keras.layers as KL
from mrcnn.triplet_loss import batch_all_triplet_loss
import tensorflow as tf
import samples.wireframe.database_actions as db_actions
from samples.wireframe.database_actions import get_known_encodings
import os
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../../")

class Model:
    def __init__(self):
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        base = KL.Input(shape=(1024,))
        x = KL.Dense(128)(base)
        x = KL.Activation('softmax')(x)
        full_model = keras.Model(inputs=base, outputs=x)
        return full_model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=self.triplet_loss,
                           metrics=['accuracy'])

    def triplet_loss(self, y_true, y_pred):
        # Reshape data
        y_true = tf.reshape(y_true, [-1])
        def_margin = tf.constant(0.3, dtype=tf.float32)

        # Run
        loss, _ = batch_all_triplet_loss(embeddings=y_pred, labels=y_true, margin=def_margin)
        return loss


#Save the losses
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.trainlosses = []
        self.vallosses = []

    def on_epoch_end(self, epoch, logs={}):
        self.trainlosses.append(logs.get('loss'))
        self.vallosses.append(logs.get('val_loss'))

"""
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8,
                                           verbose=0, mode='auto', baseline=None, restore_best_weights=True)

embeddings, labels = get_known_encodings('Database.db', 1024)
print(embeddings)

#Train the model
history = LossHistory()
model = Model()
model.model.fit(embeddings.T, labels, batch_size=50, epochs=1000, callbacks=[history, stop_early], validation_split=0.25
                , verbose=2)



#Plot the training loss
line1 = plt.plot(history.trainlosses, 'r--', label = "Training loss")
plt.plot(history.vallosses, 'b--', label = "Validation loss")
plt.legend()
plt.show()


#Save the weights
# model_path = os.path.join('/Users/BotezatuCristian/PycharmProjects/Mask_RCNN/', "weights_emb_labels.h5")
model.model.save_weights("Triplet_Res101_03.h5")
"""