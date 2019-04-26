import numpy as np
import keras
import keras.backend as K
import keras.layers as KL
from mrcnn.triplet_loss import batch_all_triplet_loss
import tensorflow as tf
import samples.wireframe.database_actions as db_actions
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
        # Resh
        # ape data
        y_true = tf.reshape(y_true, [-1])
        def_margin = tf.constant(1.0, dtype=tf.float32)

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



def get_data(database):
    """
    :return: encodings array of (1024, n)
             labels list of (n)
    """
    query = "SELECT * FROM embeddings WHERE label IS NOT NULL"
    cursor, connection = db_actions.connect(database)
    cursor.execute(query)


    result_list = cursor.fetchall()
    encodings = np.zeros((1024, len(result_list)))
    labels = []

    for i in range(len(result_list)):
        encodings[:, i] = result_list[i][0]
        labels.append(result_list[i][1])
    encodings = np.nan_to_num(encodings)
    labels = [x for x in labels]
    return encodings.astype('float32'), labels


embeddings, labels = get_data(ROOT_DIR + '/samples/wireframe/Database_emb_labels.db')

"""
#Train the model
history = LossHistory()
model = Model()
model.model.fit(embeddings.T, labels, batch_size=10, epochs=100, callbacks=[history], validation_split = 0.25)

#Plot the training loss
line1 = plt.plot(history.trainlosses, 'r--', label = "Training loss")
plt.plot(history.vallosses, 'b--', label = "Validation loss")
plt.legend()
plt.show()

#Save the weights
model_path = os.path.join(ROOT_DIR, "weights_emb_labels.h5")
model.model.save_weights(model_path)
"""