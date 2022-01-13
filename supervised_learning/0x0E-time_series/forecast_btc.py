#!/usr/bin/env python3
""" a script that creates, trains,
 and validates a keras model for the forecasting of BTC """
import tensorflow as tf
from preprocess_data import window
import matplotlib.pyplot as plt

model = tf.keras.models.Sequential(
    [tf.keras.layers.LSTM(32, return_sequences=False),
     tf.keras.layers.Dense(units=1)])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=2,
                                                  mode='min')
model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit(window.train, epochs=20,
                    validation_data=window.val,
                    callbacks=[early_stopping])

val_performance = model.evaluate(window.val)
performance = model.evaluate(window.test, verbose=0)
model.save('BTC_lstm.h5')

""" plotting accuracy metrics """

plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
