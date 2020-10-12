import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train, test   = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

print(f'Vocabulary size: {encoder.vocab_size}')

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train = train.shuffle(BUFFER_SIZE)
train = train.padded_batch(BATCH_SIZE)

test = test.padded_batch(BATCH_SIZE)

def make_model():
    model = keras.Sequential([
        keras.layers.Embedding(encoder.vocab_size, 64),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])

    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    return model

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)

    return vec

def sample_predict(sample_text, pad):
    encoded = encoder.encode(sample_text)

    if pad:
        sample_text = pad_to_size(encoded, 64)
    
    encoded = tf.cast(encoded, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded, 0))

    return (predictions)

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

model = make_model()
model.load_weights(checkpoint_path)

if 'NEED_TRAINING' in os.environ and os.environ['NEED_TRAINING']:
    history  = model.fit(train, epochs=10, validation_data=train, validation_steps=30, callbacks=[checkpoint])
    loss, acc = model.evaluate(test)

    print(f'Test loss: {loss}')
    print(f'Test accuracy: {acc}')

while True:
    sample = input('> ')

    if len(sample) > 0:
        predictions = sample_predict(sample, pad=False)

        if predictions[0] < 0:
            print("BAD: ", predictions[0])
        else:
            print("GOOD: ", predictions[0])