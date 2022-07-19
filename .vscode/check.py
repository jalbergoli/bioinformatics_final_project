import pandas as pd
import numpy as np
# Modelo Recurrentes con Embeddings a nivel de caracter
df = pd.read_csv('data/acetylcholinesterase_02_bioactivity_data_preprocessed.csv')

# Canonical_smiles To secuence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len_idx = df['canonical_smiles'].apply(len).argmax()
min_len_idx = df['canonical_smiles'].apply(len).argmin()
X = df['canonical_smiles']
y = df['pIC50']
df.head(1)
df['canonical_len'] = df['canonical_smiles'].apply(lambda x: len(x))
max_sequence_len = df['canonical_len'].max()
# Implementar tokenización y guardar en X_seq_pad el dataset tokenizado
tokenizer = Tokenizer(
    num_words = None,
    filters='',
    lower=False,
    split=' ',
    char_level=True,
    oov_token=None)

tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

X_seq_pad = pad_sequences(X_seq, maxlen=max_sequence_len)
len(tokenizer.word_index)


# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_seq_pad, y, test_size=0.2, random_state=42)


len(X_train), len(y_train), len(X_test)


# Network Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K


# Métrica
def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# vocab_size = # Completar largo del vocabulario
vocab_size = len(tokenizer.word_index)
vocab_size
max_sequence_len = df['canonical_len'].max()
max_sequence_len

embed_dim = 32
nb_words = vocab_size

model = Sequential(name='LSTM_1')
model.add(Embedding(nb_words + 1 , embed_dim, input_length=max_sequence_len, trainable=True))
model.add(Bidirectional(LSTM(100, activation='tanh')))
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dense(1))

# Implementar modelo completo
model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=[R2])
mcp = ModelCheckpoint('models/best_model_{epoch}', save_best_only=True, save_format="h5")
es = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True)
pd.set_option('display.max_columns', 50)
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), callbacks=[])
y_pred = model.predict(X_test)