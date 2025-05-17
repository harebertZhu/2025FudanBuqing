from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

EMBED_DIM = 32
LSTM_OUT = 64
max_length = 32  # 假设输入序列的最大长度是32
total_words = 10000  # 假设总词汇量是10000

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=EMBED_DIM, input_length=max_length))
model.add(LSTM(LSTM_OUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
