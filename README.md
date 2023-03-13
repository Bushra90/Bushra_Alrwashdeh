# Bushra_Alrwashdeh
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset
dataset = tfds.load('wikipedia/20190301.en', split='train', shuffle_files=True)

# Preprocess the data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
texts = [example['text'].numpy().decode('utf-8') for example in dataset]
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Extract the target data (labels)
labels = [example['label'].numpy() for example in dataset]

# Split the data into training and validation sets
batch_size = 32
train_size = int(0.8 * sample_size)
train_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[:train_size], labels[:train_size]))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[train_size:], labels[train_size:]))
val_dataset = val_dataset.batch(batch_size)

# Define and train a basic NLP model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Perform sensitivity analysis by varying hyperparameters and other modeling choices
# Vary the number of LSTM layers
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Vary the batch size
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[:train_size], labels[:train_size]))
train_dataset = train_dataset.batch(batch_size)
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Vary the optimizer and learning rate
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Vary the preprocessing technique
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
texts = [example['text'].numpy().decode('utf-8') for example in dataset]
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
train_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[:train_size], labels[:train_size]))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

train_size = int(0.8 * sample_size)
train_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[:train_size], labels[:train_size]))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[train_size:], labels[train_size:]))
val_dataset = val_dataset.batch(batch_size)

model = tf.keras.Sequential([
tf.keras.layers.Embedding(10000, 32),
tf.keras.layers.LSTM(64),
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

for epoch in range(10):
print('Epoch', epoch+1, ': loss =', history.history['loss'][epoch], ', accuracy =', history.history['accuracy'][epoch],', val_loss =', history.history['val_loss'][epoch], ', val_accuracy =', history.history['val_accuracy'][epoch]))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

train_size = int(0.8 * sample_size)
train_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[:train_size], labels[:train_size]))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences[train_size:], labels[train_size:]))
val_dataset = val_dataset.batch(batch_size)

model = tf.keras.Sequential([
tf.keras.layers.Embedding(20000, 32),
tf.keras.layers.LSTM(64),
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

for epoch in range(10):
print('Epoch', epoch+1, ': loss =', history.history['loss'][epoch], ', accuracy =', history.history['accuracy'][epoch],', val_loss =', history.history['val_loss'][epoch], ', val_accuracy =', history.history['val_accuracy'][epoch]))




#Robustness The model
# Load the dataset
dataset = tfds.load('wikipedia/20190301.en', split='train', shuffle_files=True)

# Preprocess the data
df = pd.DataFrame(tfds.as_dataframe(dataset)['text'].apply(lambda x: x.decode('utf-8')), columns=['text'])
df['text'] = df['text'].str.replace('\n', ' ').str.replace('\t', ' ').str.replace('  ', ' ').str.strip()
df['label'] = [example['label'].numpy() for example in dataset]
df['label'] = df['label'].astype(int)

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TFAutoModel.from_pretrained('bert-base-uncased')

# Tokenize the text
train_encodings = tokenizer(list(train_df['text'].values), truncation=True, padding=True)
val_encodings = tokenizer(list(val_df['text'].values), truncation=True, padding=True)

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_df['label'].values
)).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_df['label'].values
)).batch(32)

# Define and train a basic NLP model
input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

output = model({'input_ids': input_ids, 'attention_mask': attention_mask})[1]
output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)

# Perform error analysis
y_val = val_df['label'].values
y_pred = model.predict(val_dataset).squeeze()

false_positives = val_df[(y_val == 0) & (y_pred >= 0.5)]
false_negatives = val_df[(y_val == 1) & (y_pred < 0.5)]

print(f"Number of false positives: {len(false_positives)}")
print(f"Number of false negatives: {len(false_negatives)}")

from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(val_dataset).squeeze()
y_pred = np.round(y_pred)

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

cr = classification_report(y_val, y_pred)
print("Classification Report:")
print(cr)
