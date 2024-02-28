#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

tqdm.pandas()
sns.set_style('dark')
plt.rcParams['figure.figsize'] = (20,8)
plt.rcParams['font.size'] = 14


# In[2]:


pip install wordcloud


# In[16]:


import os


# In[17]:


os.chdir("C:\\Users\\Shreyash pc\\Downloads")
data = pd.read_csv("preprocessed_dataset.csv")
data.head()


# In[18]:


def text_cleaning(x):

    text = re.sub('\s+\n+', ' ', x)
    text = re.sub('[^a-zA-Z0-9\.]', ' ', text)
    text = text.split()

    text = [word for word in text]
    text = ' '.join(text)
    text = 'startseq '+text+' endseq'

    return text


# In[19]:


data['full_text'] = data['full_text'].progress_apply(text_cleaning)


# In[20]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


# In[21]:


train = data.iloc[:43000, :]
val = data.iloc[43000:49000, :].reset_index(drop=True)
test = data.iloc[52000:, :].reset_index(drop=True)


# In[22]:


tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(train['full_text'].tolist())
max_length = max(len(caption.split()) for caption in train['full_text'].tolist())


# In[23]:


df_vocab = pd.DataFrame(list(tokenizer.word_counts.items()), columns=['word','count'])
df_vocab.sort_values(by='count', ascending=False, inplace=True, ignore_index=True)
df_vocab.head()


# In[24]:


words = ""
words += " ".join(df_vocab['word'].tolist())+" "
wordcloud = WordCloud(width = 1200, height = 400,
                background_color ='black',
                min_font_size = 10).generate(words)

plt.imshow(wordcloud)
plt.title('Word Cloud of Vocbulary')
plt.show()


# In[26]:


df_vocab.describe()


# In[27]:


df_vocab[df_vocab['count']>=0]


# In[28]:


vocab_size = len(df_vocab[df_vocab['count']>=0])
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train['full_text'].tolist())


# In[29]:


data['length'] = data['full_text'].progress_apply(lambda x: len(x.split(' ')))


# In[30]:


sns.boxplot(x='length', data=data)
plt.title('IQR Analysis of Sentence Lengths')
plt.show()


# In[31]:


data.describe()


# In[32]:


max_length = 80
print(train.loc[0, 'full_text'])
print(tokenizer.texts_to_sequences([train.loc[0, 'full_text']])[0])


# In[33]:


seq = train.loc[0, 'full_text'].split()
X, y = [], []
for i in range(1,len(seq)):
    in_seq, out_seq = seq[:i], seq[i]
    X.append(' '.join(in_seq))
    y.append(out_seq)

example = pd.DataFrame(columns=['input','output'])
example['input'] = X
example['output'] = y
example


# In[34]:


class CustomDataGenerator(Sequence):

    def __init__(self, df, X_col, batch_size, tokenizer, vocab_size, max_length, shuffle=True):

        self.df = df.copy()
        self.X_col = X_col
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self,index):

        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
        X, y = self.__get_data(batch)
        return X, y

    def __get_data(self,batch):

        X, y = list(), list()
        captions = batch.loc[:, self.X_col].tolist()
        for caption in captions:
            seq = self.tokenizer.texts_to_sequences([caption])[0]
            max_len = self.max_length if len(seq) > self.max_length else len(seq)
            for i in range(1,max_len):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                X.append(in_seq)
                y.append(out_seq)

        X, y = np.array(X), np.array(y)

        return X, y


# In[35]:


train_gen = CustomDataGenerator(train, 'full_text', 16, tokenizer, vocab_size, max_length)
val_gen = CustomDataGenerator(val, 'full_text', 16, tokenizer, vocab_size, max_length)
test_gen = CustomDataGenerator(test, 'full_text', 16, tokenizer, vocab_size, max_length)


# In[36]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Embedding, LSTM, add, Concatenate, Reshape,
                                     concatenate, Bidirectional, Dense, Input)


# In[37]:


input_layer = Input(shape=(50,))
x = Embedding(vocab_size, 64)(input_layer)
x = Bidirectional(LSTM(100))(x)
output_layer = Dense(vocab_size, activation='softmax')(x)

model = Model(inputs=[input_layer], outputs=output_layer)
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam())


# In[38]:


model.summary()


# In[6]:


#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


# In[8]:


model_name = "model.h5"
checkpoint = ModelCheckpoint(model_name,
                            monitor="val_loss",
                            mode="min",
                            save_best_only = True,
                            verbose=1)

es = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * tf.math.exp(-0.1*epoch)


lr_scheduler = LearningRateScheduler(scheduler, verbose=1)


# In[27]:


#history = model.fit(train_gen, validation_data=val_gen, epochs=4, callbacks=[checkpoint, es, lr_scheduler])


# In[28]:


plt.figure(figsize=(20,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[39]:


def idx_to_word(integer,tokenizer):

    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None


# In[40]:


def predict_sentence(text, model, tokenizer, max_length):

    in_text = "startseq " + text
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict(sequence, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text+= " " + word

        if word == 'endseq':
            break

    return in_text


# In[3]:


def beam_search_predictions(text, beam_index = 3):
    in_text = "startseq " + text
    start = tokenizer.texts_to_sequences([in_text])[0]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            preds = model.predict(par_caps, verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [idx_to_word(i, tokenizer) for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


# In[32]:


#ngram
#import numpy as np
#import random

#def beam_search_predictions(text, beam_index=3, temperature=1.0, n_gram=3, max_length=50):
    # Initialize the input sequence
 #   in_text = "startseq " + text
  #  start = tokenizer.texts_to_sequences([in_text])[0]
   # start_word = [[start, 0.0]]

   # while len(start_word[0][0]) < max_length:
    #    temp = []
     #   for s in start_word:
      #      par_caps = pad_sequences([s[0]], maxlen=max_length)
       #     preds = model.predict(par_caps, verbose=0)
        #    preds = np.log(preds) / temperature
         #   exp_preds = np.exp(preds)
          #  preds = exp_preds / np.sum(exp_preds)

            # Generate multiple candidates using random sampling and n-grams
           # candidate_sequences = []
            #for _ in range(beam_index * n_gram):
             #   word_preds = np.argsort(preds[0])[-beam_index:]
              #  for w in word_preds:
               #     next_cap, prob = s[0][:], s[1]
                #    next_cap.append(w)
                 #   prob += preds[0][w]
                  #  candidate_sequences.append([next_cap, prob])

            # Randomly shuffle the candidates and select the top beam_index sequences
            #random.shuffle(candidate_sequences)
            #candidate_sequences = sorted(candidate_sequences, key=lambda l: l[1], reverse=True)[:beam_index]

            #temp.extend(candidate_sequences)

        # Sort and keep only the top beam_index sequences
        #start_word = temp
        #start_word = sorted(start_word, key=lambda l: l[1], reverse=True)[:beam_index]

    # Choose the best sequence
    #start_word = start_word[0][0]
    #intermediate_caption = [idx_to_word(i, tokenizer) for i in start_word]
    #final_caption = []

    #for i in intermediate_caption:
     #   if i != 'endseq':
      #      final_caption.append(i)
       # else:
        #    break

    #final_caption = ' '.join(final_caption[1:])
    #return final_caption


# In[41]:


sentences = ["she liked ice cream"]

for sentence in sentences:
    print("Greedy Search: ", predict_sentence(sentence, model, tokenizer, 50))
    print("Beam Search: ", beam_search_predictions(sentence))
    print("\n")


# In[ ]:




