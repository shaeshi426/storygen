#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def main():
    from tensorflow.keras.models import load_model

    # Load the trained model
    model = load_model("C:\\Users\\Shreyash pc\\Downloads\\model.h5")
    
    import pandas as pd
    import numpy as np
    import re
    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    tqdm.pandas()
    
    import os
    os.chdir("C:\\Users\\Shreyash pc\\Downloads")
    data = pd.read_csv("preprocessed_dataset.csv")
    #data.head()
    
    def text_cleaning(x):

        text = re.sub('\s+\n+', ' ', x)
        text = re.sub('[^a-zA-Z0-9\.]', ' ', text)
        text = text.split()

        text = [word for word in text]
        text = ' '.join(text)
        text = 'startseq '+text+' endseq'

        return text
    
    data['full_text'] = data['full_text'].progress_apply(text_cleaning)
    
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.utils import to_categorical
    
    train = data.iloc[:43000, :]
    val = data.iloc[43000:49000, :].reset_index(drop=True)
    test = data.iloc[52000:, :].reset_index(drop=True)
    
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(train['full_text'].tolist())
    max_length = max(len(caption.split()) for caption in train['full_text'].tolist())  #max length
    
    df_vocab = pd.DataFrame(list(tokenizer.word_counts.items()), columns=['word','count'])
    df_vocab.sort_values(by='count', ascending=False, inplace=True, ignore_index=True)
    df_vocab.head()

    max_length = 80
    
    seq = train.loc[0, 'full_text'].split()
    X, y = [], []
    for i in range(1,len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        X.append(' '.join(in_seq))
        y.append(out_seq)

    example = pd.DataFrame(columns=['input','output'])
    example['input'] = X
    example['output'] = y
    #example
    
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
        
    def idx_to_word(integer,tokenizer):

        for word, index in tokenizer.word_index.items():
            if index==integer:
                return word
        return None
    
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
    
    #tvo = input()

    print("Greedy Search: ", predict_sentence(tvo, model, tokenizer, 50))
    print("Beam Search: ", beam_search_predictions(tvo))
    
    ovd = beam_search_predictions(tvo)
    return ovd
                                  
def main1():
    ovd_value = main()
    import pandas as pd
    import numpy as np
    import os
    import time
    time.sleep(2)
    
    os.chdir("C:\\Users\\Shreyash pc\\Downloads")
    
    import torch
    from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
    
    model_name = "facebook/bart-large-cnn"
    
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = torch.load("abcd")
    
    def generate_story(judgement_text):
        inputs = tokenizer.encode("summarize: " + judgement_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=600, min_length=400, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    
    generated_story = generate_story(ovd_value)
    
    import re
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_story)

    # Take only the first four sentences
    dvt = ' '.join(sentences[:4])
    return dvt
        
#if __name__ == "__main__":
 #   tvo = input()
  #  a = main1()

def on_submit():
    processing_label.config(text="Processing......")
    window.update()  # Force update to show the label immediately

    tvo = entry.get()
    a = main1()

    output_text.config(state=tk.NORMAL, fg="white")  # Enable editing, set text color to white
    output_text.delete("1.0", tk.END)   # Clear previous content
    output_text.insert(tk.END, a)        # Insert new content
    output_text.config(state=tk.DISABLED)  # Disable editing

    processing_label.config(text="")  # Clear the processing message

# Create the main window
import tkinter as tk
window = tk.Tk()
window.title("Large Text GUI Example")
window.state("zoomed")  # Maximize the window by default
window.configure(bg="black")  # Set background color to black

# Include the title bar
window.overrideredirect(False)

# Create and place widgets with some padding and styling
label = tk.Label(window, text="Enter prompt:", font=("Arial", 12), fg="white", bg="black")
label.pack(pady=10)

entry = tk.Entry(window, font=("Arial", 12), highlightbackground="green", highlightcolor="green", highlightthickness=2)
entry.pack(pady=10)

submit_button = tk.Button(window, text="Submit", command=on_submit, font=("Arial", 12), fg="white", bg="blue")
submit_button.pack(pady=10)

processing_label = tk.Label(window, text="", font=("Arial", 12), fg="white", bg="black")
processing_label.pack(pady=10)

output_text = tk.Text(window, height=10, width=60, wrap=tk.WORD, font=("Courier", 12), fg="white", bg="black", highlightbackground="green", highlightcolor="green", highlightthickness=2)
output_text.pack(pady=10)

# Add horizontal scrollbar
x_scrollbar = tk.Scrollbar(window, orient=tk.HORIZONTAL, command=output_text.xview)
x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
output_text.config(xscrollcommand=x_scrollbar.set)

# Start the GUI event loop
window.mainloop()


# In[3]:


print(a)


# In[ ]:




