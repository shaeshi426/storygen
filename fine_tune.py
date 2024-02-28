#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd


# In[2]:


import os


# In[3]:


os.chdir("C:\\Users\\Shreyash pc\\Desktop")


# In[4]:


train_path = "IN-Abs-train.csv"
validation_path = "IN-Abs-test.csv"


# In[5]:


train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)


# In[6]:


train_df['judgement-text']=train_df['judgement-text'].apply(str)
train_df['summary-text']=train_df['summary-text'].apply(str)


# In[7]:


validation_df['judgement-text']=validation_df['judgement-text'].apply(str)
validation_df['summary-text']=validation_df['summary-text'].apply(str)


# In[8]:


import re


# In[9]:


def preprocess_text(text):
    text = re.sub(r'(\d\d\d|\d\d|\d)\.\s', ' ', text)  # removes the paragraph labels 1. or 2. etc.
    text = re.sub(r'(?<=[a-zA-Z])\.(?=\d)', '', text)  # removes dot(.) i.e File No.1063
    text = re.sub(r'(?<=\d|[a-zA-Z])\.(?=\s[\da-z])', ' ', text)  # to remove the ending dot of abbreviations
    text = re.sub(r'(?<=\d|[a-zA-Z])\.(?=\s?[\!\"\#\$\%\&\'\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~])', '', text)  # to remove the ending dot of abbreviations
    text = re.sub(r'(?<!\.)[\!\"\#\$\%\&\'\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)  # removes other punctuations
    return text

# Apply the preprocessing function to 'judgement-text' column
train_df['judgement-text'] = train_df['judgement-text'].apply(preprocess_text)

# Apply the preprocessing function to 'summary-text' column
validation_df['summary-text'] = validation_df['summary-text'].apply(preprocess_text)


# In[10]:


validation_df['judgement-text'] = validation_df['judgement-text'].apply(preprocess_text)

# Apply the preprocessing function to 'summary-text' column
train_df['summary-text'] = train_df['summary-text'].apply(preprocess_text)


# In[11]:


train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)


# In[12]:


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


# In[13]:


#def preprocess_function(examples):
#    inputs = examples['judgement-text']
#    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
#    with tokenizer.as_target_tokenizer():
#        labels = tokenizer(examples['summary-text'], max_length=119, truncation=True)

#    model_inputs["labels"] = labels["input_ids"]
#    return model_inputs
def preprocess_function(examples):
    inputs = examples['judgement-text']
    model_inputs = tokenizer(inputs, max_length=302, truncation=True, padding="max_length", return_tensors="pt")

    # Tokenize labels with an exact length of 119
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary-text'], max_length=119, truncation=True, padding="max_length", return_tensors="pt")

    # Ensure all labels have a fixed length of 119 using padding
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[14]:


# Tokenize all texts and align the labels with them.
train_dataset = train_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)


# In[15]:


# Load the pre-trained BART model
 = BartForConditionalGeneration.from_pmodelretrained("facebook/bart-large-cnn")


# In[16]:


pip install transformers[torch]


# In[17]:


pip install accelerate -U


# In[18]:


import accelerate
print(accelerate.__version__)


# In[19]:


get_ipython().system('pip install torch')


# In[20]:


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,              # Total number of training epochs to prevent memory issues
    per_device_train_batch_size=2,   # Reduced batch size per device during training
    per_device_eval_batch_size=2,    # Reduced batch size for evaluation
    warmup_steps=300,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=100,
    evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
    save_strategy="epoch",           # Save the model at the end of each epoch.
    load_best_model_at_end=True,     # Load the best model at the end of training.
    gradient_accumulation_steps=8,   # Use gradient accumulation to deal with small batch sizes
)


# In[21]:


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)


# In[22]:


trainer.train()


# In[25]:


trainer.save_model()


# In[30]:


import torch as torch


# In[38]:


torch.save(model,"abcd")


# In[ ]:




