# To fine-tune the BERT model on a text classification task, run the train_model.py script. 
# This script will load the IMDb dataset, fine-tune a pre-trained BERT model on it, and save the model in the model/trained_model/ folder.

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_data = dataset['train'].map(tokenize_function, batched=True)
val_data = dataset['test'].map(tokenize_function, batched=True)

# Convert datasets to PyTorch format
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Set training arguments
training_args = TrainingArguments(
    output_dir='./model/trained_model',          # output directory for results the model's training outputs, like intermediate checkpoints and logs, will be saved here
    num_train_epochs=3,                          # number of epochs to train the model 
    per_device_train_batch_size=16,              # batch size for training specifies that 16 samples(records) will be processed together in one forward and backward pass on each GPU device
    per_device_eval_batch_size=64,               # batch size for evaluation specifies that 64 samples(records) will be processed together in one forward pass on each GPU device
    warmup_steps=500,                            # warmup steps for learning rate scheduler 
    weight_decay=0.01,                           # strength of weight decay 
    logging_dir='./logs',                        # logging directory for training logs
    evaluation_strategy="epoch",                 # evaluation strategy to adopt during training , An epoch represents a full pass through the entire dataset during training.
    save_strategy="epoch",                       # save model every epoch 
    load_best_model_at_end=True,                 # load best model when finished training 
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
