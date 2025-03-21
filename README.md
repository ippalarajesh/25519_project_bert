# 25519_project_bert

Fine-tuning a pre-trained BERT model for text classification using Hugging Face's transformers library and deploying it with Streamlit is an excellent project. I'll walk you through the steps involved in fine-tuning BERT and deploying it with Streamlit for web interaction.
With this setup, you should be able to fine-tune BERT, save the model, and deploy it as a user-friendly web app where people can classify text sentiment.


Explanation of the `train_model.py` Script:

## Step 1: Import Libraries

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
```

- **BertTokenizer** and **BertForSequenceClassification** are imported from Hugging Face to use the BERT model and tokenizer.
- Trainer and TrainingArguments are used to handle the training process.
- **load_dataset** is used to load the **IMDb dataset**.

## Step 2: Load Dataset

```python
dataset = load_dataset("imdb")
```

The IMDb dataset is loaded using the datasets library. It contains movie reviews and their sentiment labels.

## Step 3: Initialize Tokenizer and Model

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

**BertTokenizer:** This is used to preprocess the text data. The bert-base-uncased tokenizer is used, which means the tokenizer does not distinguish between uppercase and lowercase characters.
**BertForSequenceClassification:** This is the pre-trained BERT model designed for sequence classification tasks (such as sentiment analysis). We specify `num_labels=2` for **binary classification (positive/negative)**.

## Step 4: Tokenize the Dataset

```python
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)
```

- This function tokenizes the text data, ensuring that all sequences are padded or truncated to the same length.

## Step 5: Prepare Data for Training

```python
train_data = dataset['train'].map(tokenize_function, batched=True)
val_data = dataset['test'].map(tokenize_function, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
```

- The text data is processed into token IDs and attention masks, which are required by BERT.
- We convert the dataset into PyTorch tensors, which is necessary for training with the Hugging Face `Trainer`.

## Step 6: Set Up Training Arguments

```python
training_args = TrainingArguments(
    output_dir='./model/trained_model',          
    num_train_epochs=3,                          
    per_device_train_batch_size=16,              
    per_device_eval_batch_size=64,               
    warmup_steps=500,                            
    weight_decay=0.01,                           
    logging_dir='./logs',                        
    evaluation_strategy="epoch",                 
    save_strategy="epoch",                       
    load_best_model_at_end=True,                 
)

```

- **TrainingArguments:** This object defines the settings for training, including the number of epochs, batch size, logging, saving, and evaluation strategies.

## Step 7: Initialize the Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

```

- The **Trainer** class simplifies the training loop and takes care of the training, evaluation, and saving of the model.

## Step 8: Train the Model

```python
trainer.train()

```

The **train()** method starts the training process and saves the trained model in the specified output directory.

___________________________________________________________________________________________________________________________________________________________

# Streamlit Web Application

The Streamlit app allows users to interact with the trained BERT model via a web interface.

## Explanation of the `streamlit_app.py` Script:

## Step 1: Import Libraries

```python
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

```
- **streamlit**: For creating the web app interface.
- **BertTokenizer** and **BertForSequenceClassification**: To load the `fine-tuned model` and `tokenizer`.
- **torch**:For performing the actual classification.

## Step 2: Load the Trained Model

```python
model = BertForSequenceClassification.from_pretrained('./model/trained_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

```

- The fine-tuned model and tokenizer are loaded from the saved directory (`model/trained_model`)

## Step 3: Prediction Function

```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

```

- This function takes the user input text, tokenizes it, and feeds it into the BERT model. It then returns the sentiment prediction ("Positive" or "Negative").

## Step 4: Streamlit UI

```python
st.title("BERT Sentiment Classification")
st.write("Enter a sentence to predict its sentiment")

input_text = st.text_area("Text input", "Enter text here...")
if st.button("Classify"):
    sentiment = predict_sentiment(input_text)
    st.write(f"Sentiment: {sentiment}")

```

- The `Streamlit interface` allows users to input text in a text area and click the `"Classify"` button to get the sentiment result.

# Dependencies

Here are the dependencies required to run this project:

```txt
transformers==4.49.0
datasets==3.4.1
torch==2.6.0
streamlit==1.43.2
scikit-learn==1.6.1
```

You can install all dependencies using:

```bash
pip install -r requirements.txt
```
___________________________________________________________________________________________________________________________________________________________

# How to Run / Execute

## 1.Train the model:

- Run the `train_model.py`script to fine-tune the BERT model on the IMDb dataset:

```bash
python train_model.py

```

## 2.Launch the Streamlit app:

- Run the `streamlit_app.py` script to start the web interface:

```bash
streamlit run streamlit_app.py
```


# File Structure

Here is the folder structure for this project:

```python
bert-text-classification/
│
├── streamlit_app.py                # Streamlit app for web interface
├── model/
│   └── trained_model/              # Directory where the fine-tuned model is saved
│       ├── pytorch_model.bin       # Model weights after training
│       ├── config.json             # Configuration file for the model
│       └── tokenizer_config.json   # Tokenizer configuration file
├── data/
│   └── raw_data/                   # Raw dataset if not using Hugging Face datasets
│       └── imdb_data.csv           # Sample raw dataset (if required for custom processing)
├── train_model.py                  # Script for training the BERT model
├── requirements.txt                # File to list the project dependencies
└── README.md                       # Project description and instructions

```

# Key Files:

- `streamlit_app.py`: This file contains the Streamlit web application. Users can input text and receive sentiment classification from the model.
- `train_model.py`: The script used to fine-tune BERT on a sentiment analysis dataset (e.g., IMDb) and save the trained model.
- `model/trained_model/`: This folder stores the fine-tuned model and tokenizer configuration.
- `requirements.txt:` This file lists the necessary Python dependencies for the project.
