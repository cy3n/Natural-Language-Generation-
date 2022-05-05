# Natural-Language-Generation-
NLG Project for Infineon from SUTD DBA Team 12.

# Initial solution using GPT-2 
This GPT-2 Model that we will be using was pretrained on Webtext, a dataset consisting of content from 45 million links shared by the users of reddit. 
The GPT-2 Model can be found here: https://huggingface.co/gpt2

The initial solution was meant to involve the GPT-2 model exclusively. However, the model only accepts text (strings) rather than tabulated data (dataframes), as the client demanded. To compensate, the ‘prices.csv’ data is converted into rawstring using the function data = file.read().rstrip():

# Data used 
The data used is 15 rows of data from the Prices.csv. You can find this CSV data in the file of the repository. 

# Code for Initial Solution 
from transformers import pipeline

question_answerer = pipeline("question-answering")

import csv
with open('StockPriceDate.csv', 'r') as file:
    data = file.read().rstrip() 
context2 = data

import numpy as np 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

result = question_answerer(question="What is the symbol of the least low stock ?", context=context2)
print(
    f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
)

# Reference 
GPT-2 referenced: https://huggingface.co/gpt2 
