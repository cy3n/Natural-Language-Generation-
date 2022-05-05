# Natural-Language-Generation-
NLG Project for Infineon from SUTD DBA Team 12 

# We use this code for our initial solution using GPT-2 

from transformers import pipeline
# The pipeline() automatically loads a default model and tokenizer capable of inference for your task.

question_answerer = pipeline("question-answering")
# in this case, the task here is “question-answering”
# The default model and tokenizer is identified and loaded with the weight in the checkpoint


import csv
with open('StockPriceDate.csv', 'r') as file:
    data = file.read().rstrip() 
context2 = data
# csv 15 row data from Prices.csv created in the new file StockPriceDate.csv file 
# StockPriceDate file converted into rawstring file 

import numpy as np 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#remove VisibleDeprecitionWarning that pops up into the printed output 


result = question_answerer(question="What is the symbol of the least low stock ?", context=context2)
print(
    f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
)

# Iterate over the questions and build a sequence from the text and the current question, with the correct model-specific separators token type ids and attention masks.
# Compute the softmax of the result to get probabilities over the tokens.
# Fetch the tokens from the identified start and stop values, convert those tokens to a string.
# Print the results.
# score captures the accuracy of the result
