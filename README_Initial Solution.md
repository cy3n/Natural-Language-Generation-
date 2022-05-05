# Natural-Language-Generation-
NLG Project for Infineon from SUTD DBA Team 12 

# We use this code for our initial solution using GPT-2 

The initial solution was meant to involve the GPT-2 model exclusively.
However, the model only accepts text (strings) rather than tabulated data (dataframes), as the client demanded. To compensate, the ‘prices.csv’ data is converted into rawstring using the function data = file.read().rstrip():

This GPT-2 dataset (from HuggingFace) is Trained on WebText: a dataset consisting of the text contents of 45 million links posted by ‘Reddit’ users.The model is not 
trained on financial stock data. 

Although the answer is correctly given, the accuracy score is severely impacted, which can be remedied by training the model with financial data and jargon.

