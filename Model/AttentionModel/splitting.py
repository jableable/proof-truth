import pandas as pd
from sklearn.model_selection import train_test_split
import ast

df = pd.read_csv('data_128.csv')


train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)



train_df.to_csv('train_data_128.csv', index=False)
test_df.to_csv('test_data_128.csv', index=False)