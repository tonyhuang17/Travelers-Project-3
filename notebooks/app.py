import pickle
import pandas as pd
# from flask import Flask, request, jsonify 

df = pd.read_parquet("Troop.parquet.gzip")


with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

prediction = model.predict(df)
pred = pd.Series(prediction)
df["predictions"] = pred
print(df)

