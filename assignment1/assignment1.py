import pandas as pd
from sklearn.datasets import load_iris
import os

# make sure "data" folder exists
os.makedirs("data", exist_ok=True)

# load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# save CSV into data/ folder
df.to_csv("data/iris.csv", index=False)
print("Wrote data/iris.csv")
