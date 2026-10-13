import pandas as pd

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

print("Fake Dataset Shape:")
print(fake.shape)

print("\nReal Dataset Shape:")
print(real.shape)

print("\nColumns:")
print(fake.columns)

print("\nFake Dataset Info:")
print(fake.info())
