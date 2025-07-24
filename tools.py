from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt

# Global CSV data (loaded once)
df = pd.read_csv("laptopData.csv")

@tool
def get_columns() -> list:
    """Return a list of column names from the dataset."""
    return df.columns.tolist()

