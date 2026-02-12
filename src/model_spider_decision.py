import os
import matplotlib as plt
import pandas as pd
from sklearn import metrics, tree

def importParquet(dataset_path:str, file_name:str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(dataset_path, file_name))
    return df

def main():
    
    pass

if __name__ == "__main__":
    main()