import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

def masscenter(ser):
    print(df.loc[ser.index])
    return 0

rol = df.A.rolling(window=2)
rol.apply(masscenter, raw=False)

