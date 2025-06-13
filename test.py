import pandas as pd
import os 
dir = "alice_data"
id_counter = 1
for filename in os.listdir(dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(dir, filename)
        df = pd.read_csv(file_path)
        df.insert(0, 'id', range(id_counter, id_counter + len(df)))
        df.to_csv(file_path, index=False)
        id_counter += len(df)