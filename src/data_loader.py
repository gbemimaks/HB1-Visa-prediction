
# src/data_loader.py
# src/data_loader.py

import os
import pandas as pd

def load_and_preprocess_data(csv_path, output_dir, drop_cols=None, save_name="processed_data.csv"):
    """
    Loads data from a CSV file, optionally drops specified columns, 
    saves the processed data into the output directory, and returns the DataFrame.
    
    Args:
        csv_path (str): Path to input CSV file.
        output_dir (str): Directory to save processed CSV.
        drop_cols (list, optional): Columns to drop. Defaults to None.
        save_name (str, optional): Name of the saved file. Defaults to "processed_data.csv".
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """

    # ✅ Step 1: Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Step 2: Load the CSV
    df = pd.read_csv(csv_path)

    # ✅ Step 3: Drop specified columns if given
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # ✅ Step 4: Save the processed data
    output_path = os.path.join(output_dir, save_name)
    df.to_csv(output_path, index=False)

    print(f"✅ Processed data saved at: {output_path}")

    # ✅ Step 5: Return the DataFrame
    return df
