import pandas as pd
import numpy as np

def generate_treemap_dataset(output_file="generated_treemap_dataset.csv"):
    """
    Creates a CSV with:
      - 'MainCategory': a categorical column with many unique values (e.g., 30 unique categories)
      - 'SubCategory': a secondary categorical column (few unique values)
      - 'Value': a numeric column with only a few unique values (forcing it to be ordinal)
    This should push the primary variable detection to pick 'MainCategory' as categorical.
    """
    np.random.seed(42)
    n_rows = 200

    # Create 30 unique categories and randomly assign them
    categories = [f"Category_{i+1}" for i in range(30)]
    main_cat = np.random.choice(categories, size=n_rows)

    # Create a subcategory with 3 unique values
    subcat = np.random.choice(["SubA", "SubB", "SubC"], size=n_rows)

    # Create a 'Value' column with only 5 unique values
    value_options = [10, 20, 50, 100, 200]
    values = np.random.choice(value_options, size=n_rows)

    df = pd.DataFrame({
        "MainCategory": main_cat,
        "SubCategory": subcat,
        "Value": values
    })

    # Force 'MainCategory' to be string type (it already is, but just in case)
    df["MainCategory"] = df["MainCategory"].astype(str)

    # Save the dataset to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Treemap-friendly dataset saved as '{output_file}'")
    print(df.head())

if __name__ == "__main__":
    generate_treemap_dataset()

df = pd.read_csv("generated_treemap_dataset.csv")
print(df["MainCategory"].value_counts())
