def clean_dataset(df):
    # Convert crimeaditionalinfo to Lower and strip white spaces
    df['crimeaditionalinfo'] = df['crimeaditionalinfo'].apply(lambda x: str(x).lower().strip())

    # If Category is missing, fill it with "No Category"
    df['category'] = df['category'].fillna("No Category")

    # Replace blanks with None for consistency
    df['sub_category'] = df['sub_category'].replace("", None)  
    # If Sub-Category is missing, fill it with Category
    df['sub_category'] = df['sub_category'].fillna(df['category'])

    # Remove rows where crimeaditionalinfo is empty
    df = df[df['crimeaditionalinfo'].apply(len) > 0]

    return df
