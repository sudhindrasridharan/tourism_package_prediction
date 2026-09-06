
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. DEFINE BASE PATH
# ============================================================

#BASE_PATH = os.path.dirname(
#    os.path.abspath(__file__)
#)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_PATH,
    "data",
    "tourism.csv"
)

PROCESSED_PATH = os.path.join(
    BASE_PATH,
    "data",
    "processed"
)

os.makedirs(
    PROCESSED_PATH,
    exist_ok=True
)

# ============================================================
# 2. FIND DATASET
# ============================================================

csv_files = [
    file
    for file in os.listdir(DATA_PATH)
    if file.lower().endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found in {DATA_PATH}"
    )

VAL_DATA_PATH = os.path.join(
    DATA_PATH,
    csv_files[0]
)

print("Dataset found at:")
print(VAL_DATA_PATH)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(VAL_DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

df = df.drop_duplicates()

print(
    "Shape after removing duplicates:",
    df.shape
)


# ============================================================
# 4. REMOVE UNNECESSARY COLUMNS
# ============================================================

if "CustomerID" in df.columns:

    df = df.drop(
        columns=["CustomerID"]
    )

    print("CustomerID removed.")


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["ProdTaken"]
)

y = df["ProdTaken"]


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# 7. CREATE TRAINING DATASET
# ============================================================

train_data = X_train.copy()

train_data["ProdTaken"] = y_train


# ============================================================
# 8. CREATE TEST DATASET
# ============================================================

test_data = X_test.copy()

test_data["ProdTaken"] = y_test


# ============================================================
# 9. SAVE DATASETS
# ============================================================

train_path = os.path.join(
    PROCESSED_PATH,
    "train.csv"
)

test_path = os.path.join(
    PROCESSED_PATH,
    "test.csv"
)


train_data.to_csv(
    train_path,
    index=False
)

test_data.to_csv(
    test_path,
    index=False
)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n========================================")
print("DATA PREPARATION COMPLETED")
print("========================================")

print("Training data shape:", train_data.shape)
print("Testing data shape :", test_data.shape)

print("\nTraining data saved to:")
print(train_path)

print("\nTesting data saved to:")
print(test_path)
