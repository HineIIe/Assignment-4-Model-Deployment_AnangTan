import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
   
    
    df = df.copy()

    # Extract features from cabin
    df["Deck"] = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else "Unknown")
    df["Cabin_num"] = df["Cabin"].apply(lambda x: x.split("/")[1] if pd.notna(x) else -1)
    df["Side"] = df["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else "Unknown")

    df["Cabin_num"] = pd.to_numeric(df["Cabin_num"], errors="coerce")

    # Extract group from PassengerId
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Group_size"] = df.groupby("Group")["Group"].transform("count")
    df["Solo"] = (df["Group_size"] == 1).astype(int)

    # Extract family info from Name
    df["LastName"] = df["Name"].apply(lambda x: x.split()[-1] if pd.notna(x) else "Unknown")
    df["Family_size"] = df.groupby("LastName")["LastName"].transform("count")

    # Spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    df["TotalSpending"] = df[spending_cols].sum(axis=1)

    df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

    # Spending ratios
    for col in spending_cols:
        df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)

    # Age groups
    df["Age_group"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 30, 50, 100],
        labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"]
    )

    # Missing indicators
    df["Age_missing"] = df["Age"].isna().astype(int)
    df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

    return df


def build_preprocessor(categorical_features, numerical_features):
    

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_pipeline, categorical_features),
        ("num", numerical_pipeline, numerical_features)
    ])

    return preprocessor


def preprocess_data(df: pd.DataFrame, is_train=True):
    

    df = feature_engineering(df)

    categorical_features = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "Age_group"
    ]

    numerical_features = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "Cabin_num",
        "Group_size",
        "Solo",
        "Family_size",
        "TotalSpending",
        "HasSpending",
        "NoSpending",
        "Age_missing",
        "CryoSleep_missing",
        "RoomService_ratio",
        "FoodCourt_ratio",
        "ShoppingMall_ratio",
        "Spa_ratio",
        "VRDeck_ratio"
    ]

    feature_columns = categorical_features + numerical_features

    X = df[feature_columns]

    if is_train:
        y = df["Transported"].astype(int)
        return X, y, categorical_features, numerical_features
    else:
        return X