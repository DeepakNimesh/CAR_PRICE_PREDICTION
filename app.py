import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("🚗 Car Price Prediction ML Pipeline Dashboard")

# -------------------------
# STEP 1: Problem Type
# -------------------------
st.subheader("1️⃣ Problem Type")
problem_type = st.radio("", ["Regression", "Classification"], horizontal=True)

# -------------------------
# STEP 2: Upload Data
# -------------------------
st.subheader("2️⃣ Upload Dataset")
file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    # -------------------------
    # PCA Visualization
    # -------------------------
    st.subheader("🔍 PCA Visualization")
    numeric_df = df.select_dtypes(include=np.number).dropna()

    if numeric_df.shape[1] > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_df)

        pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        fig = px.scatter(pca_df, x="PC1", y="PC2")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # STEP 3: EDA
    # -------------------------
    st.subheader("3️⃣ EDA")

    st.write(df.describe())

    col1, col2 = st.columns(2)

    with col1:
        hist_col = st.selectbox("Histogram", df.columns)
        fig = px.histogram(df, x=hist_col)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        box_col = st.selectbox("Boxplot", df.columns)
        fig = px.box(df, y=box_col)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # STEP 4: Data Cleaning
    # -------------------------
    st.subheader("4️⃣ Data Cleaning")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if st.checkbox("Handle Missing Values"):
        num_strategy = st.selectbox("Numeric Strategy", ["mean", "median"])
        cat_strategy = st.selectbox("Categorical Strategy", ["most_frequent"])

        if num_cols:
            df[num_cols] = SimpleImputer(strategy=num_strategy).fit_transform(df[num_cols])

        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy=cat_strategy).fit_transform(df[cat_cols])

        st.success("Missing values handled")

    # Encoding (CRITICAL FIX)
    if st.checkbox("Encode Categorical Features (Required for ML)"):
        df = pd.get_dummies(df, drop_first=True)
        st.success("Categorical encoding applied")

    # Refresh column lists after encoding
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Outlier removal
    if st.checkbox("Remove Outliers (IQR)"):
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1

        df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) |
                  (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

        st.success("Outliers removed (IQR)")

    if st.checkbox("Remove Outliers (Isolation Forest)"):
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(df[num_cols])
        df = df[preds == 1]

        st.success("Outliers removed (Isolation Forest)")

    st.write("Cleaned Data Shape:", df.shape)

    # Ensure target exists
    if target not in df.columns:
        st.error("Target column missing after encoding. Please reselect.")
        st.stop()

    # -------------------------
    # STEP 5: Feature Selection
    # -------------------------
    st.subheader("5️⃣ Feature Selection")

    # Ensure no categorical data remains
    if df.select_dtypes(exclude=np.number).shape[1] > 0:
        st.warning("Auto encoding remaining categorical columns")
        df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=[target])
    y = df[target]

    if st.checkbox("Apply Variance Threshold"):
        selector = VarianceThreshold(threshold=0.1)
        X = selector.fit_transform(X)
        st.success("Feature selection applied")

    # Final NaN safety check
    if np.isnan(X).sum() > 0:
        st.error("NaN values still present. Please handle missing values.")
        st.stop()

    # -------------------------
    # STEP 6: Train-Test Split
    # -------------------------
    st.subheader("6️⃣ Train-Test Split")

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # STEP 7: Model Selection
    # -------------------------
    st.subheader("7️⃣ Model Selection")

    model_name = st.selectbox("Model", [
        "Linear Regression",
        "SVM",
        "Random Forest"
    ])

    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "SVM":
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        model = SVR(kernel=kernel)

    else:
        model = RandomForestRegressor()

    # -------------------------
    # STEP 8: K-Fold
    # -------------------------
    st.subheader("8️⃣ K-Fold Validation")

    k = st.slider("K value", 2, 10, 5)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    st.write("Cross Validation Score:", np.mean(scores))

    # -------------------------
    # STEP 9: Training
    # -------------------------
    st.subheader("9️⃣ Model Performance")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write("Train Score:", train_score)
    st.write("Test Score:", test_score)

    if train_score > test_score:
        st.warning("⚠️ Overfitting detected")

    # -------------------------
    # STEP 10: Hyperparameter Tuning
    # -------------------------
    # st.subheader("🔧 Hyperparameter Tuning")

    # if st.checkbox("Run GridSearch"):

    #     if model_name == "Random Forest":
    #         params = {
    #             "n_estimators": [50, 100],
    #             "max_depth": [None, 10, 20]
    #         }

    #     elif model_name == "SVM":
    #         params = {
    #             "C": [0.1, 1, 10],
    #             "gamma": ["scale", "auto"]
    #         }

    #     else:
    #         params = {}

    #     if params:
    #         grid = GridSearchCV(model, params, cv=3)
    #         grid.fit(X_train, y_train)

    #         st.write("Best Params:", grid.best_params_)
    #         st.write("Best Score:", grid.best_score_)