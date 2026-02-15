import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="ML Classification Models", layout="wide")

st.title("ML Classification Models Comparison")
st.markdown("Upload your dataset and compare 6 different classification models!")


st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Features", df.shape[1] - 1)

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    if df.shape[0] <500:
        st.warning("The dataset has less than 500 rows.")
    if df.shape[1] < 13:
        st.warning("The dataset has less than 13 columns.")

    st.sidebar.subheader("Target Variable Selection")
    target_variable = st.sidebar.selectbox("Select the target variable", df.columns)

    if target_variable:
        x = df.drop(columns=[target_variable])
        y = df[target_variable]

        categorical_cols = x.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.info(f"Categorical columns detected: {', '.join(categorical_cols)}. They will be encoded using Label Encoding.")
            for col in categorical_cols:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col].astype(str))

        if y.dtype == 'object' or y.dtype.name == 'category':
            st.info(f"Target variable '{target_variable}' is categorical. It will be encoded using Label Encoding.")
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            st.info(f"Classes in target variable: {', '.join(le_target.classes_)}")

        n_classes = len(np.unique(y))
        if n_classes == 2:
            is_binary = True
        else:
            is_binary = False
        
        st.sidebar.subheader("Model Configuration")
        test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)/100
        random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42, step=1)

         X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.sidebar.subheader("Model Selection")
        model_options = ["Logistic Regression", "Decision Tree", "k-Nearest Neighbors", "Naive Bayes", "Random Forest", "XGBoost"]

        selected_models = st.sidebar.multiselect("Select models to compare", model_options, default=model_options)

        if st.sidebar.button("Run Models"):
            st.header("Model Performance Comparison")
            results = []
            
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
                "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                "k-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
                "XGBoost": XGBClassifier(random_state=random_state, eval_metric='logloss')
            }


        
        

        

else:
    st.info("Please upload a CSV file !")

