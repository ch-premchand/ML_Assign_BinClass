import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer



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
            st.info(f"Encoded classes: {', '.join(map(str, le_target.transform(le_target.classes_)))}")

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

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)


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

            for model_name in selected_models:
                if model_name in models:
                    with st.expander(f"{model_name}", expanded=True):
                        model = models[model_name]
                        
                        with st.spinner(f"Training {model_name}..."):
                            if model_name in ["Logistic Regression", "k-Nearest Neighbors"]:
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                                y_pred_proba = model.predict_proba(X_test_scaled)
                            else:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                y_pred_proba = model.predict_proba(X_test)
                        

                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        mcc = matthews_corrcoef(y_test, y_pred)
                        
                        if is_binary:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            st.metric("Precision", f"{precision:.4f}")
                        with col2:
                            st.metric("Recall", f"{recall:.4f}")
                            st.metric("F1 Score", f"{f1:.4f}")
                        with col3:
                            st.metric("AUC", f"{auc:.4f}")
                            st.metric("MCC", f"{mcc:.4f}")
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {model_name}')
                        st.pyplot(fig)
                        plt.close()
                        
                        results.append({
                            'Model': model_name,
                            'Accuracy': accuracy,
                            'AUC': auc,
                            'Precision': precision,
                            'Recall': recall,
                            'F1 Score': f1,
                            'MCC': mcc
                        })
            
            if results:
                st.header("Model Comparison Summary")
                results_df = pd.DataFrame(results)
                
                st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']))
                
                best_model_idx = results_df['F1 Score'].idxmax()
                best_model = results_df.loc[best_model_idx, 'Model']
                st.success(f"Best Model (by F1 Score): **{best_model}**")
                

                st.subheader("Metrics Comparison")
                metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()
                
                for idx, metric in enumerate(metrics_to_plot):
                    axes[idx].bar(results_df['Model'], results_df[metric], color='skyblue')
                    axes[idx].set_title(metric)
                    axes[idx].set_ylabel('Score')
                    axes[idx].tick_params(axis='x', rotation=45)
                    axes[idx].grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

else:
    st.info("Please upload a CSV file !")

