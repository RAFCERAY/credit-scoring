import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Scoring Crédit — Credit AES",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="metric-container"] {
        background-color: #f0f4ff;
        border: 1px solid #d0d7e3;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #1a56db; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train():
    df = pd.read_csv("data/Credit_AES.csv", sep=";", index_col="Identifiant Client")
    le = LabelEncoder()
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    X = df_enc.drop("Type de client", axis=1)
    y = df_enc["Type de client"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr.fit(X_train_s, y_train)
    rf.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr.predict_proba(X_test_s)[:, 1])
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    lr_auc = auc(lr_fpr, lr_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    return df, df_enc, X, y, lr_acc, rf_acc, lr_fpr, lr_tpr, rf_fpr, rf_tpr, lr_auc, rf_auc, importance, X_test, y_test, lr, rf, scaler, X_train

df, df_enc, X, y, lr_acc, rf_acc, lr_fpr, lr_tpr, rf_fpr, rf_tpr, lr_auc, rf_auc, importance, X_test, y_test, lr, rf, scaler, X_train = load_and_train()

st.markdown("# 💳 Scoring Crédit — Credit AES")
st.markdown("**Classification Bon/Mauvais client · Régression Logistique vs Random Forest**")
st.markdown("---")

tabs = st.tabs(["📊 Vue globale", "🤖 Modèles ML", "🔍 Prédiction client", "📈 Features"])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients total", f"{len(df):,}")
    col2.metric("Variables", f"{df.shape[1]}")
    bons = (df["Type de client"] == "Bon client").sum()
    col3.metric("Bons clients", f"{bons}")
    col4.metric("Mauvais clients", f"{len(df) - bons}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Répartition des clients**")
        vc = df["Type de client"].value_counts().reset_index()
        fig = px.pie(vc, values="count", names="Type de client",
                     hole=0.45, template="plotly_white",
                     color_discrete_sequence=["#2ecc71", "#e74c3c"])
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Aperçu des données**")
        st.dataframe(df.head(10), use_container_width=True)

with tabs[1]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy — Rég. Logistique", f"{lr_acc*100:.1f}%")
    col2.metric("AUC — Rég. Logistique", f"{lr_auc:.3f}")
    col3.metric("Accuracy — Random Forest", f"{rf_acc*100:.1f}%")
    col4.metric("AUC — Random Forest", f"{rf_auc:.3f}")
    st.markdown("---")
    st.markdown("**Courbes ROC**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode="lines",
                             name=f"Régression Logistique (AUC={lr_auc:.3f})",
                             line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode="lines",
                             name=f"Random Forest (AUC={rf_auc:.3f})",
                             line=dict(color="#3498db", width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Aléatoire", line=dict(color="gray", dash="dash")))
    fig.update_layout(template="plotly_white", height=400,
                      xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown("**Prédire le type de client**")
    st.info("Remplis les informations du client pour obtenir une prédiction.")
    col1, col2, col3 = st.columns(3)
    input_data = {}
    cols = [col1, col2, col3]
    num_cols = df_enc.drop("Type de client", axis=1).columns.tolist()
    for i, col_name in enumerate(num_cols):
        with cols[i % 3]:
            min_val = float(df_enc[col_name].min())
            max_val = float(df_enc[col_name].max())
            mean_val = float(df_enc[col_name].mean())
            input_data[col_name] = st.slider(col_name, min_val, max_val, mean_val)
    if st.button("🔮 Prédire", type="primary"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred_lr = lr.predict(input_scaled)[0]
        pred_rf = rf.predict(input_df)[0]
        proba_rf = rf.predict_proba(input_df)[0]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Régression Logistique",
                      "✅ Bon client" if pred_lr == 1 else "❌ Mauvais client")
        with col2:
            st.metric("Random Forest",
                      "✅ Bon client" if pred_rf == 1 else "❌ Mauvais client")
            st.metric("Probabilité bon client", f"{proba_rf[1]*100:.1f}%")

with tabs[3]:
    st.markdown("**Importance des variables — Random Forest**")
    fig = px.bar(importance, x="importance", y="feature",
                 orientation="h", template="plotly_white",
                 color="importance", color_continuous_scale="Blues")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*💳 Scoring Crédit — Credit AES · Rafika Cervera*")