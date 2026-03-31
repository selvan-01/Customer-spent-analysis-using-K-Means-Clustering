# ============================================
# 📊 Customer Spend Analysis - Streamlit App
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------- Page Configuration ---------
st.set_page_config(
    page_title="Customer Spend Analysis",
    page_icon="📊",
    layout="wide"
)

# --------- Custom Styling ---------
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        h1, h2, h3 {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --------- Title Section ---------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("💰 Customer Spend Analysis using K-Means")
st.write("Analyze customer income vs spending behavior using clustering.")
st.markdown('</div>', unsafe_allow_html=True)


# --------- File Upload ---------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    # --------- Dataset Preview ---------
    st.subheader("📄 Dataset Preview")
    st.dataframe(dataset.head())

    # --------- Check Required Columns ---------
    if 'INCOME' in dataset.columns and 'SPEND' in dataset.columns:

        Income = dataset['INCOME'].values
        Spend = dataset['SPEND'].values
        X = np.array(list(zip(Income, Spend)))

        # --------- Elbow Method ---------
        st.subheader("📈 Elbow Method (Find Optimal K)")

        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 11), wcss, marker='o')
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("WCSS")
        st.pyplot(fig1)

        # --------- Select K ---------
        k = st.slider("Select Number of Clusters (K)", 2, 10, 4)

        # --------- Apply KMeans ---------
        kmeans_model = KMeans(n_clusters=k, random_state=0)
        y_means = kmeans_model.fit_predict(X)

        # --------- Visualization ---------
        st.subheader("🎯 Cluster Visualization")

        fig2, ax2 = plt.subplots()

        for i in range(k):
            ax2.scatter(
                X[y_means == i, 0],
                X[y_means == i, 1],
                s=50,
                label=f'Cluster {i+1}'
            )

        # Centroids
        ax2.scatter(
            kmeans_model.cluster_centers_[:, 0],
            kmeans_model.cluster_centers_[:, 1],
            s=200,
            marker='X',
            label='Centroids'
        )

        ax2.set_title("Customer Segmentation")
        ax2.set_xlabel("Income")
        ax2.set_ylabel("Spending")
        ax2.legend()

        st.pyplot(fig2)

        # --------- Insights ---------
        st.subheader("💡 Business Insights")
        st.success("""
        - Identify high-value customers
        - Target marketing strategies
        - Improve customer retention
        - Optimize product offerings
        """)

    else:
        st.error("❌ Dataset must contain 'INCOME' and 'SPEND' columns.")

else:
    st.info("⬅️ Upload a CSV file from the sidebar to begin.")