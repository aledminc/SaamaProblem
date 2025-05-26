import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from evaluate import *

ds = pd.read_csv("data/new-thyroid.data", header=None)

ds.columns = [
    "class",    # 1 = normal, 2 = hyper, 3 = hypo
    "T3_uptake",    # % (in raw unit)
    "thyroxin",   
    "triiodothyronine",   
    "basal_TSH",
    "TSH_response"
]

# ------ Graphs to show non-normal skew ------

def skews():
    sns.histplot(ds['basal_TSH'], kde=True)
    plt.title('Basal TSH Distribution')
    plt.show()

    sns.histplot(ds['TSH_response'], kde=True)
    plt.title('TSH Response Distribution')
    plt.show()

# ------ Graphs to show relation between Basal_TSH and TSH_Response ------

def relation():
    ds['log_basal_TSH'] = np.log1p(ds['basal_TSH'])
    ds['log_TSH_response'] = np.log1p(ds['TSH_response'])

    X = ds[['log_basal_TSH']].values
    y = ds['log_TSH_response'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_pred = model.predict(x_range_poly)

    plt.scatter(X, y, alpha=0.5)
    plt.plot(x_range, y_pred, color='red')
    plt.title('Polynomial Fit (Degree 2): TSH Response vs Basal TSH')
    plt.xlabel('Log Basal TSH')
    plt.ylabel('Log TSH Response')
    plt.show()

    print("R² Score:", r2_score(y, model.predict(X_poly)))
    corr, p = spearmanr(ds['basal_TSH'], ds['TSH_response'])
    print(f"Spearman correlation (original): {corr:.3f}, p = {p:.3f}")

# ------ Real vs Synthetic Grpah Distributions for Problem Columns------

def synthReal():
    issues = ['triiodothyronine', 'basal_TSH', 'TSH_response']

    for col in issues:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(ds[col], label='Real', fill=True, alpha=0.4)
        sns.kdeplot(synthetic_data[col], label='Synthetic', fill=True, alpha=0.4, color='orange')
        plt.title(f'Distribution Comparison for {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ------ Graph to determine components for PCA ------

def compPSA():
    real_scaled = StandardScaler().fit_transform(ds.drop(columns=['class']))
    pca = PCA().fit(real_scaled)

    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Variance vs Components')
    plt.grid(True)
    plt.axhline(0.9, color='red', linestyle='--', label='90% Variance')
    plt.legend()
    plt.show()  

if __name__ == "__main__":
    #skews()
    synthReal()
    #compPSA()
    #relation()
