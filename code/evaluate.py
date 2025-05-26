from scipy.stats import ks_2samp, chi2_contingency
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from synthesize import *

# ---- Statistical Tests ----

def KS_Chi():

    results = {}

    for col in ds.columns:
        if col == 'class':  
            real_counts = ds[col].value_counts().sort_index()
            synth_counts = synthetic_data[col].value_counts().sort_index()

            all_classes = sorted(set(ds[col].unique()) | set(synthetic_data[col].unique()))
            real_counts = real_counts.reindex(all_classes, fill_value=0)
            synth_counts = synth_counts.reindex(all_classes, fill_value=0)

            chi2, p, _, _ = chi2_contingency([real_counts, synth_counts])
            results[col] = {'test': 'chi2', 'p_value': p}
        else:  
            stat, p = ks_2samp(ds[col], synthetic_data[col])
            results[col] = {'test': 'ks', 'p_value': p}


    for col, res in results.items():
        print(f"{col}: {res['test']} p-value = {res['p_value']:.4f}")

# ---- SDV Original Data Tests ----

def fromSDV():

    diagnostic = run_diagnostic(
        ds,
        synthetic_data,
        metadata
    )

    quality_report = evaluate_quality(
        ds,
        synthetic_data,
        metadata
    )

# ---- PCA and MIA ----

def privacyTest():

    pca = PCA(n_components=3) 
    real_values = ds.drop(columns=['class']).values
    synth_values = synthetic_data.drop(columns=['class']).values

    #PCA
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_values)
    synth_scaled = scaler.transform(synth_values)
    real_pca = pca.fit_transform(real_scaled)
    synth_pca = pca.transform(synth_scaled)

    distances = pairwise_distances(synth_pca, real_pca)
    min_dist = distances.min(axis=1)
    avg_min_distance = np.mean(min_dist)
    print("\nPrivacy metric (PCA + Euclidean Distance): ", avg_min_distance)

    #MIA proxy
    nn_real = NearestNeighbors(n_neighbors=2).fit(real_pca)
    distances_real, _ = nn_real.kneighbors(real_pca)
    real_real_dist = distances_real[:, 1] 

    avg_real_dist = np.mean(real_real_dist)
    print(f"Avg real-to-real NN distance: {avg_real_dist:.4f}")



if __name__ == "__main__":
    privacyTest()
    #fromSDV()
    #KS_Chi()