from scipy.stats import ks_2samp, chi2_contingency
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from synthesize import *

results = {}

for col in ds.columns:
    if col == 'class':  
        real_counts = ds[col].value_counts().sort_index()
        synth_counts = synthetic_data[col].value_counts().sort_index()

        all_classes = sorted(set(ds[col].unique()) | set(synthetic_data[col].unique()))
        real_counts = real_counts.reindex(all_classes, fill_value=0)
        synth_counts = synth_counts.reindex(all_classes, fill_value=0)

        chi2, p, _ = chi2_contingency([real_counts, synth_counts])
        results[col] = {'test': 'chi2', 'p_value': p}
    else:  
        stat, p = ks_2samp(ds[col], synthetic_data[col])
        results[col] = {'test': 'ks', 'p_value': p}


for col, res in results.items():
    print(f"{col}: {res['test']} p-value = {res['p_value']:.4f}")

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