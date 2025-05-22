import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

ds = pd.read_csv("new-thyroid.data", header=None)

ds.columns = [
    "class",    # 1 = normal, 2 = hyper, 3 = hypo
    "T3_uptake",    # % (in raw unit)
    "thyroxin",   
    "triiodothyronine",   
    "basal_TSH",
    "TSH_response"
]

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=ds)
metadata.update_column(column_name='class', sdtype='categorical')

# Decided against CTGAN after testing/training too small
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(ds)
synthetic_data = synthesizer.sample(num_rows=1000)
