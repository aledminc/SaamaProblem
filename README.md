# Saama Tech Exemplary Project

---

### **Problem Statement**

Given a small real Electronic Health Record (EHR) table (demographics, vitals, labs) and you can choose any dataset from the UC Irvine Machine Learning Data Repository.

Select any **tabular-data synthesis framework or library**, fit a generative model to the real data, and sample **1,000 synthetic patient records**.

For evaluation, compute:
- Column-wise statistical tests
- A privacy metric such as **PCA-based distance** between real and synthetic data or **record-matching risk**

---

### **Deliverable**

A script that shows:
- Your framework choice
- Model training code
- Sampling code
- Evaluation results

With short writing on:
- Synthesized data quality
- Privacy trade-offs

---

### **Execution**

Chose **Thyroid Disease dataset** from the Garavan Institute:  
https://archive.ics.uci.edu/dataset/102/thyroid+disease

Opted for `new-thyroid.[data, names]` set for synthesis:
- No missing values  
- Data is pre-cleaned  
- Contains diagnosis label (for later classification if needed)  
- Moderate size (215 records)

Used **SDV Library and CTGAN** to train the generative model, then generated 1,000 synthetic samples.
