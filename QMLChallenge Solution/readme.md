
# Team Name: Hybrids
**Team Members**
- Owais Ishtiaq Siddiqui
- Abdullah Kazi
- Hussien Shiri

# Summary
## Quantum Challenge: Feature Preparation & PCA Selection
Standardize Features:
- All 28 features are z-score standardized to mean 0, variance 1 (using StandardScaler).

- PCA Dimensionality Reduction:
  - The standardized features are projected onto a lower-dimensional space using Principal Component Analysis (PCA).
  - The number of components is set to n_qubits (e.g., 8), so the QNN input matches the quantum circuit width.

  **Example**
  `
    pca = PCA(n_components=n_qubits)
    X_pca = pca.fit_transform(X_scaled)
  `
- Over 50% of the variance is typically retained in the first 8 principal components.

- Range Map for Quantum Encoding:
  Projected PCA features are mapped to the $[0, 2\pi]$ range for compatibility with quantum gates (MinMaxScaler).
- Train/Validation Split:
  PCA is applied on the entire standardized dataset before splitting, avoiding data leakage and ensuring consistent feature transformation on all splits.

- Result:
  The input to the QNN is a compact and information-rich $[N_\text{samples},n_{\text{qubits}}]$ matrix that maximizes usable signal per qubit.

This improves quantum training stability and performance, ensuring the model uses the most informative possible input while remaining compliant with the QML challenge’s   qubit constraint!
