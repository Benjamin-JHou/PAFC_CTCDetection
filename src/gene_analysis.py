"""
Gene‑PAFC feature analysis:
Curatopes melanoma epitopes + PAFC features
correlation, stratification, clustering & permutation test.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


def load_and_clean_gene_data(curatopes_path: Path):
    """Load Curatopes gene dataset and deduplicate."""
    df = pd.read_csv(curatopes_path)
    df = df.drop_duplicates(subset=["Gene", "Peptide", "IC50"])
    df["Immunogenicity_bin"] = pd.cut(
        df["Immunogenicity"], bins=3, labels=["Low", "Medium", "High"]
    )
    return df


def stratify_and_analyze(df, pafc_features):
    """Analyze mean PAFC features by gene immunogenicity bin."""
    merged = df.merge(pafc_features, on="Gene")
    result = {}
    for level in ["Low", "Medium", "High"]:
        subset = merged[merged["Immunogenicity_bin"] == level]
        result[level] = subset.mean(numeric_only=True)
    return result


def high_immunogenicity_diff(df, pafc_features):
    """Compute mean/std differences for high immunogenicity genes."""
    high = df[df["Immunogenicity"] >= df["Immunogenicity"].quantile(0.75)]
    merged = high.merge(pafc_features, on="Gene")
    group_stats = merged.groupby("Group").mean()
    diff = group_stats.loc["Melanoma"] - group_stats.loc["Healthy"]
    ranked = diff.abs().sort_values(ascending=False)
    return ranked


def correlation_analysis(df, pafc_features, group_col="Group"):
    """Compute Pearson correlation between gene immunogenicity and PAFC features per group."""
    results = []
    for group in df[group_col].unique():
        subset = df[df[group_col] == group].merge(pafc_features, on="Gene")
        for f in pafc_features.columns.drop("Gene"):
            r, p = pearsonr(subset["Immunogenicity"], subset[f])
            results.append((group, f, r, p))
    corr_df = pd.DataFrame(results, columns=["Group", "Feature", "r", "p"])
    _, qvals, _, _ = multipletests(corr_df["p"], method="fdr_bh")
    corr_df["q"] = qvals
    return corr_df


def welch_ttest(pafc_features, group_col="Group"):
    """Welch’s t‑test & Cohen’s d between melanoma & healthy for each feature."""
    results = []
    features = pafc_features.columns.drop(["Gene", group_col])
    groups = pafc_features[group_col].unique()
    g1 = pafc_features[pafc_features[group_col] == groups[0]]
    g2 = pafc_features[pafc_features[group_col] == groups[1]]
    for f in features:
        x1, x2 = g1[f], g2[f]
        t, p = ttest_ind(x1, x2, equal_var=False)
        d = (x1.mean() - x2.mean()) / np.sqrt((x1.var() + x2.var())/2)
        results.append((f, t, p, d))
    ttest_df = pd.DataFrame(results, columns=["Feature", "t", "p", "Cohen_d"])
    _, qvals, _, _ = multipletests(ttest_df["p"], method="fdr_bh")
    ttest_df["q"] = qvals
    ttest_df["Effect_size_category"] = pd.cut(
        ttest_df["Cohen_d"].abs(),
        bins=[0, 0.1, 0.2, 0.4, np.inf],
        labels=["Negligible", "Small", "Medium", "Large"]
    )
    return ttest_df


def permutation_test(df, pafc_features, n_iter=1000):
    """Permutation test for immunogenicity‑feature correlations."""
    np.random.seed(42)
    results = []
    merged = df.merge(pafc_features, on="Gene")
    features = pafc_features.columns.drop(["Gene", "Group"])
    for f in features:
        obs_r, _ = pearsonr(merged["Immunogenicity"], merged[f])
        null = []
        for _ in range(n_iter):
            shuffled = np.random.permutation(merged["Immunogenicity"])
            r, _ = pearsonr(shuffled, merged[f])
            null.append(r)
        null = np.array(null)
        p_empirical = ((np.abs(null) >= np.abs(obs_r)).sum() + 1) / (n_iter + 1)
        results.append((f, obs_r, p_empirical))
    perm_df = pd.DataFrame(results, columns=["Feature", "r", "empirical_p"])
    _, qvals, _, _ = multipletests(perm_df["empirical_p"], method="fdr_bh")
    perm_df["q"] = qvals
    return perm_df


def hierarchical_clustering(pafc_features, method="ward"):
    """Hierarchical clustering of features."""
    data = pafc_features.drop(columns=["Gene", "Group"]).T
    dist = pdist(data, metric="euclidean")
    linkage_matrix = linkage(dist, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=data.index, leaf_rotation=90)
    plt.title("Hierarchical Clustering of PAFC Features")
    plt.tight_layout()
    plt.show()
    return linkage_matrix


def main():
    # Paths to data
    gene_csv = Path("data/curatopes_melanoma.csv")
    pafc_csv = Path("data/pafc_features.csv")  # Assume columns: Gene, Group, feat1, feat2, ...

    gene_df = load_and_clean_gene_data(gene_csv)
    pafc_df = pd.read_csv(pafc_csv)

    # Stratified analysis
    strata_means = stratify_and_analyze(gene_df, pafc_df)
    print("\n--- Mean PAFC Features by Immunogenicity Bin ---")
    for level, stats in strata_means.items():
        print(f"\n{level}:\n", stats)

    # High immunogenicity diff
    high_diff = high_immunogenicity_diff(gene_df, pafc_df)
    print("\n--- Features ranked by difference in high immunogenicity genes ---")
    print(high_diff)

    # Correlation per group
    corr_df = correlation_analysis(gene_df, pafc_df)
    print("\n--- Correlations per group ---")
    print(corr_df)

    # Welch t‑test & Cohen’s d
    ttest_df = welch_ttest(pafc_df)
    print("\n--- Welch t‑test & Effect sizes ---")
    print(ttest_df)

    # Permutation test
    perm_df = permutation_test(gene_df, pafc_df)
    print("\n--- Permutation test results ---")
    print(perm_df)

    # Clustering
    hierarchical_clustering(pafc_df)

    # Save results
    gene_df.to_csv("gene_cleaned.csv", index=False)
    corr_df.to_csv("correlations.csv", index=False)
    ttest_df.to_csv("welch_ttest.csv", index=False)
    perm_df.to_csv("permutation_test.csv", index=False)
    print("\n✅ Analysis complete. Results saved.")


if __name__ == "__main__":
    main()
