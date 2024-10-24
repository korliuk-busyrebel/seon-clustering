from sklearn.decomposition import PCA
import umap


def reduce_dimensions_optimal(df, n_components_pca=30, n_components_umap=2):
    pca = PCA(n_components=n_components_pca)
    df_pca = pca.fit_transform(df)

    reducer = umap.UMAP(n_components=n_components_umap)
    df_umap = reducer.fit_transform(df_pca)

    return df_umap
