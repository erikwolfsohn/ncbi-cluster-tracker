import glob
import os

import dendropy # type: ignore
import pandas as pd
import tqdm

import ncbi_cluster_tracker.cluster as cluster

from ncbi_cluster_tracker.logger import logger

class Cluster:
    """
    Represents a SNP cluster on NCBI Pathogen Detection and its distance matrix. 
    """
    MAX_MATRIX_SIZE = 20
    MAX_TREE_SIZE = 500

    def __init__(
            self,
            name: str,
            internal_isolates: list[str],
            total_count: int | None = None,
            max_cluster_size: int | None = None,
    ):
        self.name = name
        self.cluster_id = name.split('.')[0]
        self.internal_isolates = set(internal_isolates)
        self.external_isolates: set[str] | None = None
        self.total_count = total_count
        self.max_cluster_size = max_cluster_size
        self.snp_tree_skipped = (
            max_cluster_size is not None
            and total_count is not None
            and total_count > max_cluster_size
        )
        self.filtered_matrix_message: str | None = None
        self.filtered_matrix: pd.DataFrame | None = self._create_filtered_matrix()

    def _create_filtered_matrix(self) -> pd.DataFrame | None:
        """
        Generate distance matrix from downloaded Pathogen Detection tree,
        with number of isolates filtered down to a reasonably viewable number.
        """
        if self.snp_tree_skipped:
            self.filtered_matrix_message = (
                f'SNP distance matrix not displayed and SNP tree not downloaded '
                f'since cluster has {self.total_count} isolates, exceeding '
                f'--max-cluster-size of {self.max_cluster_size}.'
            )
            return None

        tree_file_glob = f'{self.cluster_id}*.newick_tree.newick'
        tree_path = glob.glob(
            os.path.join(os.environ['NCT_OUT_SUBDIR'], 'snps', tree_file_glob
        ))[0]
        tree = dendropy.Tree.get_from_path(tree_path, schema='newick')

        internal_taxa = {t for t in tree.taxon_namespace
                         if t.label in self.internal_isolates}

        external_taxa = {t for t in tree.taxon_namespace
                         if t.label not in self.internal_isolates}

        self.external_isolates = {t.label for t in external_taxa}

        # Arakawa warns if matrix has > 500 cells so limit matrix
        if len(tree) > self.MAX_MATRIX_SIZE:
            if len(tree.taxon_namespace) > self.MAX_TREE_SIZE:
                if 1 < len(internal_taxa) < self.MAX_MATRIX_SIZE:
                    # filter down to just the first self.MAX_TREE_SIZE taxa
                    tree.retain_taxa(internal_taxa)
                    self.filtered_matrix_message = ( 
                        f'SNP distance matrix filtered to show only internal isolates '
                        f'since there are more than {self.MAX_TREE_SIZE} isolates in the cluster.'
                    )
                elif len(internal_taxa) == 1:
                    tree = None
                    self.filtered_matrix_message = (
                        f'SNP distance matrix not displayed since there are '
                        f'more than {self.MAX_TREE_SIZE} isolates and only '
                        f'one internal isolate in the cluster.'
                    )

                else:
                    tree = None
                    self.filtered_matrix_message = (
                        f'SNP distance matrix not displayed since there are '
                        f'more than {self.MAX_TREE_SIZE} isolates in the cluster.'
                    )

            elif len(internal_taxa) == self.MAX_MATRIX_SIZE:
                # filter down taxa to just input isolates
                tree.retain_taxa(internal_taxa)
                self.filtered_matrix_message = ( 
                    f'SNP distance matrix filtered to show only the {self.MAX_MATRIX_SIZE} '
                    f'internal isolates.'
                )
            elif len(internal_taxa) < self.MAX_MATRIX_SIZE:
                # filter down to include only the nearest external isolates
                matrix = tree.phylogenetic_distance_matrix().as_data_table()
                matrix_df = pd.DataFrame.from_records(matrix._data)
                num_externals_to_keep = self.MAX_MATRIX_SIZE - len(internal_taxa)
                minimums = {}
                for external in self.external_isolates:
                    distances = matrix_df[external]
                    filtered_distances = distances.loc[
                        matrix_df.index.isin(self.internal_isolates)
                    ]
                    minimums[external] = filtered_distances.min() 
                sorted_externals = sorted(minimums.items(), key=lambda i: i[1])
                nearest_externals = [
                    k for k, _ in sorted_externals[:num_externals_to_keep]
                ]
                nearest_external_taxa = {
                    t for t in tree.taxon_namespace if t.label in nearest_externals
                }
                filtered_taxa = internal_taxa.union(nearest_external_taxa)
                tree.retain_taxa(filtered_taxa)
                self.filtered_matrix_message  = ( 
                    f'SNP distance matrix filtered to show only the '
                    f'the nearest {num_externals_to_keep} external isolate(s).'
                )
            else:
                tree = None
                self.filtered_matrix = None
                self.filtered_matrix_message = (
                    f'SNP distance matrix not displayed since there are over '
                    f'{self.MAX_MATRIX_SIZE} internal isolates in the cluster.'
                ) 
        if tree is not None:
            matrix = tree.phylogenetic_distance_matrix().as_data_table()
            matrix_df = pd.DataFrame.from_records(matrix._data)
            matrix_df.columns = matrix_df.columns.astype(str)
            matrix_df.index = matrix_df.index.astype(str)
            matrix_df = matrix_df.astype(int)

            # Sort matrix based on preorder tree traversal
            taxa = [n.taxon for n in tree if n.taxon is not None]
            labels = [n.label for n in taxa]
            matrix_df = matrix_df.reindex(index=labels, columns=labels)

            return matrix_df
        return None


def create_clusters(
        sample_sheet_df: pd.DataFrame,
        isolates_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        max_cluster_size: int | None = None,
    ) -> list[cluster.Cluster]:
    """
    Create list of all of the Clusters and their associated isolates.
    """
    clusters: list[cluster.Cluster] = []
    logger.info('Creating clusters...')
    indexed_clusters_df = clusters_df.set_index('cluster')
    if 'total_count' in indexed_clusters_df.columns:
        total_counts = indexed_clusters_df['total_count']
    elif {'internal_count', 'external_count'}.issubset(indexed_clusters_df.columns):
        # total_count isn't persisted to the clusters CSV used by --retry, so
        # fall back to the counts from the previous run.
        total_counts = (
            indexed_clusters_df['internal_count']
            + indexed_clusters_df['external_count']
        )
    else:
        total_counts = None
    for cluster_name in tqdm.tqdm(clusters_df['cluster'].tolist()):
        internal_isolates = isolates_df[
            (isolates_df['cluster'] == cluster_name)
            & (isolates_df['biosample'].isin(sample_sheet_df.index))
        ]['target_acc'].tolist()
        total_count = (
            int(total_counts[cluster_name]) if total_counts is not None else None
        )
        clusters.append(cluster.Cluster(
            cluster_name,
            internal_isolates,
            total_count=total_count,
            max_cluster_size=max_cluster_size,
        ))
    return clusters
