import argparse

from importlib.metadata import version

from typing import Sequence

from ncbi_cluster_tracker import download

def parse_args(command: Sequence[str]) -> argparse.Namespace:
    """
    Parse command-line arguments from the user.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'sample_sheet',
        help='Path to sample sheet CSV with required "biosample" column and any additional metadata columns. Use "id" column for alternate isolate IDs.',
    )
    parser.add_argument(
        '--out-dir', '-o',
        help='Path to directory to store outputs. Defaults to "./outputs/" if not specified.'
    )
    parser.add_argument(
        '--retry',
        help='Do not query BigQuery or NCBI, assumes data has already been downloaded to --out-dir.',
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        '--browser-file',
        # TODO link to instructions
        help='Path to isolates TSV or CSV downloaded from the Pathogen Detection Isolates Browser with information for all internal and external isolates. When specified, data in file will be used instead of querying the BigQuery dataset.'
    )
    parser.add_argument(
        '--keep-snp-files',
        help='Keep downloaded SNP and tree files in the output directory. By default, files are deleted after processing.',
        action='store_true',
    )
    parser.add_argument(
        '--amr',
        help='Include AMR tab in report with antimicrobial resistance genes detected by AMRFinderPlus.',
        action='store_true',
    )
    parser.add_argument(
        '--filter-amr',
        help='Only include AMR genes in provided comma-separated list of CLASS:SUBCLASS pairs in the AMR tab. Also adds filtered_amr column to Isolates and Cluster details tab and matching genes to tree labels',
        type=lambda s: [i for i in s.split(',')],
    )
    parser.add_argument(
        '--version', '-v',
        help='Print the version of ncbi-cluster-tracker and exit.',
        action='version',
        version=version('ncbi-cluster-tracker'),
    )
    parser.add_argument(
        '--ai-summary',
        help='Generate AI summaries for clusters using an LLM. Requires ANTHROPIC_API_KEY or OPENAI_API_KEY.',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--ai-provider',
        help='LLM provider to use for AI summaries. Auto-detected from environment variables if not specified.',
        choices=['anthropic', 'openai'],
        default=None,
    )
    parser.add_argument(
        '--no-ai-cache',
        help='Disable the AI summary cache and always call the LLM, even for unchanged clusters.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--no-card',
        help='Skip downloading the CARD reference index when generating AMR AI summaries.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--max-cluster-size',
        help='Skip downloading and building the SNP tree/distance matrix for any '
             'cluster with more than this many isolates, to avoid excessive memory '
             'usage and download times for extremely large clusters. Set to 0 to '
             'disable this limit. (default: %(default)s)',
        type=int,
        default=20000,
    )
    parser.add_argument(
        '--download-timeout',
        help='Timeout in seconds for each SNP tree download request. '
             '(default: %(default)s)',
        type=float,
        default=download.DEFAULT_DOWNLOAD_TIMEOUT,
    )
    parser.add_argument(
        '--download-chunk-size',
        help='Chunk size in bytes used when streaming SNP tree downloads to disk. '
             '(default: %(default)s)',
        type=int,
        default=download.DEFAULT_DOWNLOAD_CHUNK_SIZE,
    )
    mutex_group_compare = parser.add_mutually_exclusive_group()
    mutex_group_compare.add_argument(
        '--compare-dir',
        help='Path to previous output directory to detect and report new isolates.',
    )
    args = parser.parse_args(command)

    if args.retry and not args.out_dir:
        parser.error('--retry flag requires --out_dir argument')

    if args.max_cluster_size < 0:
        parser.error('--max-cluster-size must be 0 (no limit) or a positive integer')

    if args.download_timeout <= 0:
        parser.error('--download-timeout must be a positive number')

    if args.download_chunk_size <= 0:
        parser.error('--download-chunk-size must be a positive integer')

    if args.filter_amr:
        if not args.amr:
            parser.error('--filter-amr argument requires --amr flag')
        for item in args.filter_amr:
            if ':' not in item[1:-1]:
                parser.error('Each element in --filter-amr list must be in the form CLASS:SUBCLASS')
    
    return args

