import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert in infectious disease surveillance and public health microbiology. "
    "You analyze data from NCBI's Pathogen Detection system, which clusters closely related "
    "bacterial isolates by SNP distance to detect potential outbreaks.\n\n"
    "Key concepts:\n"
    "- Clusters: groups of closely related isolates with identical or nearly identical genomes\n"
    "- Internal isolates: the laboratory's own samples submitted to NCBI\n"
    "- External isolates: other labs' samples that NCBI has grouped with the internal ones\n"
    "- New isolates: isolates not present in a previous comparison run\n"
    "- SNP distance: single nucleotide polymorphism differences; lower means more closely related\n"
    "- AMR: antimicrobial resistance genes detected by AMRFinderPlus\n\n"
    "Your summaries should be concise, factual, and highlight epidemiological significance. "
    "Focus on cluster growth trends, new isolates, geographic spread, AMR concerns, and clusters "
    "that warrant attention. Avoid speculation; stick to what the data shows."
)


def _get_anthropic_client() -> object | None:
    try:
        import anthropic
    except ImportError:
        logger.warning(
            "anthropic package not installed; "
            "install it with: pip install 'ncbi-cluster-tracker[anthropic]'"
        )
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set; skipping AI summary")
        return None
    return anthropic.Anthropic()


def _get_openai_client() -> object | None:
    try:
        import openai
    except ImportError:
        logger.warning(
            "openai package not installed; "
            "install it with: pip install 'ncbi-cluster-tracker[openai]'"
        )
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set; skipping AI summary")
        return None
    return openai.OpenAI()


def get_client(provider: str | None = None) -> tuple[object, str] | None:
    """Return (client, provider_name) or None. Auto-detects from env vars if provider is None."""
    if provider == "anthropic":
        client = _get_anthropic_client()
        return (client, "anthropic") if client else None
    if provider == "openai":
        client = _get_openai_client()
        return (client, "openai") if client else None
    client = _get_anthropic_client()
    if client:
        return (client, "anthropic")
    client = _get_openai_client()
    if client:
        return (client, "openai")
    return None


def _call_llm(client: object, provider: str, user_prompt: str, max_tokens: int = 512) -> str:
    if provider == "anthropic":
        response = client.messages.create(  # type: ignore[union-attr]
            model="claude-opus-4-8",
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    if provider == "openai":
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    raise ValueError(f"Unknown provider: {provider}")


def _build_global_prompt(clusters_df: pd.DataFrame) -> str:
    cols_to_show = [c for c in [
        "cluster", "taxgroup_name", "internal_count", "external_count",
        "change", "latest_added", "earliest_year_collected", "latest_year_collected",
    ] if c in clusters_df.columns]
    table = clusters_df[cols_to_show].to_string(index=False, max_rows=50)
    return (
        f"The following table summarizes {len(clusters_df)} SNP cluster(s) detected for a public health "
        f"laboratory. Write a concise surveillance summary (3-5 bullet points) suitable for a weekly "
        f"situation report. Highlight any clusters of concern, growth trends, new clusters, and "
        f"epidemiological patterns.\n\nClusters table:\n{table}"
    )


def _build_cluster_prompt(
    cluster_name: str,
    cluster_row: pd.Series,
    isolates_df: pd.DataFrame,
) -> str:
    cluster_info = cluster_row.to_string()
    n_isolates = len(isolates_df)
    n_internal = int((isolates_df["source"] == "internal").sum()) if "source" in isolates_df.columns else "unknown"
    n_external = int((isolates_df["source"] == "external").sum()) if "source" in isolates_df.columns else "unknown"
    n_new = int((isolates_df["is_new"] == "yes").sum()) if "is_new" in isolates_df.columns else "unknown"
    isolate_preview = isolates_df.head(20).to_string(index=False)
    return (
        f"Write a 2-3 sentence summary of cluster '{cluster_name}' for a public health report. "
        f"Include: organism name, internal vs external isolate counts, new isolates if any, "
        f"geographic distribution if apparent, and whether this cluster warrants attention.\n\n"
        f"Cluster metadata:\n{cluster_info}\n\n"
        f"Isolates ({n_isolates} total; {n_internal} internal, {n_external} external; {n_new} new):\n"
        f"{isolate_preview}"
    )


def summarize_global(
    clusters_df: pd.DataFrame,
    provider: str | None = None,
) -> str | None:
    """Generate a global AI summary of all clusters. Returns markdown text or None on failure."""
    result = get_client(provider)
    if result is None:
        return None
    client, detected_provider = result
    try:
        prompt = _build_global_prompt(clusters_df)
        return _call_llm(client, detected_provider, prompt, max_tokens=512)
    except Exception as exc:
        logger.warning(f"AI global summary failed: {exc}")
        return None


def summarize_cluster(
    cluster_name: str,
    cluster_row: pd.Series,
    isolates_df: pd.DataFrame,
    provider: str | None = None,
) -> str | None:
    """Generate an AI summary for a single cluster. Returns markdown text or None on failure."""
    result = get_client(provider)
    if result is None:
        return None
    client, detected_provider = result
    try:
        prompt = _build_cluster_prompt(cluster_name, cluster_row, isolates_df)
        return _call_llm(client, detected_provider, prompt, max_tokens=256)
    except Exception as exc:
        logger.warning(f"AI summary for cluster '{cluster_name}' failed: {exc}")
        return None
