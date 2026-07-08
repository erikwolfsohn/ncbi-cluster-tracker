import hashlib
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_CONTEXT_LIMITS = {"anthropic": 200_000, "openai": 128_000}
# Headroom reserved for system prompt, output tokens, and tokenizer drift
_RESERVED_TOKENS = 5_000

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

AI_CACHE_PATH = ".ncbi_cluster_tracker_ai_cache.json"

_TIKTOKEN_FALLBACK_LOGGED = False


def _cache_key(prompt: str) -> str:
    return hashlib.sha256((SYSTEM_PROMPT + prompt).encode()).hexdigest()


def _load_cache(cache_path: str) -> dict:
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict, cache_path: str) -> None:
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _n_tokens(text: str) -> int:
    global _TIKTOKEN_FALLBACK_LOGGED
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model("gpt-4o").encode(text))
    except Exception:
        if not _TIKTOKEN_FALLBACK_LOGGED:
            logger.debug("tiktoken not available; using character-based token estimate")
            _TIKTOKEN_FALLBACK_LOGGED = True
        return len(text) // 4


def _prompt_budget(provider: str) -> int:
    return _CONTEXT_LIMITS.get(provider, 128_000) - _RESERVED_TOKENS


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


def _get_openai_api_key() -> str | None:
    vault_url = os.environ.get("AZURE_KEY_VAULT_URL")
    if vault_url:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError:
            logger.warning(
                "AZURE_KEY_VAULT_URL is set but azure packages are not installed; "
                "install them with: pip install 'ncbi-cluster-tracker[azure]'"
            )
            return None
        secret_name = os.environ.get("AZURE_OPENAI_SECRET_NAME")
        if not secret_name:
            logger.warning("AZURE_KEY_VAULT_URL is set but AZURE_OPENAI_SECRET_NAME is not; skipping AI summary")
            return None
        try:
            credential = DefaultAzureCredential(additionally_allowed_tenants=["*"])
            client = SecretClient(vault_url=vault_url, credential=credential)
            return client.get_secret(secret_name).value
        except Exception as exc:
            logger.warning(f"Failed to retrieve OpenAI API key from Azure Key Vault: {exc}")
            return None
    return os.environ.get("OPENAI_API_KEY") or None


def _get_openai_client() -> object | None:
    try:
        import openai
    except ImportError:
        logger.warning(
            "openai package not installed; "
            "install it with: pip install 'ncbi-cluster-tracker[openai]'"
        )
        return None
    api_key = _get_openai_api_key()
    if not api_key:
        logger.warning("No OpenAI API key found; skipping AI summary")
        return None
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        return openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return openai.OpenAI(api_key=api_key)


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
        logger.info(
            f"Anthropic token usage — input: {response.usage.input_tokens}, "
            f"output: {response.usage.output_tokens}"
        )
        return response.content[0].text
    if provider == "openai":
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        logger.info(
            f"OpenAI token usage — input: {response.usage.prompt_tokens}, "
            f"output: {response.usage.completion_tokens}"
        )
        return response.choices[0].message.content
    raise ValueError(f"Unknown provider: {provider}")


_GLOBAL_PRIORITY_COLS = [
    "cluster", "taxgroup_name", "internal_count", "external_count", "change", "latest_added",
]
_GLOBAL_LOW_PRIORITY_COLS = ["earliest_year_collected", "latest_year_collected", "tree_url"]


def _build_global_prompt(clusters_df: pd.DataFrame, token_budget: int) -> str | None:
    header = (
        f"The following table summarizes {len(clusters_df)} SNP cluster(s) detected for a public health "
        f"laboratory. Write a concise surveillance summary (3-5 bullet points) suitable for a weekly "
        f"situation report. Highlight any clusters of concern, growth trends, new clusters, and "
        f"epidemiological patterns.\n\nClusters table:\n"
    )
    table_budget = token_budget - _n_tokens(header)

    priority_cols = [c for c in _GLOBAL_PRIORITY_COLS if c in clusters_df.columns]
    all_cols = priority_cols + [c for c in _GLOBAL_LOW_PRIORITY_COLS if c in clusters_df.columns]
    n = len(clusters_df)

    for cols, n_rows in [
        (all_cols, n),
        (priority_cols, n),
        (priority_cols, max(n // 2, 1)),
        (priority_cols, 10),
    ]:
        table = clusters_df[cols].head(n_rows).to_csv(index=False)
        if _n_tokens(table) <= table_budget:
            if n_rows < n:
                logger.info(f"Global prompt: truncated clusters table to {n_rows}/{n} rows to fit context window")
            return header + table

    logger.warning("Clusters table too large for context window even after truncation; skipping global AI summary")
    return None


_CLUSTER_LOW_PRIORITY_COLS = ["sra_id", "bioproject_acc", "target_acc"]


def _build_cluster_prompt(
    cluster_name: str,
    cluster_row: pd.Series,
    isolates_df: pd.DataFrame,
    token_budget: int,
) -> str:
    cluster_info = cluster_row.to_frame().T.to_csv(index=False)
    n_isolates = len(isolates_df)
    n_internal = int((isolates_df["source"] == "internal").sum()) if "source" in isolates_df.columns else "unknown"
    n_external = int((isolates_df["source"] == "external").sum()) if "source" in isolates_df.columns else "unknown"
    n_new = int((isolates_df["is_new"] == "yes").sum()) if "is_new" in isolates_df.columns else "unknown"

    header = (
        f"Write a 2-3 sentence summary of cluster '{cluster_name}' for a public health report. "
        f"Include: organism name, internal vs external isolate counts, new isolates if any, "
        f"geographic distribution if apparent, and whether this cluster warrants attention.\n\n"
        f"Cluster metadata:\n{cluster_info}\n\n"
        f"Isolates ({n_isolates} total; {n_internal} internal, {n_external} external; {n_new} new):\n"
    )
    table_budget = token_budget - _n_tokens(header)

    low_pri = [c for c in _CLUSTER_LOW_PRIORITY_COLS if c in isolates_df.columns]
    df = isolates_df.drop(columns=low_pri)
    n = len(df)

    for n_rows in [n, max(n // 2, 1), 10, 0]:
        if n_rows == 0:
            logger.info(f"Cluster '{cluster_name}': omitting isolates table to fit context window")
            return header + "(isolates table omitted: too large for context window)"
        table = df.head(n_rows).to_csv(index=False)
        if _n_tokens(table) <= table_budget:
            if n_rows < n:
                logger.info(f"Cluster '{cluster_name}': truncated isolates table to {n_rows}/{n} rows")
            return header + table

    return header + "(isolates table omitted: too large for context window)"


def summarize_global(
    clusters_df: pd.DataFrame,
    provider: str | None = None,
    use_cache: bool = True,
) -> str | None:
    """Generate a global AI summary of all clusters. Returns markdown text or None on failure."""
    result = get_client(provider)
    if result is None:
        return None
    client, detected_provider = result
    try:
        budget = _prompt_budget(detected_provider)
        prompt = _build_global_prompt(clusters_df, budget)
        if prompt is None:
            return None
        if use_cache:
            cache = _load_cache(AI_CACHE_PATH)
            key = _cache_key(prompt)
            if key in cache:
                logger.info("Global prompt: using cached AI summary")
                return cache[key]
        estimated = _n_tokens(prompt) + _n_tokens(SYSTEM_PROMPT)
        logger.info(f"Global prompt: ~{estimated} estimated tokens (budget: {budget})")
        result_text = _call_llm(client, detected_provider, prompt, max_tokens=512)
        if use_cache:
            cache[key] = result_text
            _save_cache(cache, AI_CACHE_PATH)
        return result_text
    except Exception as exc:
        logger.warning(f"AI global summary failed: {exc}")
        return None


def summarize_cluster(
    cluster_name: str,
    cluster_row: pd.Series,
    isolates_df: pd.DataFrame,
    provider: str | None = None,
    use_cache: bool = True,
) -> str | None:
    """Generate an AI summary for a single cluster. Returns markdown text or None on failure."""
    result = get_client(provider)
    if result is None:
        return None
    client, detected_provider = result
    try:
        budget = _prompt_budget(detected_provider)
        prompt = _build_cluster_prompt(cluster_name, cluster_row, isolates_df, budget)
        if use_cache:
            cache = _load_cache(AI_CACHE_PATH)
            key = _cache_key(prompt)
            if key in cache:
                logger.info(f"Cluster '{cluster_name}': using cached AI summary")
                return cache[key]
        estimated = _n_tokens(prompt) + _n_tokens(SYSTEM_PROMPT)
        logger.info(f"Cluster '{cluster_name}' prompt: ~{estimated} estimated tokens (budget: {budget})")
        result_text = _call_llm(client, detected_provider, prompt, max_tokens=256)
        if use_cache:
            cache[key] = result_text
            _save_cache(cache, AI_CACHE_PATH)
        return result_text
    except Exception as exc:
        logger.warning(f"AI summary for cluster '{cluster_name}' failed: {exc}")
        return None
