import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from src.config import settings

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_K             = 5
DEFAULT_MIN_RESPONSES = 4
DATASET_NAME          = "Amod/mental_health_counseling_conversations"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(local_path: str = None) -> pd.DataFrame:
    if local_path:
        print(f"Loading dataset from local file: {local_path} ...")
        df = pd.read_csv(local_path, encoding="utf-8-sig")
    else:
        print(f"Loading dataset from HuggingFace: {DATASET_NAME} ...")
        ds = load_dataset(DATASET_NAME, split="train")
        df = ds.to_pandas()

    print(f"  Loaded {len(df):,} rows | {df['Context'].nunique():,} unique contexts\n")
    return df


# ── Embed ─────────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, batch_size=64, normalize_embeddings=True)


# ── Cluster ───────────────────────────────────────────────────────────────────

def cluster_and_pick(responses: list[str], embeddings: np.ndarray, k: int) -> list[str]:
    n        = len(responses)
    actual_k = min(k, n)

    if actual_k == n:
        return responses

    km = KMeans(n_clusters=actual_k, random_state=42, n_init="auto")
    km.fit(embeddings)

    picked = []
    for centroid in km.cluster_centers_:
        sims     = cosine_similarity([centroid], embeddings)[0]
        best_idx = int(np.argmax(sims))
        picked.append(responses[best_idx])

    return picked


# ── Main ──────────────────────────────────────────────────────────────────────

def main(k: int, min_responses: int, output: str, local_path: str = None):
    df = load_data(local_path=local_path)

    print(f"Loading embedding model: {settings.embedding_model} ...")
    model = SentenceTransformer(settings.embedding_model)
    print("  Model ready.\n")

    grouped         = df.groupby("Context", sort=False)
    unique_contexts = list(grouped.groups.keys())
    print(f"Processing {len(unique_contexts):,} unique contexts (k={k}) ...\n")

    result_rows = []
    stats       = {"clustered": 0, "kept_all": 0, "single": 0}

    for context in tqdm(unique_contexts, desc="Clustering"):
        group     = grouped.get_group(context)
        responses = group["Response"].tolist()
        n         = len(responses)

        if n == 1:
            result_rows.append({"Context": context, "Response": responses[0]})
            stats["single"] += 1

        elif n < min_responses or n <= k:
            for r in responses:
                result_rows.append({"Context": context, "Response": r})
            stats["kept_all"] += 1

        else:
            embs   = embed_texts(responses, model)
            picked = cluster_and_pick(responses, embs, k)
            for r in picked:
                result_rows.append({"Context": context, "Response": r})
            stats["clustered"] += 1

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output, index=False, encoding="utf-8")

    print(f"\n{'─'*55}")
    print(f"  Original rows      : {len(df):,}")
    print(f"  Output rows        : {len(result_df):,}")
    print(f"  Unique contexts    : {result_df['Context'].nunique():,}")
    print(f"  Reduction          : {100*(1 - len(result_df)/len(df)):.1f}%")
    print(f"  Contexts clustered : {stats['clustered']:,}")
    print(f"  Contexts kept all  : {stats['kept_all']:,}")
    print(f"  Single-response    : {stats['single']:,}")
    print(f"  Saved to           : {output}")
    print(f"{'─'*55}\n")

    print("Sample output (first context with multiple representatives):\n")
    multi = result_df.groupby("Context").filter(lambda x: len(x) > 1)
    if not multi.empty:
        sample_ctx  = multi.iloc[0]["Context"]
        sample_rows = result_df[result_df["Context"] == sample_ctx]
        print(f"Context:\n  {sample_ctx[:200]}...\n")
        for i, row in enumerate(sample_rows.itertuples(), 1):
            print(f"  Response {i}: {row.Response[:120]}...")
        print()

    return result_df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster counselor responses per context")
    parser.add_argument("--k",             type=int,  default=DEFAULT_K)
    parser.add_argument("--min_responses", type=int,  default=DEFAULT_MIN_RESPONSES)
    parser.add_argument("--output",        type=str,  default="data/mental_health_preprocessed.csv")
    parser.add_argument("--local",         type=str,  default=None,
                        help="Path to local CSV file. If omitted, loads from HuggingFace.")
    args = parser.parse_args()

    main(k=args.k, min_responses=args.min_responses, output=args.output, local_path=args.local)


# # From HuggingFace (default)
# python -m scripts.cluster_responses
#
# # From local CSV
# python -m scripts.cluster_responses --local data/raw_dataset.csv
#
# # All options
# python -m scripts.cluster_responses --local data/raw.csv --k 4 --output data/mental_health_preprocessed.csv