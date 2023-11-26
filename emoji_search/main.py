from appdirs import user_data_dir
import argparse
import os
import pickle

import clip
import numpy as np
import pyperclip
from scipy.spatial.distance import cosine
from sentence_transformers.cross_encoder import CrossEncoder
from tabulate import tabulate
import torch

PACKAGE_NAME = "emoji_search"

DATA_DIR = user_data_dir(PACKAGE_NAME)
EMBEDDINGS_FILE_NAME = "emoji_embeddings.pkl"

EMBEDDINGS_PATH = os.path.join(DATA_DIR, EMBEDDINGS_FILE_NAME)

cross_encoder_name = "cross-encoder/stsb-distilroberta-base"
embedding_model_name = "clip-vit-base32-torch"


import requests


def download_embeddings():
    if not os.path.isfile(EMBEDDINGS_PATH):
        # If the file doesn't exist, download it
        print(f"Downloading embeddings to {EMBEDDINGS_PATH}...")
        os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the DATA_DIR exists

        # Download the file
        file_id = "1ZaBN4rPRTautvk62e6l1UUBtIy51A_HQ"
        URL = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={"id": file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with open(EMBEDDINGS_PATH, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        print("Download complete.")


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def load_embeddings():
    # Ensure the embeddings are downloaded
    download_embeddings()

    # Load the embeddings
    embeddings = pickle.load(open(EMBEDDINGS_PATH, "rb"))
    return embeddings


def clip_embed_query(query):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device)

    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    return text_features.detach().cpu().numpy()[0]


def _get_basic_search_results(query, embeddings, top_2n=30):
    query_embedding = clip_embed_query(query)

    for name, props in embeddings.items():
        image_embedding = props["image_embedding"]
        name_embedding = props["name_embedding"]

        image_dist = cosine(query_embedding, image_embedding)
        name_dist = cosine(query_embedding, name_embedding)
        props["image_dist"] = image_dist
        props["name_dist"] = name_dist

        props["name"] = name
        props["description"] = props["description"]

    results = sorted(embeddings.values(), key=lambda x: x["image_dist"])

    basic_results = results[:top_2n]
    basic_results = [
        {
            "name": result["name"],
            "unicode": result["unicode"],
            "emoji": result["emoji"],
            "description": result["description"],
            "name_dist": result["name_dist"],
        }
        for result in basic_results
    ]
    return basic_results


def _reciprocal_rank(rank):
    return 1.0 / rank if rank > 0 else 0


def _get_ranks(results):
    ranks = {}
    for i, result in enumerate(results):
        name = result["name"]
        ranks[name] = i + 1
    return ranks


def _fuse_reciprocal_ranks(rank_lists):
    all_rank_ids = set()
    for ranks in rank_lists:
        all_rank_ids.update(ranks.keys())

    max_rank = len(all_rank_ids) + 1
    fused_ranks = {rid: 0 for rid in all_rank_ids}

    for ranks in rank_lists:
        for rid in all_rank_ids:
            rank = ranks.get(rid, max_rank)
            fused_ranks[rid] += _reciprocal_rank(rank)

    return sorted(fused_ranks, key=fused_ranks.get, reverse=True)


def _refine_search_results(prompt, basic_results):
    threshold = 0.1
    cross_encoder = CrossEncoder(cross_encoder_name)
    desc_corpus = [result["description"] for result in basic_results]
    name_corpus = [result["name"] for result in basic_results]

    desc_sentence_pairs = [
        [prompt, description] for description in desc_corpus
    ]

    name_sentence_pairs = [[prompt, name] for name in name_corpus]

    desc_scores = cross_encoder.predict(desc_sentence_pairs)
    name_scores = cross_encoder.predict(name_sentence_pairs)

    desc_sim_scores_argsort = reversed(np.argsort(desc_scores))
    name_sim_scores_argsort = reversed(np.argsort(name_scores))

    desc_refined_results = [
        (basic_results[i], desc_scores[i])
        for i in desc_sim_scores_argsort
        if desc_scores[i] > threshold
    ]
    name_refined_results = [
        (basic_results[i], name_scores[i])
        for i in name_sim_scores_argsort
        if name_scores[i] > threshold
    ]

    name_embedding_results = sorted(
        name_refined_results,
        key=lambda x: x[0]["name_dist"],
    )

    desc_ranks = _get_ranks([result[0] for result in desc_refined_results])
    name_ranks = _get_ranks([result[0] for result in name_refined_results])
    image_emb_ranks = _get_ranks(basic_results)
    name_emb_ranks = _get_ranks(
        [result[0] for result in name_embedding_results]
    )

    ranks_list = [desc_ranks, name_ranks, image_emb_ranks, name_emb_ranks]
    fused_ranks = _fuse_reciprocal_ranks(ranks_list)
    return fused_ranks


def _get_result_props(name, embeddings):
    unicode = embeddings[name]["unicode"]
    emoji = embeddings[name]["emoji"]
    props = {
        "name": name,
        "unicode": unicode,
        "emoji": emoji,
    }
    return props


def search(query, top_n=5):
    embeddings = load_embeddings()

    raw_query = query.lower()
    formatted_query = f"A photo of {raw_query}"

    top_2n = max(2 * top_n, 30)

    basic_results = _get_basic_search_results(
        formatted_query, embeddings, top_2n=top_2n
    )
    refined_results = _refine_search_results(raw_query, basic_results)

    refined_results = [
        _get_result_props(name, embeddings) for name in refined_results[:top_n]
    ]
    return refined_results


def _print_results(results):
    table_data = [
        (
            result["emoji"],
            result["name"],
            result["unicode"],
        )
        for result in results
    ]
    print(
        tabulate(
            table_data,
            headers=["Emoji", "Name", "Unicode"],
            tablefmt="pretty",
        )
    )


def _get_emoji_from_unicode(unicode_str):
    # Split the string at spaces and convert each part
    emoji_parts = unicode_str.split()
    emoji_chars = [
        chr(int(part.replace("U+", ""), 16)) for part in emoji_parts
    ]
    # Combine parts to form the emoji
    emoji = "".join(emoji_chars)
    return emoji


def main():

    parser = argparse.ArgumentParser(description="Search for emojis.")
    # The number of results is expected to be a separate option
    parser.add_argument(
        "--num_results",
        "-n",
        type=int,
        default=5,
        help="Number of emojis to return",
    )
    # Whether to copy the top result to the clipboard
    parser.add_argument(
        "--copy",
        "-c",
        default=False,
        help="Copy the top result to the clipboard",
    )
    # The search query will consume all other arguments
    parser.add_argument(
        "query", nargs="+", help="The search query for the emoji."
    )

    args = parser.parse_args()

    # Join all query arguments to form the full search query
    search_query = " ".join(args.query)
    num_results = args.num_results

    print(f"Searching for: {search_query}")

    results = search(search_query, top_n=num_results)
    _print_results(results)

    if args.copy:
        top_result = results[0]
        unicode = top_result["unicode"]
        emoji = _get_emoji_from_unicode(unicode)
        pyperclip.copy(emoji)
        print(f"Copied {emoji} to clipboard.")


if __name__ == "__main__":
    main()
