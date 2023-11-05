# import json
import pickle
from tabulate import tabulate
from appdirs import user_data_dir

import argparse
import os

from scipy.spatial.distance import cosine
import torch
import clip

PACKAGE_NAME = "emoji_search"

DATA_DIR = user_data_dir(PACKAGE_NAME)
EMBEDDINGS_FILE_NAME = "emoji_embeddings.pkl"

EMBEDDINGS_PATH = os.path.join(DATA_DIR, EMBEDDINGS_FILE_NAME)


import requests


def download_embeddings():
    if not os.path.isfile(EMBEDDINGS_PATH):
        # If the file doesn't exist, download it
        print(f"Downloading embeddings to {EMBEDDINGS_PATH}...")
        os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the DATA_DIR exists

        # Download the file
        file_id = "1JYM1lhgBzD9C34V-BICTzEhbW-JI9rPP"
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


def embed_query(query):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device)

    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features.detach().cpu().numpy()[0]


def search(query, top_n=5):
    embeddings = load_embeddings()

    raw_query = query.lower()
    emoji_of_text_query = f"An emoji of {raw_query}"

    emoji_of_text_query_embedding = embed_query(emoji_of_text_query)

    emoji_of_text_distances = []
    image_distances = []

    for name, props in embeddings.items():
        emoji_of_text_embedding = props["emoji_of_text_embedding"]
        image_embedding = props["image_embedding"]

        emoji_of_text_distance = cosine(
            emoji_of_text_query_embedding, emoji_of_text_embedding
        )
        image_distance = cosine(emoji_of_text_query_embedding, image_embedding)
        emoji_of_text_distances.append((name, emoji_of_text_distance))
        image_distances.append((name, image_distance))

    emoji_of_text_distances = sorted(
        emoji_of_text_distances, key=lambda x: x[1]
    )
    image_distances = sorted(image_distances, key=lambda x: x[1])

    emoji_of_text_kds = emoji_of_text_distances[:top_n]
    image_kds = image_distances[:top_n]
    kds = emoji_of_text_kds + image_kds

    results = []
    keys = []

    for kd in kds:
        name = kd[0]
        if name in keys:
            continue
        keys.append(name)
        dist = kd[1]
        unicode = embeddings[name]["unicode"]
        emoji = embeddings[name]["emoji"]
        props = {
            "name": name,
            "dist": dist,
            "unicode": unicode,
            "emoji": emoji,
        }
        results.append(props)

    results = sorted(results, key=lambda x: x["dist"])
    return results[:top_n]


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

    # Truncate distance to 3 decimal places and prepare the table data
    table_data = [
        (
            result["emoji"],
            result["name"],
            result["unicode"],
            f"{result['dist']:.3f}",
        )
        for result in results
    ]
    print(
        tabulate(
            table_data,
            headers=["Emoji", "Name", "Unicode", "Distance"],
            tablefmt="pretty",
        )
    )


if __name__ == "__main__":
    main()
