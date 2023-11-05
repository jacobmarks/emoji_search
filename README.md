# Emoji Search

Emoji Search is a Python-based CLI (Command Line Interface) application that allows users to semantically search emojis.

## Installation

You can use the package manager [pip](https://pip.pypa.io/en/stable/) to install emoji_search.

```bash
pip install emoji_search
```

Or you can install it from source:

````bash
pip install git+https://github.com/jacobmarks/emoji_search.git
```

## Usage

From the command line, use the `emoji-search` command, followed by a search term.

```bash
emoji-search beautiful sunset
````

This will return a list of emojis that most closely match the search term:

```plaintext
+-------+------------------------+---------+----------+
| Emoji |          Name          | Unicode | Distance |
+-------+------------------------+---------+----------+
|  🌇   |         sunset         | U+1F307 |  0.025   |
|  🌅   |        sunrise         | U+1F305 |  0.051   |
|  🌄   | sunrise over mountains | U+1F304 |  0.111   |
|  🌆   |   cityscape at dusk    | U+1F306 |  0.152   |
|   ☀   |          sun           | U+2600  |  0.165   |
+-------+------------------------+---------+----------+
```

You can use quotation marks around your search term if you would like, but it is not necessary.

### Flags

You can specify the number of results you would like to see by using the `-n` or `--num_results` flag.

```bash

emoji-search -n 3 sleepy
```

This will return the top 3 results:

```plaintext
+-------+---------------+---------+----------+
| Emoji |     Name      | Unicode | Distance |
+-------+---------------+---------+----------+
|  💤   |      zzz      | U+1F4A4 |  0.042   |
|  😪   |  sleepy face  | U+1F62A |  0.042   |
|  😴   | sleeping face | U+1F634 |  0.071   |
+-------+---------------+---------+----------+
```

## How it Works

Emoji Search is a semantic search engine using the [CLIP](https://github.com/openai/CLIP)
model from OpenAI. CLIP is a neural network trained on a variety of image-text pairs, and
is able to semantically match images and text.

To match emojis, Emoji Search attempts three different methods:

1. Compare the embedding for the raw search term to the embeddings for all names
   of emojis.
2. Compare the embedding of the formatted search query
   "An emoji of <search-query>" to the embeddings of similarly formatted emoji
   names.
3. Compare the embedding of the formatted search query with the embeddings of
   high-resolution images of emojis.

The results from each of these methods are combined and sorted by distance to
the search query. The distance is calculated using cosine similarity.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
