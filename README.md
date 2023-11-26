# Emoji Search

![emoji_search_CLI](https://github.com/jacobmarks/emoji_search/assets/12500356/cdc2e1ca-3243-495f-9128-c37c03c42848)

Emoji Search is a Python-based CLI (Command Line Interface) application that allows users to semantically search emojis.

## Installation

You can install it from source:

```bash
pip install git+https://github.com/jacobmarks/emoji_search.git
```

## Usage

From the command line, use the `emoji-search` command, followed by a search term.

```bash
emoji-search beautiful sunset
```

This will return a list of emojis that most closely match the search term:

```plaintext
+-------+-------------------+---------+
| Emoji |       Name        | Unicode |
+-------+-------------------+---------+
|  🌞   |   sun with face   | U+1F31E |
|  🌇   |      sunset       | U+1F307 |
|  🌅   |      sunrise      | U+1F305 |
|  🔆   |   bright button   | U+1F506 |
|  🌆   | cityscape at dusk | U+1F306 |
+-------+-------------------+---------+
```

You can use quotation marks around your search term if you would like, but it is not necessary.

### Flags

You can specify the number of results you would like to see by using the `-n` or `--num_results` flag.

```bash

emoji-search -n 3 sleepy
```

This will return the top 3 results:

```plaintext
+-------+-------------------+---------+
| Emoji |       Name        | Unicode |
+-------+-------------------+---------+
|  🌞   |   sun with face   | U+1F31E |
|  🌇   |      sunset       | U+1F307 |
|  🌅   |      sunrise      | U+1F305 |
+-------+-------------------+---------+
```

You can also add `-c` or `--copy` to copy the top result to your clipboard.

```bash
emoji-search -c happy
```

```plaintext
Searching for: a happy family
+----------+---------------------------------+------------------------------------------------------+
|  Emoji   |              Name               |                       Unicode                        |
+----------+---------------------------------+------------------------------------------------------+
|    👪    |             family              |                       U+1F46A                        |
|    👨‍👩‍👧    |    family: man, woman, girl     |        U+1F468 U+200D U+1F469 U+200D U+1F467         |
|    👩‍👦‍👦    |     family: woman, boy, boy     |        U+1F469 U+200D U+1F466 U+200D U+1F466         |
|    👩‍👩‍👧‍👦    | family: woman, woman, girl, boy | U+1F469 U+200D U+1F469 U+200D U+1F467 U+200D U+1F466 |
|    👩‍👩‍👦‍👦    | family: woman, woman, boy, boy  | U+1F469 U+200D U+1F469 U+200D U+1F466 U+200D U+1F466 |
+----------+---------------------------------+------------------------------------------------------+
Copied 👪 to clipboard.
```

## How it Works

Emoji Search is a semantic search engine that uses a three-step process to find the most relevant emojis for a given search query.

### Step 1: Rapid Filtering with CLIP

The first step is a top-level sieve that uses the [CLIP](https://github.com/openai/CLIP)
model from OpenAI. CLIP is a neural network trained on a variety of image-text pairs, and
is able to semantically match images and text.

We compare the embedding of the query to embeddings of images of emojis, generated with ESRGAN 10x upscaling
of the base64-encoded emoji images. This is done using cosine similarity, and the top candidates are
selected to move on to the next step.

### Step 2: Cross-Encoder and Re-Ranking

The second step uses a cross-encoder model, DistilRoBERTa, to rank the candidates from the first step.

Prior to inference time, we generated captions/descriptions of the emojis with GPT-4, which are of the form "A photo of ...",
and post-process them to remove the "A photo of" prefix. We then use the cross-encoder to rank the emojis.

We compare the query against two inputs: the emoji name and the emoji description. Separate rankings
are generated for each of these types of inputs. We also rank the emojis by the similarity of the
query's CLIP embedding to embeddings of the descriptions of the emojis.

### Step 3: Reciprocal Rank Fusion

The third step combines the four rankings from the first and second steps using reciprocal rank fusion. This
results in a final ranking of the emojis, which is returned to the user, potentially truncated
based on the number of results requested.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
