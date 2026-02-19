import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nvidia Learning Deep Learning: Neural Machine Translation

    ## Attribution

    This notebook is sourced from Nvidia's Learning Deep Learning, [LDL](https://github.com/NVDLI/LDL), repository. The repository contains code examples associated with a similarly named book and video series by Magnus Ekman (Book ISBN: 9780137470358; Video: 9780138177553)

    - **Original source**: [gh:NVDLI/LDL/blob/main/pt_framework/v7_7_neural_machine_translation_transformer.ipynb](https://github.com/NVDLI/LDL/blob/main/pt_framework/v7_7_neural_machine_translation_transformer.ipynb)
    - **License**: BSD 3-Clause License
    - **Copyright**: (c) 2021 NVIDIA
    - **Modifications**:
        - Stylistic changes and conversion to Marimo.
        - MPS support placeholder added. Currently uses CPU with Mac due to support issues.
        - Fixed PyTorch deprecation warning by converting boolean padding masks to additive float masks to match attention mask types.
        - Replaced TensorFlow/Keras `Tokenizer` and `text_to_word_sequence` with pure Python equivalents.
        - Pointed students to various data sources instead of the `fra.txt` mentioned in the original source.
        - Refactored the training to use helper functions
        - Added table format of how parameters affects the training memory usage (by ChatGPT)

    **Reason for resharing**: ease of access and integration into a larger collection of educational resources.

    Full MIT license text is below.

    /// attention | Attention!

    The MIT License (MIT)
    Copyright (c) 2021 NVIDIA
    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This code example demonstrates how to build a neural machine translation network. It is a sequence-to-sequence network based on a Transformer encoder-decoder architecture. More context for this code example can be found in video 7.7 "Programming Example: Machine Translation Using Transformer with PyTorch" in the video series "Learning Deep Learning: From Perceptron to Large Language Models" by Magnus Ekman (Video ISBN-13: 9780138177614).



    /// attention | Data!

    The data used to train the model is expected to be in the file ../data/fra.txt.
    ///

    We begin by importing modules that we need for the program.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import random
    import math

    from torch import Tensor
    from torch.nn import Transformer
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.nn import TransformerDecoder, TransformerDecoderLayer
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader

    return (
        DataLoader,
        Tensor,
        TensorDataset,
        Transformer,
        TransformerDecoder,
        TransformerDecoderLayer,
        TransformerEncoder,
        TransformerEncoderLayer,
        math,
        nn,
        np,
        random,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we define some constants. We specify a vocabulary size of 10,000 symbols, out of which four indices are reserved for padding, out-of-vocabulary words (denoted as UNK), START tokens, and STOP tokens. Our training corpus is large, so we set the parameter READ_LINES to the number of lines in the input file we want to use in our example (60,000). The parameter LAYER_SIZE defines the width of the intermediate fully-connected layer in the Transformer, and the embedding layers output 128 dimensions (EMBEDDING_WIDTH). We use 20% (TEST_PERCENT) of the dataset as test set and further select 20 sentences (SAMPLE_SIZE) to inspect in detail during training. We limit the length of the source and destination sentences to, at most, 60 words (MAX_LENGTH). Finally, we provide the path to the data file, where each line is expected to contain two versions of the same sentence (one in each language) separated by a tab character.
    """)
    return


@app.cell
def _(torch):
    if torch.backends.mps.is_available():
        device = torch.device("cpu")
        print("Code raises this error with MPS:")
        print("[ERROR] The operator 'aten::_nested_tensor_from_mask_left_aligned' is not currently implemented for the MPS device")
        print("Thus, on Mac, using CPU instead.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    EPOCHS = 20
    BATCH_SIZE = 128
    MAX_WORDS = 10000
    READ_LINES = 60000
    NUM_HEADS = 8
    LAYER_SIZE = 256
    EMBEDDING_WIDTH = 128
    TEST_PERCENT = 0.2
    SAMPLE_SIZE = 20
    OOV_WORD = 'UNK'
    PAD_INDEX = 0
    OOV_INDEX = 1
    START_INDEX = MAX_WORDS - 2
    STOP_INDEX = MAX_WORDS - 1
    MAX_LENGTH = 60
    SRC_DEST_FILE_NAME = 'data/fin.txt'
    return (
        BATCH_SIZE,
        EMBEDDING_WIDTH,
        EPOCHS,
        LAYER_SIZE,
        MAX_LENGTH,
        MAX_WORDS,
        NUM_HEADS,
        OOV_INDEX,
        OOV_WORD,
        PAD_INDEX,
        READ_LINES,
        SAMPLE_SIZE,
        SRC_DEST_FILE_NAME,
        START_INDEX,
        STOP_INDEX,
        TEST_PERCENT,
        device,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is KAMK addition to the material by ChatGPT. This is a tabular format of how, according to the language model, each parameter affects the memory usage of the model. 🔴 is very high, 🟠 is medium, and 🟡 is low is very low. Those with no effect have been dropped out.

    | Hyperparameter         | What it controls                    | Memory impact level         | How it affects memory                                         | Scaling behavior                          |
    | ---------------------- | ----------------------------------- | --------------------------- | ------------------------------------------------------------- | ----------------------------------------- |
    | **BATCH_SIZE**         | Number of samples per training step | 🔴  | Stores activations & gradients for all samples simultaneously | **Linear** with batch size                |
    | **MAX_LENGTH**         | Max tokens per sequence             | 🔴  | Self-attention memory grows with sequence length              | ~O(L²) for attention + linear activations |
    | **LAYER_SIZE**         | Hidden dimension of Transformer     | 🔴 | Controls size of attention & feed-forward layers              | Roughly **quadratic** with layer size     |
    | **EMBEDDING_WIDTH**    | Embedding vector size               | 🟠 | Affects embedding tables and intermediate tensors             | Linear                                    |
    | **NUM_HEADS**          | Attention heads                     | 🟠 | Splits layer size across heads; adds some overhead            | Small increase if total dim fixed         |
    | **MAX_WORDS**          | Vocabulary size                     | 🟠 | Embedding + output softmax matrices scale with vocab          | Linear with vocab                         |
    | **READ_LINES**         | Dataset size                        | 🟡 | Affects dataset storage, not batch memory                     | Linear in dataset size                    |
    """)
    return


@app.cell
def _():
    from collections import Counter

    def text_to_word_sequence(text):
        """Split text into a list of words. Lowercase, strip punctuation."""
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        text = text.lower()
        text = text.translate(str.maketrans(filters, ' ' * len(filters)))
        return [w for w in text.split() if w]

    class Tokenizer:
        """Minimal drop-in replacement for keras.preprocessing.text.Tokenizer."""
        def __init__(self, num_words=None, oov_token=None):
            self.num_words = num_words
            self.oov_token = oov_token
            self.word_index = {}
            self.index_word = {}

        def fit_on_texts(self, texts):
            word_counts = Counter()
            for text in texts:
                if isinstance(text, list):
                    word_counts.update(text)
                else:
                    word_counts.update(text_to_word_sequence(text))
            idx = 1
            if self.oov_token:
                self.word_index[self.oov_token] = idx
                self.index_word[idx] = self.oov_token
                idx += 1
            for word, _ in word_counts.most_common():
                if word == self.oov_token:
                    continue
                self.word_index[word] = idx
                self.index_word[idx] = word
                idx += 1

        def texts_to_sequences(self, texts):
            sequences = []
            oov_idx = self.word_index.get(self.oov_token, 0) if self.oov_token else 0
            max_idx = self.num_words if self.num_words else float('inf')
            for text in texts:
                words = text if isinstance(text, list) else text_to_word_sequence(text)
                seq = []
                for w in words:
                    idx = self.word_index.get(w, oov_idx)
                    if idx < max_idx:
                        seq.append(idx)
                    elif oov_idx:
                        seq.append(oov_idx)
                sequences.append(seq)
            return sequences

        def sequences_to_texts(self, sequences):
            result = []
            for seq in sequences:
                words = [self.index_word.get(idx, self.oov_token or '') for idx in seq]
                result.append(' '.join(words))
            return result

    def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
        import numpy as np

        if not sequences:
            return np.array([], dtype=dtype)

        if maxlen is None:
            maxlen = np.max([len(s) for s in sequences])

        num_samples = len(sequences)
        sample_shape = tuple()
        x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)

        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found

            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)

        return x

    return Tokenizer, pad_sequences, text_to_word_sequence


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next code snippet shows the function used to read the input data file and do some initial processing. Each line is split into two strings, where the first contains the sentence in the destination language and the second contains the sentence in the source language. We use the function text_to_word_sequence() to clean the data somewhat (make everything lowercase and remove punctuation) and split each sentence into a list of individual words. If the list (sentence) is longer than the maximum allowed length, then it is truncated.
    """)
    return


@app.cell
def _(READ_LINES, text_to_word_sequence):
    # Function to read file.
    def read_file_combined(file_name, max_len):
        file = open(file_name, 'r', encoding='utf-8')
        src_word_sequences = []
        dest_word_sequences = []
        for i, line in enumerate(file):
            if i == READ_LINES:
                break
            pair = line.split('\t')
            word_sequence = text_to_word_sequence(pair[1])
            src_word_sequence = word_sequence[0:max_len]
            src_word_sequences.append(src_word_sequence)
            word_sequence = text_to_word_sequence(pair[0])
            dest_word_sequence = word_sequence[0:max_len]
            dest_word_sequences.append(dest_word_sequence)
        file.close()
        return src_word_sequences, dest_word_sequences

    return (read_file_combined,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next code snippet shows functions used to turn sequences of words into
    sequences of tokens, and vice versa. We call tokenize() a single time for each
    language, so the argument sequences is a list of lists where each of the inner
    lists represents a sentence. The Tokenizer class assigns indices to the most
    common words and returns either these indices or the reserved OOV_INDEX
    for less common words that did not make it into the vocabulary. We tell the
    Tokenizer to use a vocabulary of 9998 (MAX_WORDS-2)—that is, use only
    indices 0 to 9997, so that we can use indices 9998 and 9999 as our START and
    STOP tokens (the Tokenizer does not support the notion of START and STOP
    tokens but does reserve index 0 to use as a padding token and index 1 for outof-
    vocabulary words). Our tokenize() function returns both the tokenized
    sequence and the Tokenizer object itself. This object will be needed anytime we
    want to convert tokens back into words.

    The function tokens_to_words() requires a Tokenizer and a list of indices. We simply check for the reserved indices: If we find a match, we replace them with hardcoded strings, and if we find no match, we let the Tokenizer convert the index to the corresponding word string. The Tokenizer expects a list of lists of indices and returns a list of strings, which is why we need to call it with [[index]] and then select the 0th element to arrive at a string.
    """)
    return


@app.cell
def _(
    MAX_WORDS,
    OOV_INDEX,
    OOV_WORD,
    PAD_INDEX,
    START_INDEX,
    STOP_INDEX,
    Tokenizer,
):
    # Functions to tokenize and un-tokenize sequences.
    def tokenize(sequences):
        # "MAX_WORDS-2" used to reserve two indices
        # for START and STOP.
        tokenizer = Tokenizer(num_words=MAX_WORDS-2,
                              oov_token=OOV_WORD)
        tokenizer.fit_on_texts(sequences)
        token_sequences = tokenizer.texts_to_sequences(sequences)
        return tokenizer, token_sequences

    def tokens_to_words(tokenizer, seq):
        word_seq = []
        for index in seq:
            if index == PAD_INDEX:
                word_seq.append('PAD')
            elif index == OOV_INDEX:
                word_seq.append(OOV_WORD)
            elif index == START_INDEX:
                word_seq.append('START')
            elif index == STOP_INDEX:
                word_seq.append('STOP')
            else:
                word_seq.append(tokenizer.sequences_to_texts(
                    [[index]])[0])
        return word_seq

    return tokenize, tokens_to_words


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given these helper functions, it is trivial to read the input data
    file and convert into tokenized sequences.
    """)
    return


@app.cell
def _(MAX_LENGTH, SRC_DEST_FILE_NAME, read_file_combined, tokenize):
    # Read file and tokenize.
    src_seq, dest_seq = read_file_combined(SRC_DEST_FILE_NAME,
                                           MAX_LENGTH)
    src_tokenizer, src_token_seq = tokenize(src_seq)
    dest_tokenizer, dest_token_seq = tokenize(dest_seq)
    return dest_token_seq, dest_tokenizer, src_token_seq, src_tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is now time to arrange the data into arrays that can be used for training and testing. The following example provides some insight into what we need as input and output for a single training example, where src_input is the input to the encoder network, dest_input is the input to the decoder network, and dest_target is the desired output from the decoder network:

    src_input = [PAD, PAD, PAD, id("je"), id("suis"), id("étudiant")]
    dest_input = [START, id("i"), id("am"), id("a"), id("student"), STOP, PAD, PAD]
    dest_target = [one_hot_id("i"), one_hot_id("am"), one_hot_id("a"), one_hot_id("student"), one_hot_id(STOP), one_hot_id(PAD), one_hot_id(PAD), one_hot_id(PAD)]

    In the example, id(string) refers to the tokenized index of the string, and one_hot_id is the one-hot encoded version of the index. We have assumed that the longest source sentence is six words, so we padded src_input to be of that length. Similarly, we have assumed that the longest destination sentence is eight words including START and STOP tokens, so we padded both dest_input and dest_target to be of that length. Note how the symbols in dest_input are offset by one location compared to the symbols in dest_target because when we later do inference, the inputs into the decoder network will be coming from the output of the network for the previous timestep. Although this example has shown the training example as being lists, in reality, they will be rows in NumPy arrays, where each array contains multiple training examples.

    The padding is done to ensure that we can use mini-batches for training. That is, all source sentences need to be the same length, and all destination sentences need to be the same length. We pad the source input at the beginning (known as prepadding) and the destination at the end (known as postpadding).

    The code snippet below shows a compact way of creating the three arrays that we need. The first two lines create two new lists, each containing the destination sequences but the first (dest_target_token_seq) also augmented with STOP_INDEX after each sequence and the second (dest_input_token_seq) augmented with both START_INDEX and STOP_INDEX. It is easy to miss that dest_input_token_seq has a STOP_INDEX, but that falls out naturally because it is created from the dest_target_token_seq for which a STOP_INDEX was just added to each sentence.

    Next, we call pad_sequences() on both the original src_input_data list (of lists) and on these two new destination lists. The pad_sequences() function pads the sequences with the PAD value and then returns a NumPy array. The default behavior of pad_sequences is to do prepadding, and we do that for the source sequence but explicitly ask for postpadding for the destination sequences.

    We conclude with converting the data type to np.int64 to match what PyTorch later requires.
    """)
    return


@app.cell
def _(
    START_INDEX,
    STOP_INDEX,
    dest_token_seq,
    np,
    pad_sequences,
    src_token_seq,
):
    # Prepare training data.
    dest_target_token_seq = [x + [STOP_INDEX] for x in dest_token_seq]
    dest_input_token_seq = [[START_INDEX] + x for x in
                            dest_target_token_seq]
    src_input_data = pad_sequences(src_token_seq)
    dest_input_data = pad_sequences(dest_input_token_seq,
                                    padding='post')
    dest_target_data = pad_sequences(
        dest_target_token_seq, padding='post', maxlen
        = len(dest_input_data[0]))

    # Convert to same precision as model.
    src_input_data = src_input_data.astype(np.int64)
    dest_input_data = dest_input_data.astype(np.int64)
    dest_target_data = dest_target_data.astype(np.int64)
    return dest_input_data, dest_target_data, src_input_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next code snippet demonstrates how we can manually split our dataset into a training dataset and a test dataset. We split the dataset by first creating a list test_indices, which contains a 20% (TEST_PERCENT) subset of all the numbers from 0 to N−1, where N is the size of our original dataset. We then create a list train_indices, which contains the remaining 80%. We can now use these lists to select a number of rows in the arrays representing the dataset and create two new collections of arrays, one to be used as training set and one to be used as test set. Finally, we create a third collection of arrays, which only contains 20 (SAMPLE_SIZE) random examples from the test dataset. We will use them to inspect the resulting translations in detail, but since that is a manual process, we limit ourselves to a small number of sentences.

    Finally, we convert the NumPy arrays to PyTorch tensors and create Dataset objects.
    """)
    return


@app.cell
def _(
    SAMPLE_SIZE,
    TEST_PERCENT,
    TensorDataset,
    dest_input_data,
    dest_target_data,
    random,
    src_input_data,
    torch,
):
    # Split into training and test set.
    rows = len(src_input_data[:,0])
    all_indices = list(range(rows))
    test_rows = int(rows * TEST_PERCENT)
    test_indices = random.sample(all_indices, test_rows)
    train_indices = [x for x in all_indices if x not in test_indices]

    train_src_input_data = src_input_data[train_indices]
    train_dest_input_data = dest_input_data[train_indices]
    train_dest_target_data = dest_target_data[train_indices]

    test_src_input_data = src_input_data[test_indices]
    test_dest_input_data = dest_input_data[test_indices]
    test_dest_target_data = dest_target_data[test_indices]

    # Create a sample of the test set that we will inspect in detail.
    test_indices = list(range(test_rows))
    sample_indices = random.sample(test_indices, SAMPLE_SIZE)
    sample_input_data = test_src_input_data[sample_indices]
    sample_target_data = test_dest_target_data[sample_indices]

    # Create Dataset objects.
    trainset = TensorDataset(torch.from_numpy(train_src_input_data),
                             torch.from_numpy(train_dest_input_data),
                             torch.from_numpy(train_dest_target_data))
    testset = TensorDataset(torch.from_numpy(test_src_input_data),
                             torch.from_numpy(test_dest_input_data),
                             torch.from_numpy(test_dest_target_data))
    return sample_input_data, sample_target_data, testset, trainset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To provide ordering information between the embeddings we need to add positional encodings to each embedding. We do this by creating a class PositionalEncoding that takes an embedding as input and adds the positional encoding.

    We calculate the positional encoding using sine and cosine as in the original Transformer paper and add it to the embedding.
    """)
    return


@app.cell
def _(Tensor, device, nn, np, torch):
    class PositionalEncoding(nn.Module):

        def __init__(self, d_model, max_len):
            super().__init__()
            i_range = np.arange(d_model).reshape(1, d_model)
            pos_range = np.arange(max_len).reshape(max_len, 1)
            sine_matrix = np.sin(1 / np.power(10000, i_range/d_model) * pos_range)
            cosine_matrix = np.cos(1 / np.power(10000, (i_range-1)/d_model) * pos_range)
            pos_matrix = np.zeros((max_len, d_model))
            for i in range(d_model):
                if (i % 2 == 0):
                    pos_matrix[:, i] = sine_matrix[:, i]
                else:
                    pos_matrix[:, i] = cosine_matrix[:, i]
            pos_matrix = pos_matrix.reshape(1, max_len, d_model).astype(np.float32)
            self.pos_matrix = torch.from_numpy(pos_matrix).to(device)

        def forward(self, x: Tensor) -> Tensor:
            return x + self.pos_matrix[:, :x.size(1)]

    return (PositionalEncoding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are now ready to build our model. It consists of an encoder part and a decoder part. The encoder consists of an Embedding layer, a PositionalEncoding layer, and two Transformer encoder modules stacked on top of each other. The decoder consists of an Embedding layer, a PositionalEncoding layer, two Transformer decoder modules stacked on top of each other, and a fully connected softmax layer. We define these as two separate models, but we will use them together as an encoder-decoder model.

    The code snippet below contains the implementation of the encoder model. The way to define the Transformer encoder modules in PyTorch is to first define a TransformerEncoderLayer object with the desired parameters, and then pass that as input to the constructor of a TransformerEncoder object that creates multiple instances stacked on top of each other.
    """)
    return


@app.cell
def _(
    EMBEDDING_WIDTH,
    LAYER_SIZE,
    MAX_LENGTH,
    MAX_WORDS,
    NUM_HEADS,
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderLayer,
    math,
    nn,
):
    # Define models.
    class EncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
            self.positional_layer = PositionalEncoding(EMBEDDING_WIDTH, MAX_LENGTH)
            enc_layer = TransformerEncoderLayer(EMBEDDING_WIDTH, NUM_HEADS, LAYER_SIZE, batch_first=True)
            self.trans_enc_layers = TransformerEncoder(enc_layer, 2)
            nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.

        def forward(self, inputs, pad_mask = None):
            x = self.embedding_layer(inputs) * math.sqrt(EMBEDDING_WIDTH)
            x = self.positional_layer(x)
            x = self.trans_enc_layers(x, src_key_padding_mask=pad_mask)
            return x

    return (EncoderModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next code snippet shows the implementation of the decoder model. In addition to the sentence in the destination language, it needs the output state from the encoder model. It also requires a mask that defines which positions a give position is allowed to self-attend to. In our use-case it is simple in that each position is only allowed to attend to prior positions, but the PyTorch implementation allows flexibility to have more complicated relationships.
    """)
    return


@app.cell
def _(
    EMBEDDING_WIDTH,
    LAYER_SIZE,
    MAX_LENGTH,
    MAX_WORDS,
    NUM_HEADS,
    PositionalEncoding,
    TransformerDecoder,
    TransformerDecoderLayer,
    math,
    nn,
):
    class DecoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
            self.positional_layer = PositionalEncoding(EMBEDDING_WIDTH, MAX_LENGTH)
            dec_layer = TransformerDecoderLayer(EMBEDDING_WIDTH, NUM_HEADS, LAYER_SIZE, batch_first=True)
            self.trans_dec_layers = TransformerDecoder(dec_layer, 2)
            nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) # Default is -1, 1.
            self.output_layer = nn.Linear(EMBEDDING_WIDTH, MAX_WORDS)

        def forward(self, embedding_inputs, state_inputs, causal_mask,  pad_mask = None):
            x = self.embedding_layer(embedding_inputs) * math.sqrt(EMBEDDING_WIDTH)
            x = self.positional_layer(x)
            x = self.trans_dec_layers(x, state_inputs, tgt_mask = causal_mask,
                                      tgt_is_causal=True, tgt_key_padding_mask=pad_mask)
            x = self.output_layer(x)
            return x

    return (DecoderModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next code snippet instantitates the two models, and creates two optimizers, one for each model. We decided to use RMSProp as optimizer because some experiments indicate that it performs better than Adam for this specific model. We use CrossEntropyLoss as usual.

    We transfer the models to the GPU and create a DataLoader object for both the training and test dataset. We have not had to do this lately because it has been included in our train_model funtion that was reused for all recent examples. We cannot use that function in this example because it does not support the more complex encoder-decoder model that we want to train.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    DataLoader,
    DecoderModel,
    EncoderModel,
    device,
    nn,
    testset,
    torch,
    trainset,
):
    encoder_model = EncoderModel()
    decoder_model = DecoderModel()

    # Loss functions and optimizer.
    encoder_optimizer = torch.optim.RMSprop(encoder_model.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.RMSprop(decoder_model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    # Using a custom training loop instead of our standard training function.
    # Transfer model to GPU.
    encoder_model.to(device)
    decoder_model.to(device)

    trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)
    return (
        decoder_model,
        decoder_optimizer,
        encoder_model,
        encoder_optimizer,
        loss_function,
        testloader,
        trainloader,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The final code snippet shows hos to train and test the model. It is very similar to the LSTM-based translation network. The key difference is the mask objects needed by the Transformer encoder and decoder modules. Both the encoder and decoder require a mask indicated, which input positions correspond to PAD tokens. Additionally, the decoder requires a mask to specify which positions it is allowed to self-attend to. This mask is created with the method generate_square_subsequent_mask(), which is provided by the Transformer class.

    The third inner loop that does autoregression is also modified. We provide the source sentence to the encoder model to create the resulting internal state and store to the variable intermediate_states. We then set the input x to the START token and use the decoder to make a prediction. We retrieve the most probable word and append it to x. We then provide this sequence to the decoder and make a new prediction. We iterate this with a gradually growing input sequence in an autoregressive manner until the model produces a STOP token or until a given number of words have been produced. Finally, we convert the produced tokenized sequences into the corresponding word sequences and print them out.
    """)
    return


@app.cell
def _(MAX_WORDS, PAD_INDEX, torch):
    def train_epoch(
        encoder_model,
        encoder_optimizer,
        decoder_model,
        decoder_optimizer,
        loss_function,
        trainloader,
        device,
        transformer_cls,
    ):
        encoder_model.train() # Set model in training mode.
        decoder_model.train() # Set model in training mode.
        train_loss = 0.0
        train_correct = 0
        train_batches = 0
        train_elems = 0
        for src_inputs, dest_inputs, dest_targets in trainloader:
            # Move data to GPU.
            src_inputs, dest_inputs, dest_targets = src_inputs.to(
                device), dest_inputs.to(device), dest_targets.to(device)

            # Create masks
            decode_input_width = dest_inputs.shape[1]
            decoder_causal_mask = transformer_cls.generate_square_subsequent_mask(
                decode_input_width, device=device, dtype=torch.float32)

            # Convert boolean masks to float masks for consistency with attn_mask
            encoder_pad_mask = (src_inputs == PAD_INDEX).float()
            encoder_pad_mask = encoder_pad_mask.masked_fill(encoder_pad_mask == 1, float('-inf'))

            decoder_pad_mask = (dest_inputs == PAD_INDEX).float()
            decoder_pad_mask = decoder_pad_mask.masked_fill(decoder_pad_mask == 1, float('-inf'))

            # Zero the parameter gradients.
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Forward pass.
            encoder_state = encoder_model(src_inputs, encoder_pad_mask)
            outputs = decoder_model(dest_inputs, encoder_state,
                                    decoder_causal_mask, decoder_pad_mask)
            loss = loss_function(outputs.view(-1, MAX_WORDS), dest_targets.view(-1))
            # Accumulate metrics.
            _, indices = torch.max(outputs.data, 2)
            train_correct += (indices == dest_targets).sum().item()
            train_elems += indices.numel()
            train_batches +=  1
            train_loss += loss.item()

            # Backward pass and update.
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        return train_loss / train_batches, train_correct / train_elems

    def validate_epoch(
        encoder_model,
        decoder_model,
        loss_function,
        testloader,
        device,
        transformer_cls,
    ):
        encoder_model.eval() # Set model in inference mode.
        decoder_model.eval() # Set model in inference mode.
        test_loss = 0.0
        test_correct = 0
        test_batches = 0
        test_elems = 0
        for src_inputs, dest_inputs, dest_targets in testloader:
            # Move data to GPU.
            src_inputs, dest_inputs, dest_targets = src_inputs.to(
                device), dest_inputs.to(device), dest_targets.to(device)

            # Create masks
            decode_input_width = dest_inputs.shape[1]
            decoder_causal_mask = transformer_cls.generate_square_subsequent_mask(
                decode_input_width, device=device, dtype=torch.float32)

             # Convert boolean masks to float masks for consistency with attn_mask
            encoder_pad_mask = (src_inputs == PAD_INDEX).float()
            encoder_pad_mask = encoder_pad_mask.masked_fill(encoder_pad_mask == 1, float('-inf'))

            decoder_pad_mask = (dest_inputs == PAD_INDEX).float()
            decoder_pad_mask = decoder_pad_mask.masked_fill(decoder_pad_mask == 1, float('-inf'))

            encoder_state = encoder_model(src_inputs, encoder_pad_mask)
            outputs = decoder_model(dest_inputs, encoder_state,
                                    decoder_causal_mask, decoder_pad_mask)
            loss = loss_function(outputs.view(-1, MAX_WORDS), dest_targets.view(-1))
            _, indices = torch.max(outputs, 2)
            test_correct += (indices == dest_targets).sum().item()
            test_elems += indices.numel()
            test_batches +=  1
            test_loss += loss.item()

        return test_loss / test_batches, test_correct / test_elems

    return train_epoch, validate_epoch


@app.cell
def _(
    EPOCHS,
    MAX_LENGTH,
    PAD_INDEX,
    START_INDEX,
    STOP_INDEX,
    Transformer,
    decoder_model,
    decoder_optimizer,
    dest_tokenizer,
    device,
    encoder_model,
    encoder_optimizer,
    loss_function,
    np,
    sample_input_data,
    sample_target_data,
    src_tokenizer,
    testloader,
    tokens_to_words,
    torch,
    train_epoch,
    trainloader,
    validate_epoch,
):
    history = {
        "test_input": [],
        "test_target": [],
        "pred_seq": []
    }

    # Train and test repeatedly.
    for i in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            encoder_model,
            encoder_optimizer,
            decoder_model,
            decoder_optimizer,
            loss_function,
            trainloader,
            device,
            Transformer,
        )
        test_loss, test_acc = validate_epoch(
            encoder_model,
            decoder_model,
            loss_function,
            testloader,
            device,
            Transformer,
        )

        print(f'Epoch {i+1}/{EPOCHS} loss: {train_loss:.4f} - acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')

        # Loop through samples to see result
        epoch_test_input = []
        epoch_test_target = []
        epoch_pred_seq = []
        for (test_input, test_target) in zip(sample_input_data,
                                             sample_target_data):
            # Run a single sentence through encoder model.
            x = np.reshape(test_input, (1, -1))
            inputs = torch.from_numpy(x)
            inputs = inputs.to(device)
            # Create padding mask for encoder (though usually single sentence doesn't have padding meaningful if not batched with others, but good practice if input had padding)
            # Here test_input might have padding if it was padded to MAX_LENGTH.
            encoder_pad_mask = (inputs == PAD_INDEX).float()
            encoder_pad_mask = encoder_pad_mask.masked_fill(encoder_pad_mask == 1, float('-inf'))

            intermediate_states = encoder_model(inputs, encoder_pad_mask)

            # Provide resulting state and START_INDEX as input
            # to decoder model.
            x = np.reshape(np.array(START_INDEX), (1, 1))
            produced_string = ''
            pred_seq = []
            for j in range(MAX_LENGTH):
                # Predict next word and capture internal state.
                decode_input_width = x.shape[1]
                decoder_causal_mask = Transformer.generate_square_subsequent_mask(
                    decode_input_width, device=device, dtype=torch.float32)
                inputs = torch.from_numpy(x)
                inputs = inputs.to(device)
                outputs = decoder_model(inputs, intermediate_states, decoder_causal_mask)
                preds = outputs.cpu().detach().numpy()[0][j]

                # Find the most probable word.
                word_index = preds.argmax()
                pred_seq.append(word_index)
                if word_index == STOP_INDEX:
                    break
                x = np.append(x, [[word_index]], axis=1)
            src_words = tokens_to_words(src_tokenizer, test_input)
            epoch_test_input.append(src_words)
            dest_words = tokens_to_words(dest_tokenizer, test_target)
            epoch_test_target.append(dest_words)
            pred_words = tokens_to_words(dest_tokenizer, pred_seq)
            epoch_pred_seq.append(pred_words)
            print(src_words)
            print(dest_words)
            print(pred_words)
            print('\n\n')
        history["test_input"].append(epoch_test_input)
        history["test_target"].append(epoch_test_target)
        history["pred_seq"].append(epoch_pred_seq)
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect epochs
    """)
    return


@app.cell
def _(EPOCHS, SAMPLE_SIZE, mo):
    epoch_slider = mo.ui.slider(start=1, stop=EPOCHS, step=1, label="Epoch #")
    sample_slider = mo.ui.slider(start=1, stop=SAMPLE_SIZE, step=1, label="Sample #")
    mo.hstack([epoch_slider, sample_slider])
    return epoch_slider, sample_slider


@app.cell(hide_code=True)
def _(epoch_slider, history, mo, sample_slider):
    # Visualize as mo.md() the chosen EPOCH and SAMPLE.
    epoch_idx = epoch_slider.value - 1
    sample_idx = sample_slider.value - 1

    content = ""
    # Check if history exists and indices are valid
    if (
        history.get("test_input") 
        and epoch_idx < len(history["test_input"])
    ):
        content += f"### Epoch {epoch_slider.value}, Sample {sample_slider.value}\n\n"

        h_inputs = history["test_input"][epoch_idx]
        h_targets = history["test_target"][epoch_idx]
        h_preds = history["pred_seq"][epoch_idx]

        if sample_idx < len(h_inputs):
            src_str = " | ".join(h_inputs[sample_idx])
            target_str = " | ".join(h_targets[sample_idx])
            pred_str = " | ".join(h_preds[sample_idx])

            content += f"- **Input:** ` {src_str} `\n"
            content += f"- **Target:** ` {target_str} `\n"
            content += f"- **Pred:** ` {pred_str} `\n\n"
            content += "---\n\n"
        else:
            content += "Sample index out of range."

    else:
        content = "No history available for this epoch yet."

    mo.md(content)
    return


if __name__ == "__main__":
    app.run()
