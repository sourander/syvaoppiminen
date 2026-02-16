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
    # PyTorch Official Tutorial

    ## Attribution

    This notebook is sourced from PyTorch's official tutorial "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention" by Sean Robertson.

    - **Original source**: [https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
    - **License**: BSD 3-Clause License
    - **Copyright**: (c) 2017-2022, PyTorch contributors
    - **Modifications**:
        - Stylistic changes and conversion to Marimo
        - Data included in the repo (see previous exercise)
        - Copied the PNG images to this repo.
        - Converted simple PNG images to ASCII or Mermaid
        - Removed some redundant guide texts (due to overlap with lesson)

    Full license text: https://github.com/pytorch/tutorials/blob/main/LICENSE

    Reason for resharing: ease of access and integration into a larger collection of educational resources.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    # NLP From Scratch: Translation with a Sequence to Sequence Network and Attention

    **Author**: [Sean Robertson](https://github.com/spro)

    This tutorials is part of a three-part series:

    -   [NLP From Scratch: Classifying Names with a Character-Level
        RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
    -   [NLP From Scratch: Generating Names with a Character-Level
        RNN](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
    -   [NLP From Scratch: Translation with a Sequence to Sequence Network
        and
        Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

    This is the third and final tutorial on doing **NLP From Scratch**,
    where we write our own classes and functions to preprocess the data to
    do our NLP modeling tasks.

    In this project we will be teaching a neural network to translate from
    French to English.

    ```plaintext
    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .
    ```

    ... to varying degrees of success.

    This is made possible by the simple but powerful idea of the [sequence
    to sequence network](https://arxiv.org/abs/1409.3215), in which two
    recurrent neural networks work together to transform one sequence to
    another. An encoder network condenses an input sequence into a vector,
    and a decoder network unfolds that vector into a new sequence.

    {mo.image(src="nb/700/images/712_seq2seqimage.png")}

    To improve upon this model we'll use an [attention
    mechanism](https://arxiv.org/abs/1409.0473), which lets the decoder
    learn to focus over a specific range of the input sequence.
    """)
    return


@app.cell
def _():
    from io import open
    import unicodedata
    import re
    import random
    from pathlib import Path

    import torch
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F

    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler

    MODEL_PATH = "models/712_seq2seq.pth"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    return (
        DataLoader,
        F,
        MODEL_PATH,
        Path,
        RandomSampler,
        TensorDataset,
        device,
        nn,
        np,
        open,
        optim,
        random,
        re,
        torch,
        unicodedata,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading data files

    The data for this project is a set of many thousands of English to
    French translation pairs.

    [This question on Open Data Stack
    Exchange](https://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages)
    pointed me to the open translation site <https://tatoeba.org/> which has
    downloads available at <https://tatoeba.org/eng/downloads> - and better
    yet, someone did the extra work of splitting language pairs into
    individual text files here: <https://www.manythings.org/anki/>

    The English to French pairs are too big to include in the repository, so
    download to `data/eng-fra.txt` before continuing. The file is a tab
    separated list of translation pairs:

    ```plaintext
    I am cold.    J'ai froid.
    ```

    ## Syväoppiminen I note

    The data has been downloaded for you by the teacher. The data was in the same Zip-file as the data for the two previous RNN exercises. Thus, you already have the file in `notebooks/data/eng-fra.txt` if you have done the previous two exercises correctly.

    ## One Hot Encoding

    Similar to the character encoding used in the character-level RNN
    tutorials, we will be representing each word in a language as a one-hot
    vector, or giant vector of zeros except for a single one (at the index
    of the word). Compared to the dozens of characters that might exist in a
    language, there are many many more words, so the encoding vector is much
    larger. We will however cheat a bit and trim the data to only use a few
    thousand words per language.

    ```plaintext
         01    03    05  06
        ┌┴┐    ┌┴┐   ┌┴┐┌┴┐
    SOS EOS the a is and or
    └┬┘     └┬┘   └┬┘
     00      02    04

    and = [ 0 0 0 0 0 1 0 ...]
    ```

    We'll need a unique index per word to use as the inputs and targets of
    the networks later. To keep track of all this we will use a helper class
    called `Lang` which has word → index (`word2index`) and index → word
    (`index2word`) dictionaries, as well as a count of each word
    `word2count` which will be used to replace rare words later.
    """)
    return


@app.cell
def _():
    SOS_token = 0
    EOS_token = 1

    class Lang:

        def __init__(self, name):
            self.name = name
            self.word2index = {}
            self.word2count = {}
            self.index2word = {0: 'SOS', 1: 'EOS'}  # Count SOS and EOS
            self.n_words = 2

        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words = self.n_words + 1
            else:
                self.word2count[word] = self.word2count[word] + 1

    return EOS_token, Lang, SOS_token


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lowercase ASCII

    The files are all in Unicode, to simplify we will turn Unicode
    characters to ASCII, make everything lowercase, and trim most
    punctuation.
    """)
    return


@app.cell
def _(re, unicodedata):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()

    return (normalizeString,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Read

    To read the data file we will split the file into lines, and then split
    lines into pairs. The files are all English → Other Language, so if we
    want to translate from Other Language → English I added the `reverse`
    flag to reverse the pairs.
    """)
    return


@app.cell
def _(Lang, normalizeString, open):
    def readLangs(lang1, lang2, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open(
            f"data/{lang1}-{lang2}.txt", 
            encoding='utf-8'
        ).read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    return (readLangs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filter

    Since there are a *lot* of example sentences and we want to train
    something quickly, we'll trim the data set to only relatively short and
    simple sentences. Here the maximum length is 10 words (that includes
    ending punctuation) and we're filtering to sentences that translate to
    the form "I am" or "He is" etc. (accounting for apostrophes replaced
    earlier).
    """)
    return


@app.cell
def _():
    MAX_LENGTH = 10

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    def filterPair(p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)


    def filterPairs(pairs):
        return [pair for pair in pairs if filterPair(pair)]

    return MAX_LENGTH, filterPairs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Process

    The full process for preparing the data is:

    -   Read text file and split into lines, split lines into pairs
    -   Normalize text, filter by length and content
    -   Make word lists from sentences in pairs
    """)
    return


@app.cell
def _(filterPairs, random, readLangs):
    def prepareData(lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))

        pairs = filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")

        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))
    return input_lang, output_lang, pairs, prepareData


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ## The Seq2Seq Model

    A Recurrent Neural Network, or RNN, is a network that operates on a
    sequence and uses its own output as input for subsequent steps.

    A [Sequence to Sequence network](https://arxiv.org/abs/1409.3215), or
    seq2seq network, or [Encoder Decoder
    network](https://arxiv.org/pdf/1406.1078v3.pdf), is a model consisting
    of two RNNs called the encoder and decoder. The encoder reads an input
    sequence and outputs a single vector, and the decoder reads that vector
    to produce an output sequence.

    {mo.image(src="nb/700/images/712_seq2seqimage.png")}

    Unlike sequence prediction with a single RNN, where every input
    corresponds to an output, the seq2seq model frees us from sequence
    length and order, which makes it ideal for translation between two
    languages.

    Consider the sentence `Je ne suis pas le chat noir` →
    `I am not the black cat`. Most of the words in the input sentence have a
    direct translation in the output sentence, but are in slightly different
    orders, e.g. `chat noir` and `black cat`. Because of the `ne/pas`
    construction there is also one more word in the input sentence. It would
    be difficult to produce a correct translation directly from the sequence
    of input words.

    With a seq2seq model the encoder creates a single vector which, in the
    ideal case, encodes the "meaning" of the input sequence into a single
    vector --- a single point in some N dimensional space of sentences.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Encoder

    The encoder of a seq2seq network is a RNN that outputs some value for
    every word from the input sentence. For every input word the encoder
    outputs a vector and a hidden state, and uses the hidden state for the
    next input word.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
    """
    graph TD
        input[input] --> embedding[embedding]
        embedding --> embedded[embedded]
        embedded --> gru[gru]

        prev_hidden[prev_hidden] --> gru

        gru --> output[output]
        gru --> hidden[hidden]

        %% styling
        classDef blue fill:#2E86DE,color:#fff,stroke:#2E86DE;
        classDef orange fill:#F39C12,color:#000,stroke:#F39C12;

        class embedding,gru blue;
        class input,embedded,prev_hidden,output,hidden orange;
    """
    ).center()
    return


@app.cell
def _(nn):
    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size, dropout_p=0.1):
            super(EncoderRNN, self).__init__()
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, input):
            embedded = self.dropout(self.embedding(input))
            output, hidden = self.gru(embedded)
            return output, hidden

    return (EncoderRNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Decoder

    The decoder is another RNN that takes the encoder output vector(s) and
    outputs a sequence of words to create the translation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple Decoder

    In the simplest seq2seq decoder we use only last output of the encoder.
    This last output is sometimes called the *context vector* as it encodes
    context from the entire sequence. This context vector is used as the
    initial hidden state of the decoder.

    At every step of decoding, the decoder is given an input token and
    hidden state. The initial input token is the start-of-string `<SOS>`
    token, and the first hidden state is the context vector (the encoder's
    last hidden state).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
    """
    graph TD
        input[input] --> embedding[embedding]
        embedding --> relu[relu]
        relu --> gru[gru]

        prev_hidden[prev_hidden] --> gru

        gru --> out[out]
        out --> softmax[softmax]
        softmax --> output[output]
        gru --> hidden[hidden]

        %% styling
        classDef blue fill:#2E86DE,color:#fff;
        classDef orange fill:#F39C12,color:#000;
        classDef green fill:#8ef312,color:#000;

        class embedding,gru blue;
        class input,prev_hidden,out,output,hidden orange;
        class relu,softmax green;
    """
    ).center()
    return


@app.cell
def _(F, MAX_LENGTH, SOS_token, device, nn, torch):
    class DecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size):
            super(DecoderRNN, self).__init__()
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.out = nn.Linear(hidden_size, output_size)

        def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
            batch_size = encoder_outputs.size(0)
            decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
            decoder_hidden = encoder_hidden
            decoder_outputs = []

            for i in range(MAX_LENGTH):
                decoder_output, decoder_hidden = ( 
                    self.forward_step(
                        decoder_input, decoder_hidden)
                )

                decoder_outputs.append(decoder_output)

                if target_tensor is not None:
                    # Teacher forcing:
                    # Feed the target as the next input
                    decoder_input = target_tensor[:, i].unsqueeze(1)
                else:
                    # Without teacher forcing: 
                    # use its own predictions as the next input
                    _, topi = decoder_output.topk(1)
                    # detach from history as input
                    decoder_input = topi.squeeze(-1).detach()  

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

            # We return `None` for consistency in the training loop
            return decoder_outputs, decoder_hidden, None

        def forward_step(self, input, hidden):
            output = self.embedding(input)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.out(output)
            return output, hidden

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I encourage you to train and observe the results of this model, but to
    save space we'll be going straight for the gold and introducing the
    Attention Mechanism.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ## Attention Decoder

    If only the context vector is passed between the encoder and decoder,
    that single vector carries the burden of encoding the entire sentence.

    Attention allows the decoder network to \"focus\" on a different part of
    the encoders outputs for every step of the decoders own outputs.
    First we calculate a set of *attention weights*. These will be
    multiplied by the encoder output vectors to create a weighted
    combination. The result (called `attn_applied` in the code) should
    contain information about that specific part of the input sequence, and
    thus help the decoder choose the right output words.

    {mo.image(src="nb/700/images/712_attention_decoder.png")}

    Calculating the attention weights is done with another feed-forward
    layer `attn`, using the decoders input and hidden state as inputs.
    Because there are sentences of all sizes in the training data, to
    actually create and train this layer we have to choose a maximum
    sentence length (input length, for encoder outputs) that it can apply
    to. Sentences of the maximum length will use all the attention weights,
    while shorter sentences will only use the first few.

    {mo.image(src="nb/700/images/712_att_dec_network.png")}

    Bahdanau attention, also known as additive attention, is a commonly used
    attention mechanism in sequence-to-sequence models, particularly in
    neural machine translation tasks. It was introduced by Bahdanau et al.
    in their paper titled [Neural Machine Translation by Jointly Learning to
    Align and Translate](https://arxiv.org/pdf/1409.0473.pdf). This
    attention mechanism employs a learned alignment model to compute
    attention scores between the encoder and decoder hidden states. It
    utilizes a feed-forward neural network to calculate alignment scores.

    However, there are alternative attention mechanisms available, such as
    Luong attention, which computes attention scores by taking the dot
    product between the decoder hidden state and the encoder hidden states.
    It does not involve the non-linear transformation used in Bahdanau
    attention.

    In this tutorial, we will be using Bahdanau attention. However, it would
    be a valuable exercise to explore modifying the attention mechanism to
    use Luong attention.
    """)
    return


@app.cell
def _(F, MAX_LENGTH, SOS_token, device, nn, torch):
    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_size):
            super(BahdanauAttention, self).__init__()
            self.Wa = nn.Linear(hidden_size, hidden_size)
            self.Ua = nn.Linear(hidden_size, hidden_size)
            self.Va = nn.Linear(hidden_size, 1)

        def forward(self, query, keys):
            scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
            scores = scores.squeeze(2).unsqueeze(1)

            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights, keys)

            return context, weights

    class AttnDecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size, dropout_p=0.1):
            super(AttnDecoderRNN, self).__init__()
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.attention = BahdanauAttention(hidden_size)
            self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
            self.out = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
            batch_size = encoder_outputs.size(0)
            decoder_input = (
                torch.empty(batch_size, 1, 
                            dtype=torch.long, device=device
               ).fill_(SOS_token)
            )

            decoder_hidden = encoder_hidden
            decoder_outputs = []
            attentions = []

            for i in range(MAX_LENGTH):
                decoder_output, decoder_hidden, attn_weights = (
                    self.forward_step(
                        decoder_input, 
                        decoder_hidden, 
                        encoder_outputs
                    )
                )
                decoder_outputs.append(decoder_output)
                attentions.append(attn_weights)

                if target_tensor is not None:
                    # Teacher forcing: 
                    # Feed the target as the next input
                    decoder_input = target_tensor[:, i].unsqueeze(1)
                else:
                    # Without teacher forcing: 
                    # use its own predictions as the next input
                    _, topi = decoder_output.topk(1)
                    # detach from history as input
                    decoder_input = topi.squeeze(-1).detach()  

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
            attentions = torch.cat(attentions, dim=1)

            return decoder_outputs, decoder_hidden, attentions


        def forward_step(self, input, hidden, encoder_outputs):
            embedded =  self.dropout(self.embedding(input))

            query = hidden.permute(1, 0, 2)
            context, attn_weights = self.attention(query, encoder_outputs)
            input_gru = torch.cat((embedded, context), dim=2)

            output, hidden = self.gru(input_gru, hidden)
            output = self.out(output)

            return output, hidden, attn_weights

    return (AttnDecoderRNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | NOTE:

    There are other forms of attention that work around the length limitation by using a relative position approach. Read about *local attention* in [Effective Approaches to Attention-based Neural MachineTranslation](https://arxiv.org/abs/1508.04025).
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training helpers

    ## Preparing Training Data
    """)
    return


@app.cell
def _(
    DataLoader,
    EOS_token,
    MAX_LENGTH,
    RandomSampler,
    TensorDataset,
    device,
    input_lang,
    np,
    output_lang,
    prepareData,
    torch,
):
    def indexesFromSentence(lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(lang, sentence):
        indexes = indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

    def tensorsFromPair(pair):
        """Note: this is never used."""
        input_tensor = tensorFromSentence(input_lang, pair[0])
        target_tensor = tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)

    def get_dataloader(batch_size):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

        n = len(pairs)
        input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = indexesFromSentence(input_lang, inp)
            tgt_ids = indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(
            torch.LongTensor(input_ids).to(device), 
            torch.LongTensor(target_ids).to(device)
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, 
            sampler=train_sampler, 
            batch_size=batch_size
        )
        return input_lang, output_lang, train_dataloader

    return get_dataloader, tensorFromSentence


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train helper function

    To train we run the input sentence through the encoder, and keep track
    of every output and the latest hidden state. Then the decoder is given
    the `<SOS>` token as its first input, and the last hidden state of the
    encoder as its first hidden state.

    *Teacher forcing* is the concept of using the real target outputs as
    each next input, instead of using the decoder's guess as the next
    input. Using teacher forcing causes it to converge faster but when the
    trained network is exploited, it may exhibit
    instability.

    You can observe outputs of teacher-forced networks that read with
    coherent grammar but wander far from the correct translation
    -intuitively it has learned to represent the output grammar and can
    *pick up* the meaning once the teacher tells it the first few words,
    but it has not properly learned how to create the sentence from the
    translation in the first place.

    Because of the freedom PyTorch's autograd gives us, we can randomly
    choose to use teacher forcing or not with a simple if statement. Turn
    `teacher_forcing_ratio` up to use more of it.
    """)
    return


@app.function
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss = total_loss + loss.item()
    return total_loss / len(dataloader)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a helper function to print time elapsed and estimated time
    remaining given the current time and progress %.
    """)
    return


@app.cell
def _():
    import time
    import math

    def asMinutes(s):
        m = math.floor(s / 60)
        s = s - m * 60
        return '%dm %ds' % (m, s)

    def timeSince(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

    return time, timeSince


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The whole training process looks like this:

    -   Start a timer
    -   Initialize optimizers and criterion
    -   Create set of training pairs
    -   Start empty losses array for plotting

    Then we call `train` many times and occasionally print the progress (%
    of examples, time so far, estimated time) and average loss.
    """)
    return


@app.cell
def _(nn, optim, showPlot, time, timeSince):
    def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0  # Reset every print_every
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)  # Reset every plot_every
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
        for epoch in range(1, n_epochs + 1):
            loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total = print_loss_total + loss
            plot_loss_total = plot_loss_total + loss
            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))
            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        showPlot(plot_losses)
        return plot_losses

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting helper func

    Plotting is done with matplotlib, using the array of loss values
    `plot_losses` saved while training.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import matplotlib.ticker as ticker

    def showPlot(points):
        plt.figure()
        fig, ax = plt.subplots()
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)  # this locator puts ticks at regular intervals
        plt.plot(points)

    return plt, showPlot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eval helper func

    Evaluation is mostly the same as training, but there are no targets so
    we simply feed the decoder's predictions back to itself for each step.
    Every time it predicts a word we add it to the output string, and if it
    predicts the EOS token we stop there. We also store the decoder's
    attention outputs for display later.
    """)
    return


@app.cell
def _(EOS_token, tensorFromSentence, torch):
    def evaluate(encoder, decoder, sentence, input_lang, output_lang):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = ( 
                decoder(encoder_outputs, encoder_hidden)
            )

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                decoded_words.append(output_lang.index2word[idx.item()])
        return decoded_words, decoder_attn

    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can evaluate random sentences from the training set and print out the
    input, target, and output to make some subjective quality judgements:
    """)
    return


@app.cell
def _(evaluate, input_lang, output_lang, pairs, random):
    def evaluateRandomly(encoder, decoder, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    return (evaluateRandomly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training and Evaluating

    With all these helper functions in place (it looks like extra work, but
    it makes it easier to run multiple experiments) we can actually
    initialize a network and start training.

    Remember that the input sentences were heavily filtered. For this small
    dataset we can use relatively small networks of 256 hidden nodes and a
    single GRU layer.
    """)
    return


@app.cell
def _(
    AttnDecoderRNN,
    EncoderRNN,
    MODEL_PATH,
    Path,
    device,
    get_dataloader,
    showPlot,
    torch,
    train,
):
    hidden_size = 128
    batch_size = 32
    input_lang_1, output_lang_1, train_dataloader = get_dataloader(batch_size)
    encoder = EncoderRNN(input_lang_1.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang_1.n_words).to(device)

    if Path(MODEL_PATH).exists():
        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Loaded model from {MODEL_PATH}")
        if 'plot_losses' in checkpoint:
            showPlot(checkpoint['plot_losses'])
    else:
        plot_losses = train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'plot_losses': plot_losses
        }, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    return decoder, encoder, input_lang_1, output_lang_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set dropout layers to `eval` mode
    """)
    return


@app.cell
def _(decoder, encoder, evaluateRandomly):
    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizing Attention

    A useful property of the attention mechanism is its highly interpretable
    outputs. Because it is used to weight specific encoder outputs of the
    input sequence, we can imagine looking where the network is focused most
    at each time step.

    You could simply run `plt.matshow(attentions)` to see attention output
    displayed as a matrix. For a better viewing experience we will do the
    extra work of adding axes and labels:
    """)
    return


@app.cell
def _(decoder, encoder, evaluate, input_lang_1, output_lang_1, plt):
    def evaluateAndShowAttention(input_sentence):
        """Evaluate translation and create attention visualization figure."""
        output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang_1, output_lang_1)

        # Print the translation
        print(f"input  = {input_sentence}")
        print(f"output = {' '.join(output_words)}")

        # Create attention heatmap
        fig, ax = plt.subplots(figsize=(8, 4))
        cax = ax.matshow(attentions[0, :len(output_words), :].cpu().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        x_labels = [''] + input_sentence.split(' ') + ['<EOS>']
        y_labels = [''] + output_words

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        return fig

    return (evaluateAndShowAttention,)


@app.cell
def _(evaluateAndShowAttention):
    evaluateAndShowAttention('il n est pas aussi grand que son pere')
    return


@app.cell
def _(evaluateAndShowAttention):
    evaluateAndShowAttention('je suis trop fatigue pour conduire')
    return


@app.cell
def _(evaluateAndShowAttention):
    evaluateAndShowAttention('je suis desole si c est une question idiote')
    return


@app.cell
def _(evaluateAndShowAttention):
    evaluateAndShowAttention('je suis reellement fiere de vous')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises (by PyTorch)

    -   Try with a different dataset
        -   Another language pair
        -   Human → Machine (e.g. IOT commands)
        -   Chat → Response
        -   Question → Answer
    -   Replace the embeddings with pretrained word embeddings such as
        `word2vec` or `GloVe`
    -   Try with more layers, more hidden units, and more sentences. Compare
        the training time and results.
    -   If you use a translation file where pairs have two of the same
        phrase (`I am test \t I am test`), you can use this as an
        autoencoder. Try this:
        -   Train as an autoencoder
        -   Save only the Encoder network
        -   Train a new Decoder for translation from there
    """)
    return


if __name__ == "__main__":
    app.run()
