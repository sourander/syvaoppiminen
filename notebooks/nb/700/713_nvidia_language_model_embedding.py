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
    # Nvidia Learning Deep Learning: Using Pretrained GloVe Embeddings

    ## Attribution

    This notebook is sourced from Nvidia's Learning Deep Learning, [LDL](https://github.com/NVDLI/LDL), repository. The repository contains code examples associated with a similarly named book and video series by Magnus Ekman (Book ISBN: 9780137470358; Video: 9780138177553)

    - **Original source**: [gh:NVDLI/LDL/blob/main/pt_framework/v6_4_language_model_embedding.ipynb](https://github.com/NVDLI/LDL/blob/main/pt_framework/v6_4_language_model_embedding.ipynb)
    - **License**: BSD 3-Clause License
    - **Copyright**: (c) 2021 NVIDIA
    - **Modifications**:
        - Stylistic changes and conversion to Marimo
        - Replaced TensorFlow/Keras `Tokenizer` and `text_to_word_sequence` with pure Python equivalents.
        - Copied the training function from [utilities.py](https://github.com/NVDLI/LDL/blob/main/pt_framework/utilities.py)
        - Separated logic from `train_model` into functions `train_epoch` and `validate_epoch`.
        - Added brief comments to code when it seems helpful to students.

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
    More context for this code example can be found in video 6.4 _"Programming Example: Language Model and Word Embeddings with PyTorch"_ in the video series _"Learning Deep Learning: From Perceptron to Large Language Models"_ by _Magnus Ekman_ (Video ISBN-13: 9780138177614).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define two new constants `MAX_WORDS` and `EMBEDDING_WIDTH` that define the max size of our vocabulary and the dimensionality of the word vectors.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 32
    BATCH_SIZE = 256
    INPUT_FILE_NAME = './data/books/frankenstein.txt'
    WINDOW_LENGTH = 40
    WINDOW_STEP = 3
    PREDICT_LENGTH = 3
    MAX_WORDS = 7500
    EMBEDDING_WIDTH = 100
    return (
        BATCH_SIZE,
        DataLoader,
        EMBEDDING_WIDTH,
        EPOCHS,
        INPUT_FILE_NAME,
        MAX_WORDS,
        PREDICT_LENGTH,
        TensorDataset,
        WINDOW_LENGTH,
        WINDOW_STEP,
        device,
        nn,
        np,
        torch,
        train_test_split,
    )


@app.cell
def _(torch):
    def train_epoch(model, device, dataloader, optimizer, loss_function, metric):
        model.train()
        train_loss = 0.0
        correct = 0
        absolute_error = 0.0
        batches = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            if metric == 'acc':
                _, indices = torch.max(outputs.data, 1)
                correct += (indices == targets).sum().item()
            elif metric == 'mae':
                absolute_error += (targets - outputs.data).abs().sum().item()

            batches += 1
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / batches
        metric_val = 0.0
        if metric == 'acc':
             metric_val = correct / (batches * dataloader.batch_size)
        elif metric == 'mae':
             metric_val = absolute_error / (batches * dataloader.batch_size)

        return train_loss, metric_val

    def validate_epoch(model, device, dataloader, loss_function, metric):
        model.eval()
        val_loss = 0.0
        correct = 0
        absolute_error = 0.0
        batches = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)

                if metric == 'acc':
                    _, indices = torch.max(outputs.data, 1)
                    correct += (indices == targets).sum().item()
                elif metric == 'mae':
                    absolute_error += (targets - outputs.data).abs().sum().item()

                batches += 1
                val_loss += loss.item()

        val_loss = val_loss / batches
        metric_val = 0.0
        if metric == 'acc':
            metric_val = correct / (batches * dataloader.batch_size)
        elif metric == 'mae':
            metric_val = absolute_error / (batches * dataloader.batch_size)

        return val_loss, metric_val

    return train_epoch, validate_epoch


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

    return Tokenizer, text_to_word_sequence


@app.cell
def _(DataLoader, train_epoch, validate_epoch):
    def train_model(model, device, epochs, batch_size, trainset, testset,
                    optimizer, loss_function, metric):
        # Transfer model to GPU.
        model.to(device)

        # Create dataloaders.
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        if metric not in ['acc', 'mae']:
            print('Error: unsupported metric')
            return

        for i in range(epochs):
            train_loss, train_metric = train_epoch(model, device, trainloader, optimizer, loss_function, metric)
            test_loss, test_metric = validate_epoch(model, device, testloader, loss_function, metric)

            if metric == 'acc':
                print(f'Epoch {i+1}/{epochs} loss: {train_loss:.4f} - acc: {train_metric:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_metric:0.4f}')
            elif metric == 'mae':
                print(f'Epoch {i+1}/{epochs} loss: {train_loss:.4f} - mae: {train_metric:0.4f} - val_loss: {test_loss:.4f} - val_mae: {test_metric:0.4f}')

        return [train_metric, test_metric]

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next code snippet first reads the input file and splits the text into a list of individual words. The latter is done by using the imported function text_to_word_sequence(), which also removes punctuation and converts the text to lowercase, so we do not need to do that manually in this example. We then create input fragments and associated target words just as in the character-based example.
    """)
    return


@app.cell
def _(INPUT_FILE_NAME, WINDOW_LENGTH, WINDOW_STEP, text_to_word_sequence):
    # Open the input file.
    file = open(INPUT_FILE_NAME, 'r', encoding='utf-8-sig')
    text = file.read()
    file.close()
    text = text_to_word_sequence(text)
    # Make lower case and split into individual words.
    fragments = []
    targets = []
    # Create training examples.
    for _i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
        fragments.append(text[_i:_i + WINDOW_LENGTH])
        targets.append(text[_i + WINDOW_LENGTH])
    return fragments, targets, text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next step is to convert the training examples into the correct format. Each input word needs to be encoded to a corresponding word index (an integer). This index will then be converted into an embedding by the Embedding layer. The target (output) word should still be one-hot encoded. To simplify how to interpret the output, we want the one-hot encoding to be done in such a way that bit N is hot when the network outputs the word corresponding to index N in the input encoding.

    We make use of the Tokenizer class. When we construct our tokenizer, we provide an argument num_words = MAX_WORDS that caps the size of the vocabulary. The tokenizer object reserves index 0 to use as a special padding value and index 1 for unknown words. The remaining 7,498 indices (MAX_WORDS was set to 7,500) are used to represent words in the vocabulary.

    The padding value (index 0) can be used to make all training examples within the same batch have the same length. The Embedding layer can be instructed to ignore this value, so the network does not train on the padding values.

    Index 1 is reserved for UNKnown (UNK) words because we have declared UNK as an out-of-vocabulary (oov) token. When using the tokenizer to convert text to tokens, any word that is not in the vocabulary will be replaced by the word UNK. Similarly, if we try to convert an index that is not assigned to a word, the tokenizer will return UNK. If we do not set the oov_token parameter, it will simply ignore such words/indices.

    After instantiating our tokenizer, we call fit_on_texts() with our entire text corpus, which will result in the tokenizer assigning indices to words. We can then use the function texts_to_sequences to convert a text string into a list of indices, where unknown words will be assigned the index 1.
    """)
    return


@app.cell
def _(MAX_WORDS, Tokenizer, fragments, np, targets, text):
    # Convert to indices.
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='UNK')
    tokenizer.fit_on_texts(text)
    fragments_indexed = tokenizer.texts_to_sequences(fragments)
    targets_indexed = tokenizer.texts_to_sequences(targets)
    X = np.array(fragments_indexed, dtype=np.int64)
    # Convert to appropriate input and output formats.
    y = np.zeros(len(targets_indexed), dtype=np.int64)
    for _i, target_index in enumerate(targets_indexed):
        y[_i] = target_index[0]
    return X, tokenizer, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Just like in the previous example, we split the data into a training set and a test set and create Dataset objects.
    """)
    return


@app.cell
def _(TensorDataset, X, torch, train_test_split, y):
    # Split into training and test set.
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.05, random_state=0)

    # Create Dataset objects.
    trainset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    testset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))
    return testset, trainset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are now ready to create the model, but we do this differently than we have done in previous examples, both to illustrate some new constructs but also to provide more flexibility in what we are building. Instead of instantiating a Sequential model with standard layers, we build a fully customized model. This is done by extending the nn.Module class. We already saw examples of that when we built custom layers, and we use a similar methodology when building a fully custom model.

    To understand the details of this model we first need to take a detour and discuss some details of how we plan to use the model to do predictions. We do this a little bit differently than in the previous example. Instead of feeding a string of symbols as input to the model, we feed it only a single symbol at a time. This is an alternative implementation compared to the previous implementation, where we repeatedly fed the model a growing sequence of characters.

    The scheme used in this example has a subtle implication, which has to do with dependencies between multiple consecutive calls to the model. In the previous example we did not have an expectation that the inputs to the first prediction should impact the second prediction. In this example, we do want that to happen. We want the LSTM layers to retain their c and h states from one call to another so that the outputs of subsequent calls to the model will depend on the prior calls to the model. We provide this functionality with our own custom methods set_state(), get_state(), and clear_state().

    We are now ready to describe the details of our model. The __init__() function will be called a single time when the model is created. When building a custom model, this is the place to put any initialization code, such as declaring the layers to be used by the model as well as any other variables that will be needed.

    We start with declaring two variables state and use_state to control how we handle initialization of the internal state of the LSTM layers (see below).

    For the model, we start with declaring an Embedding layer with MAX_WORDS as input size and EMBEDDING_WIDTH as output size. We adjust the weight initialization to use uniform random numbers between -0.05 and 0.05, as opposed to the default range of -1.0 to 1.0. We did this to match the range used in our TensorFlow examples.

    Next we declare an LSTM module containing two LSTM layers. We also add a Dropout layer, followed by two fully connected layers, with ReLU activation in-between.

    The behavior of the model is defined in the forward method. We feed the inputs to the embedding layer and the resulting outputs are temporarily stored in the variable x. Next we arrive at the custom state handling code for the LSTM layers. If our variable use_state is False (the value we initialized it to), we will call the LSTM layers with just x as input. This implies that the LSTM module will use 0 as initial h and c states. However, if use_state is set to true, we will also supply the variable self.state as input to the LSTM layers. In that case the LSTM layers will use these states as their initial states instead of 0.

    After the LSTM layers has been called, we retrieve the resulting internal state and store it in self.state so it can be used as input for the next timestep. Apart from calling detach() we also call clone(), which makes a copy of the state so it does not change under the hood by the layers themselves if they are later called with new inputs.

    We then feed the output from the top LSTM layer through the Dropout layer. The indices after variable x results in selecting the final timestep for the top LSTM layer. That is, this is equivalent to the functionality of the custom layer in c11e1_autocomplete. Given that we are building a custom model and have explicit control of how the layers are connected, there is no need to declare a custom layer.

    The rest of the forward method is straight forward and we feed the output from the Dropout layer through a Linear layer, followed by a ReLU layer, and finally another Linear layer that represents the Softmax layer but without its activation function, which is later included in the loss function.

    Now let us detail the intended use of set_state, get_state, and clear_state. At the beginning of time the use_state variable is set to False. When calling the model, it will use 0 as internal state. After each call to the model, we can retrieve the resulting state by calling get_state. If we later want the model to use this state as its initial state, we simply call set_state and supply that state to the model, before we call the model with its input data. If we later decide that we want to start with a cleared internal state again, we simply call the clear_state method. This concludes the description of the custom model.

    Next we instantiate the model, select our normal optimizer and loss function, and finally train the model. Our training process does not make use of the just described state handling and the model will simply use 0 as initial state for each call to the model during training.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    EMBEDDING_WIDTH,
    EPOCHS,
    MAX_WORDS,
    device,
    nn,
    testset,
    torch,
    train_model,
    trainset,
):
    # Define model.
    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.state = None
            self.use_state = False
            self.embedding_layer = nn.Embedding(MAX_WORDS, EMBEDDING_WIDTH)
        
            # Default is -1, 1.
            nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05) 
            self.lstm_layers = nn.LSTM(
                input_size=EMBEDDING_WIDTH, 
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                batch_first=True
            )
            self.dropout_layer = nn.Dropout(0.2)
            self.linear_layer = nn.Linear(128, 128)
            self.relu_layer = nn.ReLU()
            self.output_layer = nn.Linear(128, MAX_WORDS)

        def forward(self, inputs):
            x = self.embedding_layer(inputs)

            if(self.use_state):
                # If use_state is True, provide the LSTM with the previous 
                # state (self.state) in addition to the input (x).
                # self.state is a tuple containing: 
                # * the hidden state (h)
                # * cell state (c)
                x = self.lstm_layers(x, self.state)
            else:
                # If use_state is False (e.g. at the start 
                # of a new sentence or during training),
                # the LSTM starts with zero states.
                x = self.lstm_layers(x)
            # Store most recent internal state.
            self.state = (x[1][0].detach().clone(), x[1][1].detach().clone()) 
        
            x = self.dropout_layer(x[1][0][1])
            x = self.linear_layer(x)
            x = self.relu_layer(x)
            x = self.output_layer(x)
            return x

        # Functions to provide explicit control of LSTM state.
        def set_state(self, state):
            self.state = state
            self.use_state = True
            return

        def get_state(self):
            return self.state

        def clear_state(self):
            self.use_state = False
            return

    model = LanguageModel()

    # Loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    # Train the model.
    train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset,
                optimizer, loss_function, 'acc')

    trained_model = model
    return (trained_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After training the model, we are ready to use it to do predictions. The first loop in the code snippet below feeds an initial sequence of words to the model. Note how we call get_state after each call followed by set_state, which instructs the model to use this stat instead of reseting it before each word.

    The second loop contains the autoregression logic where we identify the word that the model predicts as highest probability and then feed that as input to the model in the next timestep. To simplify the implementation, we do not do beam search this time around but simply predict the most probable word at each timestep.

    We conclude with printing the resulting autocompleted text sequence.
    """)
    return


@app.cell
def _(PREDICT_LENGTH, device, np, tokenizer, torch, trained_model):
    # Provide beginning of sentence and
    # predict next words in a greedy manner.
    first_words = ['i', 'saw']
    first_words_indexed = tokenizer.texts_to_sequences(first_words)
    trained_model.clear_state()
    predicted_string = ''

    for _i, _word_index in enumerate(first_words_indexed):
    # Feed initial words to the model.
        x = np.zeros((1, 1), dtype=np.int64)
        x[0][0] = _word_index[0]
        predicted_string += first_words[_i]
        predicted_string += ' '
        inputs = torch.from_numpy(x)
        inputs = inputs.to(device)
        outputs = trained_model(inputs)
        y_predict = outputs.cpu().detach().numpy()[0]
        state = trained_model.get_state()
        trained_model.set_state(state)

    for _i in range(PREDICT_LENGTH):
    # Predict PREDICT_LENGTH words.
        new_word_index = np.argmax(y_predict)
        _word = tokenizer.sequences_to_texts([[new_word_index]])
        x[0][0] = new_word_index
        predicted_string += _word[0]
        predicted_string += ' '
        inputs = torch.from_numpy(x)
        inputs = inputs.to(device)
        outputs = trained_model(inputs)
        y_predict = outputs.cpu().detach().numpy()[0]
        state = trained_model.get_state()
        trained_model.set_state(state)
    print(predicted_string)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All of the preceding code had to do with building and using a language model. The next code snippet adds some functionality to explore the learned embeddings. We first read out the word embeddings from the Embedding layer by accessing the weight variable from the model's first layer, which represents the Embedding layer. We move the weights back to the CPU and convert them to NumPy format. We first have to call detach() and clone().

    We then declare a list of a number of arbitrary lookup words. This is followed by a loop that does one iteration per lookup word. The loop uses the Tokenizer to convert the lookup word to a word index, which is then used to retrieve the corresponding word embedding. The Tokenizer functions are generally assumed to work on lists. Therefore, although we work with a single word at a time, we need to provide it as a list of size 1, and then we need to retrieve element zero ([0]) from the output.

    Once we have retrieved the corresponding word embedding, we loop through all the other embeddings and calculate the Euclidean distance to the embedding for the lookup word using the NumPy function norm(). We add the distance and the corresponding word to the dictionary word_indices. Once we have calculated the distance to each word, we simply sort the distances and retrieve the five word indices that correspond to the word embeddings that are closest in vector space. We use the Tokenizer to convert these indices back to words and print them and their corresponding distances.
    """)
    return


@app.cell
def _(np, tokenizer, trained_model):
    # Explore embedding similarities.
    it = trained_model.modules()
    next(it)
    embeddings = next(it).weight
    embeddings = embeddings.detach().clone().cpu().numpy()
    lookup_words = ['the', 'of']
    for lookup_word in lookup_words:
        lookup_word_indexed = tokenizer.texts_to_sequences([lookup_word])
        print('words close to:', lookup_word)
        lookup_embedding = embeddings[lookup_word_indexed[0]]
        word_indices = {}
        for _i, embedding in enumerate(embeddings):
            distance = np.linalg.norm(embedding - lookup_embedding)
            word_indices[distance] = _i  # Calculate distances.
        for distance in sorted(word_indices.keys())[:5]:
            _word_index = word_indices[distance]
            _word = tokenizer.sequences_to_texts([[_word_index]])[0]
            print(_word + ': ', distance)
        print('')  # Print sorted by distance.
    return


if __name__ == "__main__":
    app.run()
