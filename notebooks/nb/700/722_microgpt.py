import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # microgpt

    /// attention | Copyright Notice

    This file is a copy of Andrej Karpathy's [microgpt](https://karpathy.ai/microgpt.html) model. The Markdown documentation is inspired by Sumit Pandey's Medium post [Andrej Karpathy Just Built an Entire GPT in 243 Lines of Python](https://www.towardsdeeplearning.com/andrej-karpathy-just-built-an-entire-gpt-in-243-lines-of-python-7d66cfdfa301). The last cell's graphviz implementation is an addition to the original code by [Badaszz](https://github.com/Badaszz/LLM_Grind/blob/main/microgpt-badasz.ipynb), but I replicated the results in Mermaid.

    Note that Karpathy has updated the code since Pandey wrote the blog post. The current solution is only 200 lines (as of 2026-02-23).
    ///

    This courses teacher's (Sourander) only additions are:

    * converting it to Marimo Notebook
    * moving a lot of the documentation from in-line to Markdown cells
    * Adding some new documentation, both from lectures notes and from Pandey's post.
    * Inference moved to a function. Start letter can be given.
    * data is expected to be in `data/etunimet.txt` file. Longest name in this document is 15 characters, so the context size is 16 (15 + 1 for BOS).

    **Reason for reproduction**: Aim is to try to improve readability. Make code quicker to access for the students in a familiar Marimo format.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import and Load Data
    """)
    return


@app.cell
def _():
    allowed_characters = set('abcdefghijklmnopqrstuvwxyz-äöå')
    return (allowed_characters,)


@app.cell
def _(allowed_characters):
    import math
    import random

    from pathlib import Path

    # Let there be order among chaos
    random.seed(42)

    # Dataset
    localfile = Path("data/etunimet.txt")
    if not localfile.exists():
        print("Download the file from https://avoindata.suomi.fi/data/fi/dataset/none")
        print("Keep e.g. 'Miehet kaikki' and 'Naiset kaikki' sheet names.")

    # Let there be a Dataset `docs`: list[str]
    # Drop entire name if it contains any character outside a-z + '-' + 'äöå'.
    docs = []
    n_dropped = 0
    for line in open(localfile):
        name = line.strip().lower()
        if not name:
            continue
        if all(c in allowed_characters for c in name):
            docs.append(name)
        else:
            n_dropped += 1


    print(f"Kept: {len(docs)}, Dropped: {n_dropped}")
    docs = list(set(docs))
    longest = max(docs, key=len)
    print(f"Uniques: {len(docs)} (longest: {longest} ({len(longest)})")
    random.shuffle(docs)
    print(f"first 5 lines: ", docs[:5])
    return docs, longest, math, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Count Unique Characters

    Those are the [a-z] + `BOS` (Beginning of Sequence)

    1. token id for a special Beginning of Sequence (BOS) token
    2. total number of unique tokens, +1 is for BOS

    Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    """)
    return


@app.cell
def _(docs):
    uchars = sorted(set(''.join(docs))) 
    BOS = len(uchars)            # (1)
    vocab_size = len(uchars) + 1 # (2)
    print(f"vocab size: {vocab_size}")
    return BOS, uchars, vocab_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Teacher comments
    The initial versions if microgpt included both `BOS` and `EOS`. The current solution has been optimized so that there exists only a single special token.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Autograd engine

    Here are comments to items in the code. They follow the same numbering. So list number 1 points to a comment `# (1)`

    1. scalar value of this node calculated during forward pass
    2. derivative of the loss w.r.t. this node, calculated in backward pass
    3. children of this node in the computation graph
    4. local derivative of this node w.r.t. its children

    Let there be Autograd to recursively apply the chain rule through a computation graph
    """)
    return


@app.cell
def _(math):
    class Value:
        # Python optimization for memory usage
        __slots__ = ('data', 'grad', '_children', '_local_grads')

        def __init__(self, data, children=(), local_grads=()):
            self.data = data                # (1)
            self.grad = 0                   # (2)
            self._children = children       # (3)
            self._local_grads = local_grads # (4)

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(
                self.data + other.data, (self, other), (1, 1))

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(
                self.data * other.data, (self, other), (other.data, self.data))

        def __pow__(self, other): 
            return Value(
                self.data**other, (self,), (other * self.data**(other-1),))

        def log(self): return Value(
            math.log(self.data), (self,), (1/self.data,))

        def exp(self): return Value(
            math.exp(self.data), (self,), (math.exp(self.data),))

        def relu(self): return Value(
            max(0, self.data), (self,), (float(self.data > 0),))

        def __neg__(self): return self * -1
        def __radd__(self, other): return self + other
        def __sub__(self, other): return self + (-other)
        def __rsub__(self, other): return other + (-self)
        def __rmul__(self, other): return self * other
        def __truediv__(self, other): return self * other**-1
        def __rtruediv__(self, other): return other * self**-1

        def backward(self):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._children:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            self.grad = 1
            for v in reversed(topo):
                for child, local_grad in zip(v._children, v._local_grads):
                    child.grad += local_grad * v.grad

    return (Value,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Teacher comments
    Each `Value` wraps a **single scalar number** (`float` or `int`) in its `data` field. The entire model is built by composing millions of these scalar nodes into a computation graph, one arithmetic operation at a time.

    The magic methods, named like `__this__`, define how the Value object behaves with mathematical operators. Those are...


    | magic | operator | short description |
    | ----: | :------: | :---------------- |
    | add | + | Add two values. Local gradient is 1. |
    | mul | * | Multiplies two values and use product rule for local gradient. |
    | pow | ** | Raise to power and use power rule for the local gradient. |
    | neg | -x | Negates a value by multiplying it by -1. |
    | truediv | / | Divides two values by multiplying the left operand by the right raised to -1. |

    Most typical operation in forward pass is the multiplication. Remember, if $c = ab$, then the product rule and chain rule apply, and...

    $$
    \nabla a = b \cdot \nabla c \\
    \nabla b = a \cdot \nabla c
    $$

    The division `a / b` is rewritten as `a * b**-1`. This avoids defining a separate division rule for backpropagation. Instead it reuses the already-implemented `__pow__` (power rule: $\frac{d}{db} b^{-1} = -b^{-2}$).

    The other operations that have a prefix letter `r`, like `__radd__`, are reflected. This means that instead of `val_object + 4` you do the reverse order `4 + val_object`. Thus, they simply flip the input order and use an already defined operator.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Architecture

    1. depth of the transformer neural network (number of layers)
    2. width of the network (embedding dimension)
    3. maximum context length of the attention window (note: the longest name is 15 characters)
    4. number of attention heads
    5. derived dimension of each head
    6. matrix initialization using a gaussian distribution
    7. In order, the query matrix, key matrix, output projection, fully connected layers I and II.
    8. flatten params into a single `list[Value]`
    """)
    return


@app.cell
def _(Value, longest, random, vocab_size):
    n_layer = 1                   # (1)
    n_embd = 16                   # (2)
    block_size = len(longest) + 1 # (3)
    n_head = 4                    # (4)
    head_dim = n_embd // n_head   # (5)

    matrix = lambda nout, nin, std=0.08: [
            [Value(random.gauss(0, std)) for _ in range(nin)
        ]
        for _ in range(nout)] # (6)

    # state dictionary for all learnable weights
    state_dict = {
        'wte': matrix(vocab_size, n_embd), 
        'wpe': matrix(block_size, n_embd), 
        'lm_head': matrix(vocab_size, n_embd)
    }

    for _i in range(n_layer):
        # (7)
        state_dict[f'layer{_i}.attn_wq'] = matrix(n_embd, n_embd)  # Q
        state_dict[f'layer{_i}.attn_wk'] = matrix(n_embd, n_embd)  # K
        state_dict[f'layer{_i}.attn_wv'] = matrix(n_embd, n_embd)  # V
        state_dict[f'layer{_i}.attn_wo'] = matrix(n_embd, n_embd)  # out
        state_dict[f'layer{_i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # FC
        state_dict[f'layer{_i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # FC2

    # (8)
    params = [p for mat 
              in state_dict.values()
              for row in mat for p in row]  
    print(f'num params: {len(params)}')
    return block_size, head_dim, n_head, n_layer, params, state_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Teacher comments
    Note that we have three main base matrices. Two of these are called WTE (Word Token Embedding) and WPE (Word Position Embedding), and the third is the `lm_head` (Language Model Head).

    * `WTE` has a dimensionality of `vocab_size * n_embed` (27 x 16). This is the embedding for tokens, effectively describing each letter with a 16d embedding vector.
    * `WPE` has a dimensionality of `block_size * n_embed` (16 x 16). This is the positional encoding.
    * `lm_head` has a dimensionality of `vocab_size * n_embed` (27 x 16) and is used to map the final outputs back to the vocabulary.

    In this case, we have only 1 attention layer. You can increase this count if you want. These are the Attention Blocks that are stitched after each other. Each Attention Block will contain `n_head` parallel heads. This is the Multi-Head Attention model mentioned in the lecture. Changing these values obviously affects the count of parameter and thus the time it takes to train to the model...

    * `n_layer = 1`: 4192 params. 0:40 training.
    * `n_layer = 2`: 7264 params. 1:40 training.

    The state dict contents are all parameter matrices, meaning they are weights that we will update during the training, and utilize during the forward pass.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _():
    def linear(x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    def softmax(logits):
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def rmsnorm(x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    return linear, rmsnorm, softmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Teacher comments

    **linear(x, w)**

    Performs a matrix-vector multiplication ($W \cdot x$). Notice that there is no bias term added, which is a common optimization in modern LLMs.

    **softmax(logits)**

    Converts a vector of raw, unnormalized scores into a probability distribution where all values are between 0 and 1 and sum to 1.

    **rmsnorm(x)**

    Root Mean Square Normalization. It scales the input vector so that its values have a mean square of 1. As Pandey writes: *"Before each major computation, we normalize the values so they don’t explode or shrink to zero. RMSNorm is a simpler cousin of LayerNorm (which original GPT-2 uses)."*
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define the model architecture: a function mapping tokens and parameters to logits over what comes next.

    Follow GPT-2, blessed among the GPTs, with minor differences:

    * layernorm -> rmsnorm
    * no biases
    * GeLU -> ReLU

    And the code commentaries...

    1. token embedding
    2. position embedding
    3. joint token and position embedding
    """)
    return


@app.cell
def _(head_dim, linear, n_head, n_layer, rmsnorm, softmax, state_dict):
    ## Now for the GPT2 architecture
    def gpt(token_id, pos_id, keys, values):
        tok_emb = state_dict['wte'][token_id]          # (1)
        pos_emb = state_dict['wpe'][pos_id]            # (2)
        x = [t + p for t, p in zip(tok_emb, pos_emb)]  # (3)
        x = rmsnorm(x)

        for li in range(n_layer):
            # 1) Multi-head Attention block
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f'layer{li}.attn_wq'])
            k = linear(x, state_dict[f'layer{li}.attn_wk'])
            v = linear(x, state_dict[f'layer{li}.attn_wv'])

            keys[li].append(k)
            values[li].append(v)
            x_attn = []

            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs+head_dim]
                k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+head_dim] for vi in values[li]]

                attn_logits = [
                    sum(q_h[j] * k_h[t][j] 
                        for j in range(head_dim)) / head_dim**0.5 
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)

                head_out = [
                    sum(attn_weights[t] * v_h[t][j]
                        for t in range(len(v_h))) 
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)

            x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]

            # 2) MLP block
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state_dict['lm_head'])
        return logits

    return (gpt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Teacher comments
    The last cell of Marimo Notebook contains a Mermaid diagram of this GPT model. It is suggested to look at that diagrams and this code and identify what is what. Pandey explained it like this...

    > "The MLP expands the 16-dimensional vector to 64 dimensions (4 * n_embd), applies a non-linearity (squared ReLU), and squishes it back to 16.
    >
    > Why expand and compress? Think of it like brainstorming. You first generate lots of ideas (expand to 64), filter out the bad ones (ReLU kills negatives, squaring amplifies strong signals), then summarize (compress back to 16)."
    >
    > — Sumit Pandey

    The last linear function maps the final output back to the vocabulary. These are the raw logits that you have seen in the course in all classification heads. In the training loop, we need to take `probs = softmax(logits)` to get the probability. In inference, we will also scale these by temperature, which essentially allows the user to affect the model's behaviours on a scale deterministic—creative (or random, at least).

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adam and Hyperparameters

    Let there be Adam, the blessed optimizer and its buffers
    """)
    return


@app.cell
def _(params):
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params) # first moment buffer
    v = [0.0] * len(params) # second moment buffer

    ## The training loop
    num_steps = 500
    return beta1, beta2, eps_adam, learning_rate, m, num_steps, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop

    1. Take single document, tokenize it, surround it with BOS special token on both sides
    2. Forward the token sequence through the model, building up the computation graph all the way to the loss
    3. Final average loss over the document sequence. May yours be low.
    4. Backward the loss, calculating the gradients with respect to all model parameters
    5. Adam optimizer update: update the model parameters based on the corresponding gradients. Uses linear learning rate decay.
    """)
    return


@app.cell
def _(
    BOS,
    beta1,
    beta2,
    block_size,
    docs,
    eps_adam,
    gpt,
    learning_rate,
    m,
    n_layer,
    num_steps,
    params,
    softmax,
    uchars,
    v,
):
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS] # (1)
        n = min(block_size, len(tokens) - 1)

        # (2)
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses) # (3)

        loss.backward() # (4)

        lr_t = learning_rate * (1 - step / num_steps) # (5)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        print(f"step {step+1:4d} / {num_steps:4d} |"
              f" loss {loss.data:.4f}", end='\r')
    return


@app.cell
def _(BOS, block_size, gpt, n_layer, random, softmax, uchars, vocab_size):
    def inference(num_samples=20, temperature=0.5, start_letter=None):

        if start_letter is not None and start_letter not in uchars:
            raise ValueError(f"start_letter {start_letter!r} not in vocabulary")

        print("\n--- inference (new, hallucinated names) ---")
        for sample_idx in range(num_samples):
            keys = [[] for _ in range(n_layer)]
            values = [[] for _ in range(n_layer)]

            if start_letter is not None:
                # Prime the model with BOS, then the start letter token
                first_token_id = uchars.index(start_letter)
                gpt(BOS, 0, keys, values)
                sample = [start_letter]
                token_id = first_token_id
                start_pos = 1
            else:
                token_id = BOS
                sample = []
                start_pos = 0

            for pos_id in range(start_pos, block_size):
                logits = gpt(token_id, pos_id, keys, values)
                probs = softmax([l / temperature for l in logits])
                token_id = random.choices(
                    range(vocab_size), 
                    weights=[p.data for p in probs]
                )[0]

                if token_id == BOS:
                    break    
                sample.append(uchars[token_id])
            print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

    return (inference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Teacher comments
    How does temperature work? If `temperature = 1`, the logits are unchanged, since `x / 1 = x`. If the temperature is less than 1, the division will **increase the absolute** logit values. For example, using a temperature of `0.5` will make logits `[2.0, 1.0]` double to `[4.0, 2.0]` Softmax intensifies these even further. Later on, these are being used as weights in the `random.choices()` call. Thus...

    * `temperature < 1` makes the model more deterministic ("confident")
    * `temperature > 1` makes the model more random ("unsure")

    Note that the prediction is **autoregressive**. Prediction is being fed back into the network until...

    * `BOS` is predicted, or...
    * We have predicted `block_size` amount of letters.

    Note that this *block size* is the **context length**. If this was a chatbot, the system prompt, your query, and essentially the chat history must fit into this context. Models like Anthropic Claude Sonnet 4.x, Gemini Pro 3.x, GPT 5.x and similar will have a context size of about 200k to 1M tokens, depending on the specific model. Also, the embedding size will be huge compared to ours. And, of course, their tokens are not characters `a-z` but subwords (word pieces, byte-pair encoded UTF-8 characters, ...).
    """)
    return


@app.cell
def _(inference):
    inference()
    return


@app.cell
def _(inference):
    inference(start_letter="j", temperature=0.8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extra: Plot Architecture
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
    graph TD
        Tok[Token ID] --> WTE[Token Embedding]
        Pos[Position ID] --> WPE[Position Embedding]
        WTE --> AddEmb((+))
        WPE --> AddEmb
        AddEmb --> InitNorm[RMSNorm]

        InitNorm --> BlockIn

        subgraph AttentionBlock [Attention Block x n_layer]
            BlockIn[Block Input] --> Norm1[RMSNorm]
            Norm1 --> MHA[Multi-Head Attention]
            MHA --> Add1((+))

            %% Residual 1
            BlockIn -->|Residual| Add1

            Add1 --> Norm2[RMSNorm]
            Norm2 --> MLP[MLP Block]
            MLP --> Add2((+))

            %% Residual 2
            Add1 -->|Residual| Add2
        end

        Add2 --> LMHead[LM Head]
        LMHead --> Softmax[Softmax]
        Softmax --> Output[Next Token Probabilities]
    """
    ).center()
    return


if __name__ == "__main__":
    app.run()
