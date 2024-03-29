import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

# Hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_iters = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ----------------


# Read the tiny_shakespeare.txt
with open("../../data/raw/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string


# Train and test split
data = tf.convert_to_tensor(encode(text), dtype=tf.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_slice(i):
    return data[i : i + block_size]


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = tf.random.uniform(
        shape=(batch_size,), minval=0, maxval=len(data) - block_size, dtype=tf.int32
    )
    x = tf.map_fn(get_slice, ix, dtype=tf.int32)
    y = tf.map_fn(get_slice, ix + 1, dtype=tf.int32)
    return x, y


def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = tf.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.numpy()
        out[split] = tf.math.reduce_mean(losses).numpy()
    model.train()
    return out


class Head(keras.Model):
    """
    A single head of self attention
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = keras.layers.Dense(head_size, use_bias=False, input_shape=(n_embd,))
        self.query = keras.layers.Dense(
            head_size, use_bias=False, input_shape=(n_embd,)
        )
        self.value = keras.layers.Dense(
            head_size, use_bias=False, input_shape=(n_embd,)
        )
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ tf.transpose(k, perm=[0, 2, 1]) * C**-0.5
        wei = wei
        wei = tf.nn.softmax(wei, axis=-1)
