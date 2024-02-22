import tensorflow as tf
from tensorflow import keras

# Read the tiny_shakespeare.txt
with open('../data/raw/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Print the length of the text and the first 250 characters
print(len(text))
print(text[:250])


# Create a vocabulary from the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


# Create a dictionary to convert characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(input_string):
    return [stoi[c] for c in input_string]


def decode(input_list):
    return ''.join([itos[i] for i in input_list])


# Test the encode and decode functions
print(encode('hello'))
print(decode(encode('hello')))


# Convert shakespeare to a tensor
data = tf.convert_to_tensor(encode(text), dtype=tf.int32)
# print(data.shape, data.dtype)
# print(data[:1000])


# Split the data into training and validation sets
TRAIN_SPLIT = 0.9
train_size = int(len(data) * TRAIN_SPLIT)
train_data = data[:train_size]
val_data = data[train_size:]
# print(train_data.shape, val_data.shape)


# Define block size
BLOCK_SIZE = 8
train_data[:BLOCK_SIZE + 1]


# Demonstrating training
# x = train_data[:BLOCK_SIZE]
# y = train_data[1:BLOCK_SIZE+1]
# for t in range(BLOCK_SIZE):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")


# Set seed
tf.random.set_seed(42)
BATCH_SIZE = 4
BLOCK_SIZE = 8


def get_slice(i):
    return data[i:i+BLOCK_SIZE]


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform(shape=(BATCH_SIZE,),
                           minval=0,
                           maxval=len(data) - BLOCK_SIZE,
                           dtype=tf.int32)
    x = tf.map_fn(get_slice, ix, dtype=tf.int32)
    y = tf.map_fn(get_slice, ix + 1, dtype=tf.int32)
    return x, y


xb, yb = get_batch('train')
print('inputs:')
print(xb)
print('targets:')
print(yb)

print('----')

for b in range(BATCH_SIZE):  # batch dimension
    for t in range(BLOCK_SIZE):  # time dimension
        context = xb[b, :t+1]
        context_list = list(context.numpy())
        target = yb[b, t]
        print(f"when input is {context_list} the target: {target}")


# Create Bigram Language Model


class BigramLanguageModel(keras.Model):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Using an Embedding layer similar to PyTorch's nn.Embedding
        self.token_embedding_table = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=vocab_size)

    def call(self, inputs, targets=None):
        # inputs and targets should be (B, T) tensors of integers
        # passing inputs through embedding layer
        # batch size, sequence length and embedding dimensions
        logits = self.token_embedding_table(inputs)  # (B, T, C)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = tf.reshape(logits, (B*T, C))
            targets_reshaped = tf.reshape(targets, (B*T,))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_reshaped, labels=targets_reshaped)
            loss = tf.reduce_mean(loss)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx, targets=None)  # Predict next token logits
            # Focus on the logits for the last token in the sequence
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits)  # Convert logits to probabilities
            # Sample next token from the probabilities
            idx_next = tf.random.categorical(probs, num_samples=1)
            # ensure idx_next has the same shape as idx
            idx_next = tf.cast(idx_next, idx.dtype)
            # Append sampled token to the sequence
            idx = tf.concat([idx, idx_next], axis=-1)

        return idx


model = BigramLanguageModel(vocab_size=vocab_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

idx = tf.zeros((1, 1), dtype=tf.int32)
print("idx")
print(idx)
generated_tokens = model.generate(idx, max_new_tokens=100)

print("Bigram model generated text:")
print(decode(generated_tokens.numpy()[0].tolist()))


print("Training the Bigram model")
optimizer = keras.optimizers.legacy.Adam(learning_rate=1e-3)
BATCH_SIZE = 32
n_epochs = 10
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, 100 + 1):
        xb, yb = get_batch('train')
        with tf.GradientTape() as tape:
            logits, loss = model(xb, yb)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     mean_loss(loss)
    #     for metric in metrics:
    #         metric(yb, logits)
    # for metric in [mean_loss] + metrics:
    #     metric.reset_states()
    print("Loss: ", loss)


idx = tf.zeros((1, 1), dtype=tf.int32)
generated_tokens = model.generate(idx, max_new_tokens=100)
print("Bigram model generated text:")
print(decode(generated_tokens.numpy()[0].tolist()))
