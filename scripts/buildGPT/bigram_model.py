import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from tensorflow import keras

# Read the tiny_shakespeare.txt
with open("../../data/raw/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()


# Create a vocabulary from the text
chars = sorted(list(set(text)))
vocab_size = len(chars)


# Create a dictionary to convert characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# Enconding and decoding functions
def encode(input_string):
    return [stoi[c] for c in input_string]


def decode(input_list):
    return "".join([itos[i] for i in input_list])


# Test the encode and decode functions
print(encode("hello"))
print(decode(encode("hello")))


# Convert shakespeare to a tensor
data = tf.convert_to_tensor(encode(text), dtype=tf.int32)


# Split the data into training and validation sets
TRAIN_SPLIT = 0.9
train_size = int(len(data) * TRAIN_SPLIT)
train_data = data[:train_size]
val_data = data[train_size:]


# Define block size
BLOCK_SIZE = 8
train_data[: BLOCK_SIZE + 1]


# Demonstrating training
# x = train_data[:BLOCK_SIZE]
# y = train_data[1:BLOCK_SIZE+1]
# for t in range(BLOCK_SIZE):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")


# Set seed
tf.random.set_seed(69)
BATCH_SIZE = 4
BLOCK_SIZE = 8


def get_slice(i):
    return data[i : i + BLOCK_SIZE]


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = tf.random.uniform(
        shape=(BATCH_SIZE,), minval=0, maxval=len(data) - BLOCK_SIZE, dtype=tf.int32
    )
    x = tf.map_fn(get_slice, ix, dtype=tf.int32)
    y = tf.map_fn(get_slice, ix + 1, dtype=tf.int32)
    return x, y


xb, yb = get_batch("train")
print("inputs:")
print(xb)
print("targets:")
print(yb)

print("----")

xb, yb = get_batch("train")
print("inputs:")
print(xb)
print("targets:")
print(yb)

print("----")

for b in range(BATCH_SIZE):  # batch dimension
    for t in range(BLOCK_SIZE):  # time dimension
        context = xb[b, : t + 1]
        context_list = list(context.numpy())
        target = yb[b, t]
        # print(f"when input is {context_list} the target: {target}")


# Create Bigram Language Model
class BigramLanguageModel(keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Using an Embedding layer similar to PyTorch's nn.Embedding
        self.token_embedding_table = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=vocab_size
        )

    def call(self, inputs, targets=None):
        # inputs and targets should be (B, T) tensors of integers
        # passing inputs through embedding layer
        # batch size, sequence length and embedding dimensions
        logits = self.token_embedding_table(inputs)  # (B, T, C)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = tf.reshape(logits, (B * T, C))
            targets_reshaped = tf.reshape(targets, (B * T,))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_reshaped, labels=targets_reshaped
            )
            loss = tf.reduce_mean(loss)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens given a starting token idx
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx, targets=None)  # Predict next token logits
            # Focus on the logits for the last token in the sequence
            logits = logits[:, -1, :]
            # Convert logits to probabilities
            probs = tf.nn.softmax(logits, axis=-1)
            # Sample next token from the probabilities
            idx_next = tf.random.categorical(probs, num_samples=1, dtype=tf.int32)
            # Append sampled token to the sequence
            idx = tf.concat([idx, idx_next], axis=-1)

        return idx


model = BigramLanguageModel(vocab_size=vocab_size)
# Estimate the loss with the formula below
# loss = -log(p) -> loss = -log(1/65) = 4.17
# logits, loss = model(xb, yb) # This matches well

# idx = tf.zeros((1, 1), dtype=tf.int32)
# generated_tokens = model.generate(idx, max_new_tokens=100)
# print("Bigram model generated text:")
# print(decode(generated_tokens.numpy()[0].tolist()))


print("Training the Bigram model")
optimizer = keras.optimizers.legacy.Adam(learning_rate=1e-2)
BATCH_SIZE = 64
n_epochs = 10
# mean_loss = keras.metrics.Mean()
# metrics = [keras.metrics.MeanAbsoluteError()]

loss_history = numpy.zeros(n_epochs)
# Training loop - How can i add progress bar?
for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, BATCH_SIZE + 1):
        xb, yb = get_batch("train")
        with tf.GradientTape() as tape:
            logits, loss = model(xb, yb)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("Training Loss: ", loss.numpy())
    loss_history[epoch - 1] = loss.numpy()
    # get test loss
    xb, yb = get_batch("val")
    logits, loss = model(xb, yb)
    print("Validation loss: ", loss.numpy())


idx = tf.zeros((1, 1), dtype=tf.int32)
generated_tokens = model.generate(idx, max_new_tokens=500)
print("Bigram model generated text:")
print(decode(generated_tokens.numpy()[0].tolist()))
print(model.summary())

plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("../../images/bigram_loss.png")
plt.show()
plt.close()
