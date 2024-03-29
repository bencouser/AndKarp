import keras
import matplotlib.pyplot as plt
import tensorflow as tf

if not tf.config.list_physical_devices("GPU"):
    print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
    print("Change to MyEnv!")


def load_data(file_path):
    """
    Load the data from the file path
    :param file_path: the path to the file
    :return: the text
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def encode_data(input_data):
    """
    Encode the data
    :param input_data: the input data
    :return: the encoded data
    """
    text_vec_layer = keras.layers.TextVectorization(
        split="character",
        standardize="lower",
    )
    text_vec_layer.adapt([text])
    encoded = text_vec_layer([text])[0]
    encoded -= 2
    n_tokens = text_vec_layer.vocabulary_size() - 2
    print(list(to_dataset(text_vec_layer(["To be"])[0], length=4)))
    return encoded, n_tokens


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


if __name__ == "__main__":
    text = load_data("../../data/raw/tiny_shakespeare.txt")
    print(text[:80])
    encoded, n_tokens = encode_data(text)
    print("Len of encoded data: ", len(encoded))
    print("Number of tokens: ", n_tokens)

    LENGTH = 100
    TOTAL_SAMPLES = 1_000_000
    tf.random.set_seed(69)
    train_set = to_dataset(
        encoded[:TOTAL_SAMPLES], length=LENGTH, shuffle=True, seed=42
    )
    val_set = to_dataset(encoded[TOTAL_SAMPLES:1_060_000], length=LENGTH)
    test_set = to_dataset(encoded[TOTAL_SAMPLES:], length=LENGTH)
    BATCH_SIZE = 1

    steps_per_epoch = TOTAL_SAMPLES // BATCH_SIZE // LENGTH

    model = keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
            tf.keras.layers.GRU(128, return_sequences=True),
            tf.keras.layers.Dense(n_tokens, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
    )
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        "gru_model.keras", monitor="val_accuracy", save_best_only=True
    )
    history = model.fit(
        train_set.repeat(),
        epochs=10,
        validation_data=val_set,
        callbacks=[model_ckpt],
        steps_per_epoch=steps_per_epoch,
    )
