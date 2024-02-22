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

