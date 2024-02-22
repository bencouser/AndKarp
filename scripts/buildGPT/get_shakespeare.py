import requests

# URL of the dataset to download
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the content of the response to a file in the data folder
    with open("../data/raw/tiny_shakespeare.txt", "w") as file:
        file.write(response.text)
    print("Dataset downloaded and saved successfully.")
else:
    print(
        f"Failed to download the dataset. Status code: {response.status_code}")
