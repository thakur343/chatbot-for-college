import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


class ChatBot:
    def __init__(self, intents_file='intents.json', model_file='data.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bot_name = "Sam"

        try:
            with open(intents_file, 'r') as json_data:
                self.intents = json.load(json_data)

            data = torch.load(model_file)
            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]
            self.all_words = data['all_words']
            self.tags = data['tags']
            model_state = data["model_state"]

            self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
        except FileNotFoundError as e:
            print(f"Error: Required file not found. {e}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {intents_file}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    def get_response(self, msg):
        sentence = tokenize(msg)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])

        return "I do not understand..."


# Initialize the chatbot
chatbot = ChatBot()


# Function to be called from app.py
def get_response(msg):
    return chatbot.get_response(msg)


if __name__ == "__main__":
    print(f"Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f"{chatbot.bot_name}: {resp}")