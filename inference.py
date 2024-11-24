import torch
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine
from torch.utils.data import DataLoader
import re

from encoder import WordnPositionalSelfAttentionEmbeddings


class InferencePipeline:
    def __init__(self, model, vocabulary, labels, device="cpu", context_length=600):
        """
        Initialize the inference pipeline.

        Parameters
        ----------
        model : nn.Module
            The trained PyTorch model for inference.
        vocabulary : list
            A dictionary mapping words to their indices.
        device : str, optional
            The device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
        """
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.context_length = context_length
        self.labels = labels

    def preprocess(self, text, max_length):
        """
        Preprocess the input text for the model.

        Parameters
        ----------
        text : str
            The input text to predict on.
        max_length : int, optional
            Maximum length of the tokenized sequence, by default 500.

        Returns
        -------
        torch.Tensor
            A tensor of tokenized indices.
        """
        # Tokenize and convert to indices
        text = re.sub(r"[^A-Za-z]", " ", string=text)
        text = re.sub(r"\s+", " ", string=text).strip()

        tokens = text.lower().split()
        token_indices = [self.vocabulary.index(word, 0) for word in tokens[:max_length]]

        # Pad or truncate the sequence to max_length
        if len(token_indices) < max_length:
            token_indices += [0] * (max_length - len(token_indices))  # Pad with 0
        return torch.tensor(token_indices, dtype=torch.long).unsqueeze(
            0
        )  # Add batch dimension

    def predict(self, text):
        """
        Make a prediction for a given input text.

        Parameters
        ----------
        text : str
            The input text to predict on.

        Returns
        -------
        dict
            A dictionary with class probabilities and predicted class label.
        """
        # Preprocess the input
        input_tensor = self.preprocess(text, self.context_length).to(self.device)

        # Run the model
        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocess the output
        probabilities = torch.softmax(output, dim=-1).cpu().numpy().flatten()
        predicted_class = probabilities.argmax()

        return {
            "probabilities": probabilities.tolist(),
            "predicted_class": list(self.labels.keys())[predicted_class],
        }


def create_dataframe_record(filename: str):

    pages_text = ""
    try:
        for page_layout in extract_pages(filename):
            for element in page_layout:
                if isinstance(element, (LTTextBox, LTTextLine)):
                    pages_text = pages_text + "\n" + element.get_text()

    except Exception as e:
        print(f"Error extracting text: {e,filename}")
        return None
    return pages_text


if __name__ == "__main__":
    vocabulary = torch.load("./models/vocabulary.pt", map_location="cpu")
    model = WordnPositionalSelfAttentionEmbeddings(
        vocab_size=len(vocabulary),
        network_width=128,
        num_blocks=6,
        context_length=600,
        embedding_size=512,
        hidden_size=128,
    )
    labels = torch.load("./models/labels.pt", map_location="cpu")
    model.load_state_dict(torch.load("./models/model.pt", map_location="cpu"))
    output = InferencePipeline(model, vocabulary, labels).predict(
        create_dataframe_record(
            "data_foundation/raw/test_data/fuses/=littelfuse-fuse-217-datasheet.pdf"
        )
    )
    print(output)
