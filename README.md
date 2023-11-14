# Sentiment Classifier 

Basic Sentiment Classifier using hugging face pre trained transformer. Given csv or text it will classify the sentiment of the string which can be positive, negative, or neutral.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -r requirements.txt
```

## Usage

**Run sentiment classifier in single text**
```bash
input: python sentiment_classifier.py --text_input "I hate going to that restaurant"
output: {"model_output": "negative", "confidence_score": 97.77}
```

**Run sentiment classifier in csv.** \
Note: The CSV file should have a column of "text" to indicate the column to generate sentiment
```bash
input: python sentiment_classifier.py --input_csv_path "sentiment_test_cases.csv" --output_csv_path "output_sentiment_test.csv"
output: csv file with columns text, expected_sentiment, model_output, and confidence_score
```

**Get the accuracy of the classifier.** \
Note: csv should contain columns text, expected_sentiment, model_output, and confidence_score
```bash
input: python sentiment_classifier.py --get_accuracy Y --output_csv_path output_sentiment_test.csv
output: {"Accuracy": 83.53}
```
## Author

* **Patrick La Rosa**
    * [Github](https://github.com/pgplarosa)
    * [LinkedIn](https://www.linkedin.com/in/patricklarosa)

## References
1. [Twitter-roBERTa-base for Sentiment Analysis](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
