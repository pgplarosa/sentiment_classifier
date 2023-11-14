"""
################################################################################
## Module Name: sentiment_classifier.py
## Created by: Patrick La Rosa
## Created on: 14/11/2023
##
## Reference: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
###############################################################################
"""

# import libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import json

tqdm.pandas()

def preprocess(text):
    """ Preprocess text (username and link placeholders)
    
    Parameters
    ===========
    text       : str
                 text to preprocess
    
    Returns
    ==========
    preprocess :  str
                  preprocessed text
    """
    new_text = []

    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

def load_models():
    """ load models, tokenizers, label mapping from huggingface hub
    
    Returns
    =========
    model   :   RobertaForSequenceClassification
                pre-trained model for sentiment classifier
    tokenizer : RobertaTokenizerFast
                tokenizer used for the model
    labels    : list
                list of classes for sentiment classifier
    """
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

    # Load the model from your local directory
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Download label mapping
    labels = []
    mapping_link = ("https://raw.githubusercontent.com/cardiffnlp/"
                    "tweeteval/main/datasets/sentiment/mapping.txt")
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode("utf-8").split("\n")
        csvreader = csv.reader(html, delimiter="\t")
        labels = [row[1] for row in csvreader if len(row) > 1]
        
    return model, tokenizer, labels

def generate_sentiment(text, model, tokenizer, labels):
    """ generate sentiment with confidence score given the text
    
    Parameters
    ============
    text    :   str
                input text to get sentiment
    
    Returns
    ===========
    generate_sentiment : tuple
                         label string and confidence score
    """
    # preprocess text
    text = preprocess(text)
    
    # tokenize
    encoded_input = tokenizer(text, return_tensors="pt")
    
    # get sentiments
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    return labels[ranking[0]], scores[ranking[0]]

def get_metric(df):
    accuracy = round(((df["expected_sentiment"] 
                       == df["model_output"]).mean()) * 100, 2)
    return accuracy



if __name__ == "__main__":
    parser  = ArgumentParser()
    
    parser.add_argument("--input_csv_path", default="", 
    help="path of csv file to generate sentiments")
    parser.add_argument("--output_csv_path", default="output_sentiment_test.csv", 
    help="path of csv file to output sentiments")
    parser.add_argument("--text_input", default="", 
    help="string to evaluate")
    parser.add_argument("--get_accuracy", default="N", 
    help="set to Y if you want to get accuracy, default: N")
    
    args = parser.parse_args()
    
    if args.text_input:
        model, tokenizer, labels = load_models()
        sentiment = generate_sentiment(args.text_input, model, tokenizer, labels)
        output = {"model_output":sentiment[0], 
                  "confidence_score": round(sentiment[1] * 100, 2)}
        print(json.dumps(output))
        
    if args.input_csv_path:
        model, tokenizer, labels = load_models()
        df = pd.read_csv(args.input_csv_path)
        sentiments = df["text"].progress_apply(lambda x: 
                                               generate_sentiment(x, model, 
                                                                  tokenizer, labels))
        
        df["model_output"] = sentiments.apply(lambda x: x[0])
        df["confidence_score"] = sentiments.apply(lambda x: round(x[1] * 100, 2))
        df.to_csv(args.output_csv_path, index=False)
            
    if args.get_accuracy == "Y":
        try:
            df = pd.read_csv(args.output_csv_path)
            accuracy = get_metric(df)
            accuracy = {"Accuracy": accuracy}
            print(json.dumps(accuracy))
        except (FileNotFoundError, KeyError):
            print("Make sure you have a valid output_csv_path with columns text"
                  ", expected_sentiment, model_output, and confidence_score") 
            
            
    elif not args.text_input and not args.input_csv_path:
        print("Error: Add parameters text_input or input_csv_path to generate sentiments")
            