Urdu Language Text Classification using Logistic Regression
This project is focused on building a text classification model for Urdu language data. The model preprocesses text data by performing several Natural Language Processing (NLP) techniques such as lemmatization, stopwords removal, and vectorization using Word2Vec. The final classification is performed using Logistic Regression, achieving an accuracy of 80%.

Project Overview
Steps Involved:
Data Cleaning:

Removal of punctuation, special characters, and unnecessary whitespace.
Tokenization of Urdu sentences into words.
Lemmatization:

Reducing words to their base forms for better consistency in the dataset.
Stopwords Removal:

Removing non-informative words (like "is," "the," etc.) to focus on the core content.
Word Embedding (Word2Vec):

Converting the cleaned Urdu text into vectorized form using Word2Vec, allowing the model to capture semantic relationships between words.
Modeling (Logistic Regression):

Training a logistic regression model on the vectorized data.
Achieved 80% accuracy on the validation set.
Challenges Faced
Working with Urdu language data posed several challenges:

Handling the unique grammatical and structural nuances of Urdu.
Limited availability of pre-trained language models and tools for Urdu as compared to English.
Managing complex preprocessing steps like tokenization and lemmatization for non-English scripts.
Requirements
To run this project, you'll need the following libraries:

numpy
pandas
scikit-learn
nltk
gensim
Install these dependencies using:

bash
Copy code
pip install -r requirements.txt
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/urdu-text-classification.git
Navigate to the project directory:
bash
Copy code
cd urdu-text-classification
Run the script to preprocess the data and train the model:
bash
Copy code
python train_model.py
To view a walkthrough of the project, check the video in the project folder.
Results
The Logistic Regression model achieved 80% accuracy on the validation set. Further improvements can be made by experimenting with different algorithms or fine-tuning the preprocessing techniques.

Future Work
Implementing more advanced models such as LSTM or BERT for improved accuracy.
Exploring transfer learning with pre-trained models for Urdu.
Fine-tuning hyperparameters to further boost performance.
