### News Credibility Prediction with Machine Learning 📰🤖

#### Overview:
This project focuses on predicting the credibility of news articles using machine learning techniques. It utilizes a dataset containing news articles labeled as credible or not credible to train a decision tree classifier model. The model is then deployed as a web application using Dash, where users can input news articles to get predictions about their credibility.

#### Project Structure:
- **Data Preparation**: The dataset is loaded and preprocessed to handle missing values and perform text preprocessing tasks like stemming and removing stopwords.
- **Model Training**: The preprocessed text data is used to train a decision tree classifier model.
- **Model Evaluation**: The trained model's accuracy is evaluated on a test dataset to assess its performance.
- **Model Deployment**: The trained model is serialized and saved to files for later use. Additionally, a web application is developed using Dash, allowing users to input news articles for credibility prediction.
  
#### Libraries Used:
- pandas: For data manipulation and analysis.
- scikit-learn: For machine learning tasks such as model training and evaluation.
- NLTK: For text preprocessing tasks like stemming and stopwords removal.
- Dash: For building the web application interface.

#### Files:
- `train.csv`: The dataset containing news articles and their corresponding labels.
- `app.py`: The Python script containing the Dash web application code.
- `model.pkl`: The serialized trained machine learning model.
- `vector.pkl`: The serialized TF-IDF vectorizer used for text feature extraction.

#### Usage:
1. **Training the Model**: Run the Jupyter notebook or Python script containing the model training code (`train_model.ipynb` or `train_model.py`).
2. **Deploying the Web Application**: Run the `app.py` script to start the Dash web application.
3. **Interacting with the Application**: Input a news article text into the provided textarea and click on the "Predict" button to get the credibility prediction.

#### Deployment:
The web application can be deployed locally or on a web server to make it accessible to users. Additionally, it can be integrated into existing platforms or systems for real-time credibility assessment of news articles.
