Sentiment Analysis of Movie Reviews
This project focuses on performing sentiment analysis on movie reviews. Using a Naive Bayes classification model, we can predict whether a review is positive or negative based on the text.

Steps:
Data Loading: The dataset contains movie reviews along with their sentiment labels (positive or negative). The reviews are processed using a CountVectorizer to convert text into numerical data for the machine learning model.
Sentiment Distribution: A bar chart is generated to visualize the distribution of positive and negative reviews.
Review Length Analysis: We analyze the distribution of review lengths by plotting a histogram showing how many words each review contains.
Model Training: We use the Naive Bayes algorithm to train the model using 80% of the data and evaluate it on the remaining 20%.
Model Evaluation: The model achieves an accuracy of 85%, with precision and recall values for both positive and negative reviews. A confusion matrix is plotted to show the model's performance.
Prediction on New Comments: The model can predict the sentiment of new, unseen reviews, and provides a sentiment score (probability) for each review.
Key Results:
Accuracy: 85.28%
Confusion Matrix: Visualizes true vs. predicted sentiment for test data.
New Comment Testing: You can test new reviews to predict whether they are positive or negative.
Libraries Used:
Pandas, Matplotlib, Scikit-learn, WordCloud
