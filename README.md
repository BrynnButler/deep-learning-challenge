# deep-learning-challenge
module 21

# Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received access to a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

# Process Overview

1. **Preprocessing**
   - Dropped irrelevant columns
   - One-hot encoded categorical variables using `pd.get_dummies()`
   - Scaled numerical features with `StandardScaler`

2. **Initial Model**
   - 2 hidden layers (80 & 30 neurons, ReLU)
   - Output: 1 neuron (Sigmoid)
   - Accuracy: **72.39%**

3. **Optimization**
   - Increased to 3 hidden layers
   - Added more neurons (128 → 64 → 32)
   - Switched activations, added `Dropout`, and increased training epochs
   - Accuracy remained around **72.39%**
  
# Final Thoughts

While the model didn't quite reach the target 75% accuracy, it showed strong performance with consistent training. Further tuning or testing other models (like Random Forest or XGBoost) could help increase accuracy for future iterations.
