# AlkohoLove-recommender

## Installing dependencies

Use `$ pip install -r requirements.txt`

## Local development

When running with IDE, set Run/Debug configurations with env variables:  
`DATABASE_URL` - url to MongoDB

## Docs

Docs are available under `/docs` path

## SVD Recommender

The learning method starts by retrieving user reviews and creating a rating matrix. 
Then, using the GridSearchCV object, we perform hyperparameter tuning,
namely finding the best set of parameters, i.e. the one giving the smallest RMSE error.
Having such a set, we teach the model. 
The next step is to create an anti set, i.e. a user-item combination, where the evaluation is unknown.
On the resulting set, predictions are created for each user-item pair, which are then sorted and saved.
The recommend method returns the top N saved recommendations for the specified user.


## SIM Recommender

The learning method starts by retrieving the items from the database.
A list is then created, containing a string representation of each item.
Each representation is normalized to lowercase.
In addition, each word is lemmatized, and stop-words and punctuation marks are removed.
The next step is to create an object of class Dictionary, which encapsulates the mapping between words and their identifiers.
Using the mapping, bags-of-words is created for each document.
Based on these, a similarity matrix is created.
The final step is to save the results of the values from the matrix to the database.
Only similarities greater than 0.2 are saved.

The recommendation method starts by fetching the user's reviews, in case there are none, an empty list is returned.
Then similar items to the rated ones are retrieved.
The next step is to loop through each item and calculate the prediction.
The prediction is calculated based on the weighted sum formula presented by Sarwar B., Karypis B., 
Konstan J., Riedl J. (2001) in "Item-Based Collaborative Filtering Recommendation Algorithms".
For this purpose, the second loop goes through similar rated items.
A weight is calculated, which is the difference between the item's rating and the user's average rating.
Then the multiplication result, between the weight and the similarity of the item, is added to the sum of the weighted similarity.
The similarity value is added to the sum of the ordinary similarity.
The prediction for the subject is equal to the result of dividing the weighted similarity sum by the similarity sum.
The method returns the top N recommendations.

## COMMITS CONVENTION

Commits merged into master should follow conventional
commits [convention](https://gist.github.com/Zekfad/f51cb06ac76e2457f11c80ed705c95a3). Long story
short: `<type>: <message> [<jira-ticket>]`