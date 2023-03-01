# Disaster Response Pipeline Project
### libraries used
The libraries that were used are :
nltk, pandas, sys, sqlalchemy, sklearn, pickle and re 

### motivation for the project
data engineering, ML  and NLU were applied on this project to make an analyzation of messages on the web app that were sent while there is a disaster and that will be through cleaning the data then training the model and finally deploy it

### explanation of the files in the repository 
there are three folder :
1) data:
- disaster_messages.csv:
messages dataset
- disaster_categories.csv:
categories dataset
- DisasterResponse.db:
the SQLite that has messages and categories datasets
- process_data.py:
Loads the datasets of messages and categories and then merge them after that clean and store it in a SQLite
2) models:
- train_classifier.py: 
script of ML that train the dataset
3) app:
- run.py:
flask file that run the wep app
- the templates folder that has two html files that were used to run the web app: go.html and master.html

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### summary of the results
the result is a flask web app for disaster Response Pipeline that will analyze the user entered messege via cleaing the dataset and then training it by using ML, NLU and data engineering 

### acknowledgements
All thanks to Udacity for the amazing course and Misk Academy for giving the chance to have this course.


