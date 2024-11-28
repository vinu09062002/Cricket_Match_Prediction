import pandas as pd
import os  # New import to handle directory creation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib  # For saving the model

def main():
    # Load datasets
    players = pd.read_csv('data/players.csv')
    matches = pd.read_csv('data/matches.csv')
    grounds = pd.read_csv('data/ground.csv')
    teams = pd.read_csv('data/team.csv')

    # Prepare historical match data
    matches = matches.merge(grounds, left_on='ground_id', right_on='Ground ID', how='left')
    matches = matches.merge(teams, left_on='winner_team_id', right_on='Team ID', how='left', suffixes=('', '_winner'))
    matches['outcome'] = (matches['winner_team_id'] == 6).astype(int)  # Assuming India has Team ID 6

    # Dropping unnecessary columns
    columns_to_drop = [  # Fill in with your actual columns to drop
         'slug', 'title', 'time_of_day', 'date', 'time',
       'status', 'status_description',
       'toss_choice', 
       'team_1_scoreInfo', 'team_2_scoreInfo',   
       'Team Name', 'Team Abbreviation', 'Image URL', 
    ]
    matches = matches.drop(columns=columns_to_drop)

    # Prepare features (X) and target (y)
    X = matches.drop(['match_id', 'team_1_id', 'team_2_id', 'outcome'], axis=1)
    y = matches['outcome']

    # One-Hot Encoding of categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with Grid Search
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    model = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3)
    model.fit(X_train, y_train)

    # Create the models directory if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Save the trained model
    joblib.dump(model.best_estimator_, f'{models_dir}/trained_model.pkl')
    print("Model has been trained and saved.")

if __name__ == '__main__':
    main()