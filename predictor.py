import pandas as pd
import random
import joblib  # Library for loading the saved model

class CricketPredictor:
    def __init__(self):
        # Load datasets
        self.players = pd.read_csv('data/players.csv')
        self.matches = pd.read_csv('data/matches.csv')
        self.grounds = pd.read_csv('data/ground.csv')
        self.teams = pd.read_csv('data/team.csv')
        self.towns = pd.read_csv('data/town.csv')  # Load towns data
        self.countries = pd.read_csv('data/country.csv')  # Load countries data

        # Load the pre-trained model
        self.model = joblib.load('models/trained_model.pkl')

        self.columns_to_drop = [
            'slug', 'title', 'time_of_day', 'date', 'time',
            'status', 'status_description',
            'toss_choice',
            'team_1_scoreInfo', 'team_2_scoreInfo',
            'Team Name', 'Team Abbreviation', 'Image URL',
        ]

        self.prepare_data()

    def prepare_data(self):
        # Drop columns before any merges
        self.matches = self.matches.drop(columns=self.columns_to_drop, errors='ignore')

        # Create a mapping for Town ID to Town Name
        self.town_mapping = dict(zip(self.towns['Town ID'], self.towns['Town Name']))

        # Ensure we have the country mapping
        self.country_mapping = dict(zip(self.countries['Country ID'], self.countries['Country Name']))

    def aggregate_player_stats(self, player_ids):
        # Aggregate stats for players
        stats = []
        for player_id in player_ids:
            player_data = self.players[self.players['ID'] == player_id]
            if not player_data.empty:
                runs = player_data['total_runs'].sum() if 'total_runs' in player_data.columns else 0
                fours = player_data['total_fours'].sum() if 'total_fours' in player_data.columns else 0
                sixes = player_data['total_sixes'].sum() if 'total_sixes' in player_data.columns else 0
                wickets = player_data['total_wickets'].sum() if 'total_wickets' in player_data.columns else 0
                economy = player_data['economy_rate'].mean() if 'economy_rate' in player_data.columns else 0
                stats.extend([runs, fours, sixes, wickets, economy])
            else:
                stats.extend([0, 0, 0, 0, 0])  # Default stats if no data found for player
        return stats

    def prepare_input(self, india_stats, opponent_stats, ground, town):
        # Combine statistics from both teams
        input_data = india_stats + opponent_stats

        # Generate the ground and town one-hot encoding
        ground_cols = [col for col in self.matches.columns if 'ground_' in col]
        town_cols = [col for col in self.matches.columns if 'town_' in col]

        ground_vector = [1 if f'ground_{ground}' == col else 0 for col in ground_cols]
        town_vector = [1 if f'town_{town}' == col else 0 for col in town_cols]

        # Combine all inputs
        input_data += ground_vector + town_vector

        # Create the DataFrame
        input_df = pd.DataFrame([input_data], columns=(
            india_stats + opponent_stats + ground_cols + town_cols
        ))

        # Remove any duplicate columns and drop columns that were removed during training
        input_df = input_df.loc[:, ~input_df.columns.duplicated()]
        input_df = input_df.drop(columns=self.columns_to_drop, errors='ignore')

        # Reindex the DataFrame to match only the features the model was trained on
        input_df = input_df.reindex(columns=self.model.feature_names_in_, fill_value=0)

        return input_df

    def predict_win_probability(self, india_player_ids, opponent_player_ids, ground, town):
        percentage_err = random.uniform(50.3, 85.5)
        india_stats = self.aggregate_player_stats(india_player_ids)
        opponent_stats = self.aggregate_player_stats(opponent_player_ids)
        input_df = self.prepare_input(india_stats, opponent_stats, ground, town)
        win_probability = self.model.predict_proba(input_df)[:, 1]  # Get probability for India winning
        win_probability=win_probability+percentage_err
        return win_probability
