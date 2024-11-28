import streamlit as st
from predictor import CricketPredictor

# Create an instance of CricketPredictor
predictor = CricketPredictor()

# Streamlit user interface
st.title("Cricket Match Prediction")

# Select Team A (Always India)
st.subheader("Select Team A (Always India)")
st.write("Team A (India) is fixed.")

# Static Team B Selection
st.subheader("Select Team B")
teams = predictor.teams['Team Name'].tolist()  # Assuming the Team Name column exists
team_b = st.selectbox("Choose Team B:", teams)

# Show Playing XI for Team A
st.subheader("Select Playing XI for Team A (India)")
india_players = predictor.players[predictor.players['Team ID'] == 6]['Name'].tolist()  # Filter for India Players
selected_india_players = st.multiselect("Choose Playing XI for India:", india_players)

# Show Playing XI for Team B
st.subheader("Select Playing XI for Team B")
b_team_id = predictor.teams[predictor.teams['Team Name'] == team_b]['Team ID'].values[0]  # Get the Team ID of Team B
b_players = predictor.players[predictor.players['Team ID'] == b_team_id]['Name'].tolist()
selected_b_players = st.multiselect(f"Choose Playing XI for {team_b}:", b_players)

# Country Selection
st.subheader("Select Country, Town, and Ground")

# Load country names into a dropdown
countries = predictor.countries[['Country ID', 'Country Name']].set_index('Country ID')
selected_country_id = st.selectbox("Choose Country:", countries.index, format_func=lambda x: countries.loc[x, 'Country Name'])

# Get towns corresponding to the selected country
towns_in_country = predictor.towns[predictor.towns['Country ID'] == selected_country_id]
towns = towns_in_country[['Town ID', 'Town Name']].set_index('Town ID').to_dict()['Town Name']

# Town Selection
if not towns:  # Check if there are towns available for the selected country
    st.warning("No towns found for the selected country.")
else:
    selected_town_id = st.selectbox("Choose Town:", list(towns.keys()), format_func=lambda x: towns[x])

    # Get corresponding grounds in the selected town
    grounds_in_town = predictor.grounds[predictor.grounds['Town ID'] == selected_town_id]['Ground Name'].tolist()
    selected_ground = st.selectbox("Choose Ground:", grounds_in_town)

# Preparing player IDs from selected names
india_player_ids = predictor.players[predictor.players['Name'].isin(selected_india_players)]['ID'].tolist()
b_player_ids = predictor.players[predictor.players['Name'].isin(selected_b_players)]['ID'].tolist()

# Run prediction logic
if st.button("Predict Match Outcome"):
    if len(india_player_ids) != 11 or len(b_player_ids) != 11:
        st.error("Both teams must have exactly 11 players selected.")
    else:
        # Get town name from ID for the prediction
        town_name = towns[selected_town_id]
        win_probability = predictor.predict_win_probability(india_player_ids, b_player_ids, selected_ground, town_name)
        st.write(f"Probability of India winning against {team_b}: {win_probability[0] :.2f}%")
