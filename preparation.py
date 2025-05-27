import fastf1
import pandas as pd


# Function to load all lap data for a season
def load_grand_prix_laps_data(year):
    all_laps = pd.DataFrame()  # DataFrame to collect all lap data

    # List of Grand Prix names for the given season (the 2024 season in this case)
    grand_prix_names = [
        "Bahrain", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Miami", "Imola", "Monaco",
        "Montreal", "Barcelona", "Spielberg", "Silverstone", "Hungaroring", "Spa", "Zandvoort", "Monza",
        "Baku", "Singapore", "Austin", "Mexico City", "São Paulo", "Las Vegas", "Losail", "Abu Dhabi"
    ]

    for gp_name in grand_prix_names:
        try:
            print(f"Loading data for {gp_name}...")
            # Load the race session ('R' for Race) for each Grand Prix
            session = fastf1.get_session(year, gp_name, 'R')
            session.load()  # Load session data

            # Retrieve the lap data for this session
            laps = session.laps

            # Add a column for the Grand Prix name to the laps data
            laps['GrandPrix'] = gp_name

            # Add lap data to our global DataFrame
            all_laps = pd.concat([all_laps, laps], ignore_index=True)

        except Exception as e:
            print(f"Error loading data for {gp_name}: {e}")

    return all_laps


# Load all data for the 2024 season
year = 2024
data = load_grand_prix_laps_data(year)

# Check the retrieved columns
print(data.head())  # Display the first few rows to view the data

# Save the data to a CSV file
data.to_csv('grand_prix_laps_data_2024.csv', index=False)
print("The data has been saved in 'grand_prix_laps_data_2024.csv'.")