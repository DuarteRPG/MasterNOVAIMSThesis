# Install in case its necessary
    # pip install actionable-recourse
    # pip install dice_ml
    # pip install -U git+https://github.com/joaopfonseca/ml-research.git 
    # pip install -U recourse-game
    # pip install git+https://github.com/joaopfonseca/recourse-game 
    # pip install shap
    # pip install -U git+https://github.com/joaopfonseca/ShaRP.git 
    # pip install -U git+https://github.com/DataResponsibly/ShaRP 

# Imports
import dash
import dice_ml
import math
import matplotlib.pyplot as plt
import mlresearch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import recgame  # Import the recgame library
import requests as rq
import scipy.stats as stats
import seaborn as sns
import sklearn
import streamlit as st
import shap
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
st.set_option('deprecation.showPyplotGlobalUse', False)


# From other libraries
from dash import dcc, html
from dash.dependencies import Input, Output
from itertools import cycle,islice, product
from io import BytesIO
from math import ceil, pi
from mlresearch.utils import set_matplotlib_style, parallel_loop
from os.path import join
from pandas.plotting import parallel_coordinates
from pathlib import Path
from plotly.subplots import make_subplots
from recgame.recourse import ActionableRecourse, NFeatureRecourse
from recgame.recourse import DiCE, NFeatureRecourse  # We will mainly follow DiCE results.
from sharp import ShaRP
#from sharp.qoi import QOI_OBJECTS
from sharp.qoi import get_qoi_names
from sharp.qoi import get_qoi 
from sharp.utils import scores_to_ordering
from sharp.utils import check_inputs
from sharp.visualization._waterfall import _waterfall
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

# Setting Visual Theme
set_matplotlib_style(18)

# Set the default serif font
plt.rcParams['font.family'] = 'serif'

###################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################

# 1. Displaying an image.
st.image("Sports.png")

# 2. Title for the app
st.title("What must a basketball, football or tennis player do to improve their overall ranking?")

# 3. Header
st.subheader("Datasets used in our research:")

# 4. Sub Header
items = ["(1) Basket Team NBA 2022-23 (Regular Season)", 
         "(2) Basket Player NBA 2022 (Regular Season)",
         "(3) Football Team", 
         "(4) Football Player FIFA 2022",
         "(5) ATP Season 2022", 
         "(6) WTA Season 2023"]

for item in items:
    st.write(item)

# 5. Info
st.info("Scroll down to get insights according to your selection. Select your preferences in the left side section and navegate throughout the different tabs. Note that it may take a while to process your filters.")

# 6. Sidebar Part I
st.sidebar.title("SECTIONS")
st.sidebar.header("Personalize your choice:")
Sport = st.sidebar.radio("Sport", ["Basketball", "Football", "Tennis"])

tab_titles = ['I. DiCE: General', 'II. DiCE: Individual Selection' , 'III. SHAP: General', 'IV. SHAP: Individual Selection' , 'V. SHARP', 'VI. Conclusion: DiCE vs SHAP vs SHARP']
tabs = st.tabs(tab_titles)


# Display specific information based on the selected option: Basket.
if Sport == 'Basketball':
    # Open a sidebar for additional Basketball options
    st.sidebar.subheader("Basketball Options")
    
    # Create a radio button for selecting the type (team or player)
    Team_vs_Player = st.sidebar.radio('Type Preference:', ["Team", "Player"])

    # Check if the user selects the type as Team
    if Team_vs_Player == 'Team':
        Team = st.sidebar.selectbox('Select the Team:', ('Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'))

        # df_summaries | df_totals
        df_summaries = pd.read_excel('1_NBA_Team_Stats_Regular_Season_Team_Summaries.xlsx', sheet_name= 'PBC_NBA_1947_Team_Summaries')
        df_totals = pd.read_excel('1_NBA_Team_Stats_Regular_Season_Team_Totals.xlsx', sheet_name= 'PBC_NBA_1947_Team_Totals')
        df_summaries = df_summaries[df_summaries['season'] == 2023]
        df_totals = df_totals[df_totals['season'] == 2023]
        df_summaries = df_summaries[df_summaries['team'] != 'League Average']
        df_totals = df_totals[df_totals['team'] != 'League Average']
        df_summaries.columns = df_summaries.columns.str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df_totals.columns = df_totals.columns.str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df_summaries =df_summaries.drop(columns=["season",
                                                "lg", # Not informative.
                                                "abbreviation", # Already filtered information. Discarted.
                                                "playoffs", # Irrelevant.
                                                "w", # Already filtered information. Discarted.
                                                "l", # Already filtered information. Discarted.
                                                "pw", # It is an immutable feature. It can not be changed.
                                                "pl", # It is an immutable feature. It can not be changed.
                                                "arena", # Purely informative.
                                                "attend", # Not a game related variable.
                                                "attend_g" # Not a game related variable.
                                                ]).set_index("team")
        df_totals =df_totals.drop(columns=["season",
                                                "lg", # Not informative.
                                                "abbreviation", # Already filtered information. Discarted.
                                                "playoffs", # Irrelevant.
                                                "g", # Already filtered information. Discarted.
                                                "mp", # Already filtered information. Discarted.
                                                "fg", # Informative already reflected on 'fg_percent'.
                                                "fga", # Informative already reflected on 'fg_percent'.
                                                "x3p", # Informative already reflected on 'x3p_percent'.
                                                "x3pa", # Informative already reflected on 'x3p_percent'.
                                                "x2p", # Informative already reflected on 'x2p_percent'.
                                                "x2pa", # Informative already reflected on 'x2p_percent'.
                                                "ft", # Informative already reflected on 'ft_percent'.
                                                "fta", # Informative already reflected on 'ft_percent'.
                                                ]).set_index("team")
        df_summaries['tov_percent'] = df_summaries['tov_percent'] / 100
        df_summaries['orb_percent'] = df_summaries['orb_percent'] / 100
        df_summaries['opp_tov_percent'] = df_summaries['opp_tov_percent'] / 100
        df_summaries['opp_drb_percent'] = df_summaries['opp_drb_percent'] / 100
        df = pd.merge(df_summaries, df_totals, on='team', how='inner')
        df = df.sort_values(by='pts', ascending=False)
        X = df.drop(columns=["pts"])
        y = df.pts / df.pts.max()

        # Define the dictionary mapping short names to full names
        variable_names = {"age": "Age (Team average age on 1 February 2023)",
            "mov": "Margin of Victory",
            "sos": "Strength of Schedule",
            "srs": "Simple Rating System",
            "o_rtg": "Offensive Rating",
            "d_rtg": "Defensive Rating",
            "n_rtg": "Net Rating",
            "pace": "Pace Factor",
            "f_tr": "Free Throw Rate",
            "x3p_ar": "3-Point Attempt Rate",
            "ts_percent": "True Shooting Percentage",
            "e_fg_percent": "Effective Field Goal Percentage",
            "tov_percent": "Turnover Percentage",
            "orb_percent": "Offensive Rebound Percentage",
            "ft_fga": "Free Throw Attempt Rate",
            "opp_e_fg_percent": "Opponent Effective Field Goal Percentage",
            "opp_tov_percent": "Opponent Turnover Percentage",
            "opp_drb_percent": "Opponent Defensive Rebound Percentage",
            "opp_ft_fga": "Opponent Free Throw Attempt Rate",
            "fg_percent": "Field Goal Percentage",
            "x3p_percent": "3-Point Percentage",
            "x2p_percent": "2-Point Percentage",
            "ft_percent": "Free Throw Percentage",
            "orb": "Offensive Rebounds",
            "drb": "Defensive Rebounds",
            "trb": "Total Rebounds",
            "ast": "Assists",
            "stl": "Steals",
            "blk": "Blocks",
            "tov": "Turnovers",
            "pf": "Personal Fouls"}

        # Open a sidebar for a different feature option
        Basketball_team_list = list(variable_names.keys()) # Basketball_team_list = X.columns.tolist()
        Basketball_team_list_full = list(variable_names.values())
        Basketball_team_feature_full_name = st.sidebar.selectbox('Feature in focus:', Basketball_team_list_full)
        Basketball_team_feature = [key for key, value in variable_names.items() if value == Basketball_team_feature_full_name][0] # Get the corresponding short name from the dictionary
        
        # Open a sidebar for a different feature option
        Decil = st.sidebar.selectbox('Top Ranking (%) you desire to achieve (where 0,05 means top 5%):', ('0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                                                                                                          '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9'))

        #Decil_final = 1 - float(Decil)
        Decil_final = round(1 - float(Decil), 2)
        
        Team_2 = st.sidebar.selectbox('Select a Team to compare:', ('Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'))

        # Opening our datasets
        cfs = pd.read_excel(f'cfs_1_{Decil_final}.xlsx')
        differences = pd.read_excel(f'differences_1_{Decil_final}.xlsx')
        st.write("<div style='height: 650px;'></div>", unsafe_allow_html=True)

        #if tabs == "1. General Sport Analysis":
        with tabs[0]:
            st.markdown("<h4 style='text-align: center;'>Dataset in focus</h1>", unsafe_allow_html=True)
            st.write(df)
            st.markdown("**Figure 1**: Representation of the DataFrame used. It aggregates all data used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # Concepts to take into account.
            st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Team_differences = differences.loc[Team]

            # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Teams (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # Create a histogram.
            plt.hist(differences_array, bins=20, edgecolor='black')
            plt.xlabel('Differences')
            plt.ylabel('Frequency')
            st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 4. Violin: Insights from SUGGESTED CHANGES
            differences_array = differences.values.flatten()
            # Create a violin plot
            plt.figure(figsize=(8, 6))
            sns.violinplot(y = differences_array, color='skyblue')
            plt.ylabel('Differences')
            st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 5. Density Plot: Insights from SUGGESTED CHANGES
            differences = differences.squeeze()  # Ensure it's a Series
            plt.figure(figsize=(10, 10)) 
            sns.kdeplot(data=differences, shade=True)
            plt.xlabel('(CFS - X)')
            plt.ylabel('Density')
            st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 6. Radar Chart: Average SUGGESTED CHANGES per feature
            categories = list(differences.columns) # Setting categories as a list of all "differences" column.
            values = differences.mean().values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # Plot features. 
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown("<h4 style='text-align: center;'>Radar Chart: Average SUGGESTED CHANGES per feature</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 6**: Results from DICE. Another method to represent the differences obtained. **The axis defines each difference magnitude per feature.**")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 7. SWARM: Insights from SUGGESTED CHANGES
            # sns.swarmplot(data=differences, palette='coolwarm')
            # plt.xlabel('Features')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>SWARM: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # plt.xticks(rotation=90)  # Better adjusted the rotation angle so that we can better observe feature names.
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 7**: Results from DICE. Last representation of individual differences per feature, with a clear overview on which feature vary the most. **Each point represent a single instance of the dataset**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[1]:
            # 8. Bar Plot
            fig, ax = plt.subplots()
            ax.bar(Team_differences.index, Team_differences.values)
            plt.xticks(rotation=90)  # Adjusting the angle of my axis.
            plt.xlabel('Columns')
            plt.ylabel('Values')
            st.markdown(f"<h4 style='text-align: center;'>Bar Plot for {Team}</h1>", unsafe_allow_html=True)
            st.pyplot(fig) # Displaying plot in Streamlit
            st.markdown(f"**Figure 8**: Results from DICE for **{Team}**. As described in the previous tab, it provides 'what-if' explanations for the model output, by stating **which features would need to be altered in the counterfactual scenarios** compared to the original data to achieve the desired outcomes predicted by the model.  \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 9. KDE
            differences_array = differences[Basketball_team_feature].values
            # Create KDE plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.kdeplot(differences_array, shade=True)
            plt.xlabel('Differences')
            plt.ylabel('Density')
            st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Basketball_team_feature_full_name}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Basketball_team_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Basketball_team_feature} is recommended to change**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            normalized_data_X = scaler.fit_transform(X)
            normalized_data_cfs = scaler.fit_transform(cfs)
            normalized_data_differences = scaler.fit_transform(differences)

            X_normalized = pd.DataFrame(normalized_data_X, columns=X.columns, index=X.index)
            cfs_normalized = pd.DataFrame(normalized_data_cfs, columns=cfs.columns, index=cfs.index)
            differences_normalized = pd.DataFrame(normalized_data_differences, columns=differences.columns, index=differences.index)

            # 10. Radar (per player) - INITIAL STATE
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_X_normalized = X_normalized.loc[selected_player]    
            categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_X_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            values = player_differences.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_differences_normalized = differences_normalized.loc[selected_player]    
            categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 13. Radar (per player) - RECOMMENDED STATE
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_cfs_normalized = cfs_normalized.loc[selected_player]
            categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
            # Specify the name of the player.
            selected_player = Team
            # Filter the differences "DataFrame" for the selected player.
            player_cfs_normalized = cfs_normalized.loc[selected_player]
            player_values_cfs = player_cfs_normalized.values.tolist()
            player_values_cfs += player_values_cfs[:1]
            player_X_normalized = X_normalized.loc[selected_player]
            player_values_X = player_X_normalized.values.tolist()
            player_values_X += player_values_X[:1]
            # Changing angles and categories.
            categories = list(player_cfs_normalized.index)
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            # Plot for 'cfs', that represent the desired values.
            plt.figure(figsize=(8, 8))
            plt.polar(angles, player_values_cfs, label='recommended', color='blue')
            plt.fill(angles, player_values_cfs, alpha=0.25, color='blue')
            # Plot for 'X', that represent the initial values.
            plt.polar(angles, player_values_X, label='initial', color='green')
            plt.fill(angles, player_values_X, alpha=0.25, color='green')
            plt.xticks(angles[:-1], categories)
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE and RECOMMENDED STATE: for {selected_player} - NORMALIZED</h1>", unsafe_allow_html=True)
            plt.legend()
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 14**: To obtain clear insights, we overlapped previous **INITIAL** and **RECOMMENDADED STATES** visualizations. Recapping: \n - **Blue line** represent **DESIRED** feature values (Counterfactuals); \n - **Green line** represent **INITIAL** feature values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)            


        #else:
        with tabs[2]:
            # Concepts to take into account
            st.info("SHAP: (SHapley Additive exPlanations) can be defined as a game theoretic approach to explain the output of a machine learning model. It explains the impact and the importance of each feature on model output/predictions for a specific instance. \n It provides a more interpretable view of the model's behavior and  these values can be used to gain insights on which factors mostly influence specific predictions. \n Looks at the average value and give us information.")

            # 15. SHAP Bar Plot
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            explainer = shap.Explainer(lr, X)
            shap_values = explainer(X)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values, max_display=15)
            st.pyplot()  
            st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 16. SHAP Beeswarm Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.beeswarm(shap_values, max_display=15)
            st.pyplot()
            st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.r")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            basketball_team_index_feature = Basketball_team_list.index(Basketball_team_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Basketball_team_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.scatter(shap_values[:, basketball_team_index_feature])
            st.pyplot()
            st.markdown(f"**Figure 17**: Scatter plot on feature **{Basketball_team_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Basketball_team_feature_full_name} feature, which means **'how much must {Basketball_team_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Basketball_team_feature_full_name}**.")
            st.markdown(f"This means that, for positive SHAP values, **{Basketball_team_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Basketball_team_feature_full_name} must impact negatively** the model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 18. SHAP Partial Dependence Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Basketball_team_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.partial_dependence_plot(
                Basketball_team_feature, lr.predict, X, ice=False,
                model_expected_value=True, feature_expected_value=True) 
            st.pyplot()
            st.markdown(f"**Figure 18**: Model's dependence on the feature {Basketball_team_feature_full_name}, now in the new original feature space (X). It explains ***how SHAP values of {Basketball_team_feature_full_name} vary across a dataset** and how changes in the {Basketball_team_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Basketball_team_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Basketball_team_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Basketball_team_feature_full_name}** as a certain value;")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            basketball_team_index_player = X_indexes.index(Team)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Team}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[basketball_team_index_player], max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Team}, instead of, as in the previous two graphs, focusing on feature {Basketball_team_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # Concepts to take into account
            st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # 20. SHARP: Rank vs Score
            import os
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            st.image("Rank_vs_Score_(1) Basket Team.png")
            st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**, meaning that: \n - **the higher the Team's Score, the better tends to be the its rank**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 21. SHARP: Top and Bottom 3 Individuals
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            st.image("Top_bottom_feature_importance_1.png")
            st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 22. SHARP: Feature Importance
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            st.image("Strata_boxplot_1.png")
            st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 23. Unary values in focus
            def scorer(dataset, columns=None):
                X, _ = check_inputs(dataset)                
                # Define weights for each column
                weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 
                        0.2, 0.2, 0.2, 0.2, 0.2, 
                        0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
                # Calculate the weighted sum for each row
                weighted_sum = np.sum(X * weights, axis=1)
                return weighted_sum

            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            xai = ShaRP(
                qoi="rank",
                target_function=scorer,
                measure="unary",
                sample_size=None, # sample_size=None,
                replace=False,
                random_state=42,
                verbose=1,
                n_jobs=-1)
            xai.fit(X_sharp)

            st.markdown(f"<h4 style='text-align: center;'>Unary values in focus</h1>", unsafe_allow_html=True)
            unary_values = pd.read_csv("cs_rankqoi_unary_values_(1)_basket_team.csv")
            unary_values.set_index(unary_values.columns[0], inplace=True)
            unary_values_player = unary_values.loc[Team].drop(["Score", "Ranking"])
            st.write(unary_values) #st.write(unary_values_player), if we want to filter by the player we chose.
            st.markdown("**Figure 23**: Representation of all Unary Values computed and used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 24. SHARP: Waterfall
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Waterfall Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            rank_dict = {
            "upper_bounds": None,
            "lower_bounds": None,
            "features": None,
            "data": None,  
            "base_values": 0,
            "feature_names": unary_values_player.index.tolist(),
            "values": unary_values_player}

            _waterfall(rank_dict, max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Team}. Similarly to SHAP Waterfall, it attempts to explain {Team} predictions. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            values = xai.pairwise(
                X_sharp.loc[Team].values, 
                X_sharp.loc[Team_2].values)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            pairwise_bars = xai.plot.bar(values, ax=ax)
            ax.set_ylabel("Contribution to Rank")
            ax.set_xlabel("")
            plt.xticks(rotation=90)
            st.pyplot()
            st.markdown(f"**Figure 25**: Pairwise comparison between {Team} and {Team_2}. It provides insights on which variables mostly contribute and which variables mostly harm each one. \n - **Positive values** for a certain feature, means that it **favors {Team} instead of {Team_2}**. \n - **Negative values** for a certain feature, means that it **favors {Team_2} instead of {Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[5]:
            # Extracting values per feature
            all_values = []

            # Combine values from the main plots for each feature (DiCE, SHAP and SHARP).
            for feature in rank_dict["feature_names"]:
                feature_values = {"Feature": feature}
                if feature in Team_differences.index: # Get value from Plot 1 (DiCE: Player_differences)
                    feature_values["Team_differences"] = Team_differences[feature]
                else:
                    feature_values["Team_differences"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 2 (SHAP values)
                    shap_index = rank_dict["feature_names"].index(feature)
                    feature_values["SHAP_values"] = shap_values[basketball_team_index_player].values[shap_index]
                else:
                    feature_values["SHAP_values"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 3 (SHARP: rank_dict)
                    rank_index = rank_dict["feature_names"].index(feature)
                    feature_values["Rank_dict_values"] = rank_dict["values"][rank_index]
                else:
                    feature_values["Rank_dict_values"] = None
                
                # Append to the list of all values
                all_values.append(feature_values)

            # 26. DiCE vs SHAP vs SHARP: Comparing Methods
            # Convert to DataFrame and displaying the table.
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df_values_2 = pd.DataFrame(all_values)
            df_values_2.set_index('Feature', inplace=True)
            df_values_2.columns = ["DiCE Counterfactuals", "SHAP Values", "SHARP Values"] # Renaming columns. Replacing Rank Dict Values for SHARP Values.
 
            # Highlight the top largest and smallestvalues per column
            light_green = 'background-color: rgb(144, 238, 144)'  # Light green.
            light_red = 'background-color: rgba(255, 99, 71, 0.5)'  # Light red color (with transparency)

            # Highlight the top 3 values in a Series green.
            def highlight_top1(s):
                top1 = s.nlargest(1)
                bottom1 = s.nsmallest(1)
                is_top1 = s.isin(top1)
                is_bottom1 = s.isin(bottom1)
                colors = []
                #return [light_green if v else '' for v in is_top1]
                for v in s:
                    if v in top1.values:
                        colors.append(light_green)
                    elif v in bottom1.values:
                        colors.append(light_red)
                    else:
                        colors.append('')
                return colors

            # Apply the highlight_top3 function to the DataFrame and displaying it
            df_styled_2 = df_values_2.style.apply(highlight_top1)
            st.dataframe(df_styled_2, width=900)
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 28. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 28**: Correlation matrix between DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


    # Check if the user selects the type as Player
    elif Team_vs_Player == 'Player':
        Player = st.sidebar.selectbox('Select the Player:', ('Aaron Gordon', 'Aaron Nesmith', 'Aaron Wiggins', 'AJ Griffin', 'Al Horford', 'Alec Burks', 'Aleksej Pokusevski', 'Alex Caruso', 'Alperen Þengün', 'Andrew Nembhard', 'Andrew Wiggins', 'Anfernee Simons', 'Anthony Davis', 'Anthony Edwards', 'Anthony Lamb', 'Austin Reaves', 'Austin Rivers', 'Ayo Dosunmu', 'Bam Adebayo', 'Ben Simmons', 'Bennedict Mathurin', 'Blake Wesley', 'Bobby Portis', 'Bogdan Bogdanovi?', 'Bojan Bogdanovi?', 'Bol Bol', 'Bones Hyland', 'Bradley Beal', 'Brandon Clarke', 'Brandon Ingram', 'Brook Lopez', 'Bruce Brown', 'Bryce McGowens', 'Buddy Hield', 'Cade Cunningham', 'Caleb Houstan', 'Caleb Martin', 'Cam Reddish', 'Cam Thomas', 'Cameron Johnson', 'Cameron Payne', 'Caris LeVert', 'Cedi Osman', 'Chance Comanche', 'Chris Boucher', 'Chris Duarte', 'Chris Paul', 'Christian Braun', 'Christian Wood', 'Chuma Okeke', 'CJ McCollum', 'Clint Capela', 'Coby White', 'Cody Martin', 'Cole Anthony', 'Collin Sexton', 'Corey Kispert', 'Cory Joseph', 'Daishen Nix', 'Damian Jones', 'Damian Lillard', 'Damion Lee', 'DAngelo Russell', 'Daniel Gafford', 'Daniel Theis', 'Darius Bazley', 'Darius Garland', 'David Roddy', 'Davion Mitchell', 'DeAaron Fox', 'Dean Wade', 'Deandre Ayton', 'DeAndre Hunter', 'DeAnthony Melton', 'Dejounte Murray', 'Delon Wright', 'DeMar DeRozan', 'Deni Avdija', 'Dennis Schröder', 'Dennis Smith Jr.', 'Derrick White', 'Desmond Bane', 'Devin Booker', 'Devin Vassell', 'Devonte Graham', 'Dillon Brooks', 'Domantas Sabonis', 'Donovan Mitchell', 'Donte DiVincenzo', 'Dorian Finney-Smith', 'Doug McDermott', 'Draymond Green', 'Drew Eubanks', 'Duncan Robinson', 'Dwight Powell', 'Dyson Daniels', 'Eric Gordon', 'Eugene Omoruyi', 'Evan Fournier', 'Evan Mobley', 'Franz Wagner', 'Fred VanVleet', 'Gabe Vincent', 'Gabe York', 'Gary Harris', 'Gary Payton II', 'Gary Trent Jr.', 'George Hill', 'Georges Niang', 'Giannis Antetokounmpo', 'Goran Dragi?', 'Gordon Hayward', 'Grant Williams', 'Grayson Allen', 'Hamidou Diallo', 'Harrison Barnes', 'Haywood Highsmith', 'Herbert Jones', 'Immanuel Quickley', 'Isaac Okoro', 'Isaiah Hartenstein', 'Isaiah Jackson', 'Isaiah Joe', 'Isaiah Livers', 'Isaiah Stewart', 'Ish Wainright', 'Ivica Zubac', 'Ja Morant', 'Jabari Smith Jr.', 'Jacob Gilyard', 'Jaden Ivey', 'Jaden McDaniels', 'Jae Crowder', 'JaeSean Tate', 'Jakob Poeltl', 'Jalen Brunson', 'Jalen Duren', 'Jalen Green', 'Jalen McDaniels', 'Jalen Smith', 'Jalen Suggs', 'Jalen Williams', 'Jamal Murray', 'Jamaree Bouyea', 'James Bouknight', 'James Harden', 'James Wiseman', 'Jaren Jackson Jr.', 'Jarred Vanderbilt', 'Jarrett Allen', 'Jaylen Brown', 'Jaylen Nowell', 'Jaylin Williams', 'Jayson Tatum', 'Jeenathan Williams', 'Jeff Green', 'Jerami Grant', 'Jeremiah Robinson-Earl', 'Jeremy Sochan', 'Jericho Sims', 'Jevon Carter', 'Jimmy Butler', 'Joe Harris', 'Joe Ingles', 'Joel Embiid', 'John Collins', 'John Konchar', 'John Wall', 'Johnny Davis', 'Jonas Valan?i?nas', 'Jonathan Kuminga', 'Jordan Clarkson', 'Jordan Goodwin', 'Jordan McLaughlin', 'Jordan Nwora', 'Jordan Poole', 'Jose Alvarado', 'Josh Giddey', 'Josh Green', 'Josh Hart', 'Josh Okogie', 'Josh Richardson', 'Joshua Primo', 'Jrue Holiday', 'Juan Toscano-Anderson', 'Julian Champagnie', 'Julius Randle', 'Justin Holiday', 'Justin Minaya', 'Justise Winslow', 'Jusuf Nurki?', 'Karl-Anthony Towns', 'Kawhi Leonard', 'Keegan Murray', 'Keita Bates-Diop', 'Keldon Johnson', 'Kelly Olynyk', 'Kelly Oubre Jr.', 'Kemba Walker', 'Kenrich Williams', 'Kentavious Caldwell-Pope', 'Kenyon Martin Jr.', 'Kevin Durant', 'Kevin Huerter', 'Kevin Knox', 'Kevin Love', 'Kevin Porter Jr.', 'Kevon Looney', 'Khris Middleton', 'Killian Hayes', 'Klay Thompson', 'Kris Dunn', 'Kristaps Porzi??is', 'Kyle Anderson', 'Kyle Kuzma', 'Kyle Lowry', 'Kyrie Irving', 'Lamar Stevens', 'LaMelo Ball', 'Landry Shamet', 'Larry Nance Jr.', 'Lauri Markkanen', 'LeBron James', 'Lonnie Walker IV', 'Louis King', 'Luguentz Dort', 'Luka Don?i?', 'Luka Šamani?', 'Luke Kennard', 'Mac McClung', 'Malaki Branham', 'Malcolm Brogdon', 'Malik Beasley', 'Malik Monk', 'Marcus Morris', 'Marcus Smart', 'Mark Williams', 'Markelle Fultz', 'Marvin Bagley III', 'Mason Plumlee', 'Matisse Thybulle', 'Max Strus', 'Maxi Kleber', 'Michael Porter Jr.', 'Mikal Bridges', 'Mike Conley', 'Mike Muscala', 'Mitchell Robinson', 'Mo Bamba', 'Monte Morris', 'Moritz Wagner', 'Myles Turner', 'Naji Marshall', 'Nassir Little', 'Naz Reid', 'Nic Claxton', 'Nick Richards', 'Nickeil Alexander-Walker', 'Nicolas Batum', 'Nikola Joki?', 'Nikola Vu?evi?', 'Norman Powell', 'Obi Toppin', 'Ochai Agbaji', 'OG Anunoby', 'Onyeka Okongwu', 'Oshae Brissett', 'Otto Porter Jr.', 'P.J. Tucker', 'P.J. Washington', 'Paolo Banchero', 'Pascal Siakam', 'Pat Connaughton', 'Patrick Beverley', 'Patrick Williams', 'Paul George', 'Precious Achiuwa', 'Quentin Grimes', 'R.J. Hampton', 'RaiQuan Gray', 'Reggie Bullock', 'Reggie Jackson', 'Ricky Rubio', 'RJ Barrett', 'Robert Covington', 'Robert Williams', 'Rodney McGruder', 'Romeo Langford', 'Royce ONeale', 'Rudy Gobert', 'Rui Hachimura', 'Russell Westbrook', 'Ryan Arcidiacono', 'Saben Lee', 'Saddiq Bey', 'Sam Hauser', 'Sandro Mamukelashvili', 'Santi Aldama', 'Scottie Barnes', 'Seth Curry', 'Shaedon Sharpe', 'Shai Gilgeous-Alexander', 'Shake Milton', 'Shaquille Harrison', 'Skylar Mays', 'Spencer Dinwiddie', 'Stanley Johnson', 'Stephen Curry', 'Steven Adams', 'Svi Mykhailiuk', 'T.J. McConnell', 'T.J. Warren', 'Talen Horton-Tucker', 'Tari Eason', 'Taurean Prince', 'Terance Mann', 'Terrence Ross', 'Terry Rozier', 'Théo Maledon', 'Thomas Bryant', 'Tim Hardaway Jr.', 'Tobias Harris', 'Torrey Craig', 'Trae Young', 'Tre Jones', 'Tre Mann', 'Trendon Watford', 'Trey Lyles', 'Trey Murphy III', 'Troy Brown Jr.', 'Ty Jerome', 'Tyler Herro', 'Tyrese Haliburton', 'Tyrese Maxey', 'Tyus Jones', 'Victor Oladipo', 'Walker Kessler', 'Wendell Carter Jr.', 'Wenyen Gabriel', 'Wesley Matthews', 'Will Barton', 'Xavier Tillman Sr.', 'Yuta Watanabe', 'Zach Collins', 'Zach LaVine', 'Ziaire Williams', 'Zion Williamson'))
        
        # df
        df = pd.read_excel('2_NBA_Player_Stats_Regular_Season_2022_2023.xlsx', sheet_name= 'PBC 2022_23 NBA Player Stat')
        df.info()
        df_duplicate = df[df.duplicated(subset="Player", keep=False)]
        df_duplicate = df_duplicate[df_duplicate['Tm']=='TOT']
        df_double_duplicate = df_duplicate[df_duplicate.duplicated(subset="Player", keep=False)]
        df = df[~df['Player'].duplicated(keep=False)]
        #df = df.append(df_duplicate, ignore_index=True)
        df = pd.concat([df, df_duplicate], ignore_index=True)
        df = df[df['MP'] > 15]
        df.columns = df.columns.str.replace("%", "_perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df = df.drop(columns=["rk", # Not informative.
                            "pos", # Already existed information.
                            "tm" # Not informative.
                            ]).set_index("player")
        # Eliminating redundant columns.
        df = df.drop(columns=["fg", # Information already present in fg_perc.
                            "fga", # Information already present in fg_perc.
                            "3p", # Information already present in 3p_perc.
                            "3pa", # Information already present in 3p_perc.
                            "2p", # Information already present in 2p_perc.
                            "2pa", # Information already present in 2p_perc.
                            "ft", # Information already present in ft_perc.
                            "fta" # Information already present in ft_perc.
                            ])
        df = df.sort_values(by='pts', ascending=False)
        X = df.drop(columns=["pts"])
        y = df.pts / df.pts.max()
        
        # Define the dictionary mapping short names to full names
        variable_names = {
            "age": "Player's age",
            "g": "Games played",
            "gs": "Games started",
            "mp": "Minutes played per game",
            "fg_perc": "Field goal percentage",
            "3p_perc": "3-point field goal percentage",
            "2p_perc": "2-point field goal percentage",
            "efg_perc": "Effective field goal percentage",
            "ft_perc": "Free throw percentage",
            "orb": "Offensive rebounds per game",
            "drb": "Defensive rebounds per game",
            "trb": "Total rebounds per game",
            "ast": "Assists per game",
            "stl": "Steals per game",
            "blk": "Blocks per game",
            "tov": "Turnovers per game",
            "pf": "Personal fouls per game"}

        # Open a sidebar for a different feature option
        Basketball_player_list = list(variable_names.keys()) # Basketball_player_list = X.columns.tolist()
        Basketball_player_list_full = list(variable_names.values())
        Basketball_player_feature_full_name = st.sidebar.selectbox('Feature in focus:', Basketball_player_list_full)
        Basketball_player_feature = [key for key, value in variable_names.items() if value == Basketball_player_feature_full_name][0] # Get the corresponding short name from the dictionary
        
        # Open a sidebar for a different feature option
        Decil = st.sidebar.selectbox('Top Ranking (%) you desire to achieve (where 0,05 means top 5%):', ('0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                                                                                                          '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9'))

        #Decil_final = 1 - float(Decil)
        Decil_final = round(1 - float(Decil), 2)
        
        Player_2 = st.sidebar.selectbox('Select a Team to compare:', ('Aaron Gordon', 'Aaron Nesmith', 'Aaron Wiggins', 'AJ Griffin', 'Al Horford', 'Alec Burks', 'Aleksej Pokusevski', 'Alex Caruso', 'Alperen Þengün', 'Andrew Nembhard', 'Andrew Wiggins', 'Anfernee Simons', 'Anthony Davis', 'Anthony Edwards', 'Anthony Lamb', 'Austin Reaves', 'Austin Rivers', 'Ayo Dosunmu', 'Bam Adebayo', 'Ben Simmons', 'Bennedict Mathurin', 'Blake Wesley', 'Bobby Portis', 'Bogdan Bogdanovi?', 'Bojan Bogdanovi?', 'Bol Bol', 'Bones Hyland', 'Bradley Beal', 'Brandon Clarke', 'Brandon Ingram', 'Brook Lopez', 'Bruce Brown', 'Bryce McGowens', 'Buddy Hield', 'Cade Cunningham', 'Caleb Houstan', 'Caleb Martin', 'Cam Reddish', 'Cam Thomas', 'Cameron Johnson', 'Cameron Payne', 'Caris LeVert', 'Cedi Osman', 'Chance Comanche', 'Chris Boucher', 'Chris Duarte', 'Chris Paul', 'Christian Braun', 'Christian Wood', 'Chuma Okeke', 'CJ McCollum', 'Clint Capela', 'Coby White', 'Cody Martin', 'Cole Anthony', 'Collin Sexton', 'Corey Kispert', 'Cory Joseph', 'Daishen Nix', 'Damian Jones', 'Damian Lillard', 'Damion Lee', 'DAngelo Russell', 'Daniel Gafford', 'Daniel Theis', 'Darius Bazley', 'Darius Garland', 'David Roddy', 'Davion Mitchell', 'DeAaron Fox', 'Dean Wade', 'Deandre Ayton', 'DeAndre Hunter', 'DeAnthony Melton', 'Dejounte Murray', 'Delon Wright', 'DeMar DeRozan', 'Deni Avdija', 'Dennis Schröder', 'Dennis Smith Jr.', 'Derrick White', 'Desmond Bane', 'Devin Booker', 'Devin Vassell', 'Devonte Graham', 'Dillon Brooks', 'Domantas Sabonis', 'Donovan Mitchell', 'Donte DiVincenzo', 'Dorian Finney-Smith', 'Doug McDermott', 'Draymond Green', 'Drew Eubanks', 'Duncan Robinson', 'Dwight Powell', 'Dyson Daniels', 'Eric Gordon', 'Eugene Omoruyi', 'Evan Fournier', 'Evan Mobley', 'Franz Wagner', 'Fred VanVleet', 'Gabe Vincent', 'Gabe York', 'Gary Harris', 'Gary Payton II', 'Gary Trent Jr.', 'George Hill', 'Georges Niang', 'Giannis Antetokounmpo', 'Goran Dragi?', 'Gordon Hayward', 'Grant Williams', 'Grayson Allen', 'Hamidou Diallo', 'Harrison Barnes', 'Haywood Highsmith', 'Herbert Jones', 'Immanuel Quickley', 'Isaac Okoro', 'Isaiah Hartenstein', 'Isaiah Jackson', 'Isaiah Joe', 'Isaiah Livers', 'Isaiah Stewart', 'Ish Wainright', 'Ivica Zubac', 'Ja Morant', 'Jabari Smith Jr.', 'Jacob Gilyard', 'Jaden Ivey', 'Jaden McDaniels', 'Jae Crowder', 'JaeSean Tate', 'Jakob Poeltl', 'Jalen Brunson', 'Jalen Duren', 'Jalen Green', 'Jalen McDaniels', 'Jalen Smith', 'Jalen Suggs', 'Jalen Williams', 'Jamal Murray', 'Jamaree Bouyea', 'James Bouknight', 'James Harden', 'James Wiseman', 'Jaren Jackson Jr.', 'Jarred Vanderbilt', 'Jarrett Allen', 'Jaylen Brown', 'Jaylen Nowell', 'Jaylin Williams', 'Jayson Tatum', 'Jeenathan Williams', 'Jeff Green', 'Jerami Grant', 'Jeremiah Robinson-Earl', 'Jeremy Sochan', 'Jericho Sims', 'Jevon Carter', 'Jimmy Butler', 'Joe Harris', 'Joe Ingles', 'Joel Embiid', 'John Collins', 'John Konchar', 'John Wall', 'Johnny Davis', 'Jonas Valan?i?nas', 'Jonathan Kuminga', 'Jordan Clarkson', 'Jordan Goodwin', 'Jordan McLaughlin', 'Jordan Nwora', 'Jordan Poole', 'Jose Alvarado', 'Josh Giddey', 'Josh Green', 'Josh Hart', 'Josh Okogie', 'Josh Richardson', 'Joshua Primo', 'Jrue Holiday', 'Juan Toscano-Anderson', 'Julian Champagnie', 'Julius Randle', 'Justin Holiday', 'Justin Minaya', 'Justise Winslow', 'Jusuf Nurki?', 'Karl-Anthony Towns', 'Kawhi Leonard', 'Keegan Murray', 'Keita Bates-Diop', 'Keldon Johnson', 'Kelly Olynyk', 'Kelly Oubre Jr.', 'Kemba Walker', 'Kenrich Williams', 'Kentavious Caldwell-Pope', 'Kenyon Martin Jr.', 'Kevin Durant', 'Kevin Huerter', 'Kevin Knox', 'Kevin Love', 'Kevin Porter Jr.', 'Kevon Looney', 'Khris Middleton', 'Killian Hayes', 'Klay Thompson', 'Kris Dunn', 'Kristaps Porzi??is', 'Kyle Anderson', 'Kyle Kuzma', 'Kyle Lowry', 'Kyrie Irving', 'Lamar Stevens', 'LaMelo Ball', 'Landry Shamet', 'Larry Nance Jr.', 'Lauri Markkanen', 'LeBron James', 'Lonnie Walker IV', 'Louis King', 'Luguentz Dort', 'Luka Don?i?', 'Luka Šamani?', 'Luke Kennard', 'Mac McClung', 'Malaki Branham', 'Malcolm Brogdon', 'Malik Beasley', 'Malik Monk', 'Marcus Morris', 'Marcus Smart', 'Mark Williams', 'Markelle Fultz', 'Marvin Bagley III', 'Mason Plumlee', 'Matisse Thybulle', 'Max Strus', 'Maxi Kleber', 'Michael Porter Jr.', 'Mikal Bridges', 'Mike Conley', 'Mike Muscala', 'Mitchell Robinson', 'Mo Bamba', 'Monte Morris', 'Moritz Wagner', 'Myles Turner', 'Naji Marshall', 'Nassir Little', 'Naz Reid', 'Nic Claxton', 'Nick Richards', 'Nickeil Alexander-Walker', 'Nicolas Batum', 'Nikola Joki?', 'Nikola Vu?evi?', 'Norman Powell', 'Obi Toppin', 'Ochai Agbaji', 'OG Anunoby', 'Onyeka Okongwu', 'Oshae Brissett', 'Otto Porter Jr.', 'P.J. Tucker', 'P.J. Washington', 'Paolo Banchero', 'Pascal Siakam', 'Pat Connaughton', 'Patrick Beverley', 'Patrick Williams', 'Paul George', 'Precious Achiuwa', 'Quentin Grimes', 'R.J. Hampton', 'RaiQuan Gray', 'Reggie Bullock', 'Reggie Jackson', 'Ricky Rubio', 'RJ Barrett', 'Robert Covington', 'Robert Williams', 'Rodney McGruder', 'Romeo Langford', 'Royce ONeale', 'Rudy Gobert', 'Rui Hachimura', 'Russell Westbrook', 'Ryan Arcidiacono', 'Saben Lee', 'Saddiq Bey', 'Sam Hauser', 'Sandro Mamukelashvili', 'Santi Aldama', 'Scottie Barnes', 'Seth Curry', 'Shaedon Sharpe', 'Shai Gilgeous-Alexander', 'Shake Milton', 'Shaquille Harrison', 'Skylar Mays', 'Spencer Dinwiddie', 'Stanley Johnson', 'Stephen Curry', 'Steven Adams', 'Svi Mykhailiuk', 'T.J. McConnell', 'T.J. Warren', 'Talen Horton-Tucker', 'Tari Eason', 'Taurean Prince', 'Terance Mann', 'Terrence Ross', 'Terry Rozier', 'Théo Maledon', 'Thomas Bryant', 'Tim Hardaway Jr.', 'Tobias Harris', 'Torrey Craig', 'Trae Young', 'Tre Jones', 'Tre Mann', 'Trendon Watford', 'Trey Lyles', 'Trey Murphy III', 'Troy Brown Jr.', 'Ty Jerome', 'Tyler Herro', 'Tyrese Haliburton', 'Tyrese Maxey', 'Tyus Jones', 'Victor Oladipo', 'Walker Kessler', 'Wendell Carter Jr.', 'Wenyen Gabriel', 'Wesley Matthews', 'Will Barton', 'Xavier Tillman Sr.', 'Yuta Watanabe', 'Zach Collins', 'Zach LaVine', 'Ziaire Williams', 'Zion Williamson'))

        # Opening our datasets
        cfs = pd.read_excel(f'cfs_2_{Decil_final}.xlsx')
        differences = pd.read_excel(f'differences_2_{Decil_final}.xlsx')
        st.write("<div style='height: 650px;'></div>", unsafe_allow_html=True)


        #if tabs == "1. General Sport Analysis":
        with tabs[0]:
            st.markdown("<h4 style='text-align: center;'>Dataset in focus</h1>", unsafe_allow_html=True)
            st.write(df)
            st.markdown("**Figure 1**: Representation of the DataFrame used. It aggregates all data used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # Concepts to take into account
            st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Player_differences = differences.loc[Player]

            # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # Create a histogram.
            plt.hist(differences_array, bins=20, edgecolor='black')
            plt.xlabel('Differences')
            plt.ylabel('Frequency')
            st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 4. Violin: Insights from SUGGESTED CHANGES
            differences_array = differences.values.flatten()
            # Create a violin plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.violinplot(y = differences_array, color='skyblue')
            plt.ylabel('Differences')
            st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 5. Density Plot: Insights from SUGGESTED CHANGES
            differences = differences.squeeze()  # Ensure it's a Series
            plt.figure(figsize=(10, 10))
            sns.kdeplot(data=differences, shade=True)
            plt.xlabel('(CFS - X)')
            plt.ylabel('Density')
            st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 6. Radar Chart: Average SUGGESTED CHANGES per feature
            categories = list(differences.columns) # Setting categories as a list of all "differences" column.
            values = differences.mean().values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # Plot features. 
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown("<h4 style='text-align: center;'>Radar Chart: Average SUGGESTED CHANGES per feature</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 6**: Results from DICE. Another method to represent the differences obtained. **The axis defines each difference magnitude per feature.**")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 7. SWARM: Insights from SUGGESTED CHANGES
            # sns.swarmplot(data=differences, palette='coolwarm')
            # plt.xlabel('Features')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>SWARM: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # plt.xticks(rotation=90)  # Better adjusted the rotation angle so that we can better observe feature names.
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 7**: Results from DICE. Last representation of individual differences per feature, with a clear overview on which feature vary the most. **Each point represent a single instance of the dataset**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[1]:
            # 8. Bar Plot
            fig, ax = plt.subplots()
            ax.bar(Player_differences.index, Player_differences.values)
            plt.xticks(rotation=90)  # Adjusting the angle of my axis.
            plt.xlabel('Columns')
            plt.ylabel('Values')
            st.markdown(f"<h4 style='text-align: center;'>Bar Plot for {Player}</h1>", unsafe_allow_html=True)
            st.pyplot(fig) # Displaying plot in Streamlit
            st.markdown(f"**Figure 8**: Results from DICE for **{Player}**. As described in the previous tab, it provides 'what-if' explanations for the model output, by stating **which features would need to be altered in the counterfactual scenarios** compared to the original data to achieve the desired outcomes predicted by the model.  \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 9. KDE
            differences_array = differences[Basketball_player_feature].values
            # Create KDE plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.kdeplot(differences_array, shade=True)
            plt.xlabel('Differences')
            plt.ylabel('Density')
            st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Basketball_player_feature_full_name}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Basketball_player_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Basketball_player_feature} is recommended to change**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            normalized_data_X = scaler.fit_transform(X)
            normalized_data_cfs = scaler.fit_transform(cfs)
            normalized_data_differences = scaler.fit_transform(differences)

            X_normalized = pd.DataFrame(normalized_data_X, columns=X.columns, index=X.index)
            cfs_normalized = pd.DataFrame(normalized_data_cfs, columns=cfs.columns, index=cfs.index)
            differences_normalized = pd.DataFrame(normalized_data_differences, columns=differences.columns, index=differences.index)

            # 10. Radar (per player) - INITIAL STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_X_normalized = X_normalized.loc[selected_player]    
            categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_X_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            values = player_differences.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences_normalized = differences_normalized.loc[selected_player]    
            categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 13. Radar (per player) - RECOMMENDED STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_cfs_normalized = cfs_normalized.loc[selected_player]    
            categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)            
            
            # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
            # Specify the name of the player.
            selected_player = Player
            # Filter the differences "DataFrame" for the selected player.
            player_cfs_normalized = cfs_normalized.loc[selected_player]
            player_values_cfs = player_cfs_normalized.values.tolist()
            player_values_cfs += player_values_cfs[:1]
            player_X_normalized = X_normalized.loc[selected_player]
            player_values_X = player_X_normalized.values.tolist()
            player_values_X += player_values_X[:1]
            # Changing angles and categories.
            categories = list(player_cfs_normalized.index)
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            # Plot for 'cfs', that represent the desired values.
            plt.figure(figsize=(8, 8))
            plt.polar(angles, player_values_cfs, label='recommended', color='blue')
            plt.fill(angles, player_values_cfs, alpha=0.25, color='blue')
            # Plot for 'X', that represent the initial values.
            plt.polar(angles, player_values_X, label='initial', color='green')
            plt.fill(angles, player_values_X, alpha=0.25, color='green')
            plt.xticks(angles[:-1], categories)
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE and RECOMMENDED STATE: for {selected_player} - NORMALIZED</h1>", unsafe_allow_html=True)
            plt.legend()
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 14**: To obtain clear insights, we overlapped previous **INITIAL** and **RECOMMENDADED STATES** visualizations. Recapping: \n - **Blue line** represent **DESIRED** feature values (Counterfactuals); \n - **Green line** represent **INITIAL** feature values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
                
        #else:
        with tabs[2]:
            # Concepts to take into account
            st.info("SHAP: (SHapley Additive exPlanations) can be defined as a game theoretic approach to explain the output of a machine learning model. It explains the impact and the importance of each feature on model output/predictions for a specific instance. \n It provides a more interpretable view of the model's behavior and  these values can be used to gain insights on which factors mostly influence specific predictions. \n Looks at the average value and give us information.")

            # 15. SHAP Bar Plot
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            explainer = shap.Explainer(lr, X)
            shap_values = explainer(X)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values, max_display=15)
            st.pyplot()  
            st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 16. SHAP Beeswarm Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.beeswarm(shap_values, max_display=15)
            st.pyplot()
            st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.r")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            basketball_player_index_feature = Basketball_player_list.index(Basketball_player_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Basketball_player_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.scatter(shap_values[:, basketball_player_index_feature])
            st.pyplot()
            st.markdown(f"**Figure 17**: Scatter plot on feature **{Basketball_player_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Basketball_player_feature_full_name} feature, which means **'how much must {Basketball_player_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Basketball_player_feature_full_name}**.")
            st.markdown(f"This means that, for positive SHAP values, **{Basketball_player_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Basketball_player_feature_full_name} must impact negatively** the model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 18. SHAP Partial Dependence Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Basketball_player_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.partial_dependence_plot(
                Basketball_player_feature, lr.predict, X, ice=False,
                model_expected_value=True, feature_expected_value=True) 
            st.pyplot()
            st.markdown(f"**Figure 18**: Model's dependence on the feature {Basketball_player_feature_full_name}, now in the new original feature space (X). It explains ***how SHAP values of {Basketball_player_feature_full_name} vary across a dataset** and how changes in the {Basketball_player_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Basketball_player_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Basketball_player_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Basketball_player_feature_full_name}** as a certain value;")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            basketball_player_index_player = X_indexes.index(Player)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[basketball_player_index_player])
            st.pyplot()
            st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Player}, instead of, as in the previous two graphs, focusing on feature {Basketball_player_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # Concepts to take into account
            st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # 20. SHARP: Rank vs Score
            import os
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            st.image("Rank_vs_Score_(2) Basket Player.png")
            st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**, meaning that: \n - **the higher the Player's Score, the better tends to be the its rank**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 21. SHARP: Top and Bottom 3 Individuals
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            st.image("Top_bottom_feature_importance_2.png")
            st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 22. SHARP: Feature Importance
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            st.image("Strata_boxplot_2.png")
            st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 23. Unary values in focus
            def scorer(dataset, columns=None):
                X, _ = check_inputs(dataset)                
                # Define weights for each column
                weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
                # Calculate the weighted sum for each row
                weighted_sum = np.sum(X * weights, axis=1)
                return weighted_sum

            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            xai = ShaRP(
                qoi="rank",
                target_function=scorer,
                measure="unary",
                sample_size=None, # sample_size=None,
                replace=False,
                random_state=42,
                verbose=1,
                n_jobs=-1)
            xai.fit(X_sharp)

            st.markdown(f"<h4 style='text-align: center;'>Unary values in focus</h1>", unsafe_allow_html=True)
            unary_values = pd.read_csv("cs_rankqoi_unary_values_(2)_basket_player.csv")
            unary_values.set_index(unary_values.columns[0], inplace=True)
            unary_values_player = unary_values.loc[Player].drop(["Score", "Ranking"])
            st.write(unary_values) #st.write(unary_values_player), if we want to filter by the player we chose.
            st.markdown("**Figure 23**: Representation of all Unary Values computed and used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 24. SHARP: Waterfall
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Waterfall Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            rank_dict = {
            "upper_bounds": None,
            "lower_bounds": None,
            "features": None,
            "data": None,  
            "base_values": 0,
            "feature_names": unary_values_player.index.tolist(),
            "values": unary_values_player}

            _waterfall(rank_dict, max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} predictions. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            values = xai.pairwise(
                X_sharp.loc[Player].values, 
                X_sharp.loc[Player_2].values)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            pairwise_bars = xai.plot.bar(values, ax=ax)
            ax.set_ylabel("Contribution to Rank")
            ax.set_xlabel("")
            plt.xticks(rotation=90)
            st.pyplot()
            st.markdown(f"**Figure 25**: Pairwise comparison between {Player} and {Player_2}. It provides insights on which variables mostly contribute and which variables mostly harm each one. \n - **Positive values** for a certain feature, means that it **favors {Player} instead of {Player_2}**. \n - **Negative values** for a certain feature, means that it **favors {Player_2} instead of {Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[5]:
            # Extracting values per feature
            all_values = []

            # Combine values from the main plots for each feature (DiCE, SHAP and SHARP).
            for feature in rank_dict["feature_names"]:
                feature_values = {"Feature": feature}
                if feature in Player_differences.index: # Get value from Plot 1 (DiCE: Player_differences)
                    feature_values["Player_differences"] = Player_differences[feature]
                else:
                    feature_values["Player_differences"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 2 (SHAP values)
                    shap_index = rank_dict["feature_names"].index(feature)
                    feature_values["SHAP_values"] = shap_values[basketball_player_index_player].values[shap_index]
                else:
                    feature_values["SHAP_values"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 3 (SHARP: rank_dict)
                    rank_index = rank_dict["feature_names"].index(feature)
                    feature_values["Rank_dict_values"] = rank_dict["values"][rank_index]
                else:
                    feature_values["Rank_dict_values"] = None
                
                # Append to the list of all values
                all_values.append(feature_values)

            # 26. DiCE vs SHAP vs SHARP: Comparing Methods
            # Convert to DataFrame and displaying the table.
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df_values_2 = pd.DataFrame(all_values)
            df_values_2.set_index('Feature', inplace=True)
            df_values_2.columns = ["DiCE Counterfactuals", "SHAP Values", "SHARP Values"] # Renaming columns. Replacing Rank Dict Values for SHARP Values.

            # Highlight the top largest and smallestvalues per column
            light_green = 'background-color: rgb(144, 238, 144)'  # Light green.
            light_red = 'background-color: rgba(255, 99, 71, 0.5)'  # Light red color (with transparency)

            # Highlight the top 3 values in a Series green.
            def highlight_top1(s):
                top1 = s.nlargest(1)
                bottom1 = s.nsmallest(1)
                is_top1 = s.isin(top1)
                is_bottom1 = s.isin(bottom1)
                colors = []
                #return [light_green if v else '' for v in is_top1]
                for v in s:
                    if v in top1.values:
                        colors.append(light_green)
                    elif v in bottom1.values:
                        colors.append(light_red)
                    else:
                        colors.append('')
                return colors

            # Apply the highlight_top3 function to the DataFrame and displaying it
            df_styled_2 = df_values_2.style.apply(highlight_top1)
            st.dataframe(df_styled_2, width=900)
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 28. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 28**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


elif Sport == 'Football':    
    # Create a radio button for selecting the type (team or player)
    Team_vs_Player = st.sidebar.radio('Type Preference:', ["Team", "Player"])

    # Check if the user selects the type as Team
    if Team_vs_Player == 'Team':
        Team = st.sidebar.selectbox('Select the Team:', ('AFC Bournemouth', 'Ajaccio', 'AlmerÃ­a', 'Angers SCO', 'Arsenal', 'Aston Villa', 'Atalanta', 'Athletic Club', 'AtlÃ©tico Madrid', 'Auxerre', 'Bayer 04 Leverkusen', 'Bologna', 'Borussia Dortmund', 'Borussia MÃ¶nchengladbach', 'Brentford', 'Brest', 'Brighton & Hove Albion', 'CÃ¡diz', 'Celta de Vigo', 'Chelsea', 'Clermont', 'Cremonese', 'Crystal Palace', 'Eintracht Frankfurt', 'Elche', 'Empoli', 'Espanyol', 'Everton', 'FC Augsburg', 'FC Barcelona', 'FC Bayern MÃ¼nchen', 'FC KÃ¶ln', 'FC Union Berlin', 'Fiorentina', 'FSV Mainz 05', 'Fulham', 'Getafe', 'Girona', 'Hellas Verona', 'Hertha BSC', 'Inter', 'Juventus', 'Lazio', 'Lecce', 'Leeds United', 'Leicester City', 'Lens', 'Lille', 'Liverpool', 'Lorient', 'Mallorca', 'Manchester City', 'Manchester United', 'Milan', 'Monaco', 'Montpellier', 'Monza', 'Nantes', 'Napoli', 'Newcastle United', 'Nice', 'Nottingham Forest', 'Olympique Lyonnais', 'Olympique Marseille', 'Osasuna', 'Paris Saint Germain', 'Rayo Vallecano', 'RB Leipzig', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Real Valladolid', 'Reims', 'Rennes', 'Roma', 'Salernitana', 'Sampdoria', 'Sassuolo', 'SC Freiburg', 'Schalke 04', 'Sevilla', 'Southampton', 'Spezia', 'Strasbourg', 'Torino', 'Tottenham Hotspur', 'Toulouse', 'Troyes', 'TSG Hoffenheim', 'Udinese', 'Valencia', 'VfB Stuttgart', 'VfL Bochum 1848', 'VfL Wolfsburg', 'Villarreal', 'Werder Bremen', 'West Ham United', 'Wolverhampton Wanderers'))
        
        # df
        df = pd.read_excel('3_Football_Team_FIFA_2023.xlsx', sheet_name= 'PBC 3.4_Football Team FIFA')
        df = df[df['league_id'].isin([13, 16, 19, 31, 53])]
        df = df[df['fifa_version'] == 23]
        df = df[df['fifa_update'] == 9]
        df.columns = df.columns.str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df = df.drop(columns=["team_url", # Not informative.
                            "fifa_version", # Already filtered information. Discarted.
                            "fifa_update", # Already filtered information. Discarted.
                            "fifa_update_date", # Already filtered information. Discarted.
                            "league_id", # Not informative.
                            "league_name", # Not informative.
                            "league_level", # All leagues are level 1.
                            "nationality_id", # Not informative.
                            "nationality_name", # Without values. Not informative.
                            "coach_id", # Not informative.
                            "home_stadium", # Not informative.
                            "rival_team", # Not informative.
                            "transfer_budget_eur", # Not informative.
                            "club_worth_eur", # Not informative.
                            "captain", # Only indication of the player. Not informative.
                            "def_defence_pressure", # Without values. Not informative.
                            "def_defence_aggression", # Without values. Not informative.
                            "def_defence_width", # Without values. Not informative.
                            "def_defence_defender_line", # Without values. Not informative.
                            "off_style", # Without values. Not informative.
                            "build_up_play_speed", # Without values. Not informative.
                            "build_up_play_dribbling", # Without values. Not informative.
                            "build_up_play_passing", # Without values. Not informative.
                            "build_up_play_positioning", # Without values. Not informative.
                            "chance_creation_passing", # Without values. Not informative.
                            "chance_creation_crossing", # Without values. Not informative.
                            "chance_creation_shooting", # Without values. Not informative.
                            "chance_creation_positioning"]).set_index("team_id")
        df['def_style'].replace({'Balanced': 1, 'Press After Possession Loss': 2, 'Pressure On Heavy Touch': 3}, inplace=True)
        df['off_build_up_play'].replace({'Balanced': 1, 'Fast Build Up': 2, 'Long Ball': 3, 'Slow Build Up': 4}, inplace=True)
        df['off_chance_creation'].replace({'Balanced': 1, 'Direct Passing': 2, 'Forward Runs': 3, 'Possession': 4}, inplace=True)
        X = df.drop(columns=["overall"]).set_index("team_name")
        y = df.overall / df.overall.max()
        
        # Define the dictionary mapping short names to full names
        variable_names = {
            "attack": "Attack Rating",
            "midfield": "Midfield Rating",
            "defence": "Defence Rating",
            "international_prestige": "International Prestige",
            "domestic_prestige": "Domestic Prestige",
            "starting_xi_average_age": "Starting XI Average Age",
            "whole_team_average_age": "Whole Team Average Age",
            "short_free_kick": "Short Free Kick",
            "long_free_kick": "Long Free Kick",
            "left_short_free_kick": "Left Short Free Kick",
            "right_short_free_kick": "Right Short Free Kick",
            "penalties": "Penalties",
            "left_corner": "Left Corner",
            "right_corner": "Right Corner",
            "def_style": "Defensive Style",
            "def_team_width": "Defensive Team Width",
            "def_team_depth": "Defensive Team Depth",
            "off_build_up_play": "Offensive Build-Up Play",
            "off_chance_creation": "Offensive Chance Creation",
            "off_team_width": "Offensive Team Width",
            "off_players_in_box": "Offensive Players in Box",
            "off_corners": "Offensive Corners",
            "off_free_kicks": "Offensive Free Kicks"}

        # Open a sidebar for a different feature option
        Football_team_list = list(variable_names.keys()) # Football_team_list = X.columns.tolist()
        Football_team_list_full = list(variable_names.values())
        Football_team_feature_full_name = st.sidebar.selectbox('Feature in focus:', Football_team_list_full)
        Football_team_feature = [key for key, value in variable_names.items() if value == Football_team_feature_full_name][0] # Get the corresponding short name from the dictionary
        
        # Open a sidebar for a different feature option
        Decil = st.sidebar.selectbox('Top Ranking (%) you desire to achieve (where 0,05 means top 5%):', ('0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                                                                                                          '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9'))

        #Decil_final = 1 - float(Decil)
        Decil_final = round(1 - float(Decil), 2)
        
        Team_2 = st.sidebar.selectbox('Select a Team to compare:', ('AFC Bournemouth', 'Ajaccio', 'AlmerÃ­a', 'Angers SCO', 'Arsenal', 'Aston Villa', 'Atalanta', 'Athletic Club', 'AtlÃ©tico Madrid', 'Auxerre', 'Bayer 04 Leverkusen', 'Bologna', 'Borussia Dortmund', 'Borussia MÃ¶nchengladbach', 'Brentford', 'Brest', 'Brighton & Hove Albion', 'CÃ¡diz', 'Celta de Vigo', 'Chelsea', 'Clermont', 'Cremonese', 'Crystal Palace', 'Eintracht Frankfurt', 'Elche', 'Empoli', 'Espanyol', 'Everton', 'FC Augsburg', 'FC Barcelona', 'FC Bayern MÃ¼nchen', 'FC KÃ¶ln', 'FC Union Berlin', 'Fiorentina', 'FSV Mainz 05', 'Fulham', 'Getafe', 'Girona', 'Hellas Verona', 'Hertha BSC', 'Inter', 'Juventus', 'Lazio', 'Lecce', 'Leeds United', 'Leicester City', 'Lens', 'Lille', 'Liverpool', 'Lorient', 'Mallorca', 'Manchester City', 'Manchester United', 'Milan', 'Monaco', 'Montpellier', 'Monza', 'Nantes', 'Napoli', 'Newcastle United', 'Nice', 'Nottingham Forest', 'Olympique Lyonnais', 'Olympique Marseille', 'Osasuna', 'Paris Saint Germain', 'Rayo Vallecano', 'RB Leipzig', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Real Valladolid', 'Reims', 'Rennes', 'Roma', 'Salernitana', 'Sampdoria', 'Sassuolo', 'SC Freiburg', 'Schalke 04', 'Sevilla', 'Southampton', 'Spezia', 'Strasbourg', 'Torino', 'Tottenham Hotspur', 'Toulouse', 'Troyes', 'TSG Hoffenheim', 'Udinese', 'Valencia', 'VfB Stuttgart', 'VfL Bochum 1848', 'VfL Wolfsburg', 'Villarreal', 'Werder Bremen', 'West Ham United', 'Wolverhampton Wanderers'))

        # Opening our datasets
        cfs = pd.read_excel(f'cfs_3_{Decil_final}.xlsx')
        differences = pd.read_excel(f'differences_3_{Decil_final}.xlsx')
        st.write("<div style='height: 650px;'></div>", unsafe_allow_html=True)


        #if tabs == "1. General Sport Analysis":
        with tabs[0]:
            st.markdown("<h4 style='text-align: center;'>Dataset in focus</h1>", unsafe_allow_html=True)
            st.write(df)
            st.markdown("**Figure 1**: Representation of the DataFrame used. It aggregates all data used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # Concepts to take into account
            st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Team_differences = differences.loc[Team]

            # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Teams (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # Create a histogram.
            plt.hist(differences_array, bins=20, edgecolor='black')
            plt.xlabel('Differences')
            plt.ylabel('Frequency')
            st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 4. Violin: Insights from SUGGESTED CHANGES
            differences_array = differences.values.flatten()
            # Create a violin plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.violinplot(y = differences_array, color='skyblue')
            plt.ylabel('Differences')
            st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 5. Density Plot: Insights from SUGGESTED CHANGES
            differences = differences.squeeze()  # Ensure it's a Series
            plt.figure(figsize=(10, 10))
            sns.kdeplot(data=differences, shade=True)
            plt.xlabel('(CFS - X)')
            plt.ylabel('Density')
            st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 6. Radar Chart: Average SUGGESTED CHANGES per feature
            categories = list(differences.columns) # Setting categories as a list of all "differences" column.
            values = differences.mean().values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # Plot features. 
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown("<h4 style='text-align: center;'>Radar Chart: Average SUGGESTED CHANGES per feature</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 6**: Results from DICE. Another method to represent the differences obtained. **The axis defines each difference magnitude per feature.**")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 7. SWARM: Insights from SUGGESTED CHANGES
            # sns.swarmplot(data=differences, palette='coolwarm')
            # plt.xlabel('Features')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>SWARM: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # plt.xticks(rotation=90)  # Better adjusted the rotation angle so that we can better observe feature names.
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 7**: Results from DICE. Last representation of individual differences per feature, with a clear overview on which feature vary the most. **Each point represent a single instance of the dataset**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[1]:
            # 8. Bar Plot
            fig, ax = plt.subplots()
            ax.bar(Team_differences.index, Team_differences.values)
            plt.xticks(rotation=90)  # Adjusting the angle of my axis.
            plt.xlabel('Columns')
            plt.ylabel('Values')
            st.markdown(f"<h4 style='text-align: center;'>Bar Plot for {Team}</h1>", unsafe_allow_html=True)
            st.pyplot(fig) # Displaying plot in Streamlit
            st.markdown(f"**Figure 8**: Results from DICE for **{Team}**. As described in the previous tab, it provides 'what-if' explanations for the model output, by stating **which features would need to be altered in the counterfactual scenarios** compared to the original data to achieve the desired outcomes predicted by the model.  \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 9. KDE
            differences_array = differences[Football_team_feature].values
            # Create KDE plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.kdeplot(differences_array, shade=True)
            plt.xlabel('Differences')
            plt.ylabel('Density')
            st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Football_team_feature_full_name}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Football_team_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Football_team_feature} is recommended to change**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            normalized_data_X = scaler.fit_transform(X)
            normalized_data_cfs = scaler.fit_transform(cfs)
            normalized_data_differences = scaler.fit_transform(differences)

            X_normalized = pd.DataFrame(normalized_data_X, columns=X.columns, index=X.index)
            cfs_normalized = pd.DataFrame(normalized_data_cfs, columns=cfs.columns, index=cfs.index)
            differences_normalized = pd.DataFrame(normalized_data_differences, columns=differences.columns, index=differences.index)

            # 10. Radar (per player) - INITIAL STATE
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_X_normalized = X_normalized.loc[selected_player]    
            categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_X_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            values = player_differences.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_differences_normalized = differences_normalized.loc[selected_player]    
            categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 13. Radar (per player) - RECOMMENDED STATE
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_cfs_normalized = cfs_normalized.loc[selected_player]    
            categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
            # Specify the name of the player.
            selected_player = Team
            # Filter the differences "DataFrame" for the selected player.
            player_cfs_normalized = cfs_normalized.loc[selected_player]
            player_values_cfs = player_cfs_normalized.values.tolist()
            player_values_cfs += player_values_cfs[:1]
            player_X_normalized = X_normalized.loc[selected_player]
            player_values_X = player_X_normalized.values.tolist()
            player_values_X += player_values_X[:1]
            # Changing angles and categories.
            categories = list(player_cfs_normalized.index)
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            # Plot for 'cfs', that represent the desired values.
            plt.figure(figsize=(8, 8))
            plt.polar(angles, player_values_cfs, label='recommended', color='blue')
            plt.fill(angles, player_values_cfs, alpha=0.25, color='blue')
            # Plot for 'X', that represent the initial values.
            plt.polar(angles, player_values_X, label='initial', color='green')
            plt.fill(angles, player_values_X, alpha=0.25, color='green')
            plt.xticks(angles[:-1], categories)
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE and RECOMMENDED STATE: for {selected_player} - NORMALIZED</h1>", unsafe_allow_html=True)
            plt.legend()
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 14**: To obtain clear insights, we overlapped previous **INITIAL** and **RECOMMENDADED STATES** visualizations. Recapping: \n - **Blue line** represent **DESIRED** feature values (Counterfactuals); \n - **Green line** represent **INITIAL** feature values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[2]:
            # Concepts to take into account
            st.info("SHAP: (SHapley Additive exPlanations) can be defined as a game theoretic approach to explain the output of a machine learning model. It explains the impact and the importance of each feature on model output/predictions for a specific instance. \n It provides a more interpretable view of the model's behavior and  these values can be used to gain insights on which factors mostly influence specific predictions. \n Looks at the average value and give us information.")

            # 15. SHAP Bar Plot
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            explainer = shap.Explainer(lr, X)
            shap_values = explainer(X)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values, max_display=15)
            st.pyplot()  
            st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 16. SHAP Beeswarm Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.beeswarm(shap_values, max_display=15)
            st.pyplot()
            st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.r")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Football_team_index_feature = Football_team_list.index(Football_team_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Football_team_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.scatter(shap_values[:, Football_team_index_feature])
            st.pyplot()
            st.markdown(f"**Figure 17**: Scatter plot on feature **{Football_team_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Football_team_feature_full_name} feature, which means **'how much must {Football_team_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Football_team_feature_full_name}**.")
            st.markdown(f"This means that, for positive SHAP values, **{Football_team_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Football_team_feature_full_name} must impact negatively** the model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 18. SHAP Partial Dependence Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Football_team_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.partial_dependence_plot(
                Football_team_feature, lr.predict, X, ice=False,
                model_expected_value=True, feature_expected_value=True) 
            st.pyplot()
            st.markdown(f"**Figure 18**: Model's dependence on the feature {Football_team_feature_full_name}, now in the new original feature space (X). It explains ***how SHAP values of {Football_team_feature_full_name} vary across a dataset** and how changes in the {Football_team_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Football_team_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Football_team_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Football_team_feature_full_name}** as a certain value;")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Football_team_index_player = X_indexes.index(Team)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Team}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[Football_team_index_player])
            st.pyplot()
            st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Team}, instead of, as in the previous two graphs, focusing on feature {Football_team_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # Concepts to take into account
            st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # 20. SHARP: Rank vs Score
            import os
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            st.image("Rank_vs_Score_(3) Football_Teams.png")
            st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**, meaning that: \n - **the higher the Team's Score, the better tends to be the its rank**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 21. SHARP: Top and Bottom 3 Individuals
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            st.image("Top_bottom_feature_importance_3.png")
            st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 22. SHARP: Feature Importance
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            st.image("Strata_boxplot_3.png")
            st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 23. Unary values in focus
            def scorer(dataset, columns=None):
                X, _ = check_inputs(dataset)                
                # Define weights for each column
                weights = [0.2, 0.2, 0.2, 0.2, 
                                0.2, 0.2, 
                                0.2, 0.2, 0.2,
                                0.2, 0.2, 0.2, 
                                0.2, 0.2, 0.2, 0.2,
                                0.2, 0.2, 0.2, 
                                0.2, 0.2, 0.2, 
                                0.2]
                # Calculate the weighted sum for each row
                weighted_sum = np.sum(X * weights, axis=1)
                return weighted_sum

            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            xai = ShaRP(
                qoi="rank",
                target_function=scorer,
                measure="unary",
                sample_size=None, # sample_size=None,
                replace=False,
                random_state=42,
                verbose=1,
                n_jobs=-1)
            xai.fit(X_sharp)

            st.markdown(f"<h4 style='text-align: center;'>Unary values in focus</h1>", unsafe_allow_html=True)
            unary_values = pd.read_csv("cs_rankqoi_unary_values_(3)_football_teams.csv")
            unary_values.set_index(unary_values.columns[0], inplace=True)
            unary_values_player = unary_values.loc[Team].drop(["Score", "Ranking"])
            st.write(unary_values) #st.write(unary_values_player), if we want to filter by the player we chose.
            st.markdown("**Figure 23**: Representation of all Unary Values computed and used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 24. SHARP: Waterfall
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Waterfall Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            rank_dict = {
            "upper_bounds": None,
            "lower_bounds": None,
            "features": None,
            "data": None,  
            "base_values": 0,
            "feature_names": unary_values_player.index.tolist(),
            "values": unary_values_player}

            _waterfall(rank_dict, max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Team}. Similarly to SHAP Waterfall, it attempts to explain {Team} predictions. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            values = xai.pairwise(
                X_sharp.loc[Team].values, 
                X_sharp.loc[Team_2].values)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            pairwise_bars = xai.plot.bar(values, ax=ax)
            ax.set_ylabel("Contribution to Rank")
            ax.set_xlabel("")
            plt.xticks(rotation=90)
            st.pyplot()
            st.markdown(f"**Figure 25**: Pairwise comparison between {Team} and {Team_2}. It provides insights on which variables mostly contribute and which variables mostly harm each one. \n - **Positive values** for a certain feature, means that it **favors {Team} instead of {Team_2}**. \n - **Negative values** for a certain feature, means that it **favors {Team_2} instead of {Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[5]:
            # Extracting values per feature
            all_values = []

            # Combine values from the main plots for each feature (DiCE, SHAP and SHARP).
            for feature in rank_dict["feature_names"]:
                feature_values = {"Feature": feature}
                if feature in Team_differences.index: # Get value from Plot 1 (DiCE: Player_differences)
                    feature_values["Team_differences"] = Team_differences[feature]
                else:
                    feature_values["Team_differences"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 2 (SHAP values)
                    shap_index = rank_dict["feature_names"].index(feature)
                    feature_values["SHAP_values"] = shap_values[Football_team_index_player].values[shap_index]
                else:
                    feature_values["SHAP_values"] = None
                if feature in rank_dict["feature_names"]:# Get value from Plot 3 (SHARP: rank_dict)
                    rank_index = rank_dict["feature_names"].index(feature)
                    feature_values["Rank_dict_values"] = rank_dict["values"][rank_index]
                else:
                    feature_values["Rank_dict_values"] = None
                
                # Append to the list of all values
                all_values.append(feature_values)

            # 26. DiCE vs SHAP vs SHARP: Comparing Methods
            # Convert to DataFrame and displaying the table.
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df_values_2 = pd.DataFrame(all_values)
            df_values_2.set_index('Feature', inplace=True)
            df_values_2.columns = ["DiCE Counterfactuals", "SHAP Values", "SHARP Values"] # Renaming columns. Replacing Rank Dict Values for SHARP Values.

            # Highlight the top largest and smallestvalues per column
            light_green = 'background-color: rgb(144, 238, 144)'  # Light green.
            light_red = 'background-color: rgba(255, 99, 71, 0.5)'  # Light red color (with transparency)

            # Highlight the top 3 values in a Series green.
            def highlight_top1(s):
                top1 = s.nlargest(1)
                bottom1 = s.nsmallest(1)
                is_top1 = s.isin(top1)
                is_bottom1 = s.isin(bottom1)
                colors = []
                #return [light_green if v else '' for v in is_top1]
                for v in s:
                    if v in top1.values:
                        colors.append(light_green)
                    elif v in bottom1.values:
                        colors.append(light_red)
                    else:
                        colors.append('')
                return colors

            # Apply the highlight_top3 function to the DataFrame and displaying it
            df_styled_2 = df_values_2.style.apply(highlight_top1)
            st.dataframe(df_styled_2, width=900)
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 28. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 28**: Correlation matrix between DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


    # Check if the user selects the type as Player
    elif Team_vs_Player == 'Player':
        Player = st.sidebar.selectbox('Select the Player:', ('Aime Vrsaljko', 'ä¸­äº• AA¤§', 'Ãliver Torres MuÃ±oz', 'Ãscar de Marcos Arana', 'Ãscar Esau Duarte GaitÃ¡n', 'Ãscar Gil RegaÃ±o', 'Ãscar Guido Trejo', 'Ãscar Melendo JimÃ©nez', 'Ãscar Mingueza GarcÃ­a', 'Ãscar RodrÃ­guez Arnaiz', 'Ãscar ValentÃ­n MartÃ­n Luengo', 'A¥¥A· é›…ä¹Ÿ', 'A®‰éƒ¨ è£•èµ', 'Ä°lkay GÃ¼ndoÄŸan', 'A·A³¶ æ°¸A—£', 'A†¨A®‰ A¥æ´‹', 'Ã‡aÄŸlar SÃ¶yÃ¼ncÃ¼', 'Ã‰der Gabriel MilitÃ£o', 'Ã‰douard Mendy', 'Ã‰douard Michut', 'A‰ç”° éº»ä¹Ÿ', 'ÃÃ±igo Lekue MartÃ­nez', 'ÃÃ±igo MartÃ­nez Berridi', 'ÃdÃ¡m Szalai', 'A—é‡Ž æ‹A®Ÿ', 'Ãlex Collado GutiÃ©rrez', 'Ãlex PetxarromÃ¡n', 'Ãlvaro Bastida Moya', 'Ãlvaro Borja Morata MartÃ­n', 'Ãlvaro Bravo JimÃ©nez', 'Ãlvaro FernÃ¡ndez Llorente', 'Ãlvaro GarcÃ­a Rivera', 'Ãlvaro GarcÃ­a Segovia', 'Ãlvaro GonzÃ¡lez SoberÃ³n', 'Ãlvaro JosÃ© JimÃ©nez Guerrero', 'Ãlvaro Negredo SÃ¡nchez', 'Ãlvaro Odriozola Arzalluz', 'Ãlvaro Vadillo Cifuentes', 'Ãngel Algobia Esteves', 'Ãngel FabiÃ¡n Di MarÃ­a HernÃ¡ndez', 'Ãngel JimÃ©nez Gallego', 'Ãngel LuÃ­s RodrÃ­guez DÃ­az', 'Ãngel MartÃ­n Correa', 'Ãngel Montoro SÃ¡nchez', 'Aukasz FabiaAski', 'Aukasz Skorupski', 'Aukasz Teodorczyk', 'ä¼Šè—¤ A®æ¨¹', 'ä¹…ä¿ A»ºè‹±', 'AÃ¯ssa Mandi', 'AarÃ³n Escandell Banacloche', 'AarÃ³n MartÃ­n Caricol', 'Aaron Anthony Connolly', 'Aaron Cresswell', 'Aaron Hickey', 'Aaron Lennon', 'Aaron Nassur Kamardin', 'Aaron Ramsdale', 'Aaron Ramsey', 'Aaron Wan-Bissaka', 'AbdÃ³n Prats Bastidas', 'Abdel Jalil Zaim Idriss Medioub', 'Abdelkabir Abqar', 'Abdou Diallo', 'Abdoulaye Bamba', 'Abdoulaye DoucourÃ©', 'Abdoulaye Jules Keita', 'Abdoulaye Sylla', 'Abdoulaye TourÃ©', 'Abdoulrahmane Harroui', 'Abdourahmane Barry', 'Abdul Majeed Waris', 'Abelmy Meto Silu', 'Achraf Hakimi Mouh', 'Adam Armstrong', 'Adam David Lallana', 'Adam Forshaw', 'Adam Jakubech', 'Adam MaruA¡iÄ‡', 'Adam Masina', 'Adam Ounas', 'Adam Uche Idah', 'Adam Webster', 'Adama TraorÃ© Diarra', 'Ademola Lookman', 'Adil Aouchiche', 'Adil Rami', 'Adilson Angel Abreu de Almeida Gomes', 'Admir Mehmedi', 'Adnan Januzaj', 'AdriÃ Giner Pedrosa', 'AdriÃ¡n de la Fuente Barquilla', 'AdriÃ¡n Embarba BlÃ¡zquez', 'AdriÃ¡n RodrÃ­guez GimÃ©nez', 'AdriÃ¡n San Miguel del Castillo', 'Adrian Aemper', 'Adrian Fein', 'Adrian GrbiÄ‡', 'Adrien Rabiot', 'Adrien Sebastian Perruchet Silva', 'Adrien Tameze', 'Adrien Thomasson', 'Adrien Truffert', 'æ­¦ç£Š', 'æµ…é‡Ž ä½è—¤', 'Aihen MuÃ±oz CapellÃ¡n', 'Aimar Oroz Huarte', 'Aimar Sher', 'Aimen Moueffek', 'Ainsley Maitland-Niles', 'Aitor FernÃ¡ndez Abarisketa', 'Aitor Paredes Casamichana', 'Aitor Ruibal GarcÃ­a', 'Ajdin HrustiÄ‡', 'Ajibola Alese', 'Akim Zedadka', 'Alan Godoy DomÃ­nguez', 'Alassane Alexandre PlÃ©a', 'Alban Lafont', 'Alberth JosuÃ© Elis MartÃ­nez', 'Albert-Mboyo Sambi Lokonga', 'Alberto Grassi', 'Alberto Moreno PÃ©rez', 'Alberto Moreno Rebanal', 'Alberto Perea Correoso', 'Alberto RodrÃ­guez BarÃ³', 'Alberto Soro Ãlvarez', 'Albin Ekdal', 'Aleix Febas PÃ©rez', 'Aleix Vidal Parreu', 'Alejandro Balde MartÃ­nez', 'Alejandro Berenguer Remiro', 'Alejandro Blanco SÃ¡nchez', 'Alejandro Blesa Pina', 'Alejandro Cantero SÃ¡nchez', 'Alejandro Carbonell VallÃ©s', 'Alejandro Catena MarugÃ¡n', 'Alejandro DarÃ­o GÃ³mez', 'Alejandro FernÃ¡ndez Iglesias', 'Alejandro Iturbe Encabo', 'Alejandro Remiro Gargallo', 'Alejandro RodrÃ­guez Lorite', 'Aleksa TerziÄ‡', 'Aleksandar Kolarov', 'Aleksandar Sedlar', 'Aleksander Buksa', 'Aleksandr Golovin', 'Aleksandr Kokorin', 'Aleksey Miranchuk', 'Alessandro Bastoni', 'Alessandro Berardi', 'Alessandro Buongiorno', 'Alessandro Burlamaqui', 'Alessandro Deiola', 'Alessandro Florenzi', 'Alessandro Plizzari', 'Alessandro SchÃ¶pf', 'Alessio Cragno', 'Alessio Riccardi', 'Alessio Romagnoli', 'Alex Cordaz', 'Alex Ferrari', 'Alex Iwobi', 'Alex KrÃ¡l', 'Alex McCarthy', 'Alex Meret', 'Alex Nicolao Telles', 'Alex Oxlade-Chamberlain', 'Alex Sandro Lobo Silva', 'Alexander Djiku', 'Alexander Hack', 'Alexander Isak', 'Alexander NÃ¼bel', 'Alexander SÃ¸rloth', 'Alexander Schwolow', 'Alexandre Lacazette', 'Alexandre Letellier', 'Alexandre Moreno Lopera', 'Alexandre Oukidja', 'Alexis Alejandro SÃ¡nchez SÃ¡nchez', 'Alexis Claude-Maurice', 'Alexis Laurent Patrice Roge Flips', 'Alexis Mac Allister', 'Alexis Saelemaekers', 'Alfie Devine', 'Alfonso GonzÃ¡lez MartÃ­nez', 'Alfonso Pastor Vacas', 'Alfonso Pedraza Sag', 'AlfreÃ° Finnbogason', 'Alfred Benjamin Gomis', 'Ali Reghba', 'Alidu Seidu', 'Alisson RamsÃ©s Becker', 'Alkhaly Momo CissÃ©', 'Allan Marques Loureiro', 'Allan Saint-Maximin', 'Allan Tchaptchet', 'Allan-RomÃ©o Nyom', 'Almamy TourÃ©', 'Alou Kuol', 'Alpha Sissoko', 'Alphonse Areola', 'Alphonso Boyle Davies', 'Amad Diallo TraorÃ©', 'Amadou Diawara', 'Amadou HaÃ¯dara', 'Amadou Mbengue', 'Amadou TraorÃ©', 'Amadou Zeund Georges Ba Mvom Onana', 'Amankwaa Akurugu', 'Amari Miller', 'Amath Ndiaye Diedhiou', 'Ambroise Oyongo Bitolo', 'Ã–mer Faruk Beyaz', 'Amin Younes', 'Amine Adli', 'Amine Bassi', 'Amine Gouiri', 'Amine Harit', 'Amir Rrahmani', 'Amos Pieper', 'Anastasios Donis', 'Ander Barrenetxea Muguruza', 'Ander Capa RodrÃ­guez', 'Ander Guevara Lajo', 'Ander Herrera AgÃ¼era', 'Anderson-Lenda Lucoqui', 'Andi Avdi Zeqiri', 'Andoni Gorosabel Espinosa', 'AndrÃ© Anderson Pomilio Lima da Silva', 'AndrÃ© Filipe Tavares Gomes', 'AndrÃ© Hahn', 'AndrÃ© Miguel Valente da Silva', 'AndrÃ©-Franck Zambo Anguissa', 'AndrÃ©s Alberto Andrade CedeÃ±o', 'AndrÃ©s Felipe Solano DÃ¡vila', 'AndrÃ©s MartÃ­n GarcÃ­a', 'Andrea Belotti', 'Andrea Cambiaso', 'Andrea Carboni', 'Andrea Consigli', 'Andrea Conti', 'Andrea La Mantia', 'Andrea Masiello', 'Andrea Petagna', 'Andrea Pinamonti', 'Andrea Ranocchia', 'Andrea Schiavone', 'Andreas BÃ¸dtker Christensen', 'Andreas Luthe', 'Andreas Skov Olsen', 'Andreas Voglsammer', 'Andreaw Gravillon', 'Andrei Girotto', 'Andrej KramariÄ‡', 'Andrew Abiola Omobamidele', 'Andrew Lonergan', 'Andrew Moran', 'Andrew Robertson', 'Andrey Lunev', 'Andriy Lunin', 'Andriy Yarmolenko', 'Andros Townsend', 'Andy Delort', 'Anga Dedryck Boyata', 'Angelo Fulgini', 'Angelo Obinze Ogbonna', 'Angelo Stiller', 'Angus Gunn', 'Anselmo Garcia MacNulty', 'Ansgar Knauff', 'Anssumane Fati', 'Ante Budimir', 'Ante RebiÄ‡', 'Antef Tsoungui', 'Anthony Caci', 'Anthony David Junior Elanga', 'Anthony Driscoll-Glennon', 'Anthony Gomez Mancini', 'Anthony Gordon', 'Anthony Limbombe Ekango', 'Anthony Lopes', 'Anthony Losilla', 'Anthony Mandrea', 'Anthony Martial', 'Anthony Modeste', 'Anthony RubÃ©n Lozano ColÃ³n', 'Anthony Ujah', 'Antoine Griezmann', 'Anton Ciprian TÄƒtÄƒruÈ™anu', 'Anton Stach', 'AntonÃ­n BarÃ¡k', 'Antonin Bobichon', 'Antonino Ragusa', 'Antonio BarragÃ¡n FernÃ¡ndez', 'Antonio Blanco Conde', 'Antonio Candreva', 'Antonio JosÃ© RaÃ­llo Arenas', 'Antonio JosÃ© RodrÃ­guez DÃ­az', 'Antonio Junior Vacca', 'Antonio Latorre Grueso', 'Antonio Luca Fiordilino', 'Antonio Moya Vega', 'Antonio RÃ¼diger', 'Antonio Rosati', 'Antonio SÃ¡nchez Navarro', 'Antonio Sivera SalvÃ¡', 'Antonio Zarzana PÃ©rez', 'Anwar El Ghazi', 'ArbÃ«r Zeneli', 'Archie Mair', 'Ardian Ismajli', 'Arial Benabent Mendy', 'Aridane HernÃ¡ndez UmpiÃ©rrez', 'Aritz Elustondo Irribaria', 'Arkadiusz Krystian Milik', 'Arkadiusz Reca', 'Armand LaurientÃ©', 'Armando Broja', 'Armando Izzo', 'Armel Bella Kotchap', 'Armstrong Okoflex', 'ArnÃ³r SigurÃ°sson', 'Arnaldo Antonio Sanabria Ayala', 'Arnau Tenas UreÃ±a', 'Arnaud Dominique Nordin', 'Arnaud Kalimuendo-Muinga', 'Arnaud Souquet', 'Arnaut Danjuma Groeneveld', 'Arne Maier', 'Arne Schulz', 'Arthur Desmas', 'Arthur Henrique Ramos de Oliveira Melo', 'Arthur Masuaku', 'Arthur Nicolas Theate', 'Arthur Okonkwo', 'Arturo Erasmo Vidal Pardo', 'Ashley Fletcher', 'Ashley Luke Barnes', 'Ashley Westwood', 'Ashley Young', 'Asier Illarramendi Andonegi', 'Asier Villalibre Molina', 'Asmir BegoviÄ‡', 'Assane DioussÃ© El Hadji', 'Aster Jan Vranckx', 'Atakan Karazor', 'Augusto Jorge Mateo Solari', 'AurÃ©lien TchouamÃ©ni', 'Axel Arthur Disasi', 'Axel Camblan', 'Axel Tuanzebe', 'Axel Wilfredo Werner', 'Axel Witsel', 'Aymen Barkok', 'Aymeric Laporte', 'Ayodeji Sotona', 'Ayoze PÃ©rez GutiÃ©rrez', 'Azor Matusiwa', 'Ažtefan Daniel Radu', 'AŽŸA£ A…ƒæ°—', 'Azzedine Ounahi', 'BÃ©ni Makouana', 'Bakary Adama Soumaoro', 'Bali Mumba', 'Bamidele Alli', 'Bamo MeÃ¯tÃ©', 'Bandiougou Fadiga', 'Baptiste SantamarÃ­a', 'BartAomiej DrÄ…gowski', 'Bartosz BereszyAski', 'Bartosz BiaAek', 'Bastian Oczipka', 'Batista Mendy', 'BeÃ±at Prados DÃ­az', 'Ben Bobzien', 'Ben Chilwell', 'Ben Chrisene', 'Ben Foster', 'Ben Gibson', 'Ben Godfrey', 'Ben Klefisch', 'Ben Mee', 'Benjamin AndrÃ©', 'Benjamin Bourigeaud', 'Benjamin HÃ¼bner', 'Benjamin Henrichs', 'Benjamin Johnson', 'Benjamin Lecomte', 'Benjamin Lhassine Kone', 'Benjamin Mendy', 'Benjamin Pavard', 'Benjamin Thomas Davies', 'Benjamin Uphoff', 'Benjamin White', 'Benno Schmitz', 'BenoÃ®t Badiashile Mukinayi', 'BenoÃ®t Costil', 'Berat Djimsiti', 'Bernardo Costa da Rosa', 'Bernardo Mota Veiga de Carvalho e Silva', 'Bernd Leno', 'Bertrand Isidore TraorÃ©', 'Bilal Benkhedim', 'Bilal Nadir', 'Billal Brahimi', 'Billy Gilmour', 'Billy Koumetio', 'Bingourou Kamara', 'Birger Solberg Meling', 'Bjarki Steinn Bjarkason', 'BoA¡ko Autalo', 'Bobby Adekanye', 'Bobby Thomas', 'Boris RadunoviÄ‡', 'Borja Iglesias Quintas', 'Borja Mayoral Moya', 'Borna Sosa', 'Boubacar Bernard Kamara', 'Boubacar Fall', 'Boubacar TraorÃ©', 'Boubakar KouyatÃ©', 'Boubakary SoumarÃ©', 'Boulaye Dia', 'Bouna Sarr', 'Bradley Locko', 'Brahim Abdelkader DÃ­az', 'Brais MÃ©ndez Portela', 'Bram Nuytinck', 'Brandon DominguÃ¨s', 'Brandon Soppy', 'Brandon Williams', 'Branimir Hrgota', 'Breel-Donald Embolo', 'Brendan Chardonnet', 'Brian Ebenezer Adjei Brobbey', 'Brian OlivÃ¡n Herrero', 'Brooklyn Lyons-Foster', 'Bruno AndrÃ© Cavaco JordÃ£o', 'Bruno GuimarÃ£es Rodriguez Moura', 'Bruno Miguel Borges Fernandes', 'Bruno Oliveira Bertinato', 'Bryan Cristante', 'Bryan Gil Salvatierra', 'Bryan Lasme', 'Bryan Mbeumo', 'Bryan Nokoue', 'Bryan Reynolds', 'Bukayo Saka', 'Burak YÄ±lmaz', 'CÃ©dric Brunner', 'CÃ©dric Hountondji', 'CÃ©dric Jan Itten', 'CÃ©dric Ricardo Alves Soares', 'CÃ©sar Azpilicueta Tanco', 'CÃ©sar Joel Valencia Castillo', 'CÄƒtÄƒlin CÃ®rjan', 'Caio Henrique Oliveira Silva', 'Caleb Ansah Ekuban', 'Caleb Cassius Watts', 'Callum Hudson-Odoi', 'Callum Wilson', 'Calum Chambers', 'Calvin Bombo', 'Calvin Stengs', 'Cameron Archer', 'Caoimhin Kelleher', 'Carles AleÃ±Ã¡ Castillo', 'Carles PÃ©rez Sayol', 'Carlo Pinsoglio', 'Carlos Akapo MartÃ­nez', 'Carlos Armando Gruezo Arboleda', 'Carlos Arturo Bacca Ahumada', 'Carlos Beitia Cardos', 'Carlos Clerc MartÃ­nez', 'Carlos DomÃ­nguez CÃ¡ceres', 'Carlos FernÃ¡ndez Luna', 'Carlos Henrique Venancio Casimiro', 'Carlos JoaquÃ­n Correa', 'Carlos Neva Tey', 'Carlos Soler BarragÃ¡n', 'Carney Chibueze Chukwuemeka', 'Cedric Teuchert', 'Cedric Wilfred TeguÃ­a Noubi', 'Cengiz Ãœnder', 'Cenk Tosun', 'ChÃ© Adams', 'Charalampos Lykogiannis', 'Charles Boli', 'Charles Mariano ArÃ¡nguiz Sandoval', 'Charles TraorÃ©', 'Charlie Cresswell', 'Charlie Goode', 'Charlie Taylor', 'Charlie Wiggett', 'Charly Musonda Junior', 'Cheick Oumar DoucourÃ©', 'Cheick Oumar SouarÃ©', 'Cheikh Ahmadou Bamba Mbacke Dieng', 'Cheikh Ahmet Tidian Niasse', 'Cheikh Tidiane Sabaly', 'Cheikhou KouyatÃ©', 'Chem Campbell', 'Chigozier Caleb Chukwuemeka', 'Chris FÃ¼hrich', 'Chris Smalling', 'Chrislain Iris Aurel Matsima', 'Christian Benteke Liolo', 'Christian Dannemann Eriksen', 'Christian Fernandes Marques', 'Christian FrÃ¼chtl', 'Christian GÃ¼nter', 'Christian GÃ³mez Vela', 'Christian Gabriel Oliva GimÃ©nez', 'Christian Kabasele', 'Christian Mate Pulisic', 'Christian Thers NÃ¸rgaard', 'Christoph Baumgartner', 'Christoph Kramer', 'Christoph Zimmermann', 'Christophe HÃ©relle', 'Christopher Antwi-Adjei', 'Christopher Grant Wood', 'Christopher Jeffrey Richards', 'Christopher Lenz', 'Christopher Maurice Wooh', 'Christopher Nkunku', 'Christopher Trimmel', 'Christos Tzolis', 'Ciaran Clark', 'Ciro Immobile', 'ClÃ©ment Nicolas Laurent Lenglet', 'Claudio AndrÃ©s Bravo MuÃ±oz', 'Clinton Mola', 'Cody Callum Pierre Drameh', 'Cole Palmer', 'Colin Dagba', 'Connor Roberts', 'Conor Carty', 'Conor Coady', 'Conor Gallagher', 'Conor NoÃŸ', 'Corentin Jean', 'Corentin Tolisso', 'Craig Dawson', 'Craig George Cathcart', 'CristÃ³bal Montiel RodrÃ­guez', 'Cristian Daniel Ansaldi', 'Cristian Esteban Gamboa Luna', 'Cristian Gabriel Romero', 'Cristian Molinaro', 'Cristian PortuguÃ©s Manzanera', 'Cristian Rivero Sabater', 'Cristian Tello Herrera', 'Cristiano Biraghi', 'Cristiano Lombardi', 'Cristiano Piccini', 'Cristiano Ronaldo dos Santos Aveiro', 'Crysencio Summerville', 'Curtis Jones', 'DÃ©nys Bain', 'Daan Heymans', 'DÃ­dac VilÃ¡ RossellÃ³', 'Dalbert Henrique Chagas EstevÃ£o', 'Dale Stephens', 'Daley Sinkgraven', 'DamiÃ¡n Emiliano MartÃ­nez', 'DamiÃ¡n NicolÃ¡s SuÃ¡rez SuÃ¡rez', 'Damiano Pecile', 'Damien Da Silva', 'Damir Ceter Valencia', 'Dan Burn', 'Dan Gosling', 'Dan-Axel Zagadou', 'Dane Pharrell Scarlett', 'Dani van den Heuvel', 'Daniel Amartey', 'Daniel Bachmann', 'Daniel Brosinski', 'Daniel CÃ¡rdenas LÃ­ndez', 'Daniel Caligiuri', 'Daniel Carvajal Ramos', 'Daniel Castelo Podence', 'Daniel Ceballos FernÃ¡ndez', 'Daniel CerÃ¢ntula Fuzato', 'Daniel Didavi', 'Daniel GÃ³mez AlcÃ³n', 'Daniel GarcÃ­a Carrillo', 'Daniel Ginczek', 'Daniel James', 'Daniel JosÃ© RodrÃ­guez VÃ¡zquez', 'Daniel Klein', 'Daniel Langley', 'Daniel Maldini', 'Daniel Nii Tackie Mensah Welbeck', 'Daniel Olmo Carvajal', 'Daniel Parejo MuÃ±oz', 'Daniel Plomer Gordillo', 'Daniel Raba AntolÃ­n', 'Daniel Sartori Bessa', 'Daniel Vivian Moreno', 'Daniel Wass', 'Daniel William John Ings', 'Daniele Baselli', 'Daniele Padelli', 'Daniele Rugani', 'Daniele Verde', 'Danijel PetkoviÄ‡', 'Danilo Cataldi', 'Danilo DAmbrosio', 'Danilo LuÃ­s HÃ©lio Pereira', 'Danilo Luiz da Silva', 'Danilo Teodoro Soares', 'Danny Blum', 'Danny Rose', 'Danny Vieira da Costa', 'Danny Ward', 'Dante Bonfim da Costa Santos', 'DarÃ­o Ismael Benedetto', 'DarÃ­o Poveda Romera', 'Darko BraA¡anac', 'Darko LazoviÄ‡', 'Darren Randolph', 'Darwin Daniel MachÃ­s Marcano', 'David Chidozie Okereke', 'David De Gea Quintana', 'David Edward Martin', 'David GarcÃ­a ZubirÃ­a', 'David Gil Mohedano', 'David Jason Remeseiro Salgueiro', 'David JosuÃ© JimÃ©nez Silva', 'David LÃ³pez Silva', 'David Lelle', 'David Nemeth', 'David Oberhauser', 'David Olatukunbo Alaba', 'David Ospina RamÃ­rez', 'David Pereira Da Costa', 'David Raum', 'David Raya Martin', 'David Schnegg', 'David Soria SolÃ­s', 'David Timor CopovÃ­', 'David Zima', 'Davide Biraschi', 'Davide Calabria', 'Davide Frattesi', 'Davide Santon', 'Davide Zappacosta', 'Davide Zappella', 'Davie Selke', 'Davinson SÃ¡nchez Mina', 'Davy Rouyard', 'Dayotchanculle Oswald Upamecano', 'Dean Henderson', 'Declan Rice', 'Deiver AndrÃ©s Machado Mena', 'Dejan Kulusevski', 'Dejan LjubiÄiÄ‡', 'Demarai Gray', 'Denis Athanase Bouanga', 'Denis Cheryshev', 'Denis Lemi Zakaria Lako Lado', 'Denis PetriÄ‡', 'Denis SuÃ¡rez FernÃ¡ndez', 'Denis Vavro', 'Dennis Appiah', 'Dennis Geiger', 'Dennis Jastrzembski', 'Dennis Praet', 'Dennis TÃ¸rset Johnsen', 'Denzel Justus Morris Dumfries', 'Destiny Iyenoma Udogie', 'Deyovaisio Zeefuik', 'DiadiÃ© SamassÃ©kou', 'Diant Ramaj', 'Dickson Abiama', 'Diego Carlos Santos Silva', 'Diego Demme', 'Diego Falcinelli', 'Diego FarÃ­as da Silva', 'Diego GonzÃ¡lez Polanco', 'Diego Javier Llorente RÃ­os', 'Diego JosÃ© Conde Alcolado', 'Diego LÃ³pez Noguerol', 'Diego LÃ³pez RodrÃ­guez', 'Diego Lainez Leyva', 'Diego Moreno Garbayo', 'Diego Rico Salguero', 'Diego Roberto GodÃ­n Leal', 'Diego Vicente Bri Carrazoni', 'Digbo Gnampa Habib MaÃ¯ga', 'Dilan Kumar Markanday', 'Dilane Bakwa', 'Dillon Hoogewerf', 'Dimitri Foulquier', 'Dimitri LiÃ©nard', 'Dimitri Payet', 'Dimitrios Nikolaou', 'Dimitris Giannoulis', 'Dimitry Bertaud', 'Diogo JosÃ© Teixeira da Silva', 'Dion Berisha', 'Dion Lopy', 'Divock Okoth Origi', 'DjenÃ© Dakonam Ortega', 'Djibril Fandje TourÃ©', 'Djibril SidibÃ©', 'Djibril Sow', 'DoÄŸan Alemdar', 'Dodi LukÃ©bakio', 'Domagoj BradariÄ‡', 'Domen ÄŒrnigoj', 'Domenico Berardi', 'Domenico Criscito', 'Domingos Sousa Coutinho Meneses Duarte', 'Dominic Calvert-Lewin', 'Dominic Schmidt', 'Dominic Thompson', 'Dominik Greif', 'Dominik Kohr', 'Dominik Szoboszlai', 'Dominique Heintz', 'Donny van de Beek', 'Donyell Malen', 'Dor Peretz', 'Douglas Luiz Soares de Paulo', 'DragiA¡a Gudelj', 'Dries Mertens', 'DuA¡an VlahoviÄ‡', 'Duje Ä†aleta-Car', 'DuvÃ¡n Esteban Zapata Banguera', 'Dwight Gayle', 'Dwight McNeil', 'Dylan Chambost', 'Dylan Daniel Mahmoud Bronn', 'Dynel Brown Kembo Simeu', 'é•·è°· éƒ¨èª', 'éè—¤ æ¸A¤ª', 'éè—¤ èˆª', 'Eberechi Eze', 'Ebrima Colley', 'Ebrima Darboe', 'Eddie Anthony Salcedo Mora', 'Eden Hazard', 'Eden Massouema', 'Ederson Santana de Moraes', 'Edgar Antonio MÃ©ndez Ortega', 'Edgar Badia Guardiola', 'Edgar GonzÃ¡lez Estrada', 'Edgar Paul Akouokou', 'Edgar Sevikyan', 'Edimilson Fernandes Ribeiro', 'Edin DA¾eko', 'Edinson Roberto Cavani GÃ³mez', 'Edmond FayÃ§al Tapsoba', 'Edoardo Bove', 'Edoardo Goldaniga', 'Edoardo Vergani', 'Edson AndrÃ© Sitoe', 'Eduard LÃ¶wen', 'Eduardo Camavinga', 'Edward Campbell Sutherland', 'Edward Nketiah', 'Einar Iversen', 'El Bilal TourÃ©', 'Elbasan Rashani', 'Eldin JakupoviÄ‡', 'Eldor Shomurodov', 'Elias Kratzer', 'Elijah Dixon-Bonner', 'Eliot Matazo', 'Eljif Elmas', 'Elliott Anderson', 'Ellis Simms', 'Ellyes Joris Skhiri', 'Elseid Hysaj', 'Elvis RexhbeÃ§aj', 'Emanuel Quartsin Gyasi', 'Emanuel Vignato', 'Emerson Aparecido Leite de Souza Junior', 'Emerson Palmieri dos Santos', 'Emil Audero Mulyadi', 'Emil Berggreen', 'Emil Henry Kristoffer Krafth', 'Emil Peter Forsberg', 'Emile Smith Rowe', 'Emiliano BuendÃ­a', 'Emmanuel Bonaventure Dennis', 'Emmanuel Kouadio KonÃ©', 'Emre Can', 'Emrehan Gedikli', 'Enes Ãœnal', 'Enis Bardhi', 'Enock Kwateng', 'Enock Mwepu', 'Enric Franquesa Dolz', 'Enrique Barja Afonso', 'Enrique GÃ³mez Hermoso', 'Enrique GarcÃ­a MartÃ­nez', 'Enzo Camille Alain Millot', 'Enzo Ebosse', 'Enzo Jeremy Le FÃ©e', 'Enzo Pablo Roco Roco', 'Erhan MaA¡oviÄ‡', 'Eric Bertrand Bailly', 'Eric Dier', 'Eric GarcÃ­a Martret', 'Eric Junior Dina Ebimbe', 'Eric Uhlmann', 'Erick Antonio Pulgar FarfÃ¡n', 'Erick Cathriel Cabaco Almada', 'Erik Durm', 'Erik Lamela', 'Erik Pieters', 'Erik Ross Palmer-Brown', 'Erik Thommy', 'Erion Sadiku', 'Erling Braut Haaland', 'Ermedin DemiroviÄ‡', 'Ermin BiÄakÄiÄ‡', 'Ernesto Torregrossa', 'Esey Gebreyesus', 'Esteban Ariel Saveljich', 'Ethan Ampadu', 'Ethan Pinnock', 'Etienne Capoue', 'Etienne Green', 'Etrit Berisha', 'Eugenio Pizzuto Puga', 'Evann Guessand', 'Exequiel Alejandro Palacios', 'éŽŒç”° A¤§Aœ°', 'Ezri Konsa Ngoyo', 'FÃ¡bio Daniel Soares Silva', 'FÃ¡bio Henrique Tavares', 'FÃ¡bio Pereira da Silva', 'FabiÃ¡n Ruiz PeÃ±a', 'Fabian Bredlow', 'Fabian Delph', 'Fabian Klos', 'Fabian Kunze', 'Fabian Lukas SchÃ¤r', 'Fabian RÃ¼th', 'Fabiano Parisi', 'Fabien Centonze', 'Fabien Lemoine', 'Fabio Blanco GÃ³mez', 'Fabio Depaoli', 'Fabio Quagliarella', 'Fabio Schneider', 'Facundo Axel Medina', 'Facundo Colidio', 'Facundo Pellistri Rebollo', 'Faouzi Ghoulam', 'Farid Boulaya', 'Farid El Melali', 'Federico Bernardeschi', 'Federico Bonazzoli', 'Federico Ceccherini', 'Federico Chiesa', 'Federico Di Francesco', 'Federico Dimarco', 'Federico FernÃ¡ndez', 'Federico Javier Santander Mereles', 'Federico JuliÃ¡n Fazio', 'Federico Marchetti', 'Federico Mattiello', 'Federico Peluso', 'Federico Santiago Valverde Dipetta', 'Felipe Anderson Pereira Gomes', 'Felipe Augusto de Almeida Monteiro', 'Felipe Salvador Caicedo Corozo', 'Felix Kalu Nmecha', 'Felix Passlack', 'Felix Schwarzholz', 'Ferland Mendy', 'Fernando Calero Villa', 'Fernando Francisco Reges', 'Fernando Luiz Rosa', 'Fernando MarÃ§al de Oliveira', 'Fernando MartÃ­n Forestieri', 'Fernando NiÃ±o RodrÃ­guez', 'Fernando Pacheco Flores', 'Ferran Torres GarcÃ­a', 'Fidel Chaves de la Torre', 'Fikayo Tomori', 'Filip ÄuriÄiÄ‡', 'Filip BenkoviÄ‡', 'Filip JÃ¶rgensen', 'Filip KostiÄ‡', 'Filippo Bandinelli', 'Filippo Delli Carri', 'Filippo Melegoni', 'Filippo Romagna', 'Filippo Tripi', 'Finley Stevens', 'Finn Gilbert Dahmen', 'Flavien Tait', 'Flavius David Daniliuc', 'Florent Da Silva', 'Florent Mollet', 'Florent Ogier', 'Florentino Ibrain Morris LuÃ­s', 'Florian Christian Neuhaus', 'Florian Grillitsch', 'Florian Kainz', 'Florian KrÃ¼ger', 'Florian Lejeune', 'Florian MÃ¼ller', 'Florian Niederlechner', 'Florian Palmowski', 'Florian Richard Wirtz', 'Florian Sotoca', 'Florian Tardieu', 'Florin Andone', 'Flynn Clarke', 'FodÃ© Ballo-TourÃ©', 'FodÃ© DoucourÃ©', 'Folarin Balogun', 'FrÃ©dÃ©ric Guilbert', 'FrÃ©dÃ©ric Veseli', 'Francesc FÃbregas i Soler', 'Francesco Acerbi', 'Francesco Bardi', 'Francesco Caputo', 'Francesco Cassata', 'Francesco Di Tacchio', 'Francesco Forte', 'Francesco Magnanelli', 'Francesco Rossi', 'Francis Coquelin', 'Francisco AlcÃ¡cer GarcÃ­a', 'Francisco AndrÃ©s Sierralta Carvallo', 'Francisco AntÃ³nio Machado Mota Castro TrincÃ£o', 'Francisco Casilla CortÃ©s', 'Francisco FemenÃ­a Far', 'Francisco Javier Hidalgo GÃ³mez', 'Francisco JosÃ© BeltrÃ¡n Peinado', 'Francisco JosÃ© GarcÃ­a Torres', 'Francisco MÃ©rida PÃ©rez', 'Francisco Manuel Gonzalez Verjara', 'Francisco RomÃ¡n AlarcÃ³n SuÃ¡rez', 'Franck Bilal RibÃ©ry', 'Franck Honorat', 'Franck Yannick KessiÃ©', 'Franco Daryl Tongya Heubang', 'Franco Emanuel Cervi', 'Franco MatÃ­as Russo Panos', 'Frank Ogochukwu Onyeka', 'FransÃ©rgio Rodrigues Barbosa', 'Fraser Forster', 'Fraser Hornby', 'Freddie Woodman', 'Frederico Rodrigues de Paula Santos', 'Frederik Franck Winther', 'Frederik Riis RÃ¸nnow', 'Frenkie de Jong', 'GÃ¶ktan GÃ¼rpÃ¼z', 'GaÃ«l Kakuta', 'GaÃ«tan Laborde', 'GaÃ«tan Poussin', 'Gabriel Armando de Abreu', 'Gabriel dos Santos MagalhÃ£es', 'Gabriel Fernando de Jesus', 'Gabriel Gudmundsson', 'Gabriel MoisÃ©s Antunes da Silva', 'Gabriel Mutombo Kupa', 'Gabriel Nascimento Resende BrazÃ£o', 'Gabriel Teodoro Martinelli Silva', 'Gabriele Corbo', 'Gabriele Zappa', 'Gaetano Castrovilli', 'Gaizka Campos BahÃ­llo', 'Gareth Frank Bale', 'Gary Alexis Medel Soto', 'GastÃ³n Rodrigo Pereiro LÃ³pez', 'Gauthier Gallon', 'Gautier Larsonneur', 'Gelson Dany Batalha Martins', 'Geoffrey Edwin Kondogbia', 'George McEachran', 'Georginio Rutter', 'Georginio Wijnaldum', 'GerÃ³nimo Rulli', 'Gerard Deulofeu LÃ¡zaro', 'Gerard Gumbau Garriga', 'Gerard Moreno BalaguerÃ³', 'Gerard PiquÃ© BernabÃ©u', 'GermÃ¡n Alejandro Pezzella', 'GermÃ¡n SÃ¡nchez Barahona', 'Gerrit Holtmann', 'Gerson Leal Rodrigues Gouveia', 'Gerson Santos da Silva', 'Gerzino Nyamsi', 'Ghislain Konan', 'Giacomo Bonaventura', 'Giacomo Raspadori', 'Giacomo Satalino', 'Gian Marco Ferrari', 'Giangiacomo Magnani', 'Gianluca Busio', 'Gianluca Caprari', 'Gianluca Frabotta', 'Gianluca Gaetano', 'Gian-Luca Itter', 'Gianluca Mancini', 'Gianluca Pegolo', 'Gianluca Scamacca', 'Gianluca SchÃ¤fer', 'Gian-Luca Waldschmidt', 'Gianluigi Donnarumma', 'Gianmarco Cangiano', 'Gianmarco Zigoni', 'Gideon Jung', 'Gideon Mensah', 'Gil-Linnart Walther', 'Giorgi Mamardashvili', 'Giorgio Altare', 'Giorgio Chiellini', 'Giorgos Kyriakopoulos', 'Giovani Lo Celso', 'Giovanni Alejandro Reyna', 'Giovanni Crociata', 'Giovanni Di Lorenzo', 'Giovanni Pablo Simeone', 'Giulian Biancone', 'Giuliano Simeone', 'Giulio Maggiore', 'Giuseppe Pezzella', 'Gleison Bremer Silva Nascimento', 'Gnaly Maxwel Cornet', 'GonÃ§alo Manuel Ganchinho Guedes', 'GonÃ§alo Mendes PaciÃªncia', 'Gonzalo Ariel Montiel', 'Gonzalo Cacicedo VerdÃº', 'Gonzalo Escalante', 'Gonzalo JuliÃ¡n Melero Manzanares', 'Gonzalo Villar del Fraile', 'Gor Manvelyan', 'Goran Pandev', 'GrÃ©goire Coudert', 'Granit Xhaka', 'Grant Hanley', 'Gregoire Defrel', 'Gregor Kobel', 'Gregorio Gracia SÃ¡nchez', 'Grigoris Kastanos', 'Grischa PrÃ¶mel', 'Guglielmo Vicario', 'Guido Guerrieri', 'Guido Marcelo Carrillo', 'Guido RodrÃ­guez', 'Guilherme Magro Pires Ramos', 'Guillermo Alfonso MaripÃ¡n Loaysa', 'Gylfi ÃžÃ³r SigurÃ°sson', 'HÃ¥vard Kallevik Nielsen', 'HÃ¥vard Nordtveit', 'HÃ©ctor BellerÃ­n Moruno', 'HÃ©ctor Junior Firpo AdamÃ©s', 'HÃ©ctor Miguel Herrera LÃ³pez', 'HÃ©lder Wander Sousa Azevedo Costa', 'Habib Ali Keita', 'Habib Diarra', 'Habibou Mouhamadou Diallo', 'Hakan Ã‡alhanoÄŸlu', 'Hakim Ziyech', 'Hamadi Al Ghaddioui', 'Hamari TraorÃ©', 'Hamed Junior TraorÃ¨', 'Hamza Choudhury', 'Hannes Wolf', 'Hannibal Mejbri', 'Hans Carl Ludwig Augustinsson', 'Hans Fredrik Jensen', 'Hans Hateboer', 'Hans Nunoo Sarpei', 'Haris Belkebla', 'Harold Moukoudi', 'Harrison Ashby', 'Harry Kane', 'Harry Lewis', 'Harry Maguire', 'Harry Winks', 'Harvey David White', 'Harvey Elliott', 'Harvey Lewis Barnes', 'Hassane Kamara', 'Hayden Lindley', 'Haydon Roberts', 'Helibelton Palacios Zapata', 'Henrikh Mkhitaryan', 'Henrique Silva Milagres', 'Henry Wise', 'Herbert Bockhorn', 'HernÃ¢ni Jorge Santos Fortes', 'Hernani Azevedo JÃºnior', 'Hianga Mananga Mbock', 'Hicham Boudaoui', 'Hirving Rodrigo Lozano Bahena', 'Houboulang Mendes', 'Houssem Aouar', 'Hugo Bueno LÃ³pez', 'Hugo Duro Perales', 'Hugo Ekitike', 'Hugo GuillamÃ³n SammartÃ­n', 'Hugo Lloris', 'Hugo Magnetti', 'Hugo Mallo Novegil', 'Hugo Novoa Ramos', 'ì•ìš°ì˜ Woo Yeong Jeong', 'ìí˜ì¤€ çŽé‰ä¿Š', 'ì†í¥ë¯¼ A­™A…´æ…œ', 'ì´ê°•ì¸ Kang-In Lee', 'ì´ìž¬ì± æŽAœ¨æˆ', 'IÃ±aki Williams Arthuer', 'IÃ±igo PÃ©rez Soto', 'IÃ±igo Ruiz de Galarreta Etxeberria', 'Iago Amaral Borduchi', 'Iago Aspas Juncal', 'Ibrahim Amadou', 'Ibrahim Karamoko', 'Ibrahim Yalatif Diabate', 'Ibrahima BaldÃ©', 'Ibrahima Diallo', 'Ibrahima KonatÃ©', 'Ibrahima Mbaye', 'Ibrahima Niane', 'Ibrahima Sissoko', 'Iddrisu Baba Mohammed', 'Idrissa Gana Gueye', 'Ignacio Monreal Eraso', 'Ignacio PeÃ±a Sotorres', 'Ignacio Pussetto', 'Ignacio RamÃ³n del Valle', 'Ignacio Vidal Miralles', 'Ignatius Kpene Ganago', 'Igor JÃºlio dos Santos de Paulo', 'Igor Silva de Almeida', 'Igor ZubeldÃ­a Elorza', 'Ihlas Bebou', 'Ihsan Sacko', 'Iker Benito SÃ¡nchez', 'Iker Losada Aragunde', 'Iker Muniain GoÃ±i', 'Iker Pozo La Rosa', 'Iker Recio Ortega', 'Ilan Kais Kebbal', 'Ilija Nestorovski', 'Illan Meslier', 'Imran Louza', 'IonuÈ› Andrei Radu', 'Irvin Cardona', 'Isaac Ajayi Success', 'Isaac CarcelÃ©n Valencia', 'Isaac Hayden', 'Isaac Lihadji', 'Isaac PalazÃ³n Camacho', 'Ishak Belfodil', 'Islam Slimani', 'IsmaÃ¯la Sarr', 'IsmaÃ«l Bennacer', 'IsmaÃ«l Boura', 'IsmaÃ«l Gharbi', 'IsmaÃ«l TraorÃ©', 'Ismael Ruiz SÃ¡nchez', 'Ismail Joshua Jakobs', 'Ismaila PathÃ© Ciss', 'Israel Salazar PÃ­riz', 'Issa Diop', 'Issa KaborÃ©', 'Issouf Sissokho', 'í™©ì˜ì¡° Ui Jo Hwang', 'í™©í¬ì°¬ é»A–œç¿', 'IvÃ¡n Alejo Peralta', 'IvÃ¡n Balliu Campeny', 'IvÃ¡n Bravo Castro', 'IvÃ¡n Chapela LÃ³pez', 'IvÃ¡n JosÃ© Marcone', 'IvÃ¡n MartÃ­n NÃºÃ±ez', 'IvÃ¡n MartÃ­nez GonzÃ¡lvez', 'IvÃ¡n Mauricio Arboleda', 'IvÃ¡n Romero de Ãvila', 'Ivan AaponjiÄ‡', 'Ivan IliÄ‡', 'Ivan PeriA¡iÄ‡', 'Ivan Provedel', 'Ivan RadovanoviÄ‡', 'Ivan RakitiÄ‡', 'Ivan Toney', 'Ivo GrbiÄ‡', 'Ivor Pandur', 'JÃ©rÃ´me Boateng', 'JÃ©rÃ´me Hergault', 'JÃ©rÃ´me Phojo', 'JÃ©rÃ´me Roussillon', 'JÃ©rÃ©mie Boga', 'JÃ©rÃ©my Doku', 'JÃ©rÃ©my Le Douaron', 'JÃ©rÃ©my Morel', 'JÃ©rÃ©my Pied', 'JÃ©rÃ©my Pierre SincÃ¨re Gelin', 'JÃ¶rgen Joakim Nilsson', 'JÃ¼rgen Locadia', 'JaÃ¯ro Riedewald', 'Jack Butland', 'Jack Clarke', 'Jack Cork', 'Jack de Vries', 'Jack Grealish', 'Jack Harrison', 'Jack Jenkins', 'Jack Stephens', 'Jack Young', 'Jacob Barrett Laursen', 'Jacob Bedeau', 'Jacob Bruun Larsen', 'Jacob Michael Italiano', 'Jacob Montes', 'Jacob Murphy', 'Jacob Ramsey', 'Jacopo Dezi', 'Jacopo Furlan', 'Jacopo Sala', 'Jaden Philogene-Bidace', 'Jadon Sancho', 'Jaime Mata Arnaiz', 'Jakob Busk Jensen', 'Jakob Lungi SÃ¸rensen', 'Jakub Jankto', 'Jakub Piotr Moder', 'Jamaal Lascelles', 'Jamal Baptiste', 'Jamal Lewis', 'Jamal Musiala', 'James David RodrÃ­guez Rubio', 'James Furlong', 'James Justin', 'James Maddison', 'James McArthur', 'James McAtee', 'James Norris', 'James Olayinka', 'James Philip Milner', 'James Tarkowski', 'James Tomkins', 'James Ward-Prowse', 'Jamie Leweling', 'Jamie Shackleton', 'Jamie Vardy', 'Jan A½ambA¯rek', 'Jan Bednarek', 'Jan Jakob Olschowsky', 'Jan MorÃ¡vek', 'Jan Oblak', 'Jan SchrÃ¶der', 'Jan Thielmann', 'Jan Thilo Kehrer', 'Janik Haberer', 'Janis Antiste', 'Jannes-Kilian Horn', 'Jannik Vestergaard', 'Janni-Luca Serra', 'Jannis Lang', 'Japhet Tanganga', 'JarosAaw PrzemysAaw Jach', 'Jarrad Branthwaite', 'Jarrod Bowen', 'Jason Berthomier', 'Jason Denayer', 'Jason Steele', 'Jasper Cillessen', 'Jaume DomÃ©nech SÃ¡nchez', 'Jaume Grau Ciscar', 'Jaume Vicent Costa JordÃ¡', 'JavairÃ´ Dilrosun', 'Javier Castro UrdÃ­n', 'Javier DÃ­az SÃ¡nchez', 'Javier GÃ³mez Castroverde', 'Javier GalÃ¡n Gil', 'Javier LÃ³pez Carballo', 'Javier LlabrÃ©s Exposito', 'Javier Manquillo GaitÃ¡n', 'Javier MartÃ­nez Calvo', 'Javier MatÃ­as Pastore', 'Javier Ontiveros Parra', 'Javier Puado DÃ­az', 'Javier Serrano MartÃ­nez', 'Jay Rodriguez', 'Jayden Jezairo Braaf', 'Jean Emile Junior Onana Onana', 'Jean Harisson Marcelin', 'Jean Lucas de Souza Oliveira', 'Jean-Charles Castelletto', 'Jean-Clair Dimitri Roger Todibo', 'Jean-Claude Billong', 'Jean-Daniel Dave Lewis Akpa Akpro', 'Jean-Eric Maxim Choupo-Moting', 'Jean-EudÃ¨s Aholou', 'Jean-KÃ©vin Augustin', 'Jean-KÃ©vin Duverne', 'Jean-Louis Leca', 'Jean-Paul BoÃ«tius', 'Jean-Philippe Gbamin', 'Jean-Philippe Krasso', 'Jean-Philippe Mateta', 'Jean-Ricner Bellegarde', 'Jean-Victor Makengo', 'Jed Steer', 'Jeff Patrick Hendrick', 'Jeff Reine-AdÃ©laÃ¯de', 'Jeffrey Gouweleeuw', 'Jeffrey Schlupp', 'Jeison FabiÃ¡n Murillo CerÃ³n', 'Jens Castrop', 'Jens Grahl', 'Jens JÃ¸nsson', 'Jens Petter Hauge', 'Jens Stryger Larsen', 'Jerdy Schouten', 'Jere Juhani Uronen', 'JeremÃ­as Conan Ledesma', 'Jeremiah St. Juste', 'Jeremie Agyekum Frimpong', 'Jeremy Dudziak', 'Jeremy Ngakia', 'Jeremy Sarmiento', 'Jeremy Toljan', 'Jeroen Zoet', 'JesÃºs Areso Blanco', 'JesÃºs JoaquÃ­n FernÃ¡ndez SÃ¡ez de la Torre', 'JesÃºs Navas GonzÃ¡lez', 'JesÃºs VÃ¡zquez Alcalde', 'JesÃºs Vallejo LÃ¡zaro', 'Jesper GrÃ¦nge LindstrÃ¸m', 'Jesse Lingard', 'Jessic GaÃ¯tan Ngankam', 'Jessy Moulin', 'Jesurun Rak-Sakyi', 'Jetro Willems', 'Jim Ã‰milien Ngowet Allevinah', 'Jimmy Briand', 'Jimmy Cabot', 'Jimmy Giraudon', 'JoA¡ko Gvardiol', 'JoÃ£o FÃ©lix Sequeira', 'JoÃ£o Filipe Iria Santos Moutinho', 'JoÃ£o Paulo Santos da Costa', 'JoÃ£o Pedro Cavaco Cancelo', 'JoÃ£o Pedro Geraldino dos Santos GalvÃ£o', 'JoÃ£o Pedro Junqueira de Jesus', 'JoÃ«l Andre Job Matip', 'JoÃ«l Ivo Veltman', 'Joachim Christian Andersen', 'Joakim MÃ¦hle Pedersen', 'Joan GarcÃ­a Pons', 'Joan JordÃ¡n Moreno', 'Joan Sastre Vanrell', 'JoaquÃ­n JosÃ© MarÃ­n Ruiz', 'JoaquÃ­n Navarro JimÃ©nez', 'JoaquÃ­n SÃ¡nchez RodrÃ­guez', 'Jodel Dossou', 'Joe Gelhardt', 'Joe Gomez', 'Joe Rodon', 'Joel Chukwuma Obi', 'Joel Ideho', 'Joel LÃ³pez Salguero', 'Joel Robles BlÃ¡zquez', 'Joel Ward', 'Joelinton Cassio ApolinÃ¡rio de Lira', 'Jofre Carreras PagÃ¨s', 'Johan AndrÃ©s Mojica Palacio', 'Johan Felipe VÃ¡squez Ibarra', 'Johan Gastien', 'Johann Berg GuÃ°mundsÂ­son', 'John Anthony Brooks', 'John Guidetti', 'John Joe Patrick Finn Benoa', 'John McGinn', 'John Nwankwo Chetauya Donald Okeh', 'John Ruddy', 'John Stones', 'Jokin Ezkieta Mendiburu', 'Jon Ander Garrido Moracia', 'Jon Guridi Aldalur', 'Jon McCracken', 'Jon Moncayola Tollar', 'Jon Morcillo Conesa', 'Jon Pacheco Dozagarat', 'Jon Sillero Monreal', 'JonÃ¡s Ramalho Chimeno', 'Jonas Hector', 'Jonas Hofmann', 'Jonas Kersken', 'Jonas Martin', 'Jonas Michelbrink', 'Jonas Omlin', 'Jonas Urbig', 'Jonatan Carmona Ãlamo', 'Jonathan Bamba', 'Jonathan Castro Otto', 'Jonathan Christian David', 'Jonathan Clauss', 'Jonathan Cristian Silva', 'Jonathan DamiÃ¡n Iglesias Abreu', 'Jonathan Gradit', 'Jonathan Grant Evans', 'Jonathan Michael Burkardt', 'Jonathan RodrÃ­guez MenÃ©ndez', 'Jonathan Russell', 'Jonathan Schmid', 'Jonathan Tah', 'Jonjo Shelvey', 'Jonjoe Kenny', 'JorÃ¨s Rahou', 'Jordan Ferri', 'Jordan Henderson', 'Jordan Holsgrove', 'Jordan KÃ©vin Amavi', 'Jordan Lotomba', 'Jordan Meyer', 'Jordan Pickford', 'Jordan Pierre Ayew', 'Jordan Tell', 'Jordan Torunarigha', 'Jordan Veretout', 'Jordan Zacharie Lukaku Menama Mokelenge', 'Jordi Alba Ramos', 'Jordi Bongard', 'Jordi Mboula Queralt', 'Jorge AndÃºjar Moreno', 'Jorge Cuenca Barreno', 'Jorge de Frutos SebastiÃ¡n', 'Jorge Filipe Soares Silva', 'Jorge MerÃ© PÃ©rez', 'Jorge MiramÃ³n Santagertrudis', 'Jorge Molina Vidal', 'Jorge Padilla Soler', 'Jorge ResurrecciÃ³n Merodio', 'Joris Chotard', 'Joris Gnagnon', 'JosÃ© Ãngel Carmona Navarro', 'JosÃ© Ãngel Esmoris Tasende', 'JosÃ© Ãngel GÃ³mez CampaÃ±a', 'JosÃ© Ãngel Pozo la Rosa', 'JosÃ© Ãngel ValdÃ©s DÃ­az', 'JosÃ© Alejandro MartÃ­n ValerÃ³n', 'JosÃ© Alonso Lara', 'JosÃ© AndrÃ©s Guardado HernÃ¡ndez', 'JosÃ© Antonio FerrÃ¡ndez Pomares', 'JosÃ© Antonio Morente Oliva', 'JosÃ© David Menargues', 'JosÃ© Diogo Dalot Teixeira', 'JosÃ© Ignacio FernÃ¡ndez Iglesias', 'JosÃ© Juan MacÃ­as GuzmÃ¡n', 'JosÃ© LuÃ­s GayÃ PeÃ±a', 'JosÃ© Luis Morales Nogales', 'JosÃ© Luis Palomino', 'JosÃ© Luis SanmartÃ­n Mato', 'JosÃ© Manuel Cabrera LÃ³pez', 'JosÃ© Manuel FontÃ¡n MondragÃ³n', 'JosÃ© Manuel Reina PÃ¡ez', 'JosÃ© Manuel RodrÃ­guez Benito', 'JosÃ© Manuel SÃ¡nchez GuillÃ©n', 'JosÃ© MarÃ­a CallejÃ³n Bueno', 'JosÃ© MarÃ­a GimÃ©nez de Vargas', 'JosÃ© MarÃ­a MartÃ­n-Bejarano Serrano', 'JosÃ© MarÃ­a Relucio Gallego', 'JosÃ© MartÃ­n CÃ¡ceres Silva', 'JosÃ© Miguel da Rocha Fonte', 'JosÃ© Pedro Malheiro de SÃ¡', 'JosÃ© RaÃºl GutiÃ©rrez Parejo', 'JosÃ© SÃ¡nchez MartÃ­nez', 'JosÃ© SalomÃ³n RondÃ³n GimÃ©nez', 'Joscha Wosz', 'Jose LuÃ­s GarcÃ­a VayÃ¡', 'Joseba ZaldÃºa Bengoetxea', 'Josep GayÃ MartÃ­nez', 'Josep MartÃ­nez Riera', 'Joseph Aidoo', 'Joseph Alfred Duncan', 'Joseph Scally', 'Joseph Shaun Hodge', 'Joseph Willock', 'Josh Brooking', 'Josh Brownhill', 'Josh Maja', 'Joshua Brenet', 'Joshua Christian Kojo King', 'Joshua Felix Okpoda Eppiah', 'Joshua Thomas Sargent', 'Joshua Walter Kimmich', 'Josip Brekalo', 'Josip IliÄiÄ‡', 'Josip StaniA¡iÄ‡', 'JosuÃ© Albert', 'Josuha Guilavogui', 'Juan AgustÃ­n Musso', 'Juan Antonio Iglesias SÃ¡nchez', 'Juan Bernat Velasco', 'Juan Camilo HernÃ¡ndez SuÃ¡rez', 'Juan Cruz Ãlvaro Armada', 'Juan Cruz DÃ­az EspÃ³sito', 'Juan Flere Pizzuti', 'Juan Guilherme Nunes Jesus', 'Juan Guillermo Cuadrado Bello', 'Juan Ignacio RamÃ­rez Polero', 'Juan Manuel Mata GarcÃ­a', 'Juan Manuel PÃ©rez Ruiz', 'Juan Marcos Foyth', 'Juan Miguel JimÃ©nez LÃ³pez', 'Juan Miranda GonzÃ¡lez', 'Juan Torres Ruiz', 'Jude Victor William Bellingham', 'Judilson Mamadu TuncarÃ¡ Gomes', 'Julen Agirrezabala', 'Jules KoundÃ©', 'Julian Albrecht', 'Julian Baumgartlinger', 'Julian Brandt', 'Julian Chabot', 'Julian Draxler', 'Julian Green', 'Julian Jeanvier', 'Julian Philipp Frommann', 'Julian Pollersbeck', 'Julian Ryerson', 'Julien Boyer', 'Julien Faussurier', 'Julien Laporte', 'Julius Pfennig', 'Junior Castello Lukeba', 'Junior Morau Kadile', 'Junior Wakalible Lago', 'Junior Walter Messias', 'Juraj Kucka', 'Jurgen Ekkelenkamp', 'Justin Hoogma', 'Justin Kluivert', 'Justin Smith', 'KÃ©vin Boma', 'KÃ©vin Gameiro', 'KÃ©vin Malcuit', 'KÃ©vin Manuel Rodrigues', 'KÃ©vin NDoram', 'Kaan Ayhan', 'Kaan Kurt', 'Kacper UrbaAski', 'Kai Lukas Havertz', 'Kaio Jorge Pinto Ramos', 'Kaito Mizuta', 'Kalidou Koulibaly', 'Kalifa Coulibaly', 'Kalvin Phillips', 'Kamaldeen Sulemana', 'Karim Azamoum', 'Karim Bellarabi', 'Karim Benzema', 'Karim Onisiwo', 'Karim Rekik', 'Karl Brillant Toko Ekambi', 'Karl Darlow', 'Karol Fila', 'Karol Linetty', 'Kasim Adams Nuhu', 'Kasper Dolberg Rasmussen', 'Kasper Peter Schmeichel', 'Kayky da Silva Chagas', 'Kays Ruiz-Atil', 'Keanan Bennetts', 'Keidi Bare', 'Keinan Davis', 'Keita BaldÃ© Diao', 'Kelechi Promise Iheanacho', 'Kelvin Amian Adou', 'Ken Nlata Sema', 'Ken Remi Stefan Strandberg', 'Kenny McLean', 'Kepa Arrizabalaga Revuelta', 'Kerem Demirbay', 'Keven Schlotterbeck', 'Kevin AndrÃ©s Agudelo Ardila', 'Kevin Behrens', 'Kevin Bonifazi', 'Kevin Danso', 'Kevin De Bruyne', 'Kevin John Ufuoma Akpoguma', 'Kevin Kampl', 'Kevin Lasagna', 'Kevin Long', 'Kevin MÃ¶hwald', 'Kevin Piscopo', 'Kevin RÃ¼egg', 'Kevin Schade', 'Kevin StÃ¶ger', 'Kevin Strootman', 'Kevin Trapp', 'Kevin VÃ¡zquez ComesaÃ±a', 'Kevin Vogt', 'Kevin Volland', 'Kevin-Prince Boateng', 'Keylor Navas Gamboa', 'Kgaogelo Chauke', 'KhÃ©phren Thuram-Ulien', 'Kieran Dowell', 'Kieran Tierney', 'Kieran Trippier', 'Kiernan Dewsbury-Hall', 'Ki-Jana Delano Hoever', 'Kiliann Sildillia', 'Kimberly Ezekwem', 'Kingsley Dogo Michael', 'Kingsley Ehizibue', 'Kingsley Fobi', 'Kingsley Junior Coman', 'Kingsley Schindler', 'Kjell Scherpen', 'Koba LeÃ¯n Koindredi', 'Koen Casteels', 'Konrad de la Fuente', 'Konrad Laimer', 'Konstantinos Manolas', 'Konstantinos Mavropanos', 'Konstantinos Stafylidis', 'Konstantinos Tsimikas', 'Koray GÃ¼nter', 'Kortney Hause', 'Kouadio-Yves Dabila', 'Kouassi Ryan Sessegnon', 'KrÃ©pin Diatta', 'Kristijan JakiÄ‡', 'Kristoffer Askildsen', 'Kristoffer Vassbakk Ajer', 'Kristoffer-August Sundquist Klaesson', 'Krisztofer HorvÃ¡th', 'Krzysztof PiÄ…tek', 'Kurt Happy Zouma', 'Kwadwo Baah', 'Kyle Alex John', 'Kyle Walker', 'Kyle Walker-Peters', 'Kylian MbappÃ© Lottin', 'LÃ¡szlÃ³ BÃ©nes', 'LÃ©o Dubois', 'LÃ©o Leroy', 'LÃ©o PÃ©trot', 'LÃ©vy Koffi Djidji', 'Lamare Bogarde', 'Landry Nany Dimata', 'Lars Edi Stindl', 'Lassana Coulibaly', 'Lasse GÃ¼nther', 'Lasse RieÃŸ', 'Lasse Schulz', 'Laurent Abergel', 'Laurent Koscielny', 'Laurenz Dehl', 'Lautaro de LeÃ³n Billar', 'Lautaro Javier MartÃ­nez', 'Lautaro Marco Spatz', 'Layvin Kurzawa', 'Lazar SamardA¾iÄ‡', 'Leander Dendoncker', 'Leandro Barreiro Martins', 'Leandro Daniel Cabrera SasÃ­a', 'Leandro Daniel Paredes', 'Leandro Trossard', 'Lebo Mothiba', 'Lee Grant', 'Lennart Czyborra', 'Lenny Jean-Pierre Pintor', 'Lenny Joseph', 'Lenny Lacroix', 'Leo Atulac', 'Leo Fuhr Hjelde', 'Leon Bailey Butler', 'Leon Christoph Goretzka', 'Leon Valentin Schaffran', 'Leonardo Bonucci', 'Leonardo CÃ©sar Jardim', 'Leonardo Capezzi', 'Leonardo de Souza Sena', 'Leonardo JuliÃ¡n Balerdi Rossa', 'Leonardo Mancuso', 'Leonardo Pavoletti', 'Leonardo RomÃ¡n Riquelme', 'Leonardo Spinazzola', 'Leroy Aziz SanÃ©', 'Lesley Chimuanya Ugochukwu', 'Levi Jeremiah Lumeka', 'Levin Mete Ã–ztunali', 'Lewis Baker', 'Lewis Bate', 'Lewis Dobbin', 'Lewis Dunk', 'Lewis Gordon', 'Lewis Paul Jimmy Richards', 'Lewis Richardson', 'Liam Cooper', 'Liam Delap', 'Liam Gibbs', 'Liam Henderson', 'Liam McCarron', 'Lilian Brassier', 'Lilian Egloff', 'Linus Gechter', 'Lionel AndrÃ©s Messi Cuccittini', 'Lisandru Tramoni', 'LluÃ­s Andreu i Ruiz', 'LluÃ­s Recasens Vives', 'LoÃ¯c BadÃ©', 'Lorenz Assignon', 'Lorenzo Andrenacci', 'Lorenzo De Silvestri', 'Lorenzo Insigne', 'Lorenzo JesÃºs MorÃ³n GarcÃ­a', 'Lorenzo MontipÃ²', 'Lorenzo Pellegrini', 'Lorenzo Tonelli', 'Lorenzo Venuti', 'Loris Karius', 'Loris Mouyokolo', 'Louis Jordan Beyer', 'Louis Munteanu', 'Louis Schaub', 'Lovro Majer', 'Luan Peres Petroni', 'LuÃ­s Manuel Arantes Maximiano', 'Luca Ceppitelli', 'Luca Jannis Kilian', 'Luca Lezzerini', 'Luca Netz', 'Luca Palmiero', 'Luca Pellegrini', 'Luca Philipp', 'Luca Ranieri', 'Luca Zinedine Zidane', 'Lucas Ariel BoyÃ©', 'Lucas Ariel Ocampos', 'Lucas BergstrÃ¶m', 'Lucas Bonelli', 'Lucas Da Cunha', 'Lucas Digne', 'Lucas FranÃ§ois Bernard HernÃ¡ndez Pi', 'Lucas Gourna-Douath', 'Lucas HÃ¶ler', 'Lucas Margueron', 'Lucas MartÃ­nez Quarta', 'Lucas NicolÃ¡s Alario', 'Lucas PÃ©rez MartÃ­nez', 'Lucas Perrin', 'Lucas Pezzini Leiva', 'Lucas Rodrigues Moura da Silva', 'Lucas Silva Melo', 'Lucas Simon Pierre Tousart', 'Lucas Tolentino Coelho de Lima', 'Lucas TorrÃ³ Marset', 'Lucas Torreira Di Pascua', 'Lucas VÃ¡zquez Iglesias', 'Lucien Jefferson Agoume', 'Ludovic Ajorque', 'Ludovic Blas', 'Luis Alberto Romero Alconchel', 'Luis Alberto SuÃ¡rez DÃ­az', 'Luis Alfonso Abram Ugarelli', 'Luis Alfonso Espino GarcÃ­a', 'Luis Carbonell Artajona', 'Luis Enrique Carrasco Acosta', 'Luis Ezequiel Ãvila', 'Luis Federico LÃ³pez AndÃºgar', 'Luis Fernando Muriel Fruto', 'Luis Hartwig', 'Luis Henrique Tomaz de Lima', 'Luis Javier SuÃ¡rez Charris', 'Luis JesÃºs Rioja GonzÃ¡lez', 'Luis Milla Manzanares', 'Luis Thomas Binks', 'Luiz Felipe Ramos Marchi', 'Luiz Frello Filho Jorge', 'Luka Bogdan', 'Luka JoviÄ‡', 'Luka MilivojeviÄ‡', 'Luka ModriÄ‡', 'Luka RaÄiÄ‡', 'LukÃ¡A¡ HaraslÃ­n', 'LukÃ¡A¡ HrÃ¡deckÃ½', 'Lukas KÃ¼bler', 'Lukas KlÃ¼nter', 'Lukas Manuel Klostermann', 'Lukas Nmecha', 'Lukas Rupp', 'Luke Ayling', 'Luke Bolton', 'Luke James Cundle', 'Luke Matheson', 'Luke Mbete', 'Luke Shaw', 'Luke Thomas', 'Luuk de Jong', 'Lyanco Evangelista Silveira Neves VojnoviÄ‡', 'MÃ¡rio Rui Silva Duarte', 'MÃ¡rton DÃ¡rdai', 'MÃ«rgim Vojvoda', 'MÃ©saque Geremias DjÃº', 'Mads Bech SÃ¸rensen', 'Mads Bidstrup', 'Mads Pedersen', 'Mads Roerslev Rasmussen', 'Magnus Warming', 'MahamÃ© Siby', 'Mahdi Camara', 'Mahmoud Ahmed Ibrahim Hassan', 'Mahmoud Dahoud', 'Maksim Paskotsi', 'Malachi Fagan-Walcott', 'Malang Sarr', 'Malcolm Barcola', 'Malcom Bokele', 'Malik Tillman', 'Malo Gusto', 'Mama Samba BaldÃ©', 'Mamadou Camara', 'Mamadou Coulibaly', 'Mamadou DoucourÃ©', 'Mamadou Lamine Gueye', 'Mamadou Loum NDiaye', 'Mamadou Sakho', 'Mamadou Sylla Diallo', 'Mamor Niang', 'Manolo Gabbiadini', 'Manolo Portanova', 'Manuel Agudo DurÃ¡n', 'Manuel Cabit', 'Manuel GarcÃ­a Alonso', 'Manuel Gulde', 'Manuel Javier Vallejo GalvÃ¡n', 'Manuel Lanzini', 'Manuel Lazzari', 'Manuel Locatelli', 'Manuel Morlanes AriÃ±o', 'Manuel Navarro SÃ¡nchez', 'Manuel Nazaretian', 'Manuel Obafemi Akanji', 'Manuel Peter Neuer', 'Manuel Prietl', 'Manuel Reina RodrÃ­guez', 'Manuel Riemann', 'Manuel SÃ¡nchez de la PeÃ±a', 'Manuel Trigueros MuÃ±oz', 'Marash Kumbulla', 'Marc Albrighton', 'Marc Bartra Aregall', 'Marc Cucurella Saseta', 'Marc GuÃ©hi', 'Marc Roca JunquÃ©', 'Marc-AndrÃ© ter Stegen', 'Marc-AurÃ¨le Caillard', 'Marcel Edwin Rodrigues Lavinier', 'Marcel Halstenberg', 'Marcel Sabitzer', 'Marcel Schmelzer', 'Marcelo AntÃ´nio Guedes Filho', 'Marcelo BrozoviÄ‡', 'Marcelo Josemir Saracchi Pintos', 'Marcelo Pitaluga', 'Marcelo Vieira da Silva JÃºnior', 'Marcin BuAka', 'Marco Asensio Willemsen', 'Marco Benassi', 'Marco Bizot', 'Marco Davide Faraoni', 'Marco John', 'Marco MeyerhÃ¶fer', 'Marco Modolo', 'Marco Reus', 'Marco Richter', 'Marco Silvestri', 'Marco Sportiello', 'Marco Verratti', 'Marc-Oliver Kempf', 'Marcos Alonso Mendoza', 'Marcos AndrÃ© de Sousa MendonÃ§a', 'Marcos AoÃ¡s CorrÃªa', 'Marcos Javier AcuÃ±a', 'Marcos Llorente Moreno', 'Marcos Mauro LÃ³pez GutiÃ©rrez', 'Marcus Bettinelli', 'Marcus Coco', 'Marcus Forss', 'Marcus Ingvartsen', 'Marcus Lilian Thuram-Ulien', 'Marcus Rashford', 'Mariano DÃ­az MejÃ­a', 'Marin PongraÄiÄ‡', 'Mario Gaspar PÃ©rez MartÃ­nez', 'Mario Hermoso Canseco', 'Mario HernÃ¡ndez FernÃ¡ndez', 'Mario PaA¡aliÄ‡', 'Mario RenÃ© Junior Lemina', 'Mario SuÃ¡rez Mata', 'Marius Adamonis', 'Marius Funk', 'Marius Liesegang', 'Marius Wolf', 'Mark Flekken', 'Mark Gillespie', 'Mark Helm', 'Mark Noble', 'Mark Uth', 'Marko ArnautoviÄ‡', 'Marko DmitroviÄ‡', 'Marko Pjaca', 'Marko Rog', 'Marshall Nyasha Munetsi', 'MartÃ­n Aguirregabiria Padilla', 'MartÃ­n Manuel CalderÃ³n GÃ³mez', 'MartÃ­n Merquelanz Castellanos', 'MartÃ­n Montoya Torralbo', 'MartÃ­n Pascual Castillo', 'MartÃ­n Satriano', 'MartÃ­n Zubimendi IbÃ¡Ã±ez', 'Marten Elco de Roon', 'Martin Ã˜degaard', 'Martin Braithwaite Christensen', 'Martin DÃºbravka', 'Martin ErliÄ‡', 'Martin Hinteregger', 'Martin Hongla Yma', 'Martin Kelly', 'Martin PeÄar', 'Martin Terrier', 'Martin Valjent', 'Marvelous Nakamba', 'Marvin Ayhan Obuz', 'Marvin Elimbi', 'Marvin Friedrich', 'Marvin Olawale Akinlabi Park', 'Marvin Plattenhardt', 'Marvin SchwÃ¤be', 'Marvin Stefaniak', 'Marvin Zeegelaar', 'Marwin Hitz', 'Mason Greenwood', 'Mason Holgate', 'Mason Mount', 'Massadio HaÃ¯dara', 'MatÄ›j Vydra', 'MatÃ­as Ezequiel Dituro', 'MatÃ­as Vecino Falero', 'MatÃ­as ViÃ±a', 'Mateo Klimowicz', 'Mateo KovaÄiÄ‡', 'Mateu Jaume Morey BauzÃ¡', 'Mateusz Andrzej Klich', 'MathÃ­as Olivera Miramontes', 'MathÃ­as SebastiÃ¡n SuÃ¡rez SuÃ¡rez', 'Matheus Henrique de Souza', 'Matheus Pereira da Silva', 'Matheus Santos Carneiro Da Cunha', 'Matheus Soares Thuler', 'Mathew David Ryan', 'Mathias Antonsen Normann', 'Mathias Jattah-Njie JÃ¸rgensen', 'Mathias Jensen', 'Mathias Pereira Lage', 'Mathieu Cafaro', 'Mathis Bruns', 'Mathys Saban', 'Matija NastasiÄ‡', 'Matis Carvalho', 'Mato Jajalo', 'Matondo-Merveille Papela', 'Mats Hummels', 'Matt Ritchie', 'Matt Targett', 'MattÃ©o Elias Kenzo Guendouzi OliÃ©', 'Matteo Cancellieri', 'Matteo Darmian', 'Matteo Gabbia', 'Matteo Lovato', 'Matteo Pessina', 'Matteo Politano', 'Matteo Ruggeri', 'Matthew Bondswell', 'Matthew Hoppe', 'Matthew James Doherty', 'Matthew Lowton', 'Matthew Miazga', 'Matthias Ginter', 'Matthias KÃ¶bbing', 'Matthieu Dreyer', 'Matthieu Udol', 'Matthijs de Ligt', 'Matthis Abline', 'Mattia Aramu', 'Mattia Bani', 'Mattia Caldara', 'Mattia De Sciglio', 'Mattia Destro', 'Mattia Pagliuca', 'Mattia Perin', 'Mattia Viti', 'Mattia Zaccagni', 'Mattias Olof Svanberg', 'Matty Cash', 'Matz Sels', 'Maurice Dominick ÄŒoviÄ‡', 'Maurizio Pochettino', 'Mauro Emanuel Icardi Rivero', 'Mauro Wilney Arambarri Rosa', 'Max Bennet Kruse', 'Max Christiansen', 'Max Svensson RÃ­o', 'Max Thompson', 'Maxence Caqueret', 'Maxence Lacroix', 'Maxence Rivera', 'Maxim Leitsch', 'Maxime EstÃ¨ve', 'Maxime Gonalons', 'Maxime Le Marchand', 'Maxime Lopez', 'Maximilian Arnold', 'Maximilian Bauer', 'Maximilian Eggestein', 'Maximilian Kilman', 'Maximilian MittelstÃ¤dt', 'Maximilian Philipp', 'Maximiliano GÃ³mez GonzÃ¡lez', 'Maximillian James Aarons', 'Maxwell Haygarth', 'MBala Nzola', 'MBaye Babacar Niang', 'Mehdi Bourabia', 'Mehdi Zerkane', 'Mehmet Ibrahimi', 'Mehmet Zeki Ã‡elik', 'Meiko Sponsel', 'Melayro Chakewno Jalaino Bogarde', 'Melingo Kevin Mbabu', 'Melvin Michel Maxence Bard', 'Memphis Depay', 'Merih Demiral', 'Meritan Shabani', 'Mert MÃ¼ldÃ¼r', 'Mert-Yusuf Torlak', 'Metehan GÃ¼Ã§lÃ¼', 'MichaÃ«l Bruno Dominique Cuisance', 'Michael Esser', 'Michael Gregoritsch', 'Michael Keane', 'Michael McGovern', 'Michael Olise', 'Michael Svoboda', 'Michail Antonio', 'MickaÃ«l NadÃ©', 'MickaÃ«l Ramon Malsa', 'Micky van de Ven', 'Miguel Ãngelo da Silva Rocha', 'Miguel Ãngel AlmirÃ³n Rejala', 'Miguel Ãngel Leal DÃ­az', 'Miguel Ãngel Trauco Saavedra', 'Miguel Baeza PÃ©rez', 'Miguel de la Fuente Escudero', 'Miguel GutiÃ©rrez Ortega', 'Miguel Juan Llambrich', 'Miguel LuÃ­s Pinto Veloso', 'Mihailo RistiÄ‡', 'Mijat GaÄ‡inoviÄ‡', 'Mika SchrÃ¶ers', 'Mike Maignan', 'Mikel Balenziaga Oruesagasti', 'Mikel Merino ZazÃ³n', 'Mikel Oyarzabal Ugarte', 'Mikel Vesga Arruti', 'Mikkel Krogh Damsgaard', 'Milan Akriniar', 'Milan ÄuriÄ‡', 'Milan Badelj', 'MiloA¡ PantoviÄ‡', 'Milot Rashica', 'Milutin OsmajiÄ‡', 'Mitchel Bakker', 'Mitchell Dijks', 'Mitchell van Bergen', 'MoÃ¯se Dion Sahi', 'Mohamed Amine Elyounoussi', 'Mohamed Amine Ihattaren', 'Mohamed Lamine Bayo', 'Mohamed Naser Elsayed Elneny', 'Mohamed SaÃ¯d Benrahma', 'Mohamed Salah Ghaly', 'Mohamed Salim Fares', 'Mohamed Salisu Abdul Karim', 'Mohamed Simakan', 'Mohamed-Ali Cho', 'Mohammed Sangare', 'MoisÃ©s GÃ³mez Bordonado', 'Moise Bioty Kean', 'Molla WaguÃ©', 'Moreto Moro CassamÃ¡', 'Morgan Boyes', 'Morgan Sanson', 'Morgan Schneiderlin', 'Moriba Kourouma Kourouma', 'Moritz Jenz', 'Morten Thorsby', 'Moses Daddy-Ajala Simon', 'Mouctar Diakhaby', 'Moussa DembÃ©lÃ©', 'Moussa Diaby', 'Moussa Djenepo', 'Moussa Doumbia', 'Moussa NiakhatÃ©', 'Moussa Sissoko', 'Moussa WaguÃ©', 'Moustapha Mbow', 'Munas Dabbur', 'Munir El Haddadi Mohamed', 'Musa Barrow', 'Myles Peart-Harris', 'Myron Boadu', 'Myziane Maolida', 'NÃ©lson Cabral Semedo', 'NÃ©stor Alejandro AraÃºjo Razo', 'NaÃ«l Jaby', 'Nabil Fekir', 'Nabili Zoubdi Touaizi', 'Naby KeÃ¯ta', 'Nadiem Amiri', 'Nadir Zortea', 'Nahitan Michel NÃ¡ndez Acosta', 'Nahuel Molina Lucero', 'Nahuel Noll', 'Nampalys Mendy', 'Nanitamo Jonathan IkonÃ©', 'Naouirou Ahamada', 'Nassim Chadli', 'Nathan AkÃ©', 'Nathan Bitumazala', 'Nathan De Medina', 'Nathan Ferguson', 'Nathan Michael Collins', 'Nathan Redmond', 'Nathan Tella', 'NathanaÃ«l Mbuku', 'Nathaniel Edwin Clyne', 'Nathaniel Phillips', 'Nayef Aguerd', 'NDri Philippe Koffi', 'Neal Maupay', 'Neco Williams', 'Nedim Bajrami', 'Nemanja Gudelj', 'Nemanja MaksimoviÄ‡', 'Nemanja MatiÄ‡', 'Nemanja Radoja', 'Neyder Yessy Lozano RenterÃ­a', 'Neymar da Silva Santos JÃºnior', 'NGolo KantÃ©', 'NGuessan Rominigue KouamÃ©', 'Nicholas Gioacchini', 'Nicholas Williams Arthuer', 'Nick Pope', 'Nick Viergever', 'Nico Elvedi', 'Nico Schlotterbeck', 'Nico Schulz', 'Nicola Domenico Sansone', 'Nicola Murru', 'Nicola Ravaglia', 'Nicola Zalewski', 'NicolÃ¡s GonzÃ¡lez Iglesias', 'NicolÃ¡s IvÃ¡n GonzÃ¡lez', 'NicolÃ¡s MartÃ­n DomÃ­nguez', 'NicolÃ¡s Melamed Ribaudo', 'NicolÃ¡s Serrano Galdeano', 'NicolÃ² Barella', 'NicolÃ² Casale', 'NicolÃ² Fagioli', 'NicolÃ² Rovella', 'NicolÃ² Zaniolo', 'Nicolas De PrÃ©ville', 'Nicolas HÃ¶fler', 'Nicolas Louis Marcel Cozza', 'Nicolas PÃ©pÃ©', 'Nicolas Pallois', 'Nicolas Penneteau', 'Nicolas Thibault Haas', 'Niki Emil Antonio MÃ¤enpÃ¤Ã¤', 'Nikita Iosifov', 'Niklas Bernd Dorsch', 'Niklas Hauptmann', 'Niklas Klinger', 'Niklas Lomb', 'Niklas SÃ¼le', 'Niklas Stark', 'Niklas Tauer', 'Niko GieÃŸelmann', 'Nikola KaliniÄ‡', 'Nikola MaksimoviÄ‡', 'Nikola MaraA¡', 'Nikola MilenkoviÄ‡', 'Nikola VlaA¡iÄ‡', 'Nikola VukÄeviÄ‡', 'Nikolas Terkelsen Nartey', 'Nile Omari Mckenzi John', 'Nils Petersen', 'Nils Seufert', 'Nils-Jonathan KÃ¶rber', 'Nishan Connell Burkart', 'Nnamdi Collins', 'NoÃ© Sow', 'Noah Atubolu', 'Noah Fatar', 'Noah Joel Sarenren Bazee', 'Noah KÃ¶nig', 'Noah Katterbach', 'Noah Nadje', 'Noah WeiÃŸhaupt', 'Norbert GyÃ¶mbÃ©r', 'Norberto Bercique Gomes Betuncal', 'Norberto Murara Neto', 'Nordi Mukiele Mulere', 'Nuno Albertino Varela Tavares', 'Nuno Alexandre Tavares Mendes', 'Nya Jerome Kirby', 'Obite Evan NDicka', 'Odel Offiah', 'Odilon Kossounou', 'Odsonne Ã‰douard', 'Oghenekaro Peter Etebo', 'Ohis Felix Uduokhai', 'Oier OlazÃ¡bal Paredes', 'Oier Sanjurjo MatÃ©', 'Oier Zarraga EgaÃ±a', 'Oihan Sancet Tirapu', 'Okay YokuAŸlu', 'Oleksandr Zinchenko', 'Oliver Batista Meier', 'Oliver Baumann', 'Oliver Bosworth', 'Oliver Christensen', 'Oliver Skipp', 'Oliver Webber', 'Olivier Giroud', 'Ollie Watkins', 'Oludare Olufunwa', 'Omar Colley', 'Omar El Hilali', 'Omar Federico Alderete FernÃ¡ndez', 'Omar Khaled Mohamed Marmoush', 'Omar Mascarell GonzÃ¡lez', 'Omar Tyrell Crawford Richards', 'Omer Hanin', 'Ondrej Duda', 'Onyinye Wilfred Ndidi', 'Opa Nguette', 'Orel Mangala', 'Orestis Spyridon Karnezis', 'Oriol Busquets Mas', 'Oriol Romeu Vidal', 'Orlando RubÃ©n YÃ¡Ã±ez Alabart', 'Osman Bukari', 'Ossama Ashley', 'Osvaldo Pedro Capemba', 'OtÃ¡vio Henrique Passos Santos', 'Oualid El Hajjam', 'Ouparine Djoco', 'Ousmane Ba', 'Ousmane DembÃ©lÃ©', 'Oussama Idrissi', 'Oussama Targhalline', 'Owen Dodgson', 'Ozan Muhammed Kabak', 'Ozan Tufan', 'PÃ©pÃ© Bonet Kapambu', 'PÃ©ter GulÃ¡csi', 'Pablo ÃÃ±iguez de Heredia Larraz', 'Pablo Carmine Maffeo Becerra', 'Pablo Daniel Piatti', 'Pablo Fornals Malla', 'Pablo GÃ¡lvez Miranda', 'Pablo GozÃ¡lbez Gilabert', 'Pablo IbÃ¡Ã±ez Lumbreras', 'Pablo MarÃ­ Villar', 'Pablo MartÃ­n PÃ¡ez Gavira', 'Pablo MartÃ­n PicÃ³n Ãlvaro', 'Pablo MartÃ­nez AndrÃ©s', 'Pablo PÃ©rez Rico', 'Pablo Paulino Rosario', 'Pablo Valencia GarcÃ­a', 'Panagiotis Retsos', 'Paolo Ghiglione', 'Paolo Pancrazio FaragÃ²', 'Paolo Sciortino', 'Pape Alassane Gueye', 'Pape Cheikh Diop Gueye', 'Pape Matar Sarr', 'Pape Ndiaga Yade', 'Pascal GroÃŸ', 'Pascal Juan Estrada', 'Pascal Stenzel', 'Pascal Struijk', 'Pasquale Mazzocchi', 'Patricio GabarrÃ³n Gil', 'Patricio Nehuen PÃ©rez', 'Patrick Bamford', 'Patrick Cutrone', 'Patrick Herrmann', 'Patrick Osterhage', 'Patrick Roberts', 'Patrick Wimmer', 'Patrik Schick', 'Patryk Dziczek', 'Patson Daka', 'Pau Francisco Torres', 'Pau LÃ³pez Sabata', 'Paul Baysse', 'Paul Dummett', 'Paul Grave', 'Paul Jaeckel', 'Paul Jean FranÃ§ois Bernardoni', 'Paul Nardi', 'Paul Nebel', 'Paul Pogba', 'Paul Seguin', 'Paulo Bruno Exequiel Dybala', 'Paulo Henrique Sampaio Filho', 'Paulo OtÃ¡vio Rosa da Silva', 'Pavao Pervan', 'Pavel KadeA™Ã¡bek', 'PaweA Kamil JaroszyAski', 'PaweA Marek Dawidowicz', 'PaweA Marek WszoAek', 'Pedro Bigas Rigo', 'Pedro Chirivella Burgos', 'Pedro Eliezer RodrÃ­guez Ledesma', 'Pedro Filipe TeodÃ³sio Mendes', 'Pedro GonzÃ¡lez LÃ³pez', 'Pedro Lomba Neto', 'Pedro Mba Obiang Avomo', 'Pedro Ortiz Bernat', 'Pelenda Joshua Tunga Dasilva', 'Pere Joan GarcÃ­a BauzÃ', 'Pere Milla PeÃ±a', 'Pere Pons Riera', 'Peru Nolaskoain Esnal', 'Pervis JosuÃ© EstupiÃ±Ã¡n Tenorio', 'Petar MiÄ‡in', 'Petar StojanoviÄ‡', 'Petar Zovko', 'Peter PekarÃ­k', 'Petko Hristov', 'Phil Bardsley', 'Phil Jones', 'Philana Tinotenda Kadewere', 'Philip Ankhrah', 'Philip Foden', 'Philipp FÃ¶rster', 'Philipp Lienhart', 'Philipp Pentke', 'Philipp Schulze', 'Philipp Tschauner', 'Philippe Coutinho Correia', 'Philippe Sandler', 'Phillipp Klement', 'Pierluigi Gollini', 'Piero MartÃ­n HincapiÃ© Reyna', 'Pierre Kazeye Rommel Kalulu Kyatengwa', 'Pierre Lees-Melou', 'Pierre-Emerick Emiliano FranÃ§ois Aubameyang', 'Pierre-Emile Kordt HÃ¸jbjerg', 'Pierre-Emmanuel Ekwah Elimby', 'Pierre-Yves Hamel', 'Pierrick Capelle', 'Pietro Boer', 'Pietro Ceccaroni', 'Pietro Pellegri', 'Pietro Terracciano', 'Piotr Sebastian ZieliAski', 'Pol Mikel Lirola Kosok', 'Pontus Jansson', 'Predrag RajkoviÄ‡', 'Presnel Kimpembe', 'PrzemysAaw Frankowski', 'PrzemysAaw PAacheta', 'Quentin Boisgard', 'Quentin Merlin', 'RÃ©mi Oudin', 'RÃ©my Descamps', 'RaÃºl Albiol Tortajada', 'RaÃºl Alonso JimÃ©nez RodrÃ­guez', 'RaÃºl De TomÃ¡s GÃ³mez', 'RaÃºl GarcÃ­a de Haro', 'RaÃºl GarcÃ­a Escudero', 'RaÃºl Moro Prescoli', 'RaÃºl Torrente Navarro', 'Radamel Falcao GarcÃ­a ZÃ¡rate', 'Rade KruniÄ‡', 'RadosAaw Majecki', 'Radu Matei DrÄƒguÈ™in', 'RafaA Gikiewicz', 'Rafael AlcÃ¢ntara do Nascimento', 'Rafael Alexandre ConceiÃ§Ã£o LeÃ£o', 'Rafael Czichos', 'Rafael JimÃ©nez Jarque', 'Rafael Mir Vicente', 'Rafael Santos BorrÃ© Maury', 'Rafael TolÃ³i', 'Rafiki Said Ahamada', 'RÄƒzvan Gabriel Marin', 'Ragnar Ache', 'Raheem Sterling', 'RamÃ³n RodrÃ­guez JimÃ©nez', 'Ramiro Guerra Pereyra', 'Ramon Olamilekan Azeez', 'Ramy Bensebaini', 'Ramzi Aya', 'Randal Kolo Muani', 'Randy Nteka', 'Rani Khedira', 'RÃºben Diogo da Silva Neves', 'RÃºben dos Santos Gato Alves Dias', 'RÃºben Miguel Nunes Vezo', 'Raoul Bellanova', 'RaphaÃ«l Adelino JosÃ© Guerreiro', 'RaphaÃ«l Varane', 'Raphael Dias Belloli', 'Raphael Framberger', 'Rayan AÃ¯t Nouri', 'Rayan Mathis Cherki', 'Redwan BourlÃ¨s', 'Reece Hannam', 'Reece James', 'Reece Oxford', 'Reinier Jesus Carvalho', 'Reinildo Isnard Mandava', 'Remi Matthews', 'Remi Savage', 'Remo Marco Freuler', 'Renan Augusto Lodi dos Santos', 'Renato Fabrizio Tapia Cortijo', 'Renato JÃºnior Luz Sanches', 'Renato Steffen', 'Renaud Emond', 'Renaud Ripart', 'Rey Manaj', 'Ricard Puig MartÃ­', 'Ricardo Domingos Barbosa Pereira', 'Ricardo IvÃ¡n RodrÃ­guez Araya', 'Ricardo LuÃ­s Chaby Mangas', 'Riccardo Bocalon', 'Riccardo Calafiori', 'Riccardo Fiamozzi', 'Riccardo Gagliolo', 'Riccardo Ladinetti', 'Riccardo Marchizza', 'Riccardo Orsolini', 'Riccardo Saponara', 'Riccardo Sottil', 'Richarlison de Andrade', 'Rick Karsdorp', 'Rick van Drongelen', 'Rico Henry', 'Ridgeciano Delano Haps', 'Ridle Bote Baku', 'Riyad Mahrez', 'Riza Durmisi', 'Rob Elliot', 'Rob Holding', 'Robert Andrich', 'Robert Gumny', 'Robert Harker', 'Robert Lewandowski', 'Robert Lynch SanchÃ©z', 'Robert Navarro MuÃ±oz', 'Robert Skov', 'Robert Street', 'Robert Tesche', 'Roberto Firmino Barbosa de Oliveira', 'Roberto Gagliardini', 'Roberto GonzÃ¡lez BayÃ³n', 'Roberto IbÃ¡Ã±ez Castro', 'Roberto Massimo', 'Roberto Maximiliano Pereyra', 'Roberto Piccoli', 'Roberto Pirrello', 'Roberto Soldado Rillo', 'Roberto Soriano', 'Roberto SuÃ¡rez Pier', 'Roberto Torres Morales', 'Robin Everardus Gosens', 'Robin Friedrich', 'Robin Hack', 'Robin Knoche', 'Robin Koch', 'Robin Le Normand', 'Robin Luca Kehr', 'Robin Zentner', 'Robson Alves de Barros', 'Rocco Ascone', 'Rocky Bushiri Kisonga', 'Rodrigo AndrÃ©s Battaglia', 'Rodrigo Bentancur ColmÃ¡n', 'Rodrigo HernÃ¡ndez Cascante', 'Rodrigo Javier De Paul', 'Rodrigo Moreno Machado', 'Rodrigo Nascimento FranÃ§a', 'Rodrigo SÃ¡nchez RodrÃ­guez', 'Rodrigue Casimir Ninga', 'Rodrygo Silva de Goes', 'RogÃ©rio Oliveira da Silva', 'Roger IbaÃ±ez Da Silva', 'Roger MartÃ­ Salvador', 'Rok VodiA¡ek', 'Roland Sallai', 'Rolando Mandragora', 'Roli Pereira de Sa', 'Romain Del Castillo', 'Romain Faivre', 'Romain Hamouma', 'Romain Jules Salin', 'Romain Perraud', 'Romain SaÃ¯ss', 'Romain Thomas', 'Roman BÃ¼rki', 'Romario RÃ¶sch', 'Romelu Lukaku Menama', 'Romeo Lavia', 'RonaÃ«l Julien Pierre-Gabriel', 'Ronald Federico AraÃºjo da Silva', 'Ronaldo Augusto Vieira Nan', 'Ross Barkley', 'RubÃ©n Blanco Veiga', 'RubÃ©n de TomÃ¡s GÃ³mez', 'RubÃ©n Duarte SÃ¡nchez', 'RubÃ©n GarcÃ­a Santos', 'RubÃ©n PeÃ±a JimÃ©nez', 'RubÃ©n Rochina Naixes', 'RubÃ©n Sobrino Pozuelo', 'Ruben Aguilar', 'Ruben Estephan Vargas MartÃ­nez', 'Ruben Loftus-Cheek', 'Rui Pedro dos Santos PatrÃ­cio', 'Rui Tiago Dantas da Silva', 'Rune Almenning Jarstein', 'Ruslan Malinovskyi', 'Ruwen WerthmÃ¼ller', 'Ryad Boudebouz', 'Ryan Astley', 'Ryan Bertrand', 'Ryan Bouallak', 'Ryan Cassidy', 'Ryan Finnigan', 'Ryan Fraser', 'Ryan Fredericks', 'SÃ©amus Coleman', 'SÃ©bastien Cibois', 'SÃ©bastien Corchia', 'SÃ©bastien RÃ©not', 'SÃ©kou Mara', 'SaA¡a KalajdA¾iÄ‡', 'SaA¡a LukiÄ‡', 'SaÃ¯dou Sow', 'SaÃ®f-Eddine Khaoui', 'Saad Agouzoul', 'SaÃºl ÃÃ­guez EsclÃ¡pez', 'SaÃºl GarcÃ­a Cabrero', 'Sacha Delaye', 'Sada Thioub', 'Sadik Fofana', 'Sadio ManÃ©', 'Salih Ã–zcan', 'Salim Ben Seghir', 'Salis Abdul Samed', 'Salomon Junior Sambia', 'Salvador Ferrer Canals', 'Salvador SÃ¡nchez Ponce', 'Salvador Sevilla LÃ³pez', 'Salvatore Sirigu', 'Sam Byram', 'Sam Greenwood', 'Sam Lammers', 'Sam McClelland', 'Sam McQueen', 'Saman Ghoddos', 'Sambou Sissoko', 'Samir Caetano de Souza Santos', 'Samir HandanoviÄ', 'Samuel Castillejo Azuaga', 'Samuel Chimerenka Chukwueze', 'Samuel Edozie', 'Samuel Kalu Ojim', 'Samuel Loric', 'Samuel Moutoussamy', 'Samuel Yves Umtiti', 'Samuele Damiani', 'Samuele Ricci', 'Sander Johan Christiansen', 'Sandro RamÃ­rez Castillo', 'Sandro Tonali', 'Sanjin PrciÄ‡', 'Santiago Arias Naranjo', 'Santiago Arzamendia Duarte', 'Santiago ComesaÃ±a Veiga', 'Santiago Eneme Bocari', 'Santiago Lionel AscacÃ­bar', 'Santiago Mina Lorenzo', 'Santiago RenÃ© MuÃ±Ã³z Robles', 'Sargis Adamyan', 'Sascha Burchert', 'Saulo Igor Decarli', 'Sava-Arangel ÄŒestiÄ‡', 'Scott Brian Banks', 'Scott Carson', 'Scott McTominay', 'Sead KolaA¡inac', 'Sean Longstaff', 'Sean McGurk', 'Sebastiaan Bornauw', 'Sebastian Andersson', 'Sebastian De Maio', 'Sebastian Griesbeck', 'Sebastian Polter', 'Sebastian Rode', 'Sebastian Rudy', 'Sebastian Vasiliadis', 'Sebastian Wiktor Walukiewicz', 'Sebastiano Luperto', 'Seko Fofana', 'Sepe Elye Wahi', 'Serge David Gnabry', 'Sergej MilinkoviÄ‡-SaviÄ‡', 'Sergi CanÃ³s TenÃ©s', 'Sergi Darder Moll', 'Sergi GÃ³mez SolÃ', 'SergiÃ±o Gianni Dest', 'Sergio Arratia Lechosa', 'Sergio Arribas Calvo', 'Sergio Asenjo AndrÃ©s', 'Sergio Barcia Larenxeira', 'Sergio Busquets i Burgos', 'Sergio Camus Perojo', 'Sergio Canales Madrazo', 'Sergio Duvan CÃ³rdova Lezama', 'Sergio Escudero Palomo', 'Sergio Guardiola Navarro', 'Sergio Guerrero Romero', 'Sergio Herrera PirÃ³n', 'Sergio Leonel AgÃ¼ero del Castillo', 'Sergio Lozano Lluch', 'Sergio Moreno MartÃ­nez', 'Sergio Postigo Redondo', 'Sergio Ramos GarcÃ­a', 'Sergio ReguilÃ³n RodrÃ­guez', 'Sergio Rico GonzÃ¡lez', 'Sergio Roberto Carnicer', 'Serhou Yadaly Guirassy', 'Shandon Baptiste', 'Shane Patrick Long', 'Shane Patrick Michael Duffy', 'Sheraldo Becker', 'Shkodran Mustafi', 'Shola Maxwell Shoretire', 'Sikou NiakatÃ©', 'Sil Swinkels', 'Silas Katompa Mvumpa', 'SilvÃ¨re Ganvoula Mboussy', 'Silvan Dominic Widmer', 'Simeon Tochukwu Nwankwo', 'Simon Asta', 'Simon Brady Ngapandouetnbu', 'Simon Thorup KjÃ¦r', 'Simon Zoller', 'Simone Aresti', 'Simone Bastoni', 'Simone Edera', 'Simone Romagnoli', 'Simone Verdi', 'Simone Zaza', 'Sinaly DiomandÃ©', 'Sofian Kiyine', 'Sofiane Alakouch', 'Sofiane Boufal', 'Sofiane Diop', 'Sofyan Amrabat', 'Solomon March', 'Soma Zsombor Novothny', 'Souleyman Doumbia', 'Soumaila Coulibaly', 'StÃ©phane Bahoken', 'StÃ©phane Imad Diarra', 'Stanislav Lobotka', 'Stefan Bajic', 'Stefan Bell', 'Stefan de Vrij', 'Stefan Ilsanker', 'Stefan Lainer', 'Stefan MitroviÄ‡', 'Stefan Ortega Moreno', 'Stefan Posch', 'Stefan SaviÄ‡', 'Stefano Sabelli', 'Stefano Sensi', 'Stefano Sturaro', 'Stefanos Kapino', 'Steffen Tigges', 'Stephan El Shaarawy', 'Stephan FÃ¼rstner', 'Stephy Alvaro Mavididi', 'Stevan JovetiÄ‡', 'Steve Mandanda', 'Steve Michel MouniÃ©', 'Steven Alzate', 'Steven Charles Bergwijn', 'Steven NKemboanza Mike Christopher Nzonzi', 'Stian Rode Gregersen', 'Stole Dimitrievski', 'Strahinja PavloviÄ‡', 'Stuart Armstrong', 'Stuart Dallas', 'Suat Serdar', 'Suleiman Abdullahi', 'Sven Botman', 'Sven Ulreich', 'Sydney van Hooijdonk', 'Szymon Piotr A»urkowski', 'TÃ©ji Tedy Savanier', 'Taiwo Michael Awoniyi', 'Tammy Bakumo-Abraham', 'Tanguy Coulibaly', 'Tanguy NdombÃ¨lÃ© Alvaro', 'Tanguy-Austin Nianzou Kouassi', 'Tanner Tessmann', 'Tariq Lamptey', 'Tariq Uwakwe', 'Tarique Fosu', 'Tarsis Bonga', 'Taylor Anthony Booth', 'Taylor Richards', 'Teddy Bartouche-Selbonne', 'Teddy Boulhendi', 'Teden Mengi', 'Teemu Pukki', 'Temitayo Olufisayo Olaoluwa Aina', 'Terem Igobor Moffi', 'Teun Koopmeiners', 'Thanawat Suengchitthawon', 'Theo Bernard FranÃ§ois HernÃ¡ndez', 'Theo Walcott', 'Thiago AlcÃ¢ntara do Nascimento', 'Thiago Emiliano da Silva', 'Thiago Galhardo do Nascimento Rocha', 'Thiago Henrique Mendes Ribeiro', 'Thibault Tamas', 'Thibaut Courtois', 'Thibo Baeten', 'Thierry Rendall Correia', 'Thierry Small', 'Thomas Callens', 'Thomas Clayton', 'Thomas Delaine', 'Thomas Dickson-Peters', 'Thomas Foket', 'Thomas Fontaine', 'Thomas Henry', 'Thomas Joseph Delaney', 'Thomas Lemar', 'Thomas MÃ¼ller', 'Thomas Mangani', 'Thomas Meunier', 'Thomas Monconduit', 'Thomas Partey', 'Thomas Strakosha', 'Thorgan Hazard', 'Tiago Emanuel EmbalÃ³ DjalÃ³', 'Tiago Manuel Dias Correia', 'Tidiane Malbec', 'TiemouÃ© Bakayoko', 'Tim Akinola', 'Tim Civeja', 'Tim Krul', 'Tim Lemperle', 'Timo Baumgartl', 'Timo Bernd HÃ¼bers', 'Timo Horn', 'Timo Werner', 'TimothÃ© Rupil', 'TimothÃ©e Joseph PembÃ©lÃ©', 'TimothÃ©e Kolodziejczak', 'Timothy Castagne', 'Timothy Chandler', 'Timothy Evans Fosu-Mensah', 'Timothy Tarpeh Weah', 'Timothy Tillman', 'Titouan Thomas', 'Tjark Ernst', 'Tobias Raschl', 'Tobias Sippel', 'Tobias Strobl', 'Todd Cantwell', 'Tolgay Ali Arslan', 'Tom Cleverley', 'Tom Davies', 'Tom Heaton', 'Tom Lacoux', 'Tom Weilandt', 'Toma BaA¡iÄ‡', 'TomÃ¡A¡ Koubek', 'TomÃ¡A¡ OstrÃ¡k', 'TomÃ¡A¡ SouÄek', 'TomÃ¡s Eduardo RincÃ³n HernÃ¡ndez', 'TomÃ¡s JesÃºs AlarcÃ³n Vergara', 'TomÃ¡s Pina Isla', 'Tommaso Augello', 'Tommaso Pobega', 'Toni Herrero Oliva', 'Toni Kroos', 'Tony Jantschke', 'Torben MÃ¼sel', 'Trent Alexander-Arnold', 'Trevoh Chalobah', 'Tristan DingomÃ©', 'Tudor Cristian BÄƒluA£Äƒ', 'Tyler Onyango', 'Tyler Roberts', 'Tyler Shaan Adams', 'Tymoteusz Puchacz', 'Tyrick Mitchell', 'Tyrone Mings', 'Tyronne Ebuehi', 'Ugo Bertelli', 'Ulrick Brad Eneme Ella', 'Unai GarcÃ­a Lugea', 'Unai LÃ³pez Cabrera', 'Unai NÃºÃ±ez Gestoso', 'Unai SimÃ³n Mendibil', 'Unai Vencedor Paris', 'UroA¡ RaÄiÄ‡', 'VÃ­ctor Camarasa Ferrando', 'VÃ­ctor Christopher De Baunbaug', 'VÃ­ctor Chust GarcÃ­a', 'VÃ­ctor David DÃ­az Miguel', 'VÃ­ctor Laguardia Cisneros', 'VÃ­ctor MachÃ­n PÃ©rez', 'VÃ­ctor RuÃ­z Torre', 'ValÃ¨re Germain', 'Valentin Rongier', 'Valentino Lesieur', 'Valentino Livramento', 'Valerio Verre', 'Valon Behrami', 'Valon Berisha', 'Vanja MilinkoviÄ‡-SaviÄ‡', 'Varazdat Haroyan', 'Vasilios Konstantinos Lampropoulos', 'Vedat Muriqi', 'Vicente Guaita Panadero', 'Vicente Iborra de la Fuente', 'Victor JÃ¶rgen Nilsson LindelÃ¶f', 'Victor James Osimhen', 'Vid Belec', 'Viktor Kovalenko', 'Viljami Sinisalo', 'Vilmos TamÃ¡s Orban', 'VinÃ­cius JosÃ© PaixÃ£o de Oliveira JÃºnior', 'Vincent Le Goff', 'Vincent Manceau', 'Vincent Pajot', 'Vincenzo Fiorillo', 'Vincenzo Grifo', 'Virgil van Dijk', 'Vital Manuel NSimba', 'Vitaly Janelt', 'Vito Mannone', 'Vlad Iulian ChiricheÈ™', 'VladimÃ­r Coufal', 'VladimÃ­r Darida', 'Vladislav Cherny', 'Vontae Daley-Campbell', 'Wadi Ibrahim Suzuki', 'Wahbi Khazri', 'Wahidullah Faghir', 'Wajdi Kechrida', 'Walace Souza Silva', 'Waldemar Anton', 'Walim Lgharbi', 'Walter Daniel BenÃ­tez', 'Waniss TaÃ¯bi', 'Warmed Omari', 'Warren TchimbembÃ©', 'Wayne Robert Hennessey', 'Wesley Fofana', 'Wesley SaÃ¯d', 'Weston James Earl McKennie', 'Wilfried Stephane Singo', 'Wilfried Zaha', 'Will Hughes', 'Will Norris', 'Willem Geubbels', 'William Alain AndrÃ© Gabriel Saliba', 'William Anthony Patrick Smallbone', 'William de Asevedo Furtado', 'William Mikelbrencis', 'William Silva de Carvalho', 'William Troost-Ekong', 'Willian JosÃ© da Silva', 'Willy-Arnaud Zobo Boly', 'Wilson Isidor', 'Winston Wiremu Reid', 'Wissam Ben Yedder', 'Wladimiro Falcone', 'Wojciech Tomasz SzczÄ™sny', 'Wout Faes', 'Wout Weghorst', 'Wuilker FariÃ±ez Aray', 'Wylan Cyprien', 'Xaver Schlager', 'Xavi Simons', 'Xavier Chavalerin', 'Xherdan Shaqiri', 'YÄ±ldÄ±rÄ±m Mert Ã‡etin', 'Yacine Adli', 'Yacine Qasmi', 'Yan Brice Eteki', 'Yan Valery', 'Yangel Clemente Herrera Ravelo', 'Yanis Guermouche', 'Yann Sommer', 'Yannick Cahuzac', 'Yannick Ferreira Carrasco', 'Yannick Gerhardt', 'Yannick Pandor', 'Yannik Keitel', 'Yannis MBemba', 'Yasser Larouci', 'Yassin FÃ©kir', 'Yassine Bounou', 'Yayah Kallon', 'Yehvann Diouf', 'Yeray Ãlvarez LÃ³pez', 'Yeremi JesÃºs Santos Pino', 'Yerry Fernando Mina GonzÃ¡lez', 'Yerson Mosquera Valdelamar', 'Yoane Wissa', 'Yoann Salmier', 'Yoann Touzghar', 'Yohann Magnin', 'Youcef Atal', 'Youri Tielemans', 'Youssef En-Nesyri', 'Youssef Maleh', 'Youssouf Fofana', 'Youssouf KonÃ©', 'Youssouf Sabaly', 'Youssouph Mamadou Badji', 'Yunis Abdelhamid', 'Yunus Dimoara Musah', 'Yuri Berchiche Izeta', 'Yussif Raman Chibsah', 'Yussuf Yurary Poulsen', 'Yusuf Demir', 'Yusuf YazÄ±cÄ±', 'Yvan Neyou Noupa', 'Yvann MaÃ§on', 'Yves Bissouma', 'Zack Thomas Steffen', 'Zak Emmerson', 'Zane Monlouis', 'Zaydou Youssouf', 'ZinÃ©dine Machach', 'ZinÃ©dine Ould Khaled', 'Zinho Vanheusden', 'Zlatan IbrahimoviÄ‡'))
        
        # df
        df = pd.read_excel('4_Football_Player_FIFA 2022.xlsx', sheet_name= 'PBC players_22')
        df = df[df['player_positions'] != 'GK']
        selected_leagues = ['English Premier League', 'Spain Primera Division']
        df = df[df['league_name'].isin(selected_leagues)]
        df.columns = df.columns.str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df = df.drop(columns=["player_url", # Not informative.
                            "short_name", # Already existed information.
                            "player_positions", # Not informative.
                            "potential", # Not informative.
                            "value_eur", # Contratual information discarted.
                            "wage_eur", # Contractual information discarted.
                            "dob", # Already existed information.
                            "club_team_id", # Not informative.
                            "club_name", # Not informative.
                            "league_name", # Not informative.
                            "league_level", # Not informative.
                            "club_position", # Not informative.
                            "club_jersey_number", # Not informative.
                            "club_loaned_from", # Not informative.
                            "club_joined", # Not informative.
                            "club_contract_valid_until", # Not informative.
                            "nationality_id", # Not informative.
                            "nationality_name", # Not informative.
                            "nation_team_id", # Not informative.
                            "nation_position", # Not informative.
                            "nation_jersey_number", # Not informative.
                            "preferred_foot", # Not informative.
                            "body_type", # Not informative.
                            "real_face", # Not informative.
                            "release_clause_eur", # Contractual information discarted.
                            "release_clause_eur", # Contractual information discarted.
                            "player_tags", # Not informative.
                            "player_traits", # Not informative.
                            "goalkeeping_diving", # GOALKEEPING POSITION
                            "goalkeeping_handling", # GOALKEEPING POSITION
                            "goalkeeping_kicking", # GOALKEEPING POSITION
                            "goalkeeping_positioning", # GOALKEEPING POSITION
                            "goalkeeping_reflexes", # GOALKEEPING POSITION
                            "goalkeeping_speed", # GOALKEEPING POSITION
                            "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm", # Not informative.
                            "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk", # Not informative.
                            "player_face_url", # Not informative.
                            "club_logo_url", # Not informative.
                            "club_flag_url", # Not informative.
                            "nation_logo_url", # Not informative.
                            "nation_flag_url"]).set_index("sofifa_id")
        df[['work_rate_attacking', 'work_rate_defensive']] = df['work_rate'].str.split('/', expand=True).replace({'Low': 1, 'Medium': 2, 'High': 3})
        df = df.drop(columns=["work_rate"]) # Not informative.
        df.isnull().sum()[df.isnull().sum() > 0]
        df.fillna(0, inplace=True)
        X = df.drop(columns=["overall"]).set_index("long_name")
        y = df.overall / df.overall.max()
        
        # Define the dictionary mapping short names to full names
        variable_names = {
            "age": "Age",
            "height_cm": "Height (in centimeters)",
            "weight_kg": "Weight (in kilograms)",
            "weak_foot": "Weak Foot Rating",
            "skill_moves": "Skill Moves Rating",
            "international_reputation": "International Reputation",
            "pace": "Pace",
            "shooting": "Shooting",
            "passing": "Passing",
            "dribbling": "Dribbling",
            "defending": "Defending",
            "physic": "Physic",
            "attacking_crossing": "Attacking Crossing",
            "attacking_finishing": "Attacking Finishing",
            "attacking_heading_accuracy": "Attacking Heading Accuracy",
            "attacking_short_passing": "Attacking Short Passing",
            "attacking_volleys": "Attacking Volleys",
            "skill_dribbling": "Skill Dribbling",
            "skill_curve": "Skill Curve",
            "skill_fk_accuracy": "Skill Free Kick Accuracy",
            "skill_long_passing": "Skill Long Passing",
            "skill_ball_control": "Skill Ball Control",
            "movement_acceleration": "Movement Acceleration",
            "movement_sprint_speed": "Movement Sprint Speed",
            "movement_agility": "Movement Agility",
            "movement_reactions": "Movement Reactions",
            "movement_balance": "Movement Balance",
            "power_shot_power": "Power Shot Power",
            "power_jumping": "Power Jumping",
            "power_stamina": "Power Stamina",
            "power_strength": "Power Strength",
            "power_long_shots": "Power Long Shots",
            "mentality_aggression": "Mentality Aggression",
            "mentality_interceptions": "Mentality Interceptions",
            "mentality_positioning": "Mentality Positioning",
            "mentality_vision": "Mentality Vision",
            "mentality_penalties": "Mentality Penalties",
            "mentality_composure": "Mentality Composure",
            "defending_marking_awareness": "Defending Marking Awareness",
            "defending_standing_tackle": "Defending Standing Tackle",
            "defending_sliding_tackle": "Defending Sliding Tackle",
            "work_rate_attacking": "Work Rate Attacking",
            "work_rate_defensive": "Work Rate Defensive"}

        # Open a sidebar for a different feature option
        Football_player_list = list(variable_names.keys()) # Football_player_list = X.columns.tolist()
        Football_player_list_full = list(variable_names.values())
        Football_player_feature_full_name = st.sidebar.selectbox('Feature in focus:', Football_player_list_full)
        Football_player_feature = [key for key, value in variable_names.items() if value == Football_player_feature_full_name][0] # Get the corresponding short name from the dictionary
        
        # Open a sidebar for a different feature option
        Decil = st.sidebar.selectbox('Top Ranking (%) you desire to achieve (where 0,05 means top 5%):', ('0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                                                                                                          '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9'))

        #Decil_final = 1 - float(Decil)
        Decil_final = round(1 - float(Decil), 2)
        
        Player_2 = st.sidebar.selectbox('Select a Player to compare:', ('Aime Vrsaljko', 'ä¸­äº• AA¤§', 'Ãliver Torres MuÃ±oz', 'Ãscar de Marcos Arana', 'Ãscar Esau Duarte GaitÃ¡n', 'Ãscar Gil RegaÃ±o', 'Ãscar Guido Trejo', 'Ãscar Melendo JimÃ©nez', 'Ãscar Mingueza GarcÃ­a', 'Ãscar RodrÃ­guez Arnaiz', 'Ãscar ValentÃ­n MartÃ­n Luengo', 'A¥¥A· é›…ä¹Ÿ', 'A®‰éƒ¨ è£•èµ', 'Ä°lkay GÃ¼ndoÄŸan', 'A·A³¶ æ°¸A—£', 'A†¨A®‰ A¥æ´‹', 'Ã‡aÄŸlar SÃ¶yÃ¼ncÃ¼', 'Ã‰der Gabriel MilitÃ£o', 'Ã‰douard Mendy', 'Ã‰douard Michut', 'A‰ç”° éº»ä¹Ÿ', 'ÃÃ±igo Lekue MartÃ­nez', 'ÃÃ±igo MartÃ­nez Berridi', 'ÃdÃ¡m Szalai', 'A—é‡Ž æ‹A®Ÿ', 'Ãlex Collado GutiÃ©rrez', 'Ãlex PetxarromÃ¡n', 'Ãlvaro Bastida Moya', 'Ãlvaro Borja Morata MartÃ­n', 'Ãlvaro Bravo JimÃ©nez', 'Ãlvaro FernÃ¡ndez Llorente', 'Ãlvaro GarcÃ­a Rivera', 'Ãlvaro GarcÃ­a Segovia', 'Ãlvaro GonzÃ¡lez SoberÃ³n', 'Ãlvaro JosÃ© JimÃ©nez Guerrero', 'Ãlvaro Negredo SÃ¡nchez', 'Ãlvaro Odriozola Arzalluz', 'Ãlvaro Vadillo Cifuentes', 'Ãngel Algobia Esteves', 'Ãngel FabiÃ¡n Di MarÃ­a HernÃ¡ndez', 'Ãngel JimÃ©nez Gallego', 'Ãngel LuÃ­s RodrÃ­guez DÃ­az', 'Ãngel MartÃ­n Correa', 'Ãngel Montoro SÃ¡nchez', 'Aukasz FabiaAski', 'Aukasz Skorupski', 'Aukasz Teodorczyk', 'ä¼Šè—¤ A®æ¨¹', 'ä¹…ä¿ A»ºè‹±', 'AÃ¯ssa Mandi', 'AarÃ³n Escandell Banacloche', 'AarÃ³n MartÃ­n Caricol', 'Aaron Anthony Connolly', 'Aaron Cresswell', 'Aaron Hickey', 'Aaron Lennon', 'Aaron Nassur Kamardin', 'Aaron Ramsdale', 'Aaron Ramsey', 'Aaron Wan-Bissaka', 'AbdÃ³n Prats Bastidas', 'Abdel Jalil Zaim Idriss Medioub', 'Abdelkabir Abqar', 'Abdou Diallo', 'Abdoulaye Bamba', 'Abdoulaye DoucourÃ©', 'Abdoulaye Jules Keita', 'Abdoulaye Sylla', 'Abdoulaye TourÃ©', 'Abdoulrahmane Harroui', 'Abdourahmane Barry', 'Abdul Majeed Waris', 'Abelmy Meto Silu', 'Achraf Hakimi Mouh', 'Adam Armstrong', 'Adam David Lallana', 'Adam Forshaw', 'Adam Jakubech', 'Adam MaruA¡iÄ‡', 'Adam Masina', 'Adam Ounas', 'Adam Uche Idah', 'Adam Webster', 'Adama TraorÃ© Diarra', 'Ademola Lookman', 'Adil Aouchiche', 'Adil Rami', 'Adilson Angel Abreu de Almeida Gomes', 'Admir Mehmedi', 'Adnan Januzaj', 'AdriÃ Giner Pedrosa', 'AdriÃ¡n de la Fuente Barquilla', 'AdriÃ¡n Embarba BlÃ¡zquez', 'AdriÃ¡n RodrÃ­guez GimÃ©nez', 'AdriÃ¡n San Miguel del Castillo', 'Adrian Aemper', 'Adrian Fein', 'Adrian GrbiÄ‡', 'Adrien Rabiot', 'Adrien Sebastian Perruchet Silva', 'Adrien Tameze', 'Adrien Thomasson', 'Adrien Truffert', 'æ­¦ç£Š', 'æµ…é‡Ž ä½è—¤', 'Aihen MuÃ±oz CapellÃ¡n', 'Aimar Oroz Huarte', 'Aimar Sher', 'Aimen Moueffek', 'Ainsley Maitland-Niles', 'Aitor FernÃ¡ndez Abarisketa', 'Aitor Paredes Casamichana', 'Aitor Ruibal GarcÃ­a', 'Ajdin HrustiÄ‡', 'Ajibola Alese', 'Akim Zedadka', 'Alan Godoy DomÃ­nguez', 'Alassane Alexandre PlÃ©a', 'Alban Lafont', 'Alberth JosuÃ© Elis MartÃ­nez', 'Albert-Mboyo Sambi Lokonga', 'Alberto Grassi', 'Alberto Moreno PÃ©rez', 'Alberto Moreno Rebanal', 'Alberto Perea Correoso', 'Alberto RodrÃ­guez BarÃ³', 'Alberto Soro Ãlvarez', 'Albin Ekdal', 'Aleix Febas PÃ©rez', 'Aleix Vidal Parreu', 'Alejandro Balde MartÃ­nez', 'Alejandro Berenguer Remiro', 'Alejandro Blanco SÃ¡nchez', 'Alejandro Blesa Pina', 'Alejandro Cantero SÃ¡nchez', 'Alejandro Carbonell VallÃ©s', 'Alejandro Catena MarugÃ¡n', 'Alejandro DarÃ­o GÃ³mez', 'Alejandro FernÃ¡ndez Iglesias', 'Alejandro Iturbe Encabo', 'Alejandro Remiro Gargallo', 'Alejandro RodrÃ­guez Lorite', 'Aleksa TerziÄ‡', 'Aleksandar Kolarov', 'Aleksandar Sedlar', 'Aleksander Buksa', 'Aleksandr Golovin', 'Aleksandr Kokorin', 'Aleksey Miranchuk', 'Alessandro Bastoni', 'Alessandro Berardi', 'Alessandro Buongiorno', 'Alessandro Burlamaqui', 'Alessandro Deiola', 'Alessandro Florenzi', 'Alessandro Plizzari', 'Alessandro SchÃ¶pf', 'Alessio Cragno', 'Alessio Riccardi', 'Alessio Romagnoli', 'Alex Cordaz', 'Alex Ferrari', 'Alex Iwobi', 'Alex KrÃ¡l', 'Alex McCarthy', 'Alex Meret', 'Alex Nicolao Telles', 'Alex Oxlade-Chamberlain', 'Alex Sandro Lobo Silva', 'Alexander Djiku', 'Alexander Hack', 'Alexander Isak', 'Alexander NÃ¼bel', 'Alexander SÃ¸rloth', 'Alexander Schwolow', 'Alexandre Lacazette', 'Alexandre Letellier', 'Alexandre Moreno Lopera', 'Alexandre Oukidja', 'Alexis Alejandro SÃ¡nchez SÃ¡nchez', 'Alexis Claude-Maurice', 'Alexis Laurent Patrice Roge Flips', 'Alexis Mac Allister', 'Alexis Saelemaekers', 'Alfie Devine', 'Alfonso GonzÃ¡lez MartÃ­nez', 'Alfonso Pastor Vacas', 'Alfonso Pedraza Sag', 'AlfreÃ° Finnbogason', 'Alfred Benjamin Gomis', 'Ali Reghba', 'Alidu Seidu', 'Alisson RamsÃ©s Becker', 'Alkhaly Momo CissÃ©', 'Allan Marques Loureiro', 'Allan Saint-Maximin', 'Allan Tchaptchet', 'Allan-RomÃ©o Nyom', 'Almamy TourÃ©', 'Alou Kuol', 'Alpha Sissoko', 'Alphonse Areola', 'Alphonso Boyle Davies', 'Amad Diallo TraorÃ©', 'Amadou Diawara', 'Amadou HaÃ¯dara', 'Amadou Mbengue', 'Amadou TraorÃ©', 'Amadou Zeund Georges Ba Mvom Onana', 'Amankwaa Akurugu', 'Amari Miller', 'Amath Ndiaye Diedhiou', 'Ambroise Oyongo Bitolo', 'Ã–mer Faruk Beyaz', 'Amin Younes', 'Amine Adli', 'Amine Bassi', 'Amine Gouiri', 'Amine Harit', 'Amir Rrahmani', 'Amos Pieper', 'Anastasios Donis', 'Ander Barrenetxea Muguruza', 'Ander Capa RodrÃ­guez', 'Ander Guevara Lajo', 'Ander Herrera AgÃ¼era', 'Anderson-Lenda Lucoqui', 'Andi Avdi Zeqiri', 'Andoni Gorosabel Espinosa', 'AndrÃ© Anderson Pomilio Lima da Silva', 'AndrÃ© Filipe Tavares Gomes', 'AndrÃ© Hahn', 'AndrÃ© Miguel Valente da Silva', 'AndrÃ©-Franck Zambo Anguissa', 'AndrÃ©s Alberto Andrade CedeÃ±o', 'AndrÃ©s Felipe Solano DÃ¡vila', 'AndrÃ©s MartÃ­n GarcÃ­a', 'Andrea Belotti', 'Andrea Cambiaso', 'Andrea Carboni', 'Andrea Consigli', 'Andrea Conti', 'Andrea La Mantia', 'Andrea Masiello', 'Andrea Petagna', 'Andrea Pinamonti', 'Andrea Ranocchia', 'Andrea Schiavone', 'Andreas BÃ¸dtker Christensen', 'Andreas Luthe', 'Andreas Skov Olsen', 'Andreas Voglsammer', 'Andreaw Gravillon', 'Andrei Girotto', 'Andrej KramariÄ‡', 'Andrew Abiola Omobamidele', 'Andrew Lonergan', 'Andrew Moran', 'Andrew Robertson', 'Andrey Lunev', 'Andriy Lunin', 'Andriy Yarmolenko', 'Andros Townsend', 'Andy Delort', 'Anga Dedryck Boyata', 'Angelo Fulgini', 'Angelo Obinze Ogbonna', 'Angelo Stiller', 'Angus Gunn', 'Anselmo Garcia MacNulty', 'Ansgar Knauff', 'Anssumane Fati', 'Ante Budimir', 'Ante RebiÄ‡', 'Antef Tsoungui', 'Anthony Caci', 'Anthony David Junior Elanga', 'Anthony Driscoll-Glennon', 'Anthony Gomez Mancini', 'Anthony Gordon', 'Anthony Limbombe Ekango', 'Anthony Lopes', 'Anthony Losilla', 'Anthony Mandrea', 'Anthony Martial', 'Anthony Modeste', 'Anthony RubÃ©n Lozano ColÃ³n', 'Anthony Ujah', 'Antoine Griezmann', 'Anton Ciprian TÄƒtÄƒruÈ™anu', 'Anton Stach', 'AntonÃ­n BarÃ¡k', 'Antonin Bobichon', 'Antonino Ragusa', 'Antonio BarragÃ¡n FernÃ¡ndez', 'Antonio Blanco Conde', 'Antonio Candreva', 'Antonio JosÃ© RaÃ­llo Arenas', 'Antonio JosÃ© RodrÃ­guez DÃ­az', 'Antonio Junior Vacca', 'Antonio Latorre Grueso', 'Antonio Luca Fiordilino', 'Antonio Moya Vega', 'Antonio RÃ¼diger', 'Antonio Rosati', 'Antonio SÃ¡nchez Navarro', 'Antonio Sivera SalvÃ¡', 'Antonio Zarzana PÃ©rez', 'Anwar El Ghazi', 'ArbÃ«r Zeneli', 'Archie Mair', 'Ardian Ismajli', 'Arial Benabent Mendy', 'Aridane HernÃ¡ndez UmpiÃ©rrez', 'Aritz Elustondo Irribaria', 'Arkadiusz Krystian Milik', 'Arkadiusz Reca', 'Armand LaurientÃ©', 'Armando Broja', 'Armando Izzo', 'Armel Bella Kotchap', 'Armstrong Okoflex', 'ArnÃ³r SigurÃ°sson', 'Arnaldo Antonio Sanabria Ayala', 'Arnau Tenas UreÃ±a', 'Arnaud Dominique Nordin', 'Arnaud Kalimuendo-Muinga', 'Arnaud Souquet', 'Arnaut Danjuma Groeneveld', 'Arne Maier', 'Arne Schulz', 'Arthur Desmas', 'Arthur Henrique Ramos de Oliveira Melo', 'Arthur Masuaku', 'Arthur Nicolas Theate', 'Arthur Okonkwo', 'Arturo Erasmo Vidal Pardo', 'Ashley Fletcher', 'Ashley Luke Barnes', 'Ashley Westwood', 'Ashley Young', 'Asier Illarramendi Andonegi', 'Asier Villalibre Molina', 'Asmir BegoviÄ‡', 'Assane DioussÃ© El Hadji', 'Aster Jan Vranckx', 'Atakan Karazor', 'Augusto Jorge Mateo Solari', 'AurÃ©lien TchouamÃ©ni', 'Axel Arthur Disasi', 'Axel Camblan', 'Axel Tuanzebe', 'Axel Wilfredo Werner', 'Axel Witsel', 'Aymen Barkok', 'Aymeric Laporte', 'Ayodeji Sotona', 'Ayoze PÃ©rez GutiÃ©rrez', 'Azor Matusiwa', 'Ažtefan Daniel Radu', 'AŽŸA£ A…ƒæ°—', 'Azzedine Ounahi', 'BÃ©ni Makouana', 'Bakary Adama Soumaoro', 'Bali Mumba', 'Bamidele Alli', 'Bamo MeÃ¯tÃ©', 'Bandiougou Fadiga', 'Baptiste SantamarÃ­a', 'BartAomiej DrÄ…gowski', 'Bartosz BereszyAski', 'Bartosz BiaAek', 'Bastian Oczipka', 'Batista Mendy', 'BeÃ±at Prados DÃ­az', 'Ben Bobzien', 'Ben Chilwell', 'Ben Chrisene', 'Ben Foster', 'Ben Gibson', 'Ben Godfrey', 'Ben Klefisch', 'Ben Mee', 'Benjamin AndrÃ©', 'Benjamin Bourigeaud', 'Benjamin HÃ¼bner', 'Benjamin Henrichs', 'Benjamin Johnson', 'Benjamin Lecomte', 'Benjamin Lhassine Kone', 'Benjamin Mendy', 'Benjamin Pavard', 'Benjamin Thomas Davies', 'Benjamin Uphoff', 'Benjamin White', 'Benno Schmitz', 'BenoÃ®t Badiashile Mukinayi', 'BenoÃ®t Costil', 'Berat Djimsiti', 'Bernardo Costa da Rosa', 'Bernardo Mota Veiga de Carvalho e Silva', 'Bernd Leno', 'Bertrand Isidore TraorÃ©', 'Bilal Benkhedim', 'Bilal Nadir', 'Billal Brahimi', 'Billy Gilmour', 'Billy Koumetio', 'Bingourou Kamara', 'Birger Solberg Meling', 'Bjarki Steinn Bjarkason', 'BoA¡ko Autalo', 'Bobby Adekanye', 'Bobby Thomas', 'Boris RadunoviÄ‡', 'Borja Iglesias Quintas', 'Borja Mayoral Moya', 'Borna Sosa', 'Boubacar Bernard Kamara', 'Boubacar Fall', 'Boubacar TraorÃ©', 'Boubakar KouyatÃ©', 'Boubakary SoumarÃ©', 'Boulaye Dia', 'Bouna Sarr', 'Bradley Locko', 'Brahim Abdelkader DÃ­az', 'Brais MÃ©ndez Portela', 'Bram Nuytinck', 'Brandon DominguÃ¨s', 'Brandon Soppy', 'Brandon Williams', 'Branimir Hrgota', 'Breel-Donald Embolo', 'Brendan Chardonnet', 'Brian Ebenezer Adjei Brobbey', 'Brian OlivÃ¡n Herrero', 'Brooklyn Lyons-Foster', 'Bruno AndrÃ© Cavaco JordÃ£o', 'Bruno GuimarÃ£es Rodriguez Moura', 'Bruno Miguel Borges Fernandes', 'Bruno Oliveira Bertinato', 'Bryan Cristante', 'Bryan Gil Salvatierra', 'Bryan Lasme', 'Bryan Mbeumo', 'Bryan Nokoue', 'Bryan Reynolds', 'Bukayo Saka', 'Burak YÄ±lmaz', 'CÃ©dric Brunner', 'CÃ©dric Hountondji', 'CÃ©dric Jan Itten', 'CÃ©dric Ricardo Alves Soares', 'CÃ©sar Azpilicueta Tanco', 'CÃ©sar Joel Valencia Castillo', 'CÄƒtÄƒlin CÃ®rjan', 'Caio Henrique Oliveira Silva', 'Caleb Ansah Ekuban', 'Caleb Cassius Watts', 'Callum Hudson-Odoi', 'Callum Wilson', 'Calum Chambers', 'Calvin Bombo', 'Calvin Stengs', 'Cameron Archer', 'Caoimhin Kelleher', 'Carles AleÃ±Ã¡ Castillo', 'Carles PÃ©rez Sayol', 'Carlo Pinsoglio', 'Carlos Akapo MartÃ­nez', 'Carlos Armando Gruezo Arboleda', 'Carlos Arturo Bacca Ahumada', 'Carlos Beitia Cardos', 'Carlos Clerc MartÃ­nez', 'Carlos DomÃ­nguez CÃ¡ceres', 'Carlos FernÃ¡ndez Luna', 'Carlos Henrique Venancio Casimiro', 'Carlos JoaquÃ­n Correa', 'Carlos Neva Tey', 'Carlos Soler BarragÃ¡n', 'Carney Chibueze Chukwuemeka', 'Cedric Teuchert', 'Cedric Wilfred TeguÃ­a Noubi', 'Cengiz Ãœnder', 'Cenk Tosun', 'ChÃ© Adams', 'Charalampos Lykogiannis', 'Charles Boli', 'Charles Mariano ArÃ¡nguiz Sandoval', 'Charles TraorÃ©', 'Charlie Cresswell', 'Charlie Goode', 'Charlie Taylor', 'Charlie Wiggett', 'Charly Musonda Junior', 'Cheick Oumar DoucourÃ©', 'Cheick Oumar SouarÃ©', 'Cheikh Ahmadou Bamba Mbacke Dieng', 'Cheikh Ahmet Tidian Niasse', 'Cheikh Tidiane Sabaly', 'Cheikhou KouyatÃ©', 'Chem Campbell', 'Chigozier Caleb Chukwuemeka', 'Chris FÃ¼hrich', 'Chris Smalling', 'Chrislain Iris Aurel Matsima', 'Christian Benteke Liolo', 'Christian Dannemann Eriksen', 'Christian Fernandes Marques', 'Christian FrÃ¼chtl', 'Christian GÃ¼nter', 'Christian GÃ³mez Vela', 'Christian Gabriel Oliva GimÃ©nez', 'Christian Kabasele', 'Christian Mate Pulisic', 'Christian Thers NÃ¸rgaard', 'Christoph Baumgartner', 'Christoph Kramer', 'Christoph Zimmermann', 'Christophe HÃ©relle', 'Christopher Antwi-Adjei', 'Christopher Grant Wood', 'Christopher Jeffrey Richards', 'Christopher Lenz', 'Christopher Maurice Wooh', 'Christopher Nkunku', 'Christopher Trimmel', 'Christos Tzolis', 'Ciaran Clark', 'Ciro Immobile', 'ClÃ©ment Nicolas Laurent Lenglet', 'Claudio AndrÃ©s Bravo MuÃ±oz', 'Clinton Mola', 'Cody Callum Pierre Drameh', 'Cole Palmer', 'Colin Dagba', 'Connor Roberts', 'Conor Carty', 'Conor Coady', 'Conor Gallagher', 'Conor NoÃŸ', 'Corentin Jean', 'Corentin Tolisso', 'Craig Dawson', 'Craig George Cathcart', 'CristÃ³bal Montiel RodrÃ­guez', 'Cristian Daniel Ansaldi', 'Cristian Esteban Gamboa Luna', 'Cristian Gabriel Romero', 'Cristian Molinaro', 'Cristian PortuguÃ©s Manzanera', 'Cristian Rivero Sabater', 'Cristian Tello Herrera', 'Cristiano Biraghi', 'Cristiano Lombardi', 'Cristiano Piccini', 'Cristiano Ronaldo dos Santos Aveiro', 'Crysencio Summerville', 'Curtis Jones', 'DÃ©nys Bain', 'Daan Heymans', 'DÃ­dac VilÃ¡ RossellÃ³', 'Dalbert Henrique Chagas EstevÃ£o', 'Dale Stephens', 'Daley Sinkgraven', 'DamiÃ¡n Emiliano MartÃ­nez', 'DamiÃ¡n NicolÃ¡s SuÃ¡rez SuÃ¡rez', 'Damiano Pecile', 'Damien Da Silva', 'Damir Ceter Valencia', 'Dan Burn', 'Dan Gosling', 'Dan-Axel Zagadou', 'Dane Pharrell Scarlett', 'Dani van den Heuvel', 'Daniel Amartey', 'Daniel Bachmann', 'Daniel Brosinski', 'Daniel CÃ¡rdenas LÃ­ndez', 'Daniel Caligiuri', 'Daniel Carvajal Ramos', 'Daniel Castelo Podence', 'Daniel Ceballos FernÃ¡ndez', 'Daniel CerÃ¢ntula Fuzato', 'Daniel Didavi', 'Daniel GÃ³mez AlcÃ³n', 'Daniel GarcÃ­a Carrillo', 'Daniel Ginczek', 'Daniel James', 'Daniel JosÃ© RodrÃ­guez VÃ¡zquez', 'Daniel Klein', 'Daniel Langley', 'Daniel Maldini', 'Daniel Nii Tackie Mensah Welbeck', 'Daniel Olmo Carvajal', 'Daniel Parejo MuÃ±oz', 'Daniel Plomer Gordillo', 'Daniel Raba AntolÃ­n', 'Daniel Sartori Bessa', 'Daniel Vivian Moreno', 'Daniel Wass', 'Daniel William John Ings', 'Daniele Baselli', 'Daniele Padelli', 'Daniele Rugani', 'Daniele Verde', 'Danijel PetkoviÄ‡', 'Danilo Cataldi', 'Danilo DAmbrosio', 'Danilo LuÃ­s HÃ©lio Pereira', 'Danilo Luiz da Silva', 'Danilo Teodoro Soares', 'Danny Blum', 'Danny Rose', 'Danny Vieira da Costa', 'Danny Ward', 'Dante Bonfim da Costa Santos', 'DarÃ­o Ismael Benedetto', 'DarÃ­o Poveda Romera', 'Darko BraA¡anac', 'Darko LazoviÄ‡', 'Darren Randolph', 'Darwin Daniel MachÃ­s Marcano', 'David Chidozie Okereke', 'David De Gea Quintana', 'David Edward Martin', 'David GarcÃ­a ZubirÃ­a', 'David Gil Mohedano', 'David Jason Remeseiro Salgueiro', 'David JosuÃ© JimÃ©nez Silva', 'David LÃ³pez Silva', 'David Lelle', 'David Nemeth', 'David Oberhauser', 'David Olatukunbo Alaba', 'David Ospina RamÃ­rez', 'David Pereira Da Costa', 'David Raum', 'David Raya Martin', 'David Schnegg', 'David Soria SolÃ­s', 'David Timor CopovÃ­', 'David Zima', 'Davide Biraschi', 'Davide Calabria', 'Davide Frattesi', 'Davide Santon', 'Davide Zappacosta', 'Davide Zappella', 'Davie Selke', 'Davinson SÃ¡nchez Mina', 'Davy Rouyard', 'Dayotchanculle Oswald Upamecano', 'Dean Henderson', 'Declan Rice', 'Deiver AndrÃ©s Machado Mena', 'Dejan Kulusevski', 'Dejan LjubiÄiÄ‡', 'Demarai Gray', 'Denis Athanase Bouanga', 'Denis Cheryshev', 'Denis Lemi Zakaria Lako Lado', 'Denis PetriÄ‡', 'Denis SuÃ¡rez FernÃ¡ndez', 'Denis Vavro', 'Dennis Appiah', 'Dennis Geiger', 'Dennis Jastrzembski', 'Dennis Praet', 'Dennis TÃ¸rset Johnsen', 'Denzel Justus Morris Dumfries', 'Destiny Iyenoma Udogie', 'Deyovaisio Zeefuik', 'DiadiÃ© SamassÃ©kou', 'Diant Ramaj', 'Dickson Abiama', 'Diego Carlos Santos Silva', 'Diego Demme', 'Diego Falcinelli', 'Diego FarÃ­as da Silva', 'Diego GonzÃ¡lez Polanco', 'Diego Javier Llorente RÃ­os', 'Diego JosÃ© Conde Alcolado', 'Diego LÃ³pez Noguerol', 'Diego LÃ³pez RodrÃ­guez', 'Diego Lainez Leyva', 'Diego Moreno Garbayo', 'Diego Rico Salguero', 'Diego Roberto GodÃ­n Leal', 'Diego Vicente Bri Carrazoni', 'Digbo Gnampa Habib MaÃ¯ga', 'Dilan Kumar Markanday', 'Dilane Bakwa', 'Dillon Hoogewerf', 'Dimitri Foulquier', 'Dimitri LiÃ©nard', 'Dimitri Payet', 'Dimitrios Nikolaou', 'Dimitris Giannoulis', 'Dimitry Bertaud', 'Diogo JosÃ© Teixeira da Silva', 'Dion Berisha', 'Dion Lopy', 'Divock Okoth Origi', 'DjenÃ© Dakonam Ortega', 'Djibril Fandje TourÃ©', 'Djibril SidibÃ©', 'Djibril Sow', 'DoÄŸan Alemdar', 'Dodi LukÃ©bakio', 'Domagoj BradariÄ‡', 'Domen ÄŒrnigoj', 'Domenico Berardi', 'Domenico Criscito', 'Domingos Sousa Coutinho Meneses Duarte', 'Dominic Calvert-Lewin', 'Dominic Schmidt', 'Dominic Thompson', 'Dominik Greif', 'Dominik Kohr', 'Dominik Szoboszlai', 'Dominique Heintz', 'Donny van de Beek', 'Donyell Malen', 'Dor Peretz', 'Douglas Luiz Soares de Paulo', 'DragiA¡a Gudelj', 'Dries Mertens', 'DuA¡an VlahoviÄ‡', 'Duje Ä†aleta-Car', 'DuvÃ¡n Esteban Zapata Banguera', 'Dwight Gayle', 'Dwight McNeil', 'Dylan Chambost', 'Dylan Daniel Mahmoud Bronn', 'Dynel Brown Kembo Simeu', 'é•·è°· éƒ¨èª', 'éè—¤ æ¸A¤ª', 'éè—¤ èˆª', 'Eberechi Eze', 'Ebrima Colley', 'Ebrima Darboe', 'Eddie Anthony Salcedo Mora', 'Eden Hazard', 'Eden Massouema', 'Ederson Santana de Moraes', 'Edgar Antonio MÃ©ndez Ortega', 'Edgar Badia Guardiola', 'Edgar GonzÃ¡lez Estrada', 'Edgar Paul Akouokou', 'Edgar Sevikyan', 'Edimilson Fernandes Ribeiro', 'Edin DA¾eko', 'Edinson Roberto Cavani GÃ³mez', 'Edmond FayÃ§al Tapsoba', 'Edoardo Bove', 'Edoardo Goldaniga', 'Edoardo Vergani', 'Edson AndrÃ© Sitoe', 'Eduard LÃ¶wen', 'Eduardo Camavinga', 'Edward Campbell Sutherland', 'Edward Nketiah', 'Einar Iversen', 'El Bilal TourÃ©', 'Elbasan Rashani', 'Eldin JakupoviÄ‡', 'Eldor Shomurodov', 'Elias Kratzer', 'Elijah Dixon-Bonner', 'Eliot Matazo', 'Eljif Elmas', 'Elliott Anderson', 'Ellis Simms', 'Ellyes Joris Skhiri', 'Elseid Hysaj', 'Elvis RexhbeÃ§aj', 'Emanuel Quartsin Gyasi', 'Emanuel Vignato', 'Emerson Aparecido Leite de Souza Junior', 'Emerson Palmieri dos Santos', 'Emil Audero Mulyadi', 'Emil Berggreen', 'Emil Henry Kristoffer Krafth', 'Emil Peter Forsberg', 'Emile Smith Rowe', 'Emiliano BuendÃ­a', 'Emmanuel Bonaventure Dennis', 'Emmanuel Kouadio KonÃ©', 'Emre Can', 'Emrehan Gedikli', 'Enes Ãœnal', 'Enis Bardhi', 'Enock Kwateng', 'Enock Mwepu', 'Enric Franquesa Dolz', 'Enrique Barja Afonso', 'Enrique GÃ³mez Hermoso', 'Enrique GarcÃ­a MartÃ­nez', 'Enzo Camille Alain Millot', 'Enzo Ebosse', 'Enzo Jeremy Le FÃ©e', 'Enzo Pablo Roco Roco', 'Erhan MaA¡oviÄ‡', 'Eric Bertrand Bailly', 'Eric Dier', 'Eric GarcÃ­a Martret', 'Eric Junior Dina Ebimbe', 'Eric Uhlmann', 'Erick Antonio Pulgar FarfÃ¡n', 'Erick Cathriel Cabaco Almada', 'Erik Durm', 'Erik Lamela', 'Erik Pieters', 'Erik Ross Palmer-Brown', 'Erik Thommy', 'Erion Sadiku', 'Erling Braut Haaland', 'Ermedin DemiroviÄ‡', 'Ermin BiÄakÄiÄ‡', 'Ernesto Torregrossa', 'Esey Gebreyesus', 'Esteban Ariel Saveljich', 'Ethan Ampadu', 'Ethan Pinnock', 'Etienne Capoue', 'Etienne Green', 'Etrit Berisha', 'Eugenio Pizzuto Puga', 'Evann Guessand', 'Exequiel Alejandro Palacios', 'éŽŒç”° A¤§Aœ°', 'Ezri Konsa Ngoyo', 'FÃ¡bio Daniel Soares Silva', 'FÃ¡bio Henrique Tavares', 'FÃ¡bio Pereira da Silva', 'FabiÃ¡n Ruiz PeÃ±a', 'Fabian Bredlow', 'Fabian Delph', 'Fabian Klos', 'Fabian Kunze', 'Fabian Lukas SchÃ¤r', 'Fabian RÃ¼th', 'Fabiano Parisi', 'Fabien Centonze', 'Fabien Lemoine', 'Fabio Blanco GÃ³mez', 'Fabio Depaoli', 'Fabio Quagliarella', 'Fabio Schneider', 'Facundo Axel Medina', 'Facundo Colidio', 'Facundo Pellistri Rebollo', 'Faouzi Ghoulam', 'Farid Boulaya', 'Farid El Melali', 'Federico Bernardeschi', 'Federico Bonazzoli', 'Federico Ceccherini', 'Federico Chiesa', 'Federico Di Francesco', 'Federico Dimarco', 'Federico FernÃ¡ndez', 'Federico Javier Santander Mereles', 'Federico JuliÃ¡n Fazio', 'Federico Marchetti', 'Federico Mattiello', 'Federico Peluso', 'Federico Santiago Valverde Dipetta', 'Felipe Anderson Pereira Gomes', 'Felipe Augusto de Almeida Monteiro', 'Felipe Salvador Caicedo Corozo', 'Felix Kalu Nmecha', 'Felix Passlack', 'Felix Schwarzholz', 'Ferland Mendy', 'Fernando Calero Villa', 'Fernando Francisco Reges', 'Fernando Luiz Rosa', 'Fernando MarÃ§al de Oliveira', 'Fernando MartÃ­n Forestieri', 'Fernando NiÃ±o RodrÃ­guez', 'Fernando Pacheco Flores', 'Ferran Torres GarcÃ­a', 'Fidel Chaves de la Torre', 'Fikayo Tomori', 'Filip ÄuriÄiÄ‡', 'Filip BenkoviÄ‡', 'Filip JÃ¶rgensen', 'Filip KostiÄ‡', 'Filippo Bandinelli', 'Filippo Delli Carri', 'Filippo Melegoni', 'Filippo Romagna', 'Filippo Tripi', 'Finley Stevens', 'Finn Gilbert Dahmen', 'Flavien Tait', 'Flavius David Daniliuc', 'Florent Da Silva', 'Florent Mollet', 'Florent Ogier', 'Florentino Ibrain Morris LuÃ­s', 'Florian Christian Neuhaus', 'Florian Grillitsch', 'Florian Kainz', 'Florian KrÃ¼ger', 'Florian Lejeune', 'Florian MÃ¼ller', 'Florian Niederlechner', 'Florian Palmowski', 'Florian Richard Wirtz', 'Florian Sotoca', 'Florian Tardieu', 'Florin Andone', 'Flynn Clarke', 'FodÃ© Ballo-TourÃ©', 'FodÃ© DoucourÃ©', 'Folarin Balogun', 'FrÃ©dÃ©ric Guilbert', 'FrÃ©dÃ©ric Veseli', 'Francesc FÃbregas i Soler', 'Francesco Acerbi', 'Francesco Bardi', 'Francesco Caputo', 'Francesco Cassata', 'Francesco Di Tacchio', 'Francesco Forte', 'Francesco Magnanelli', 'Francesco Rossi', 'Francis Coquelin', 'Francisco AlcÃ¡cer GarcÃ­a', 'Francisco AndrÃ©s Sierralta Carvallo', 'Francisco AntÃ³nio Machado Mota Castro TrincÃ£o', 'Francisco Casilla CortÃ©s', 'Francisco FemenÃ­a Far', 'Francisco Javier Hidalgo GÃ³mez', 'Francisco JosÃ© BeltrÃ¡n Peinado', 'Francisco JosÃ© GarcÃ­a Torres', 'Francisco MÃ©rida PÃ©rez', 'Francisco Manuel Gonzalez Verjara', 'Francisco RomÃ¡n AlarcÃ³n SuÃ¡rez', 'Franck Bilal RibÃ©ry', 'Franck Honorat', 'Franck Yannick KessiÃ©', 'Franco Daryl Tongya Heubang', 'Franco Emanuel Cervi', 'Franco MatÃ­as Russo Panos', 'Frank Ogochukwu Onyeka', 'FransÃ©rgio Rodrigues Barbosa', 'Fraser Forster', 'Fraser Hornby', 'Freddie Woodman', 'Frederico Rodrigues de Paula Santos', 'Frederik Franck Winther', 'Frederik Riis RÃ¸nnow', 'Frenkie de Jong', 'GÃ¶ktan GÃ¼rpÃ¼z', 'GaÃ«l Kakuta', 'GaÃ«tan Laborde', 'GaÃ«tan Poussin', 'Gabriel Armando de Abreu', 'Gabriel dos Santos MagalhÃ£es', 'Gabriel Fernando de Jesus', 'Gabriel Gudmundsson', 'Gabriel MoisÃ©s Antunes da Silva', 'Gabriel Mutombo Kupa', 'Gabriel Nascimento Resende BrazÃ£o', 'Gabriel Teodoro Martinelli Silva', 'Gabriele Corbo', 'Gabriele Zappa', 'Gaetano Castrovilli', 'Gaizka Campos BahÃ­llo', 'Gareth Frank Bale', 'Gary Alexis Medel Soto', 'GastÃ³n Rodrigo Pereiro LÃ³pez', 'Gauthier Gallon', 'Gautier Larsonneur', 'Gelson Dany Batalha Martins', 'Geoffrey Edwin Kondogbia', 'George McEachran', 'Georginio Rutter', 'Georginio Wijnaldum', 'GerÃ³nimo Rulli', 'Gerard Deulofeu LÃ¡zaro', 'Gerard Gumbau Garriga', 'Gerard Moreno BalaguerÃ³', 'Gerard PiquÃ© BernabÃ©u', 'GermÃ¡n Alejandro Pezzella', 'GermÃ¡n SÃ¡nchez Barahona', 'Gerrit Holtmann', 'Gerson Leal Rodrigues Gouveia', 'Gerson Santos da Silva', 'Gerzino Nyamsi', 'Ghislain Konan', 'Giacomo Bonaventura', 'Giacomo Raspadori', 'Giacomo Satalino', 'Gian Marco Ferrari', 'Giangiacomo Magnani', 'Gianluca Busio', 'Gianluca Caprari', 'Gianluca Frabotta', 'Gianluca Gaetano', 'Gian-Luca Itter', 'Gianluca Mancini', 'Gianluca Pegolo', 'Gianluca Scamacca', 'Gianluca SchÃ¤fer', 'Gian-Luca Waldschmidt', 'Gianluigi Donnarumma', 'Gianmarco Cangiano', 'Gianmarco Zigoni', 'Gideon Jung', 'Gideon Mensah', 'Gil-Linnart Walther', 'Giorgi Mamardashvili', 'Giorgio Altare', 'Giorgio Chiellini', 'Giorgos Kyriakopoulos', 'Giovani Lo Celso', 'Giovanni Alejandro Reyna', 'Giovanni Crociata', 'Giovanni Di Lorenzo', 'Giovanni Pablo Simeone', 'Giulian Biancone', 'Giuliano Simeone', 'Giulio Maggiore', 'Giuseppe Pezzella', 'Gleison Bremer Silva Nascimento', 'Gnaly Maxwel Cornet', 'GonÃ§alo Manuel Ganchinho Guedes', 'GonÃ§alo Mendes PaciÃªncia', 'Gonzalo Ariel Montiel', 'Gonzalo Cacicedo VerdÃº', 'Gonzalo Escalante', 'Gonzalo JuliÃ¡n Melero Manzanares', 'Gonzalo Villar del Fraile', 'Gor Manvelyan', 'Goran Pandev', 'GrÃ©goire Coudert', 'Granit Xhaka', 'Grant Hanley', 'Gregoire Defrel', 'Gregor Kobel', 'Gregorio Gracia SÃ¡nchez', 'Grigoris Kastanos', 'Grischa PrÃ¶mel', 'Guglielmo Vicario', 'Guido Guerrieri', 'Guido Marcelo Carrillo', 'Guido RodrÃ­guez', 'Guilherme Magro Pires Ramos', 'Guillermo Alfonso MaripÃ¡n Loaysa', 'Gylfi ÃžÃ³r SigurÃ°sson', 'HÃ¥vard Kallevik Nielsen', 'HÃ¥vard Nordtveit', 'HÃ©ctor BellerÃ­n Moruno', 'HÃ©ctor Junior Firpo AdamÃ©s', 'HÃ©ctor Miguel Herrera LÃ³pez', 'HÃ©lder Wander Sousa Azevedo Costa', 'Habib Ali Keita', 'Habib Diarra', 'Habibou Mouhamadou Diallo', 'Hakan Ã‡alhanoÄŸlu', 'Hakim Ziyech', 'Hamadi Al Ghaddioui', 'Hamari TraorÃ©', 'Hamed Junior TraorÃ¨', 'Hamza Choudhury', 'Hannes Wolf', 'Hannibal Mejbri', 'Hans Carl Ludwig Augustinsson', 'Hans Fredrik Jensen', 'Hans Hateboer', 'Hans Nunoo Sarpei', 'Haris Belkebla', 'Harold Moukoudi', 'Harrison Ashby', 'Harry Kane', 'Harry Lewis', 'Harry Maguire', 'Harry Winks', 'Harvey David White', 'Harvey Elliott', 'Harvey Lewis Barnes', 'Hassane Kamara', 'Hayden Lindley', 'Haydon Roberts', 'Helibelton Palacios Zapata', 'Henrikh Mkhitaryan', 'Henrique Silva Milagres', 'Henry Wise', 'Herbert Bockhorn', 'HernÃ¢ni Jorge Santos Fortes', 'Hernani Azevedo JÃºnior', 'Hianga Mananga Mbock', 'Hicham Boudaoui', 'Hirving Rodrigo Lozano Bahena', 'Houboulang Mendes', 'Houssem Aouar', 'Hugo Bueno LÃ³pez', 'Hugo Duro Perales', 'Hugo Ekitike', 'Hugo GuillamÃ³n SammartÃ­n', 'Hugo Lloris', 'Hugo Magnetti', 'Hugo Mallo Novegil', 'Hugo Novoa Ramos', 'ì•ìš°ì˜ Woo Yeong Jeong', 'ìí˜ì¤€ çŽé‰ä¿Š', 'ì†í¥ë¯¼ A­™A…´æ…œ', 'ì´ê°•ì¸ Kang-In Lee', 'ì´ìž¬ì± æŽAœ¨æˆ', 'IÃ±aki Williams Arthuer', 'IÃ±igo PÃ©rez Soto', 'IÃ±igo Ruiz de Galarreta Etxeberria', 'Iago Amaral Borduchi', 'Iago Aspas Juncal', 'Ibrahim Amadou', 'Ibrahim Karamoko', 'Ibrahim Yalatif Diabate', 'Ibrahima BaldÃ©', 'Ibrahima Diallo', 'Ibrahima KonatÃ©', 'Ibrahima Mbaye', 'Ibrahima Niane', 'Ibrahima Sissoko', 'Iddrisu Baba Mohammed', 'Idrissa Gana Gueye', 'Ignacio Monreal Eraso', 'Ignacio PeÃ±a Sotorres', 'Ignacio Pussetto', 'Ignacio RamÃ³n del Valle', 'Ignacio Vidal Miralles', 'Ignatius Kpene Ganago', 'Igor JÃºlio dos Santos de Paulo', 'Igor Silva de Almeida', 'Igor ZubeldÃ­a Elorza', 'Ihlas Bebou', 'Ihsan Sacko', 'Iker Benito SÃ¡nchez', 'Iker Losada Aragunde', 'Iker Muniain GoÃ±i', 'Iker Pozo La Rosa', 'Iker Recio Ortega', 'Ilan Kais Kebbal', 'Ilija Nestorovski', 'Illan Meslier', 'Imran Louza', 'IonuÈ› Andrei Radu', 'Irvin Cardona', 'Isaac Ajayi Success', 'Isaac CarcelÃ©n Valencia', 'Isaac Hayden', 'Isaac Lihadji', 'Isaac PalazÃ³n Camacho', 'Ishak Belfodil', 'Islam Slimani', 'IsmaÃ¯la Sarr', 'IsmaÃ«l Bennacer', 'IsmaÃ«l Boura', 'IsmaÃ«l Gharbi', 'IsmaÃ«l TraorÃ©', 'Ismael Ruiz SÃ¡nchez', 'Ismail Joshua Jakobs', 'Ismaila PathÃ© Ciss', 'Israel Salazar PÃ­riz', 'Issa Diop', 'Issa KaborÃ©', 'Issouf Sissokho', 'í™©ì˜ì¡° Ui Jo Hwang', 'í™©í¬ì°¬ é»A–œç¿', 'IvÃ¡n Alejo Peralta', 'IvÃ¡n Balliu Campeny', 'IvÃ¡n Bravo Castro', 'IvÃ¡n Chapela LÃ³pez', 'IvÃ¡n JosÃ© Marcone', 'IvÃ¡n MartÃ­n NÃºÃ±ez', 'IvÃ¡n MartÃ­nez GonzÃ¡lvez', 'IvÃ¡n Mauricio Arboleda', 'IvÃ¡n Romero de Ãvila', 'Ivan AaponjiÄ‡', 'Ivan IliÄ‡', 'Ivan PeriA¡iÄ‡', 'Ivan Provedel', 'Ivan RadovanoviÄ‡', 'Ivan RakitiÄ‡', 'Ivan Toney', 'Ivo GrbiÄ‡', 'Ivor Pandur', 'JÃ©rÃ´me Boateng', 'JÃ©rÃ´me Hergault', 'JÃ©rÃ´me Phojo', 'JÃ©rÃ´me Roussillon', 'JÃ©rÃ©mie Boga', 'JÃ©rÃ©my Doku', 'JÃ©rÃ©my Le Douaron', 'JÃ©rÃ©my Morel', 'JÃ©rÃ©my Pied', 'JÃ©rÃ©my Pierre SincÃ¨re Gelin', 'JÃ¶rgen Joakim Nilsson', 'JÃ¼rgen Locadia', 'JaÃ¯ro Riedewald', 'Jack Butland', 'Jack Clarke', 'Jack Cork', 'Jack de Vries', 'Jack Grealish', 'Jack Harrison', 'Jack Jenkins', 'Jack Stephens', 'Jack Young', 'Jacob Barrett Laursen', 'Jacob Bedeau', 'Jacob Bruun Larsen', 'Jacob Michael Italiano', 'Jacob Montes', 'Jacob Murphy', 'Jacob Ramsey', 'Jacopo Dezi', 'Jacopo Furlan', 'Jacopo Sala', 'Jaden Philogene-Bidace', 'Jadon Sancho', 'Jaime Mata Arnaiz', 'Jakob Busk Jensen', 'Jakob Lungi SÃ¸rensen', 'Jakub Jankto', 'Jakub Piotr Moder', 'Jamaal Lascelles', 'Jamal Baptiste', 'Jamal Lewis', 'Jamal Musiala', 'James David RodrÃ­guez Rubio', 'James Furlong', 'James Justin', 'James Maddison', 'James McArthur', 'James McAtee', 'James Norris', 'James Olayinka', 'James Philip Milner', 'James Tarkowski', 'James Tomkins', 'James Ward-Prowse', 'Jamie Leweling', 'Jamie Shackleton', 'Jamie Vardy', 'Jan A½ambA¯rek', 'Jan Bednarek', 'Jan Jakob Olschowsky', 'Jan MorÃ¡vek', 'Jan Oblak', 'Jan SchrÃ¶der', 'Jan Thielmann', 'Jan Thilo Kehrer', 'Janik Haberer', 'Janis Antiste', 'Jannes-Kilian Horn', 'Jannik Vestergaard', 'Janni-Luca Serra', 'Jannis Lang', 'Japhet Tanganga', 'JarosAaw PrzemysAaw Jach', 'Jarrad Branthwaite', 'Jarrod Bowen', 'Jason Berthomier', 'Jason Denayer', 'Jason Steele', 'Jasper Cillessen', 'Jaume DomÃ©nech SÃ¡nchez', 'Jaume Grau Ciscar', 'Jaume Vicent Costa JordÃ¡', 'JavairÃ´ Dilrosun', 'Javier Castro UrdÃ­n', 'Javier DÃ­az SÃ¡nchez', 'Javier GÃ³mez Castroverde', 'Javier GalÃ¡n Gil', 'Javier LÃ³pez Carballo', 'Javier LlabrÃ©s Exposito', 'Javier Manquillo GaitÃ¡n', 'Javier MartÃ­nez Calvo', 'Javier MatÃ­as Pastore', 'Javier Ontiveros Parra', 'Javier Puado DÃ­az', 'Javier Serrano MartÃ­nez', 'Jay Rodriguez', 'Jayden Jezairo Braaf', 'Jean Emile Junior Onana Onana', 'Jean Harisson Marcelin', 'Jean Lucas de Souza Oliveira', 'Jean-Charles Castelletto', 'Jean-Clair Dimitri Roger Todibo', 'Jean-Claude Billong', 'Jean-Daniel Dave Lewis Akpa Akpro', 'Jean-Eric Maxim Choupo-Moting', 'Jean-EudÃ¨s Aholou', 'Jean-KÃ©vin Augustin', 'Jean-KÃ©vin Duverne', 'Jean-Louis Leca', 'Jean-Paul BoÃ«tius', 'Jean-Philippe Gbamin', 'Jean-Philippe Krasso', 'Jean-Philippe Mateta', 'Jean-Ricner Bellegarde', 'Jean-Victor Makengo', 'Jed Steer', 'Jeff Patrick Hendrick', 'Jeff Reine-AdÃ©laÃ¯de', 'Jeffrey Gouweleeuw', 'Jeffrey Schlupp', 'Jeison FabiÃ¡n Murillo CerÃ³n', 'Jens Castrop', 'Jens Grahl', 'Jens JÃ¸nsson', 'Jens Petter Hauge', 'Jens Stryger Larsen', 'Jerdy Schouten', 'Jere Juhani Uronen', 'JeremÃ­as Conan Ledesma', 'Jeremiah St. Juste', 'Jeremie Agyekum Frimpong', 'Jeremy Dudziak', 'Jeremy Ngakia', 'Jeremy Sarmiento', 'Jeremy Toljan', 'Jeroen Zoet', 'JesÃºs Areso Blanco', 'JesÃºs JoaquÃ­n FernÃ¡ndez SÃ¡ez de la Torre', 'JesÃºs Navas GonzÃ¡lez', 'JesÃºs VÃ¡zquez Alcalde', 'JesÃºs Vallejo LÃ¡zaro', 'Jesper GrÃ¦nge LindstrÃ¸m', 'Jesse Lingard', 'Jessic GaÃ¯tan Ngankam', 'Jessy Moulin', 'Jesurun Rak-Sakyi', 'Jetro Willems', 'Jim Ã‰milien Ngowet Allevinah', 'Jimmy Briand', 'Jimmy Cabot', 'Jimmy Giraudon', 'JoA¡ko Gvardiol', 'JoÃ£o FÃ©lix Sequeira', 'JoÃ£o Filipe Iria Santos Moutinho', 'JoÃ£o Paulo Santos da Costa', 'JoÃ£o Pedro Cavaco Cancelo', 'JoÃ£o Pedro Geraldino dos Santos GalvÃ£o', 'JoÃ£o Pedro Junqueira de Jesus', 'JoÃ«l Andre Job Matip', 'JoÃ«l Ivo Veltman', 'Joachim Christian Andersen', 'Joakim MÃ¦hle Pedersen', 'Joan GarcÃ­a Pons', 'Joan JordÃ¡n Moreno', 'Joan Sastre Vanrell', 'JoaquÃ­n JosÃ© MarÃ­n Ruiz', 'JoaquÃ­n Navarro JimÃ©nez', 'JoaquÃ­n SÃ¡nchez RodrÃ­guez', 'Jodel Dossou', 'Joe Gelhardt', 'Joe Gomez', 'Joe Rodon', 'Joel Chukwuma Obi', 'Joel Ideho', 'Joel LÃ³pez Salguero', 'Joel Robles BlÃ¡zquez', 'Joel Ward', 'Joelinton Cassio ApolinÃ¡rio de Lira', 'Jofre Carreras PagÃ¨s', 'Johan AndrÃ©s Mojica Palacio', 'Johan Felipe VÃ¡squez Ibarra', 'Johan Gastien', 'Johann Berg GuÃ°mundsÂ­son', 'John Anthony Brooks', 'John Guidetti', 'John Joe Patrick Finn Benoa', 'John McGinn', 'John Nwankwo Chetauya Donald Okeh', 'John Ruddy', 'John Stones', 'Jokin Ezkieta Mendiburu', 'Jon Ander Garrido Moracia', 'Jon Guridi Aldalur', 'Jon McCracken', 'Jon Moncayola Tollar', 'Jon Morcillo Conesa', 'Jon Pacheco Dozagarat', 'Jon Sillero Monreal', 'JonÃ¡s Ramalho Chimeno', 'Jonas Hector', 'Jonas Hofmann', 'Jonas Kersken', 'Jonas Martin', 'Jonas Michelbrink', 'Jonas Omlin', 'Jonas Urbig', 'Jonatan Carmona Ãlamo', 'Jonathan Bamba', 'Jonathan Castro Otto', 'Jonathan Christian David', 'Jonathan Clauss', 'Jonathan Cristian Silva', 'Jonathan DamiÃ¡n Iglesias Abreu', 'Jonathan Gradit', 'Jonathan Grant Evans', 'Jonathan Michael Burkardt', 'Jonathan RodrÃ­guez MenÃ©ndez', 'Jonathan Russell', 'Jonathan Schmid', 'Jonathan Tah', 'Jonjo Shelvey', 'Jonjoe Kenny', 'JorÃ¨s Rahou', 'Jordan Ferri', 'Jordan Henderson', 'Jordan Holsgrove', 'Jordan KÃ©vin Amavi', 'Jordan Lotomba', 'Jordan Meyer', 'Jordan Pickford', 'Jordan Pierre Ayew', 'Jordan Tell', 'Jordan Torunarigha', 'Jordan Veretout', 'Jordan Zacharie Lukaku Menama Mokelenge', 'Jordi Alba Ramos', 'Jordi Bongard', 'Jordi Mboula Queralt', 'Jorge AndÃºjar Moreno', 'Jorge Cuenca Barreno', 'Jorge de Frutos SebastiÃ¡n', 'Jorge Filipe Soares Silva', 'Jorge MerÃ© PÃ©rez', 'Jorge MiramÃ³n Santagertrudis', 'Jorge Molina Vidal', 'Jorge Padilla Soler', 'Jorge ResurrecciÃ³n Merodio', 'Joris Chotard', 'Joris Gnagnon', 'JosÃ© Ãngel Carmona Navarro', 'JosÃ© Ãngel Esmoris Tasende', 'JosÃ© Ãngel GÃ³mez CampaÃ±a', 'JosÃ© Ãngel Pozo la Rosa', 'JosÃ© Ãngel ValdÃ©s DÃ­az', 'JosÃ© Alejandro MartÃ­n ValerÃ³n', 'JosÃ© Alonso Lara', 'JosÃ© AndrÃ©s Guardado HernÃ¡ndez', 'JosÃ© Antonio FerrÃ¡ndez Pomares', 'JosÃ© Antonio Morente Oliva', 'JosÃ© David Menargues', 'JosÃ© Diogo Dalot Teixeira', 'JosÃ© Ignacio FernÃ¡ndez Iglesias', 'JosÃ© Juan MacÃ­as GuzmÃ¡n', 'JosÃ© LuÃ­s GayÃ PeÃ±a', 'JosÃ© Luis Morales Nogales', 'JosÃ© Luis Palomino', 'JosÃ© Luis SanmartÃ­n Mato', 'JosÃ© Manuel Cabrera LÃ³pez', 'JosÃ© Manuel FontÃ¡n MondragÃ³n', 'JosÃ© Manuel Reina PÃ¡ez', 'JosÃ© Manuel RodrÃ­guez Benito', 'JosÃ© Manuel SÃ¡nchez GuillÃ©n', 'JosÃ© MarÃ­a CallejÃ³n Bueno', 'JosÃ© MarÃ­a GimÃ©nez de Vargas', 'JosÃ© MarÃ­a MartÃ­n-Bejarano Serrano', 'JosÃ© MarÃ­a Relucio Gallego', 'JosÃ© MartÃ­n CÃ¡ceres Silva', 'JosÃ© Miguel da Rocha Fonte', 'JosÃ© Pedro Malheiro de SÃ¡', 'JosÃ© RaÃºl GutiÃ©rrez Parejo', 'JosÃ© SÃ¡nchez MartÃ­nez', 'JosÃ© SalomÃ³n RondÃ³n GimÃ©nez', 'Joscha Wosz', 'Jose LuÃ­s GarcÃ­a VayÃ¡', 'Joseba ZaldÃºa Bengoetxea', 'Josep GayÃ MartÃ­nez', 'Josep MartÃ­nez Riera', 'Joseph Aidoo', 'Joseph Alfred Duncan', 'Joseph Scally', 'Joseph Shaun Hodge', 'Joseph Willock', 'Josh Brooking', 'Josh Brownhill', 'Josh Maja', 'Joshua Brenet', 'Joshua Christian Kojo King', 'Joshua Felix Okpoda Eppiah', 'Joshua Thomas Sargent', 'Joshua Walter Kimmich', 'Josip Brekalo', 'Josip IliÄiÄ‡', 'Josip StaniA¡iÄ‡', 'JosuÃ© Albert', 'Josuha Guilavogui', 'Juan AgustÃ­n Musso', 'Juan Antonio Iglesias SÃ¡nchez', 'Juan Bernat Velasco', 'Juan Camilo HernÃ¡ndez SuÃ¡rez', 'Juan Cruz Ãlvaro Armada', 'Juan Cruz DÃ­az EspÃ³sito', 'Juan Flere Pizzuti', 'Juan Guilherme Nunes Jesus', 'Juan Guillermo Cuadrado Bello', 'Juan Ignacio RamÃ­rez Polero', 'Juan Manuel Mata GarcÃ­a', 'Juan Manuel PÃ©rez Ruiz', 'Juan Marcos Foyth', 'Juan Miguel JimÃ©nez LÃ³pez', 'Juan Miranda GonzÃ¡lez', 'Juan Torres Ruiz', 'Jude Victor William Bellingham', 'Judilson Mamadu TuncarÃ¡ Gomes', 'Julen Agirrezabala', 'Jules KoundÃ©', 'Julian Albrecht', 'Julian Baumgartlinger', 'Julian Brandt', 'Julian Chabot', 'Julian Draxler', 'Julian Green', 'Julian Jeanvier', 'Julian Philipp Frommann', 'Julian Pollersbeck', 'Julian Ryerson', 'Julien Boyer', 'Julien Faussurier', 'Julien Laporte', 'Julius Pfennig', 'Junior Castello Lukeba', 'Junior Morau Kadile', 'Junior Wakalible Lago', 'Junior Walter Messias', 'Juraj Kucka', 'Jurgen Ekkelenkamp', 'Justin Hoogma', 'Justin Kluivert', 'Justin Smith', 'KÃ©vin Boma', 'KÃ©vin Gameiro', 'KÃ©vin Malcuit', 'KÃ©vin Manuel Rodrigues', 'KÃ©vin NDoram', 'Kaan Ayhan', 'Kaan Kurt', 'Kacper UrbaAski', 'Kai Lukas Havertz', 'Kaio Jorge Pinto Ramos', 'Kaito Mizuta', 'Kalidou Koulibaly', 'Kalifa Coulibaly', 'Kalvin Phillips', 'Kamaldeen Sulemana', 'Karim Azamoum', 'Karim Bellarabi', 'Karim Benzema', 'Karim Onisiwo', 'Karim Rekik', 'Karl Brillant Toko Ekambi', 'Karl Darlow', 'Karol Fila', 'Karol Linetty', 'Kasim Adams Nuhu', 'Kasper Dolberg Rasmussen', 'Kasper Peter Schmeichel', 'Kayky da Silva Chagas', 'Kays Ruiz-Atil', 'Keanan Bennetts', 'Keidi Bare', 'Keinan Davis', 'Keita BaldÃ© Diao', 'Kelechi Promise Iheanacho', 'Kelvin Amian Adou', 'Ken Nlata Sema', 'Ken Remi Stefan Strandberg', 'Kenny McLean', 'Kepa Arrizabalaga Revuelta', 'Kerem Demirbay', 'Keven Schlotterbeck', 'Kevin AndrÃ©s Agudelo Ardila', 'Kevin Behrens', 'Kevin Bonifazi', 'Kevin Danso', 'Kevin De Bruyne', 'Kevin John Ufuoma Akpoguma', 'Kevin Kampl', 'Kevin Lasagna', 'Kevin Long', 'Kevin MÃ¶hwald', 'Kevin Piscopo', 'Kevin RÃ¼egg', 'Kevin Schade', 'Kevin StÃ¶ger', 'Kevin Strootman', 'Kevin Trapp', 'Kevin VÃ¡zquez ComesaÃ±a', 'Kevin Vogt', 'Kevin Volland', 'Kevin-Prince Boateng', 'Keylor Navas Gamboa', 'Kgaogelo Chauke', 'KhÃ©phren Thuram-Ulien', 'Kieran Dowell', 'Kieran Tierney', 'Kieran Trippier', 'Kiernan Dewsbury-Hall', 'Ki-Jana Delano Hoever', 'Kiliann Sildillia', 'Kimberly Ezekwem', 'Kingsley Dogo Michael', 'Kingsley Ehizibue', 'Kingsley Fobi', 'Kingsley Junior Coman', 'Kingsley Schindler', 'Kjell Scherpen', 'Koba LeÃ¯n Koindredi', 'Koen Casteels', 'Konrad de la Fuente', 'Konrad Laimer', 'Konstantinos Manolas', 'Konstantinos Mavropanos', 'Konstantinos Stafylidis', 'Konstantinos Tsimikas', 'Koray GÃ¼nter', 'Kortney Hause', 'Kouadio-Yves Dabila', 'Kouassi Ryan Sessegnon', 'KrÃ©pin Diatta', 'Kristijan JakiÄ‡', 'Kristoffer Askildsen', 'Kristoffer Vassbakk Ajer', 'Kristoffer-August Sundquist Klaesson', 'Krisztofer HorvÃ¡th', 'Krzysztof PiÄ…tek', 'Kurt Happy Zouma', 'Kwadwo Baah', 'Kyle Alex John', 'Kyle Walker', 'Kyle Walker-Peters', 'Kylian MbappÃ© Lottin', 'LÃ¡szlÃ³ BÃ©nes', 'LÃ©o Dubois', 'LÃ©o Leroy', 'LÃ©o PÃ©trot', 'LÃ©vy Koffi Djidji', 'Lamare Bogarde', 'Landry Nany Dimata', 'Lars Edi Stindl', 'Lassana Coulibaly', 'Lasse GÃ¼nther', 'Lasse RieÃŸ', 'Lasse Schulz', 'Laurent Abergel', 'Laurent Koscielny', 'Laurenz Dehl', 'Lautaro de LeÃ³n Billar', 'Lautaro Javier MartÃ­nez', 'Lautaro Marco Spatz', 'Layvin Kurzawa', 'Lazar SamardA¾iÄ‡', 'Leander Dendoncker', 'Leandro Barreiro Martins', 'Leandro Daniel Cabrera SasÃ­a', 'Leandro Daniel Paredes', 'Leandro Trossard', 'Lebo Mothiba', 'Lee Grant', 'Lennart Czyborra', 'Lenny Jean-Pierre Pintor', 'Lenny Joseph', 'Lenny Lacroix', 'Leo Atulac', 'Leo Fuhr Hjelde', 'Leon Bailey Butler', 'Leon Christoph Goretzka', 'Leon Valentin Schaffran', 'Leonardo Bonucci', 'Leonardo CÃ©sar Jardim', 'Leonardo Capezzi', 'Leonardo de Souza Sena', 'Leonardo JuliÃ¡n Balerdi Rossa', 'Leonardo Mancuso', 'Leonardo Pavoletti', 'Leonardo RomÃ¡n Riquelme', 'Leonardo Spinazzola', 'Leroy Aziz SanÃ©', 'Lesley Chimuanya Ugochukwu', 'Levi Jeremiah Lumeka', 'Levin Mete Ã–ztunali', 'Lewis Baker', 'Lewis Bate', 'Lewis Dobbin', 'Lewis Dunk', 'Lewis Gordon', 'Lewis Paul Jimmy Richards', 'Lewis Richardson', 'Liam Cooper', 'Liam Delap', 'Liam Gibbs', 'Liam Henderson', 'Liam McCarron', 'Lilian Brassier', 'Lilian Egloff', 'Linus Gechter', 'Lionel AndrÃ©s Messi Cuccittini', 'Lisandru Tramoni', 'LluÃ­s Andreu i Ruiz', 'LluÃ­s Recasens Vives', 'LoÃ¯c BadÃ©', 'Lorenz Assignon', 'Lorenzo Andrenacci', 'Lorenzo De Silvestri', 'Lorenzo Insigne', 'Lorenzo JesÃºs MorÃ³n GarcÃ­a', 'Lorenzo MontipÃ²', 'Lorenzo Pellegrini', 'Lorenzo Tonelli', 'Lorenzo Venuti', 'Loris Karius', 'Loris Mouyokolo', 'Louis Jordan Beyer', 'Louis Munteanu', 'Louis Schaub', 'Lovro Majer', 'Luan Peres Petroni', 'LuÃ­s Manuel Arantes Maximiano', 'Luca Ceppitelli', 'Luca Jannis Kilian', 'Luca Lezzerini', 'Luca Netz', 'Luca Palmiero', 'Luca Pellegrini', 'Luca Philipp', 'Luca Ranieri', 'Luca Zinedine Zidane', 'Lucas Ariel BoyÃ©', 'Lucas Ariel Ocampos', 'Lucas BergstrÃ¶m', 'Lucas Bonelli', 'Lucas Da Cunha', 'Lucas Digne', 'Lucas FranÃ§ois Bernard HernÃ¡ndez Pi', 'Lucas Gourna-Douath', 'Lucas HÃ¶ler', 'Lucas Margueron', 'Lucas MartÃ­nez Quarta', 'Lucas NicolÃ¡s Alario', 'Lucas PÃ©rez MartÃ­nez', 'Lucas Perrin', 'Lucas Pezzini Leiva', 'Lucas Rodrigues Moura da Silva', 'Lucas Silva Melo', 'Lucas Simon Pierre Tousart', 'Lucas Tolentino Coelho de Lima', 'Lucas TorrÃ³ Marset', 'Lucas Torreira Di Pascua', 'Lucas VÃ¡zquez Iglesias', 'Lucien Jefferson Agoume', 'Ludovic Ajorque', 'Ludovic Blas', 'Luis Alberto Romero Alconchel', 'Luis Alberto SuÃ¡rez DÃ­az', 'Luis Alfonso Abram Ugarelli', 'Luis Alfonso Espino GarcÃ­a', 'Luis Carbonell Artajona', 'Luis Enrique Carrasco Acosta', 'Luis Ezequiel Ãvila', 'Luis Federico LÃ³pez AndÃºgar', 'Luis Fernando Muriel Fruto', 'Luis Hartwig', 'Luis Henrique Tomaz de Lima', 'Luis Javier SuÃ¡rez Charris', 'Luis JesÃºs Rioja GonzÃ¡lez', 'Luis Milla Manzanares', 'Luis Thomas Binks', 'Luiz Felipe Ramos Marchi', 'Luiz Frello Filho Jorge', 'Luka Bogdan', 'Luka JoviÄ‡', 'Luka MilivojeviÄ‡', 'Luka ModriÄ‡', 'Luka RaÄiÄ‡', 'LukÃ¡A¡ HaraslÃ­n', 'LukÃ¡A¡ HrÃ¡deckÃ½', 'Lukas KÃ¼bler', 'Lukas KlÃ¼nter', 'Lukas Manuel Klostermann', 'Lukas Nmecha', 'Lukas Rupp', 'Luke Ayling', 'Luke Bolton', 'Luke James Cundle', 'Luke Matheson', 'Luke Mbete', 'Luke Shaw', 'Luke Thomas', 'Luuk de Jong', 'Lyanco Evangelista Silveira Neves VojnoviÄ‡', 'MÃ¡rio Rui Silva Duarte', 'MÃ¡rton DÃ¡rdai', 'MÃ«rgim Vojvoda', 'MÃ©saque Geremias DjÃº', 'Mads Bech SÃ¸rensen', 'Mads Bidstrup', 'Mads Pedersen', 'Mads Roerslev Rasmussen', 'Magnus Warming', 'MahamÃ© Siby', 'Mahdi Camara', 'Mahmoud Ahmed Ibrahim Hassan', 'Mahmoud Dahoud', 'Maksim Paskotsi', 'Malachi Fagan-Walcott', 'Malang Sarr', 'Malcolm Barcola', 'Malcom Bokele', 'Malik Tillman', 'Malo Gusto', 'Mama Samba BaldÃ©', 'Mamadou Camara', 'Mamadou Coulibaly', 'Mamadou DoucourÃ©', 'Mamadou Lamine Gueye', 'Mamadou Loum NDiaye', 'Mamadou Sakho', 'Mamadou Sylla Diallo', 'Mamor Niang', 'Manolo Gabbiadini', 'Manolo Portanova', 'Manuel Agudo DurÃ¡n', 'Manuel Cabit', 'Manuel GarcÃ­a Alonso', 'Manuel Gulde', 'Manuel Javier Vallejo GalvÃ¡n', 'Manuel Lanzini', 'Manuel Lazzari', 'Manuel Locatelli', 'Manuel Morlanes AriÃ±o', 'Manuel Navarro SÃ¡nchez', 'Manuel Nazaretian', 'Manuel Obafemi Akanji', 'Manuel Peter Neuer', 'Manuel Prietl', 'Manuel Reina RodrÃ­guez', 'Manuel Riemann', 'Manuel SÃ¡nchez de la PeÃ±a', 'Manuel Trigueros MuÃ±oz', 'Marash Kumbulla', 'Marc Albrighton', 'Marc Bartra Aregall', 'Marc Cucurella Saseta', 'Marc GuÃ©hi', 'Marc Roca JunquÃ©', 'Marc-AndrÃ© ter Stegen', 'Marc-AurÃ¨le Caillard', 'Marcel Edwin Rodrigues Lavinier', 'Marcel Halstenberg', 'Marcel Sabitzer', 'Marcel Schmelzer', 'Marcelo AntÃ´nio Guedes Filho', 'Marcelo BrozoviÄ‡', 'Marcelo Josemir Saracchi Pintos', 'Marcelo Pitaluga', 'Marcelo Vieira da Silva JÃºnior', 'Marcin BuAka', 'Marco Asensio Willemsen', 'Marco Benassi', 'Marco Bizot', 'Marco Davide Faraoni', 'Marco John', 'Marco MeyerhÃ¶fer', 'Marco Modolo', 'Marco Reus', 'Marco Richter', 'Marco Silvestri', 'Marco Sportiello', 'Marco Verratti', 'Marc-Oliver Kempf', 'Marcos Alonso Mendoza', 'Marcos AndrÃ© de Sousa MendonÃ§a', 'Marcos AoÃ¡s CorrÃªa', 'Marcos Javier AcuÃ±a', 'Marcos Llorente Moreno', 'Marcos Mauro LÃ³pez GutiÃ©rrez', 'Marcus Bettinelli', 'Marcus Coco', 'Marcus Forss', 'Marcus Ingvartsen', 'Marcus Lilian Thuram-Ulien', 'Marcus Rashford', 'Mariano DÃ­az MejÃ­a', 'Marin PongraÄiÄ‡', 'Mario Gaspar PÃ©rez MartÃ­nez', 'Mario Hermoso Canseco', 'Mario HernÃ¡ndez FernÃ¡ndez', 'Mario PaA¡aliÄ‡', 'Mario RenÃ© Junior Lemina', 'Mario SuÃ¡rez Mata', 'Marius Adamonis', 'Marius Funk', 'Marius Liesegang', 'Marius Wolf', 'Mark Flekken', 'Mark Gillespie', 'Mark Helm', 'Mark Noble', 'Mark Uth', 'Marko ArnautoviÄ‡', 'Marko DmitroviÄ‡', 'Marko Pjaca', 'Marko Rog', 'Marshall Nyasha Munetsi', 'MartÃ­n Aguirregabiria Padilla', 'MartÃ­n Manuel CalderÃ³n GÃ³mez', 'MartÃ­n Merquelanz Castellanos', 'MartÃ­n Montoya Torralbo', 'MartÃ­n Pascual Castillo', 'MartÃ­n Satriano', 'MartÃ­n Zubimendi IbÃ¡Ã±ez', 'Marten Elco de Roon', 'Martin Ã˜degaard', 'Martin Braithwaite Christensen', 'Martin DÃºbravka', 'Martin ErliÄ‡', 'Martin Hinteregger', 'Martin Hongla Yma', 'Martin Kelly', 'Martin PeÄar', 'Martin Terrier', 'Martin Valjent', 'Marvelous Nakamba', 'Marvin Ayhan Obuz', 'Marvin Elimbi', 'Marvin Friedrich', 'Marvin Olawale Akinlabi Park', 'Marvin Plattenhardt', 'Marvin SchwÃ¤be', 'Marvin Stefaniak', 'Marvin Zeegelaar', 'Marwin Hitz', 'Mason Greenwood', 'Mason Holgate', 'Mason Mount', 'Massadio HaÃ¯dara', 'MatÄ›j Vydra', 'MatÃ­as Ezequiel Dituro', 'MatÃ­as Vecino Falero', 'MatÃ­as ViÃ±a', 'Mateo Klimowicz', 'Mateo KovaÄiÄ‡', 'Mateu Jaume Morey BauzÃ¡', 'Mateusz Andrzej Klich', 'MathÃ­as Olivera Miramontes', 'MathÃ­as SebastiÃ¡n SuÃ¡rez SuÃ¡rez', 'Matheus Henrique de Souza', 'Matheus Pereira da Silva', 'Matheus Santos Carneiro Da Cunha', 'Matheus Soares Thuler', 'Mathew David Ryan', 'Mathias Antonsen Normann', 'Mathias Jattah-Njie JÃ¸rgensen', 'Mathias Jensen', 'Mathias Pereira Lage', 'Mathieu Cafaro', 'Mathis Bruns', 'Mathys Saban', 'Matija NastasiÄ‡', 'Matis Carvalho', 'Mato Jajalo', 'Matondo-Merveille Papela', 'Mats Hummels', 'Matt Ritchie', 'Matt Targett', 'MattÃ©o Elias Kenzo Guendouzi OliÃ©', 'Matteo Cancellieri', 'Matteo Darmian', 'Matteo Gabbia', 'Matteo Lovato', 'Matteo Pessina', 'Matteo Politano', 'Matteo Ruggeri', 'Matthew Bondswell', 'Matthew Hoppe', 'Matthew James Doherty', 'Matthew Lowton', 'Matthew Miazga', 'Matthias Ginter', 'Matthias KÃ¶bbing', 'Matthieu Dreyer', 'Matthieu Udol', 'Matthijs de Ligt', 'Matthis Abline', 'Mattia Aramu', 'Mattia Bani', 'Mattia Caldara', 'Mattia De Sciglio', 'Mattia Destro', 'Mattia Pagliuca', 'Mattia Perin', 'Mattia Viti', 'Mattia Zaccagni', 'Mattias Olof Svanberg', 'Matty Cash', 'Matz Sels', 'Maurice Dominick ÄŒoviÄ‡', 'Maurizio Pochettino', 'Mauro Emanuel Icardi Rivero', 'Mauro Wilney Arambarri Rosa', 'Max Bennet Kruse', 'Max Christiansen', 'Max Svensson RÃ­o', 'Max Thompson', 'Maxence Caqueret', 'Maxence Lacroix', 'Maxence Rivera', 'Maxim Leitsch', 'Maxime EstÃ¨ve', 'Maxime Gonalons', 'Maxime Le Marchand', 'Maxime Lopez', 'Maximilian Arnold', 'Maximilian Bauer', 'Maximilian Eggestein', 'Maximilian Kilman', 'Maximilian MittelstÃ¤dt', 'Maximilian Philipp', 'Maximiliano GÃ³mez GonzÃ¡lez', 'Maximillian James Aarons', 'Maxwell Haygarth', 'MBala Nzola', 'MBaye Babacar Niang', 'Mehdi Bourabia', 'Mehdi Zerkane', 'Mehmet Ibrahimi', 'Mehmet Zeki Ã‡elik', 'Meiko Sponsel', 'Melayro Chakewno Jalaino Bogarde', 'Melingo Kevin Mbabu', 'Melvin Michel Maxence Bard', 'Memphis Depay', 'Merih Demiral', 'Meritan Shabani', 'Mert MÃ¼ldÃ¼r', 'Mert-Yusuf Torlak', 'Metehan GÃ¼Ã§lÃ¼', 'MichaÃ«l Bruno Dominique Cuisance', 'Michael Esser', 'Michael Gregoritsch', 'Michael Keane', 'Michael McGovern', 'Michael Olise', 'Michael Svoboda', 'Michail Antonio', 'MickaÃ«l NadÃ©', 'MickaÃ«l Ramon Malsa', 'Micky van de Ven', 'Miguel Ãngelo da Silva Rocha', 'Miguel Ãngel AlmirÃ³n Rejala', 'Miguel Ãngel Leal DÃ­az', 'Miguel Ãngel Trauco Saavedra', 'Miguel Baeza PÃ©rez', 'Miguel de la Fuente Escudero', 'Miguel GutiÃ©rrez Ortega', 'Miguel Juan Llambrich', 'Miguel LuÃ­s Pinto Veloso', 'Mihailo RistiÄ‡', 'Mijat GaÄ‡inoviÄ‡', 'Mika SchrÃ¶ers', 'Mike Maignan', 'Mikel Balenziaga Oruesagasti', 'Mikel Merino ZazÃ³n', 'Mikel Oyarzabal Ugarte', 'Mikel Vesga Arruti', 'Mikkel Krogh Damsgaard', 'Milan Akriniar', 'Milan ÄuriÄ‡', 'Milan Badelj', 'MiloA¡ PantoviÄ‡', 'Milot Rashica', 'Milutin OsmajiÄ‡', 'Mitchel Bakker', 'Mitchell Dijks', 'Mitchell van Bergen', 'MoÃ¯se Dion Sahi', 'Mohamed Amine Elyounoussi', 'Mohamed Amine Ihattaren', 'Mohamed Lamine Bayo', 'Mohamed Naser Elsayed Elneny', 'Mohamed SaÃ¯d Benrahma', 'Mohamed Salah Ghaly', 'Mohamed Salim Fares', 'Mohamed Salisu Abdul Karim', 'Mohamed Simakan', 'Mohamed-Ali Cho', 'Mohammed Sangare', 'MoisÃ©s GÃ³mez Bordonado', 'Moise Bioty Kean', 'Molla WaguÃ©', 'Moreto Moro CassamÃ¡', 'Morgan Boyes', 'Morgan Sanson', 'Morgan Schneiderlin', 'Moriba Kourouma Kourouma', 'Moritz Jenz', 'Morten Thorsby', 'Moses Daddy-Ajala Simon', 'Mouctar Diakhaby', 'Moussa DembÃ©lÃ©', 'Moussa Diaby', 'Moussa Djenepo', 'Moussa Doumbia', 'Moussa NiakhatÃ©', 'Moussa Sissoko', 'Moussa WaguÃ©', 'Moustapha Mbow', 'Munas Dabbur', 'Munir El Haddadi Mohamed', 'Musa Barrow', 'Myles Peart-Harris', 'Myron Boadu', 'Myziane Maolida', 'NÃ©lson Cabral Semedo', 'NÃ©stor Alejandro AraÃºjo Razo', 'NaÃ«l Jaby', 'Nabil Fekir', 'Nabili Zoubdi Touaizi', 'Naby KeÃ¯ta', 'Nadiem Amiri', 'Nadir Zortea', 'Nahitan Michel NÃ¡ndez Acosta', 'Nahuel Molina Lucero', 'Nahuel Noll', 'Nampalys Mendy', 'Nanitamo Jonathan IkonÃ©', 'Naouirou Ahamada', 'Nassim Chadli', 'Nathan AkÃ©', 'Nathan Bitumazala', 'Nathan De Medina', 'Nathan Ferguson', 'Nathan Michael Collins', 'Nathan Redmond', 'Nathan Tella', 'NathanaÃ«l Mbuku', 'Nathaniel Edwin Clyne', 'Nathaniel Phillips', 'Nayef Aguerd', 'NDri Philippe Koffi', 'Neal Maupay', 'Neco Williams', 'Nedim Bajrami', 'Nemanja Gudelj', 'Nemanja MaksimoviÄ‡', 'Nemanja MatiÄ‡', 'Nemanja Radoja', 'Neyder Yessy Lozano RenterÃ­a', 'Neymar da Silva Santos JÃºnior', 'NGolo KantÃ©', 'NGuessan Rominigue KouamÃ©', 'Nicholas Gioacchini', 'Nicholas Williams Arthuer', 'Nick Pope', 'Nick Viergever', 'Nico Elvedi', 'Nico Schlotterbeck', 'Nico Schulz', 'Nicola Domenico Sansone', 'Nicola Murru', 'Nicola Ravaglia', 'Nicola Zalewski', 'NicolÃ¡s GonzÃ¡lez Iglesias', 'NicolÃ¡s IvÃ¡n GonzÃ¡lez', 'NicolÃ¡s MartÃ­n DomÃ­nguez', 'NicolÃ¡s Melamed Ribaudo', 'NicolÃ¡s Serrano Galdeano', 'NicolÃ² Barella', 'NicolÃ² Casale', 'NicolÃ² Fagioli', 'NicolÃ² Rovella', 'NicolÃ² Zaniolo', 'Nicolas De PrÃ©ville', 'Nicolas HÃ¶fler', 'Nicolas Louis Marcel Cozza', 'Nicolas PÃ©pÃ©', 'Nicolas Pallois', 'Nicolas Penneteau', 'Nicolas Thibault Haas', 'Niki Emil Antonio MÃ¤enpÃ¤Ã¤', 'Nikita Iosifov', 'Niklas Bernd Dorsch', 'Niklas Hauptmann', 'Niklas Klinger', 'Niklas Lomb', 'Niklas SÃ¼le', 'Niklas Stark', 'Niklas Tauer', 'Niko GieÃŸelmann', 'Nikola KaliniÄ‡', 'Nikola MaksimoviÄ‡', 'Nikola MaraA¡', 'Nikola MilenkoviÄ‡', 'Nikola VlaA¡iÄ‡', 'Nikola VukÄeviÄ‡', 'Nikolas Terkelsen Nartey', 'Nile Omari Mckenzi John', 'Nils Petersen', 'Nils Seufert', 'Nils-Jonathan KÃ¶rber', 'Nishan Connell Burkart', 'Nnamdi Collins', 'NoÃ© Sow', 'Noah Atubolu', 'Noah Fatar', 'Noah Joel Sarenren Bazee', 'Noah KÃ¶nig', 'Noah Katterbach', 'Noah Nadje', 'Noah WeiÃŸhaupt', 'Norbert GyÃ¶mbÃ©r', 'Norberto Bercique Gomes Betuncal', 'Norberto Murara Neto', 'Nordi Mukiele Mulere', 'Nuno Albertino Varela Tavares', 'Nuno Alexandre Tavares Mendes', 'Nya Jerome Kirby', 'Obite Evan NDicka', 'Odel Offiah', 'Odilon Kossounou', 'Odsonne Ã‰douard', 'Oghenekaro Peter Etebo', 'Ohis Felix Uduokhai', 'Oier OlazÃ¡bal Paredes', 'Oier Sanjurjo MatÃ©', 'Oier Zarraga EgaÃ±a', 'Oihan Sancet Tirapu', 'Okay YokuAŸlu', 'Oleksandr Zinchenko', 'Oliver Batista Meier', 'Oliver Baumann', 'Oliver Bosworth', 'Oliver Christensen', 'Oliver Skipp', 'Oliver Webber', 'Olivier Giroud', 'Ollie Watkins', 'Oludare Olufunwa', 'Omar Colley', 'Omar El Hilali', 'Omar Federico Alderete FernÃ¡ndez', 'Omar Khaled Mohamed Marmoush', 'Omar Mascarell GonzÃ¡lez', 'Omar Tyrell Crawford Richards', 'Omer Hanin', 'Ondrej Duda', 'Onyinye Wilfred Ndidi', 'Opa Nguette', 'Orel Mangala', 'Orestis Spyridon Karnezis', 'Oriol Busquets Mas', 'Oriol Romeu Vidal', 'Orlando RubÃ©n YÃ¡Ã±ez Alabart', 'Osman Bukari', 'Ossama Ashley', 'Osvaldo Pedro Capemba', 'OtÃ¡vio Henrique Passos Santos', 'Oualid El Hajjam', 'Ouparine Djoco', 'Ousmane Ba', 'Ousmane DembÃ©lÃ©', 'Oussama Idrissi', 'Oussama Targhalline', 'Owen Dodgson', 'Ozan Muhammed Kabak', 'Ozan Tufan', 'PÃ©pÃ© Bonet Kapambu', 'PÃ©ter GulÃ¡csi', 'Pablo ÃÃ±iguez de Heredia Larraz', 'Pablo Carmine Maffeo Becerra', 'Pablo Daniel Piatti', 'Pablo Fornals Malla', 'Pablo GÃ¡lvez Miranda', 'Pablo GozÃ¡lbez Gilabert', 'Pablo IbÃ¡Ã±ez Lumbreras', 'Pablo MarÃ­ Villar', 'Pablo MartÃ­n PÃ¡ez Gavira', 'Pablo MartÃ­n PicÃ³n Ãlvaro', 'Pablo MartÃ­nez AndrÃ©s', 'Pablo PÃ©rez Rico', 'Pablo Paulino Rosario', 'Pablo Valencia GarcÃ­a', 'Panagiotis Retsos', 'Paolo Ghiglione', 'Paolo Pancrazio FaragÃ²', 'Paolo Sciortino', 'Pape Alassane Gueye', 'Pape Cheikh Diop Gueye', 'Pape Matar Sarr', 'Pape Ndiaga Yade', 'Pascal GroÃŸ', 'Pascal Juan Estrada', 'Pascal Stenzel', 'Pascal Struijk', 'Pasquale Mazzocchi', 'Patricio GabarrÃ³n Gil', 'Patricio Nehuen PÃ©rez', 'Patrick Bamford', 'Patrick Cutrone', 'Patrick Herrmann', 'Patrick Osterhage', 'Patrick Roberts', 'Patrick Wimmer', 'Patrik Schick', 'Patryk Dziczek', 'Patson Daka', 'Pau Francisco Torres', 'Pau LÃ³pez Sabata', 'Paul Baysse', 'Paul Dummett', 'Paul Grave', 'Paul Jaeckel', 'Paul Jean FranÃ§ois Bernardoni', 'Paul Nardi', 'Paul Nebel', 'Paul Pogba', 'Paul Seguin', 'Paulo Bruno Exequiel Dybala', 'Paulo Henrique Sampaio Filho', 'Paulo OtÃ¡vio Rosa da Silva', 'Pavao Pervan', 'Pavel KadeA™Ã¡bek', 'PaweA Kamil JaroszyAski', 'PaweA Marek Dawidowicz', 'PaweA Marek WszoAek', 'Pedro Bigas Rigo', 'Pedro Chirivella Burgos', 'Pedro Eliezer RodrÃ­guez Ledesma', 'Pedro Filipe TeodÃ³sio Mendes', 'Pedro GonzÃ¡lez LÃ³pez', 'Pedro Lomba Neto', 'Pedro Mba Obiang Avomo', 'Pedro Ortiz Bernat', 'Pelenda Joshua Tunga Dasilva', 'Pere Joan GarcÃ­a BauzÃ', 'Pere Milla PeÃ±a', 'Pere Pons Riera', 'Peru Nolaskoain Esnal', 'Pervis JosuÃ© EstupiÃ±Ã¡n Tenorio', 'Petar MiÄ‡in', 'Petar StojanoviÄ‡', 'Petar Zovko', 'Peter PekarÃ­k', 'Petko Hristov', 'Phil Bardsley', 'Phil Jones', 'Philana Tinotenda Kadewere', 'Philip Ankhrah', 'Philip Foden', 'Philipp FÃ¶rster', 'Philipp Lienhart', 'Philipp Pentke', 'Philipp Schulze', 'Philipp Tschauner', 'Philippe Coutinho Correia', 'Philippe Sandler', 'Phillipp Klement', 'Pierluigi Gollini', 'Piero MartÃ­n HincapiÃ© Reyna', 'Pierre Kazeye Rommel Kalulu Kyatengwa', 'Pierre Lees-Melou', 'Pierre-Emerick Emiliano FranÃ§ois Aubameyang', 'Pierre-Emile Kordt HÃ¸jbjerg', 'Pierre-Emmanuel Ekwah Elimby', 'Pierre-Yves Hamel', 'Pierrick Capelle', 'Pietro Boer', 'Pietro Ceccaroni', 'Pietro Pellegri', 'Pietro Terracciano', 'Piotr Sebastian ZieliAski', 'Pol Mikel Lirola Kosok', 'Pontus Jansson', 'Predrag RajkoviÄ‡', 'Presnel Kimpembe', 'PrzemysAaw Frankowski', 'PrzemysAaw PAacheta', 'Quentin Boisgard', 'Quentin Merlin', 'RÃ©mi Oudin', 'RÃ©my Descamps', 'RaÃºl Albiol Tortajada', 'RaÃºl Alonso JimÃ©nez RodrÃ­guez', 'RaÃºl De TomÃ¡s GÃ³mez', 'RaÃºl GarcÃ­a de Haro', 'RaÃºl GarcÃ­a Escudero', 'RaÃºl Moro Prescoli', 'RaÃºl Torrente Navarro', 'Radamel Falcao GarcÃ­a ZÃ¡rate', 'Rade KruniÄ‡', 'RadosAaw Majecki', 'Radu Matei DrÄƒguÈ™in', 'RafaA Gikiewicz', 'Rafael AlcÃ¢ntara do Nascimento', 'Rafael Alexandre ConceiÃ§Ã£o LeÃ£o', 'Rafael Czichos', 'Rafael JimÃ©nez Jarque', 'Rafael Mir Vicente', 'Rafael Santos BorrÃ© Maury', 'Rafael TolÃ³i', 'Rafiki Said Ahamada', 'RÄƒzvan Gabriel Marin', 'Ragnar Ache', 'Raheem Sterling', 'RamÃ³n RodrÃ­guez JimÃ©nez', 'Ramiro Guerra Pereyra', 'Ramon Olamilekan Azeez', 'Ramy Bensebaini', 'Ramzi Aya', 'Randal Kolo Muani', 'Randy Nteka', 'Rani Khedira', 'RÃºben Diogo da Silva Neves', 'RÃºben dos Santos Gato Alves Dias', 'RÃºben Miguel Nunes Vezo', 'Raoul Bellanova', 'RaphaÃ«l Adelino JosÃ© Guerreiro', 'RaphaÃ«l Varane', 'Raphael Dias Belloli', 'Raphael Framberger', 'Rayan AÃ¯t Nouri', 'Rayan Mathis Cherki', 'Redwan BourlÃ¨s', 'Reece Hannam', 'Reece James', 'Reece Oxford', 'Reinier Jesus Carvalho', 'Reinildo Isnard Mandava', 'Remi Matthews', 'Remi Savage', 'Remo Marco Freuler', 'Renan Augusto Lodi dos Santos', 'Renato Fabrizio Tapia Cortijo', 'Renato JÃºnior Luz Sanches', 'Renato Steffen', 'Renaud Emond', 'Renaud Ripart', 'Rey Manaj', 'Ricard Puig MartÃ­', 'Ricardo Domingos Barbosa Pereira', 'Ricardo IvÃ¡n RodrÃ­guez Araya', 'Ricardo LuÃ­s Chaby Mangas', 'Riccardo Bocalon', 'Riccardo Calafiori', 'Riccardo Fiamozzi', 'Riccardo Gagliolo', 'Riccardo Ladinetti', 'Riccardo Marchizza', 'Riccardo Orsolini', 'Riccardo Saponara', 'Riccardo Sottil', 'Richarlison de Andrade', 'Rick Karsdorp', 'Rick van Drongelen', 'Rico Henry', 'Ridgeciano Delano Haps', 'Ridle Bote Baku', 'Riyad Mahrez', 'Riza Durmisi', 'Rob Elliot', 'Rob Holding', 'Robert Andrich', 'Robert Gumny', 'Robert Harker', 'Robert Lewandowski', 'Robert Lynch SanchÃ©z', 'Robert Navarro MuÃ±oz', 'Robert Skov', 'Robert Street', 'Robert Tesche', 'Roberto Firmino Barbosa de Oliveira', 'Roberto Gagliardini', 'Roberto GonzÃ¡lez BayÃ³n', 'Roberto IbÃ¡Ã±ez Castro', 'Roberto Massimo', 'Roberto Maximiliano Pereyra', 'Roberto Piccoli', 'Roberto Pirrello', 'Roberto Soldado Rillo', 'Roberto Soriano', 'Roberto SuÃ¡rez Pier', 'Roberto Torres Morales', 'Robin Everardus Gosens', 'Robin Friedrich', 'Robin Hack', 'Robin Knoche', 'Robin Koch', 'Robin Le Normand', 'Robin Luca Kehr', 'Robin Zentner', 'Robson Alves de Barros', 'Rocco Ascone', 'Rocky Bushiri Kisonga', 'Rodrigo AndrÃ©s Battaglia', 'Rodrigo Bentancur ColmÃ¡n', 'Rodrigo HernÃ¡ndez Cascante', 'Rodrigo Javier De Paul', 'Rodrigo Moreno Machado', 'Rodrigo Nascimento FranÃ§a', 'Rodrigo SÃ¡nchez RodrÃ­guez', 'Rodrigue Casimir Ninga', 'Rodrygo Silva de Goes', 'RogÃ©rio Oliveira da Silva', 'Roger IbaÃ±ez Da Silva', 'Roger MartÃ­ Salvador', 'Rok VodiA¡ek', 'Roland Sallai', 'Rolando Mandragora', 'Roli Pereira de Sa', 'Romain Del Castillo', 'Romain Faivre', 'Romain Hamouma', 'Romain Jules Salin', 'Romain Perraud', 'Romain SaÃ¯ss', 'Romain Thomas', 'Roman BÃ¼rki', 'Romario RÃ¶sch', 'Romelu Lukaku Menama', 'Romeo Lavia', 'RonaÃ«l Julien Pierre-Gabriel', 'Ronald Federico AraÃºjo da Silva', 'Ronaldo Augusto Vieira Nan', 'Ross Barkley', 'RubÃ©n Blanco Veiga', 'RubÃ©n de TomÃ¡s GÃ³mez', 'RubÃ©n Duarte SÃ¡nchez', 'RubÃ©n GarcÃ­a Santos', 'RubÃ©n PeÃ±a JimÃ©nez', 'RubÃ©n Rochina Naixes', 'RubÃ©n Sobrino Pozuelo', 'Ruben Aguilar', 'Ruben Estephan Vargas MartÃ­nez', 'Ruben Loftus-Cheek', 'Rui Pedro dos Santos PatrÃ­cio', 'Rui Tiago Dantas da Silva', 'Rune Almenning Jarstein', 'Ruslan Malinovskyi', 'Ruwen WerthmÃ¼ller', 'Ryad Boudebouz', 'Ryan Astley', 'Ryan Bertrand', 'Ryan Bouallak', 'Ryan Cassidy', 'Ryan Finnigan', 'Ryan Fraser', 'Ryan Fredericks', 'SÃ©amus Coleman', 'SÃ©bastien Cibois', 'SÃ©bastien Corchia', 'SÃ©bastien RÃ©not', 'SÃ©kou Mara', 'SaA¡a KalajdA¾iÄ‡', 'SaA¡a LukiÄ‡', 'SaÃ¯dou Sow', 'SaÃ®f-Eddine Khaoui', 'Saad Agouzoul', 'SaÃºl ÃÃ­guez EsclÃ¡pez', 'SaÃºl GarcÃ­a Cabrero', 'Sacha Delaye', 'Sada Thioub', 'Sadik Fofana', 'Sadio ManÃ©', 'Salih Ã–zcan', 'Salim Ben Seghir', 'Salis Abdul Samed', 'Salomon Junior Sambia', 'Salvador Ferrer Canals', 'Salvador SÃ¡nchez Ponce', 'Salvador Sevilla LÃ³pez', 'Salvatore Sirigu', 'Sam Byram', 'Sam Greenwood', 'Sam Lammers', 'Sam McClelland', 'Sam McQueen', 'Saman Ghoddos', 'Sambou Sissoko', 'Samir Caetano de Souza Santos', 'Samir HandanoviÄ', 'Samuel Castillejo Azuaga', 'Samuel Chimerenka Chukwueze', 'Samuel Edozie', 'Samuel Kalu Ojim', 'Samuel Loric', 'Samuel Moutoussamy', 'Samuel Yves Umtiti', 'Samuele Damiani', 'Samuele Ricci', 'Sander Johan Christiansen', 'Sandro RamÃ­rez Castillo', 'Sandro Tonali', 'Sanjin PrciÄ‡', 'Santiago Arias Naranjo', 'Santiago Arzamendia Duarte', 'Santiago ComesaÃ±a Veiga', 'Santiago Eneme Bocari', 'Santiago Lionel AscacÃ­bar', 'Santiago Mina Lorenzo', 'Santiago RenÃ© MuÃ±Ã³z Robles', 'Sargis Adamyan', 'Sascha Burchert', 'Saulo Igor Decarli', 'Sava-Arangel ÄŒestiÄ‡', 'Scott Brian Banks', 'Scott Carson', 'Scott McTominay', 'Sead KolaA¡inac', 'Sean Longstaff', 'Sean McGurk', 'Sebastiaan Bornauw', 'Sebastian Andersson', 'Sebastian De Maio', 'Sebastian Griesbeck', 'Sebastian Polter', 'Sebastian Rode', 'Sebastian Rudy', 'Sebastian Vasiliadis', 'Sebastian Wiktor Walukiewicz', 'Sebastiano Luperto', 'Seko Fofana', 'Sepe Elye Wahi', 'Serge David Gnabry', 'Sergej MilinkoviÄ‡-SaviÄ‡', 'Sergi CanÃ³s TenÃ©s', 'Sergi Darder Moll', 'Sergi GÃ³mez SolÃ', 'SergiÃ±o Gianni Dest', 'Sergio Arratia Lechosa', 'Sergio Arribas Calvo', 'Sergio Asenjo AndrÃ©s', 'Sergio Barcia Larenxeira', 'Sergio Busquets i Burgos', 'Sergio Camus Perojo', 'Sergio Canales Madrazo', 'Sergio Duvan CÃ³rdova Lezama', 'Sergio Escudero Palomo', 'Sergio Guardiola Navarro', 'Sergio Guerrero Romero', 'Sergio Herrera PirÃ³n', 'Sergio Leonel AgÃ¼ero del Castillo', 'Sergio Lozano Lluch', 'Sergio Moreno MartÃ­nez', 'Sergio Postigo Redondo', 'Sergio Ramos GarcÃ­a', 'Sergio ReguilÃ³n RodrÃ­guez', 'Sergio Rico GonzÃ¡lez', 'Sergio Roberto Carnicer', 'Serhou Yadaly Guirassy', 'Shandon Baptiste', 'Shane Patrick Long', 'Shane Patrick Michael Duffy', 'Sheraldo Becker', 'Shkodran Mustafi', 'Shola Maxwell Shoretire', 'Sikou NiakatÃ©', 'Sil Swinkels', 'Silas Katompa Mvumpa', 'SilvÃ¨re Ganvoula Mboussy', 'Silvan Dominic Widmer', 'Simeon Tochukwu Nwankwo', 'Simon Asta', 'Simon Brady Ngapandouetnbu', 'Simon Thorup KjÃ¦r', 'Simon Zoller', 'Simone Aresti', 'Simone Bastoni', 'Simone Edera', 'Simone Romagnoli', 'Simone Verdi', 'Simone Zaza', 'Sinaly DiomandÃ©', 'Sofian Kiyine', 'Sofiane Alakouch', 'Sofiane Boufal', 'Sofiane Diop', 'Sofyan Amrabat', 'Solomon March', 'Soma Zsombor Novothny', 'Souleyman Doumbia', 'Soumaila Coulibaly', 'StÃ©phane Bahoken', 'StÃ©phane Imad Diarra', 'Stanislav Lobotka', 'Stefan Bajic', 'Stefan Bell', 'Stefan de Vrij', 'Stefan Ilsanker', 'Stefan Lainer', 'Stefan MitroviÄ‡', 'Stefan Ortega Moreno', 'Stefan Posch', 'Stefan SaviÄ‡', 'Stefano Sabelli', 'Stefano Sensi', 'Stefano Sturaro', 'Stefanos Kapino', 'Steffen Tigges', 'Stephan El Shaarawy', 'Stephan FÃ¼rstner', 'Stephy Alvaro Mavididi', 'Stevan JovetiÄ‡', 'Steve Mandanda', 'Steve Michel MouniÃ©', 'Steven Alzate', 'Steven Charles Bergwijn', 'Steven NKemboanza Mike Christopher Nzonzi', 'Stian Rode Gregersen', 'Stole Dimitrievski', 'Strahinja PavloviÄ‡', 'Stuart Armstrong', 'Stuart Dallas', 'Suat Serdar', 'Suleiman Abdullahi', 'Sven Botman', 'Sven Ulreich', 'Sydney van Hooijdonk', 'Szymon Piotr A»urkowski', 'TÃ©ji Tedy Savanier', 'Taiwo Michael Awoniyi', 'Tammy Bakumo-Abraham', 'Tanguy Coulibaly', 'Tanguy NdombÃ¨lÃ© Alvaro', 'Tanguy-Austin Nianzou Kouassi', 'Tanner Tessmann', 'Tariq Lamptey', 'Tariq Uwakwe', 'Tarique Fosu', 'Tarsis Bonga', 'Taylor Anthony Booth', 'Taylor Richards', 'Teddy Bartouche-Selbonne', 'Teddy Boulhendi', 'Teden Mengi', 'Teemu Pukki', 'Temitayo Olufisayo Olaoluwa Aina', 'Terem Igobor Moffi', 'Teun Koopmeiners', 'Thanawat Suengchitthawon', 'Theo Bernard FranÃ§ois HernÃ¡ndez', 'Theo Walcott', 'Thiago AlcÃ¢ntara do Nascimento', 'Thiago Emiliano da Silva', 'Thiago Galhardo do Nascimento Rocha', 'Thiago Henrique Mendes Ribeiro', 'Thibault Tamas', 'Thibaut Courtois', 'Thibo Baeten', 'Thierry Rendall Correia', 'Thierry Small', 'Thomas Callens', 'Thomas Clayton', 'Thomas Delaine', 'Thomas Dickson-Peters', 'Thomas Foket', 'Thomas Fontaine', 'Thomas Henry', 'Thomas Joseph Delaney', 'Thomas Lemar', 'Thomas MÃ¼ller', 'Thomas Mangani', 'Thomas Meunier', 'Thomas Monconduit', 'Thomas Partey', 'Thomas Strakosha', 'Thorgan Hazard', 'Tiago Emanuel EmbalÃ³ DjalÃ³', 'Tiago Manuel Dias Correia', 'Tidiane Malbec', 'TiemouÃ© Bakayoko', 'Tim Akinola', 'Tim Civeja', 'Tim Krul', 'Tim Lemperle', 'Timo Baumgartl', 'Timo Bernd HÃ¼bers', 'Timo Horn', 'Timo Werner', 'TimothÃ© Rupil', 'TimothÃ©e Joseph PembÃ©lÃ©', 'TimothÃ©e Kolodziejczak', 'Timothy Castagne', 'Timothy Chandler', 'Timothy Evans Fosu-Mensah', 'Timothy Tarpeh Weah', 'Timothy Tillman', 'Titouan Thomas', 'Tjark Ernst', 'Tobias Raschl', 'Tobias Sippel', 'Tobias Strobl', 'Todd Cantwell', 'Tolgay Ali Arslan', 'Tom Cleverley', 'Tom Davies', 'Tom Heaton', 'Tom Lacoux', 'Tom Weilandt', 'Toma BaA¡iÄ‡', 'TomÃ¡A¡ Koubek', 'TomÃ¡A¡ OstrÃ¡k', 'TomÃ¡A¡ SouÄek', 'TomÃ¡s Eduardo RincÃ³n HernÃ¡ndez', 'TomÃ¡s JesÃºs AlarcÃ³n Vergara', 'TomÃ¡s Pina Isla', 'Tommaso Augello', 'Tommaso Pobega', 'Toni Herrero Oliva', 'Toni Kroos', 'Tony Jantschke', 'Torben MÃ¼sel', 'Trent Alexander-Arnold', 'Trevoh Chalobah', 'Tristan DingomÃ©', 'Tudor Cristian BÄƒluA£Äƒ', 'Tyler Onyango', 'Tyler Roberts', 'Tyler Shaan Adams', 'Tymoteusz Puchacz', 'Tyrick Mitchell', 'Tyrone Mings', 'Tyronne Ebuehi', 'Ugo Bertelli', 'Ulrick Brad Eneme Ella', 'Unai GarcÃ­a Lugea', 'Unai LÃ³pez Cabrera', 'Unai NÃºÃ±ez Gestoso', 'Unai SimÃ³n Mendibil', 'Unai Vencedor Paris', 'UroA¡ RaÄiÄ‡', 'VÃ­ctor Camarasa Ferrando', 'VÃ­ctor Christopher De Baunbaug', 'VÃ­ctor Chust GarcÃ­a', 'VÃ­ctor David DÃ­az Miguel', 'VÃ­ctor Laguardia Cisneros', 'VÃ­ctor MachÃ­n PÃ©rez', 'VÃ­ctor RuÃ­z Torre', 'ValÃ¨re Germain', 'Valentin Rongier', 'Valentino Lesieur', 'Valentino Livramento', 'Valerio Verre', 'Valon Behrami', 'Valon Berisha', 'Vanja MilinkoviÄ‡-SaviÄ‡', 'Varazdat Haroyan', 'Vasilios Konstantinos Lampropoulos', 'Vedat Muriqi', 'Vicente Guaita Panadero', 'Vicente Iborra de la Fuente', 'Victor JÃ¶rgen Nilsson LindelÃ¶f', 'Victor James Osimhen', 'Vid Belec', 'Viktor Kovalenko', 'Viljami Sinisalo', 'Vilmos TamÃ¡s Orban', 'VinÃ­cius JosÃ© PaixÃ£o de Oliveira JÃºnior', 'Vincent Le Goff', 'Vincent Manceau', 'Vincent Pajot', 'Vincenzo Fiorillo', 'Vincenzo Grifo', 'Virgil van Dijk', 'Vital Manuel NSimba', 'Vitaly Janelt', 'Vito Mannone', 'Vlad Iulian ChiricheÈ™', 'VladimÃ­r Coufal', 'VladimÃ­r Darida', 'Vladislav Cherny', 'Vontae Daley-Campbell', 'Wadi Ibrahim Suzuki', 'Wahbi Khazri', 'Wahidullah Faghir', 'Wajdi Kechrida', 'Walace Souza Silva', 'Waldemar Anton', 'Walim Lgharbi', 'Walter Daniel BenÃ­tez', 'Waniss TaÃ¯bi', 'Warmed Omari', 'Warren TchimbembÃ©', 'Wayne Robert Hennessey', 'Wesley Fofana', 'Wesley SaÃ¯d', 'Weston James Earl McKennie', 'Wilfried Stephane Singo', 'Wilfried Zaha', 'Will Hughes', 'Will Norris', 'Willem Geubbels', 'William Alain AndrÃ© Gabriel Saliba', 'William Anthony Patrick Smallbone', 'William de Asevedo Furtado', 'William Mikelbrencis', 'William Silva de Carvalho', 'William Troost-Ekong', 'Willian JosÃ© da Silva', 'Willy-Arnaud Zobo Boly', 'Wilson Isidor', 'Winston Wiremu Reid', 'Wissam Ben Yedder', 'Wladimiro Falcone', 'Wojciech Tomasz SzczÄ™sny', 'Wout Faes', 'Wout Weghorst', 'Wuilker FariÃ±ez Aray', 'Wylan Cyprien', 'Xaver Schlager', 'Xavi Simons', 'Xavier Chavalerin', 'Xherdan Shaqiri', 'YÄ±ldÄ±rÄ±m Mert Ã‡etin', 'Yacine Adli', 'Yacine Qasmi', 'Yan Brice Eteki', 'Yan Valery', 'Yangel Clemente Herrera Ravelo', 'Yanis Guermouche', 'Yann Sommer', 'Yannick Cahuzac', 'Yannick Ferreira Carrasco', 'Yannick Gerhardt', 'Yannick Pandor', 'Yannik Keitel', 'Yannis MBemba', 'Yasser Larouci', 'Yassin FÃ©kir', 'Yassine Bounou', 'Yayah Kallon', 'Yehvann Diouf', 'Yeray Ãlvarez LÃ³pez', 'Yeremi JesÃºs Santos Pino', 'Yerry Fernando Mina GonzÃ¡lez', 'Yerson Mosquera Valdelamar', 'Yoane Wissa', 'Yoann Salmier', 'Yoann Touzghar', 'Yohann Magnin', 'Youcef Atal', 'Youri Tielemans', 'Youssef En-Nesyri', 'Youssef Maleh', 'Youssouf Fofana', 'Youssouf KonÃ©', 'Youssouf Sabaly', 'Youssouph Mamadou Badji', 'Yunis Abdelhamid', 'Yunus Dimoara Musah', 'Yuri Berchiche Izeta', 'Yussif Raman Chibsah', 'Yussuf Yurary Poulsen', 'Yusuf Demir', 'Yusuf YazÄ±cÄ±', 'Yvan Neyou Noupa', 'Yvann MaÃ§on', 'Yves Bissouma', 'Zack Thomas Steffen', 'Zak Emmerson', 'Zane Monlouis', 'Zaydou Youssouf', 'ZinÃ©dine Machach', 'ZinÃ©dine Ould Khaled', 'Zinho Vanheusden', 'Zlatan IbrahimoviÄ‡'))

        # Opening our datasets
        cfs = pd.read_excel(f'cfs_4_{Decil_final}.xlsx')
        differences = pd.read_excel(f'differences_4_{Decil_final}.xlsx')
        st.write("<div style='height: 650px;'></div>", unsafe_allow_html=True)


        #if tabs == "1. General Sport Analysis":
        with tabs[0]:
            st.markdown("<h4 style='text-align: center;'>Dataset in focus</h1>", unsafe_allow_html=True)
            st.write(df)
            st.markdown("**Figure 1**: Representation of the DataFrame used. It aggregates all data used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # Concepts to take into account
            st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Player_differences = differences.loc[Player]

            # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # Create a histogram.
            plt.hist(differences_array, bins=20, edgecolor='black')
            plt.xlabel('Differences')
            plt.ylabel('Frequency')
            st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 4. Violin: Insights from SUGGESTED CHANGES
            differences_array = differences.values.flatten()
            # Create a violin plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.violinplot(y = differences_array, color='skyblue')
            plt.ylabel('Differences')
            st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 5. Density Plot: Insights from SUGGESTED CHANGES
            differences = differences.squeeze()  # Ensure it's a Series
            plt.figure(figsize=(10, 10)) 
            sns.kdeplot(data=differences, shade=True)
            plt.xlabel('(CFS - X)')
            plt.ylabel('Density')
            st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 6. Radar Chart: Average SUGGESTED CHANGES per feature
            categories = list(differences.columns) # Setting categories as a list of all "differences" column.
            values = differences.mean().values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # Plot features. 
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown("<h4 style='text-align: center;'>Radar Chart: Average SUGGESTED CHANGES per feature</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 6**: Results from DICE. Another method to represent the differences obtained. **The axis defines each difference magnitude per feature.**")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 7. SWARM: Insights from SUGGESTED CHANGES
            # sns.swarmplot(data=differences, palette='coolwarm')
            # plt.xlabel('Features')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>SWARM: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # plt.xticks(rotation=90)  # Better adjusted the rotation angle so that we can better observe feature names.
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 7**: Results from DICE. Last representation of individual differences per feature, with a clear overview on which feature vary the most. **Each point represent a single instance of the dataset**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[1]:
            # 8. Bar Plot
            fig, ax = plt.subplots()
            ax.bar(Player_differences.index, Player_differences.values)
            plt.xticks(rotation=90)  # Adjusting the angle of my axis.
            plt.xlabel('Columns')
            plt.ylabel('Values')
            st.markdown(f"<h4 style='text-align: center;'>Bar Plot for {Player}</h1>", unsafe_allow_html=True)
            st.pyplot(fig) # Displaying plot in Streamlit
            st.markdown(f"**Figure 8**: Results from DICE for **{Player}**. As described in the previous tab, it provides 'what-if' explanations for the model output, by stating **which features would need to be altered in the counterfactual scenarios** compared to the original data to achieve the desired outcomes predicted by the model.  \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 9. KDE
            differences_array = differences[Football_player_feature].values
            # Create KDE plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.kdeplot(differences_array, shade=True)
            plt.xlabel('Differences')
            plt.ylabel('Density')
            st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Football_player_feature_full_name}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Football_player_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Football_player_feature} is recommended to change**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            normalized_data_X = scaler.fit_transform(X)
            normalized_data_cfs = scaler.fit_transform(cfs)
            normalized_data_differences = scaler.fit_transform(differences)

            X_normalized = pd.DataFrame(normalized_data_X, columns=X.columns, index=X.index)
            cfs_normalized = pd.DataFrame(normalized_data_cfs, columns=cfs.columns, index=cfs.index)
            differences_normalized = pd.DataFrame(normalized_data_differences, columns=differences.columns, index=differences.index)

            # 10. Radar (per player) - INITIAL STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_X_normalized = X_normalized.loc[selected_player]    
            categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_X_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            values = player_differences.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences_normalized = differences_normalized.loc[selected_player]    
            categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 13. Radar (per player) - RECOMMENDED STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_cfs_normalized = cfs_normalized.loc[selected_player]    
            categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                        
            # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
            # Specify the name of the player.
            selected_player = Player
            # Filter the differences "DataFrame" for the selected player.
            player_cfs_normalized = cfs_normalized.loc[selected_player]
            player_values_cfs = player_cfs_normalized.values.tolist()
            player_values_cfs += player_values_cfs[:1]
            player_X_normalized = X_normalized.loc[selected_player]
            player_values_X = player_X_normalized.values.tolist()
            player_values_X += player_values_X[:1]
            # Changing angles and categories.
            categories = list(player_cfs_normalized.index)
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            # Plot for 'cfs', that represent the desired values.
            plt.figure(figsize=(8, 8))
            plt.polar(angles, player_values_cfs, label='recommended', color='blue')
            plt.fill(angles, player_values_cfs, alpha=0.25, color='blue')
            # Plot for 'X', that represent the initial values.
            plt.polar(angles, player_values_X, label='initial', color='green')
            plt.fill(angles, player_values_X, alpha=0.25, color='green')
            plt.xticks(angles[:-1], categories)
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE and RECOMMENDED STATE: for {selected_player} - NORMALIZED</h1>", unsafe_allow_html=True)
            plt.legend()
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 14**: To obtain clear insights, we overlapped previous **INITIAL** and **RECOMMENDADED STATES** visualizations. Recapping: \n - **Blue line** represent **DESIRED** feature values (Counterfactuals); \n - **Green line** represent **INITIAL** feature values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
                
        #else:
        with tabs[2]:
            # Concepts to take into account
            st.info("SHAP: (SHapley Additive exPlanations) can be defined as a game theoretic approach to explain the output of a machine learning model. It explains the impact and the importance of each feature on model output/predictions for a specific instance. \n It provides a more interpretable view of the model's behavior and  these values can be used to gain insights on which factors mostly influence specific predictions. \n Looks at the average value and give us information.")

            # 15. SHAP Bar Plot
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            explainer = shap.Explainer(lr, X)
            shap_values = explainer(X)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values, max_display=15)
            st.pyplot()  
            st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 16. SHAP Beeswarm Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.beeswarm(shap_values, max_display=15)
            st.pyplot()
            st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.r")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Football_player_index_feature = Football_player_list.index(Football_player_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Football_player_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.scatter(shap_values[:, Football_player_index_feature])
            st.pyplot()
            st.markdown(f"**Figure 17**: Scatter plot on feature **{Football_player_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Football_player_feature_full_name} feature, which means **'how much must {Football_player_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Football_player_feature_full_name}**.")
            st.markdown(f"This means that, for positive SHAP values, **{Football_player_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Football_player_feature_full_name} must impact negatively** the model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 18. SHAP Partial Dependence Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Football_player_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.partial_dependence_plot(
                Football_player_feature, lr.predict, X, ice=False,
                model_expected_value=True, feature_expected_value=True) 
            st.pyplot()
            st.markdown(f"**Figure 18**: Model's dependence on the feature {Football_player_feature_full_name}, now in the new original feature space (X). It explains ***how SHAP values of {Football_player_feature_full_name} vary across a dataset** and how changes in the {Football_player_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Football_player_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Football_player_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Football_player_feature_full_name}** as a certain value;")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Football_player_index_player = X_indexes.index(Player)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[Football_player_index_player])
            st.pyplot()
            st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Player}, instead of, as in the previous two graphs, focusing on feature {Football_player_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # Concepts to take into account
            st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # 20. SHARP: Rank vs Score
            import os
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            st.image("Rank_vs_Score_(4) Football Player.png")
            st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**, meaning that: \n - **the higher the Player's Score, the better tends to be the its rank**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 21. SHARP: Top and Bottom 3 Individuals
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            st.image("Top_bottom_feature_importance_4.png")
            st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 22. SHARP: Feature Importance
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            st.image("Strata_boxplot_4.png")
            st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 23. Unary values in focus
            def scorer(dataset, columns=None):
                X, _ = check_inputs(dataset)                
                # Define weights for each column
                weights = [0.2, 0.2, 0.2, 0.2, 0.2, 
                                0.2, 0.2, 0.2, 0.2, 0.2, 
                                0.2, 0.2, 0.2, 0.2,
                                0.2, 0.2,
                                0.2, 0.2, 0.2,
                                0.2, 0.2, 0.2, 
                                0.2, 0.2, 0.2, 
                                0.2, 0.2, 0.2, 
                                0.2, 0.2, 0.2, 0.2,
                                0.2, 0.2,
                                0.2, 0.2, 0.2, 
                                0.2, 0.2, 
                                0.2, 0.2, 
                                0.2, 0.2]
                # Calculate the weighted sum for each row
                weighted_sum = np.sum(X * weights, axis=1)
                return weighted_sum

            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            xai = ShaRP(
                qoi="rank",
                target_function=scorer,
                measure="unary",
                sample_size=None, # sample_size=None,
                replace=False,
                random_state=42,
                verbose=1,
                n_jobs=-1)
            xai.fit(X_sharp)

            st.markdown(f"<h4 style='text-align: center;'>Unary values in focus</h1>", unsafe_allow_html=True)
            unary_values = pd.read_csv("cs_rankqoi_unary_values_(4)_football_player.csv")
            unary_values.set_index(unary_values.columns[0], inplace=True)
            unary_values_player = unary_values.loc[Player].drop(["Score", "Ranking"])
            st.write(unary_values) #st.write(unary_values_player), if we want to filter by the player we chose.
            st.markdown("**Figure 23**: Representation of all Unary Values computed and used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 24. SHARP: Waterfall
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Waterfall Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            rank_dict = {
            "upper_bounds": None,
            "lower_bounds": None,
            "features": None,
            "data": None,  
            "base_values": 0,
            "feature_names": unary_values_player.index.tolist(),
            "values": unary_values_player}

            _waterfall(rank_dict, max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} predictions. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            values = xai.pairwise(
                X_sharp.loc[Player].values, 
                X_sharp.loc[Player_2].values)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            pairwise_bars = xai.plot.bar(values, ax=ax)
            ax.set_ylabel("Contribution to Rank")
            ax.set_xlabel("")
            plt.xticks(rotation=90)
            st.pyplot()
            st.markdown(f"**Figure 25**: Pairwise comparison between {Player} and {Player_2}. It provides insights on which variables mostly contribute and which variables mostly harm each one. \n - **Positive values** for a certain feature, means that it **favors {Player} instead of {Player_2}**. \n - **Negative values** for a certain feature, means that it **favors {Player_2} instead of {Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[5]:
            # Extracting values per feature
            all_values = []

            # Combine values from the main plots for each feature (DiCE, SHAP and SHARP).
            for feature in rank_dict["feature_names"]:
                feature_values = {"Feature": feature}
                if feature in Player_differences.index: # Get value from Plot 1 (DiCE: Player_differences)
                    feature_values["Player_differences"] = Player_differences[feature]
                else:
                    feature_values["Player_differences"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 2 (SHAP values)
                    shap_index = rank_dict["feature_names"].index(feature)
                    feature_values["SHAP_values"] = shap_values[Football_player_index_player].values[shap_index]
                else:
                    feature_values["SHAP_values"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 3 (SHARP: rank_dict)
                    rank_index = rank_dict["feature_names"].index(feature)
                    feature_values["Rank_dict_values"] = rank_dict["values"][rank_index]
                else:
                    feature_values["Rank_dict_values"] = None
                
                # Append to the list of all values
                all_values.append(feature_values)

            # 26. DiCE vs SHAP vs SHARP: Comparing Methods
            # Convert to DataFrame and displaying the table.
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df_values_2 = pd.DataFrame(all_values)
            df_values_2.set_index('Feature', inplace=True)
            df_values_2.columns = ["DiCE Counterfactuals", "SHAP Values", "SHARP Values"] # Renaming columns. Replacing Rank Dict Values for SHARP Values.

            # Highlight the top largest and smallestvalues per column
            light_green = 'background-color: rgb(144, 238, 144)'  # Light green.
            light_red = 'background-color: rgba(255, 99, 71, 0.5)'  # Light red color (with transparency)

            # Highlight the top 3 values in a Series green.
            def highlight_top1(s):
                top1 = s.nlargest(1)
                bottom1 = s.nsmallest(1)
                is_top1 = s.isin(top1)
                is_bottom1 = s.isin(bottom1)
                colors = []
                #return [light_green if v else '' for v in is_top1]
                for v in s:
                    if v in top1.values:
                        colors.append(light_green)
                    elif v in bottom1.values:
                        colors.append(light_red)
                    else:
                        colors.append('')
                return colors

            # Apply the highlight_top3 function to the DataFrame and displaying it
            df_styled_2 = df_values_2.style.apply(highlight_top1)
            st.dataframe(df_styled_2, width=900)
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 28. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 28**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


elif Sport == 'Tennis':     
    # Create a radio button for selecting the type (team or player)
    Male_vs_Female = st.sidebar.radio('Gender Preference:', ["Male", "Female"])

    # Check if the user selects the Male Player
    if Male_vs_Female == 'Male':
        Player = st.sidebar.selectbox('Select the Player:', ('Adrian Mannarino', 'Albert Ramos-Vinolas', 'Alejandro Davidovich Fokina', 'Alejandro Tabilo', 'Alex de Minaur', 'Alex Molcan', 'Alexander Bublik', 'Alexander Zverev', 'Alexei Popyrin', 'Aljaz Bedene', 'Andrey Rublev', 'Andy Murray', 'Arthur Rinderknech', 'Aslan Karatsev', 'Ben Shelton', 'Benjamin Bonzi', 'Benoit Paire', 'Bernabe Zapata Miralles', 'Borna Coric', 'Botic van de Zandschulp', 'Brandon Nakashima', 'Cameron Norrie', 'Carlos Alcaraz', 'Casper Ruud', 'Cristian Garin', 'Daniel Evans', 'Daniil Medvedev', 'David Goffin', 'Denis Shapovalov', 'Diego Schwartzman', 'Dominic Thiem', 'Dominik Koepfer', 'Dusan Lajovic', 'Egor Gerasimov', 'Emil Ruusuvuori', 'Fabio Fognini', 'Federico Coria', 'Federico Delbonis', 'Felix Auger-Aliassime', 'Filip Krajinovic', 'Frances Tiafoe', 'Francisco Cerundolo', 'Gael Monfils', 'Gianluca Mager', 'Grigor Dimitrov', 'Holger Rune', 'Hubert Hurkacz', 'Ilya Ivashka', 'J.J. Wolf', 'Jack Draper', 'James Duckworth', 'Jan-Lennard Struff', 'Jannik Sinner', 'Jaume Munar', 'Jenson Brooksby', 'Jeremy Chardy', 'Jiri Lehecka', 'Joao Sousa', 'John Isner', 'John Isner', 'John Millman', 'Jordan Thompson', 'Karen Khachanov', 'Kei Nishikori', 'Laslo Djere', 'Lloyd Harris', 'Lorenzo Musetti', 'Lorenzo Sonego', 'Mackenzie McDonald', 'Marco Cecchinato', 'Marcos Giron', 'Marin Cilic', 'Marton Fucsovics', 'Matteo Berrettini', 'Maxime Cressy', 'Mikael Ymer', 'Milos Raonic', 'Miomir Kecmanovic', 'Nick Kyrgios', 'Nicolas Jarry', 'Nikoloz Basilashvili', 'Novak Djokovic', 'Oscar Otte', 'Pablo Carreno Busta', 'Pedro Martinez', 'Rafael Nadal', 'Reilly Opelka', 'Ricardas Berankis', 'Richard Gasquet', 'Roberto Bautista Agut', 'Sebastian Baez', 'Sebastian Korda', 'Soonwoo Kwon', 'Stefano Travaglia', 'Stefanos Tsitsipas', 'Steve Johnson', 'Tallon Griekspoor', 'Taro Daniel', 'Taylor Fritz', 'Thiago Monteiro', 'Tomas Martin Etcheverry', 'Tommy Paul', 'Ugo Humbert', 'Yoshihito Nishioka'))
        
        # df_serve
        df_serve = pd.read_excel('5_ATP_info.xlsx', sheet_name= 'Serve 2022')
        
        # df_return
        df_return = pd.read_excel('5_ATP_info.xlsx', sheet_name= 'Return 2022')
        
        # df_under_pressure
        df_underpressure = pd.read_excel('5_ATP_info.xlsx', sheet_name= 'UnderPressure 2022')
        df_serve.columns = df_serve.columns.str.replace("©", "").str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df_return.columns = df_return.columns.str.replace("©", "").str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df_underpressure.columns = df_underpressure.columns.str.replace("©", "").str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df_serve.rename(columns={'serve_standing_player2': 'player'}, inplace=True)
        df_return.rename(columns={'return_standing_player2': 'player'}, inplace=True)
        df_underpressure.rename(columns={'under_pressure_standing_player2': 'player'}, inplace=True)
        df_return = df_return.drop(columns=["perc_break_points_converted"]) # Since this variable is already present in "df_underpressure" dataset.
        df = pd.merge(df_serve, df_return, on='player', how='inner')
        df = pd.merge(df, df_underpressure, on='player', how='inner')
        df['final_rating'] = df['serve_rating'] + df['return_rating'] + df['under_pressure_rating']
        df.drop(['serve_rating', 'return_rating', 'under_pressure_rating'], axis=1, inplace=True)
        df = df.sort_values(by='final_rating', ascending=False)
        X = df.drop(columns=["serve_standing_player", "return_standing_player", "under_pressure_standing_player", "final_rating"]).set_index("player")
        y = df.final_rating / df.final_rating.max()
        
        # Define the dictionary mapping short names to full names
        variable_names = {
        'perc_1st_serve': " 1st Serve (%)", 
        'perc_1st_serve_points_won': "1st Serve Points Won (%)",
        'perc_2nd_serve_points_won': "2nd Serve Points Won (%)",
        'perc_service_games_won': "Service Games Won (%)",
        'avg_aces__match': "Average Aces/Match",
        'avg_double_faults_match': "Average Double Faults/Match",
        'perc_1st_serve_return_points_won': "1st Serve Returns Points Won (%)",
        'perc_2nd_serve_return_points_won': "2nd Serve Returns Points Won (%)",
        'perc_return_games_won': "Return Games Won (%)", 
        'perc_break_points_converted_x': "Break Points Converted (%)",
        'perc_break_points_converted_y': "Break Points Converted (%) - Eliminar",
        'perc_break_points_saved': "Break Points Saved (%)",
        'perc_tie_breaks_won': "Tie Breaks Won (%)",
        'perc_deciding_sets_won': "Deciding Sets Won (%)"}

        # Open a sidebar for a different feature option
        Tennis_male_list = list(variable_names.keys()) # Tennis_male_list = X.columns.tolist()
        Tennis_male_list_full = list(variable_names.values())
        Tennis_male_feature_full_name = st.sidebar.selectbox('Feature in focus:', Tennis_male_list_full)
        Tennis_male_feature = [key for key, value in variable_names.items() if value == Tennis_male_feature_full_name][0] # Get the corresponding short name from the dictionary
        
        # Open a sidebar for a different feature option
        Decil = st.sidebar.selectbox('Top Ranking (%) you desire to achieve (where 0,05 means top 5%):', ('0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                                                                                                          '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9'))

        #Decil_final = 1 - float(Decil)
        Decil_final = round(1 - float(Decil), 2)
        
        Player_2 = st.sidebar.selectbox('Select a ATP tennis player to compare:', ('Adrian Mannarino', 'Albert Ramos-Vinolas', 'Alejandro Davidovich Fokina', 'Alejandro Tabilo', 'Alex de Minaur', 'Alex Molcan', 'Alexander Bublik', 'Alexander Zverev', 'Alexei Popyrin', 'Aljaz Bedene', 'Andrey Rublev', 'Andy Murray', 'Arthur Rinderknech', 'Aslan Karatsev', 'Ben Shelton', 'Benjamin Bonzi', 'Benoit Paire', 'Bernabe Zapata Miralles', 'Borna Coric', 'Botic van de Zandschulp', 'Brandon Nakashima', 'Cameron Norrie', 'Carlos Alcaraz', 'Casper Ruud', 'Cristian Garin', 'Daniel Evans', 'Daniil Medvedev', 'David Goffin', 'Denis Shapovalov', 'Diego Schwartzman', 'Dominic Thiem', 'Dominik Koepfer', 'Dusan Lajovic', 'Egor Gerasimov', 'Emil Ruusuvuori', 'Fabio Fognini', 'Federico Coria', 'Federico Delbonis', 'Felix Auger-Aliassime', 'Filip Krajinovic', 'Frances Tiafoe', 'Francisco Cerundolo', 'Gael Monfils', 'Gianluca Mager', 'Grigor Dimitrov', 'Holger Rune', 'Hubert Hurkacz', 'Ilya Ivashka', 'J.J. Wolf', 'Jack Draper', 'James Duckworth', 'Jan-Lennard Struff', 'Jannik Sinner', 'Jaume Munar', 'Jenson Brooksby', 'Jeremy Chardy', 'Jiri Lehecka', 'Joao Sousa', 'John Isner', 'John Isner', 'John Millman', 'Jordan Thompson', 'Karen Khachanov', 'Kei Nishikori', 'Laslo Djere', 'Lloyd Harris', 'Lorenzo Musetti', 'Lorenzo Sonego', 'Mackenzie McDonald', 'Marco Cecchinato', 'Marcos Giron', 'Marin Cilic', 'Marton Fucsovics', 'Matteo Berrettini', 'Maxime Cressy', 'Mikael Ymer', 'Milos Raonic', 'Miomir Kecmanovic', 'Nick Kyrgios', 'Nicolas Jarry', 'Nikoloz Basilashvili', 'Novak Djokovic', 'Oscar Otte', 'Pablo Carreno Busta', 'Pedro Martinez', 'Rafael Nadal', 'Reilly Opelka', 'Ricardas Berankis', 'Richard Gasquet', 'Roberto Bautista Agut', 'Sebastian Baez', 'Sebastian Korda', 'Soonwoo Kwon', 'Stefano Travaglia', 'Stefanos Tsitsipas', 'Steve Johnson', 'Tallon Griekspoor', 'Taro Daniel', 'Taylor Fritz', 'Thiago Monteiro', 'Tomas Martin Etcheverry', 'Tommy Paul', 'Ugo Humbert', 'Yoshihito Nishioka'))

        # Opening our datasets
        cfs = pd.read_excel(f'cfs_5_{Decil_final}.xlsx')
        differences = pd.read_excel(f'differences_5_{Decil_final}.xlsx')
        st.write("<div style='height: 650px;'></div>", unsafe_allow_html=True)


        #if tabs == "1. General Sport Analysis":
        with tabs[0]:
            st.markdown("<h4 style='text-align: center;'>Dataset in focus</h1>", unsafe_allow_html=True)
            st.write(df)
            st.markdown("**Figure 1**: Representation of the DataFrame used. It aggregates all data used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # Concepts to take into account
            st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Player_differences = differences.loc[Player]

            # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # Create a histogram.
            plt.hist(differences_array, bins=20, edgecolor='black')
            plt.xlabel('Differences')
            plt.ylabel('Frequency')
            st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 4. Violin: Insights from SUGGESTED CHANGES
            differences_array = differences.values.flatten()
            # Create a violin plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.violinplot(y = differences_array, color='skyblue')
            plt.ylabel('Differences')
            st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
        
            # 5. Density Plot: Insights from SUGGESTED CHANGES
            differences = differences.squeeze()  # Ensure it's a Series
            plt.figure(figsize=(10, 10)) 
            sns.kdeplot(data=differences, shade=True)
            plt.xlabel('(CFS - X)')
            plt.ylabel('Density')
            st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 6. Radar Chart: Average SUGGESTED CHANGES per feature
            categories = list(differences.columns) # Setting categories as a list of all "differences" column.
            values = differences.mean().values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # Plot features. 
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown("<h4 style='text-align: center;'>Radar Chart: Average SUGGESTED CHANGES per feature</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 6**: Results from DICE. Another method to represent the differences obtained. **The axis defines each difference magnitude per feature.**")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 7. SWARM: Insights from SUGGESTED CHANGES
            # sns.swarmplot(data=differences, palette='coolwarm')
            # plt.xlabel('Features')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>SWARM: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # plt.xticks(rotation=90)  # Better adjusted the rotation angle so that we can better observe feature names.
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 7**: Results from DICE. Last representation of individual differences per feature, with a clear overview on which feature vary the most. **Each point represent a single instance of the dataset**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[1]:
            # 8. Bar Plot
            fig, ax = plt.subplots()
            ax.bar(Player_differences.index, Player_differences.values)
            plt.xticks(rotation=90)  # Adjusting the angle of my axis.
            plt.xlabel('Columns')
            plt.ylabel('Values')
            st.markdown(f"<h4 style='text-align: center;'>Bar Plot for {Player}</h1>", unsafe_allow_html=True)
            st.pyplot(fig) # Displaying plot in Streamlit
            st.markdown(f"**Figure 8**: Results from DICE for **{Player}**. As described in the previous tab, it provides 'what-if' explanations for the model output, by stating **which features would need to be altered in the counterfactual scenarios** compared to the original data to achieve the desired outcomes predicted by the model.  \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 9. KDE
            differences_array = differences[Tennis_male_feature].values
            # Create KDE plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.kdeplot(differences_array, shade=True)
            plt.xlabel('Differences')
            plt.ylabel('Density')
            st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Tennis_male_feature_full_name}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Tennis_male_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Tennis_male_feature} is recommended to change**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            normalized_data_X = scaler.fit_transform(X)
            normalized_data_cfs = scaler.fit_transform(cfs)
            normalized_data_differences = scaler.fit_transform(differences)

            X_normalized = pd.DataFrame(normalized_data_X, columns=X.columns, index=X.index)
            cfs_normalized = pd.DataFrame(normalized_data_cfs, columns=cfs.columns, index=cfs.index)
            differences_normalized = pd.DataFrame(normalized_data_differences, columns=differences.columns, index=differences.index)

            # 10. Radar (per player) - INITIAL STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_X_normalized = X_normalized.loc[selected_player]    
            categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_X_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            values = player_differences.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences_normalized = differences_normalized.loc[selected_player]    
            categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 13. Radar (per player) - RECOMMENDED STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_cfs_normalized = cfs_normalized.loc[selected_player]    
            categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
            # Specify the name of the player.
            selected_player = Player
            # Filter the differences "DataFrame" for the selected player.
            player_cfs_normalized = cfs_normalized.loc[selected_player]
            player_values_cfs = player_cfs_normalized.values.tolist()
            player_values_cfs += player_values_cfs[:1]
            player_X_normalized = X_normalized.loc[selected_player]
            player_values_X = player_X_normalized.values.tolist()
            player_values_X += player_values_X[:1]
            # Changing angles and categories.
            categories = list(player_cfs_normalized.index)
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            # Plot for 'cfs', that represent the desired values.
            plt.figure(figsize=(8, 8))
            plt.polar(angles, player_values_cfs, label='recommended', color='blue')
            plt.fill(angles, player_values_cfs, alpha=0.25, color='blue')
            # Plot for 'X', that represent the initial values.
            plt.polar(angles, player_values_X, label='initial', color='green')
            plt.fill(angles, player_values_X, alpha=0.25, color='green')
            plt.xticks(angles[:-1], categories)
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE and RECOMMENDED STATE: for {selected_player} - NORMALIZED</h1>", unsafe_allow_html=True)
            plt.legend()
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 14**: To obtain clear insights, we overlapped previous **INITIAL** and **RECOMMENDADED STATES** visualizations. Recapping: \n - **Blue line** represent **DESIRED** feature values (Counterfactuals); \n - **Green line** represent **INITIAL** feature values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
                
        #else:
        with tabs[2]:
            # Concepts to take into account
            st.info("SHAP: (SHapley Additive exPlanations) can be defined as a game theoretic approach to explain the output of a machine learning model. It explains the impact and the importance of each feature on model output/predictions for a specific instance. \n It provides a more interpretable view of the model's behavior and  these values can be used to gain insights on which factors mostly influence specific predictions. \n Looks at the average value and give us information.")

            # 15. SHAP Bar Plot
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            explainer = shap.Explainer(lr, X)
            shap_values = explainer(X)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values, max_display=15)
            st.pyplot()  
            st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 16. SHAP Beeswarm Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.beeswarm(shap_values, max_display=15)
            st.pyplot()
            st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.r")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Tennis_male_index_feature = Tennis_male_list.index(Tennis_male_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Tennis_male_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.scatter(shap_values[:, Tennis_male_index_feature])
            st.pyplot()
            st.markdown(f"**Figure 17**: Scatter plot on feature **{Tennis_male_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Tennis_male_feature_full_name} feature, which means **'how much must {Tennis_male_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Tennis_male_feature_full_name}**.")
            st.markdown(f"This means that, for positive SHAP values, **{Tennis_male_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Tennis_male_feature_full_name} must impact negatively** the model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 18. SHAP Partial Dependence Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Tennis_male_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.partial_dependence_plot(
                Tennis_male_feature, lr.predict, X, ice=False,
                model_expected_value=True, feature_expected_value=True) 
            st.pyplot()
            st.markdown(f"**Figure 18**: Model's dependence on the feature {Tennis_male_feature_full_name}, now in the new original feature space (X). It explains ***how SHAP values of {Tennis_male_feature_full_name} vary across a dataset** and how changes in the {Tennis_male_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Tennis_male_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Tennis_male_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Tennis_male_feature_full_name}** as a certain value;")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Tennis_male_index_player = X_indexes.index(Player)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[Tennis_male_index_player])
            st.pyplot()
            st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Player}, instead of, as in the previous two graphs, focusing on feature {Tennis_male_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # Concepts to take into account
            st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # 20. SHARP: Rank vs Score
            import os
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            st.image("Rank_vs_Score_(5) ATP.png")
            st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**, meaning that: \n - **the higher the Player's Score, the better tends to be the its rank**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 21. SHARP: Top and Bottom 3 Individuals
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            st.image("Top_bottom_feature_importance_5.png")
            st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 22. SHARP: Feature Importance
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            st.image("Strata_boxplot_5.png")
            st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 23. Unary values in focus
            def scorer(dataset, columns=None):
                X, _ = check_inputs(dataset)                
                # Define weights for each column
                weights = [0.2, 0.2, 0.2, 0.2, 
                           0.5, -0.5, 
                           0.2, 0.2, 0.2, 0.2, 
                           0.2, 0.2, 0.2]
                # Calculate the weighted sum for each row
                weighted_sum = np.sum(X * weights, axis=1)
                return weighted_sum

            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            xai = ShaRP(
                qoi="rank",
                target_function=scorer,
                measure="unary",
                sample_size=None, # sample_size=None,
                replace=False,
                random_state=42,
                verbose=1,
                n_jobs=-1)
            xai.fit(X_sharp)

            st.markdown(f"<h4 style='text-align: center;'>Unary values in focus</h1>", unsafe_allow_html=True)
            unary_values = pd.read_csv("cs_rankqoi_unary_values_(5)_atp.csv")
            unary_values.set_index(unary_values.columns[0], inplace=True)
            unary_values_player = unary_values.loc[Player].drop(["Score", "Ranking"])
            st.write(unary_values) #st.write(unary_values_player), if we want to filter by the player we chose.
            st.markdown("**Figure 23**: Representation of all Unary Values computed and used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 24. SHARP: Waterfall
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Waterfall Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            rank_dict = {
            "upper_bounds": None,
            "lower_bounds": None,
            "features": None,
            "data": None,  
            "base_values": 0,
            "feature_names": unary_values_player.index.tolist(),
            "values": unary_values_player}

            _waterfall(rank_dict, max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} predictions. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            values = xai.pairwise(
                X_sharp.loc[Player].values, 
                X_sharp.loc[Player_2].values)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            pairwise_bars = xai.plot.bar(values, ax=ax)
            ax.set_ylabel("Contribution to Rank")
            ax.set_xlabel("")
            plt.xticks(rotation=90)
            st.pyplot()
            st.markdown(f"**Figure 25**: Pairwise comparison between {Player} and {Player_2}. It provides insights on which variables mostly contribute and which variables mostly harm each one. \n - **Positive values** for a certain feature, means that it **favors {Player} instead of {Player_2}**. \n - **Negative values** for a certain feature, means that it **favors {Player_2} instead of {Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[5]:
            # Extracting values per feature
            all_values = []

            # Combine values from the main plots for each feature (DiCE, SHAP and SHARP).
            for feature in rank_dict["feature_names"]:
                feature_values = {"Feature": feature}            
                if feature in Player_differences.index: # Get value from Plot 1 (DiCE: Player_differences)
                    feature_values["Player_differences"] = Player_differences[feature]
                else:
                    feature_values["Player_differences"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 2 (SHAP values)
                    shap_index = rank_dict["feature_names"].index(feature)
                    feature_values["SHAP_values"] = shap_values[Tennis_male_index_player].values[shap_index]
                else:
                    feature_values["SHAP_values"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 3 (SHARP: rank_dict)
                    rank_index = rank_dict["feature_names"].index(feature)
                    feature_values["Rank_dict_values"] = rank_dict["values"][rank_index]
                else:
                    feature_values["Rank_dict_values"] = None
                
                # Append to the list of all values
                all_values.append(feature_values)

            # 26. DiCE vs SHAP vs SHARP: Comparing Methods
            # Convert to DataFrame and displaying the table.
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df_values_2 = pd.DataFrame(all_values)
            df_values_2.set_index('Feature', inplace=True)
            df_values_2.columns = ["DiCE Counterfactuals", "SHAP Values", "SHARP Values"] # Renaming columns. Replacing Rank Dict Values for SHARP Values.

            # Highlight the top largest and smallestvalues per column
            light_green = 'background-color: rgb(144, 238, 144)'  # Light green.
            light_red = 'background-color: rgba(255, 99, 71, 0.5)'  # Light red color (with transparency)

            # Highlight the top 3 values in a Series green.
            def highlight_top1(s):
                top1 = s.nlargest(1)
                bottom1 = s.nsmallest(1)
                is_top1 = s.isin(top1)
                is_bottom1 = s.isin(bottom1)
                colors = []
                #return [light_green if v else '' for v in is_top1]
                for v in s:
                    if v in top1.values:
                        colors.append(light_green)
                    elif v in bottom1.values:
                        colors.append(light_red)
                    else:
                        colors.append('')
                return colors

            # Apply the highlight_top3 function to the DataFrame and displaying it
            df_styled_2 = df_values_2.style.apply(highlight_top1)
            st.dataframe(df_styled_2, width=900)
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 28. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 28**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


    # Check if the user selects the Female Player
    elif Male_vs_Female == 'Female':
        Player = st.sidebar.selectbox('Select a WTA tennis player:', ('Aliaksandra Sasnovich', 'Alycia Parks', 'Ana Bogdan', 'Anastasia Pavlyuchenkova', 'Anastasia Potapova', 'Anhelina Kalinina', 'Anna Blinkova', 'Anna Kalinskaya', 'Anna Karolina Schmiedlova', 'Arantxa Rus', 'Aryna Sabalenka', 'Ashlyn Krueger', 'Barbora Krejcikova', 'Beatriz Haddad Maia', 'Belinda Bencic', 'Bernarda Pera', 'Bianca Andreescu', 'Camila Giorgi', 'Camila Osorio', 'Caroline Dolehide', 'Caroline Garcia', 'Claire Liu', 'Clara Burel', 'Coco Gauff', 'Cristina Bucsa', 'Danielle Collins', 'Daria Kasatkina', 'Diana Shnaider', 'Diane Parry', 'Donna Vekic', 'Ekaterina Alexandrova', 'Elena Rybakina', 'Elina Avanesyan', 'Elina Svitolina', 'Elisabetta Cocciaretto', 'Elise Mertens', 'Emina Bektas', 'Emma Navarro', 'Greet Minnen', 'Iga Swiatek', 'Irina-Camelia Begu', 'Jaqueline Cristian', 'Jasmine Paolini', 'Jelena Ostapenko', 'Jessica Pegula', 'Jodie Burrage', 'Kamilla Rakhimova', 'Karolina Muchova', 'Karolina Pliskova', 'Katerina Siniakova', 'Katie Boulter', 'Kayla Day', 'Laura Siegemund', 'Lauren Davis', 'Lesia Tsurenko', 'Leylah Fernandez', 'Lin Zhu', 'Linda Fruhvirtova', 'Linda Noskova', 'Liudmila Samsonova', 'Lucia Bronzetti', 'Madison Keys', 'Magda Linette', 'Magdalena Frech', 'Maria Sakkari', 'Marie Bouzkova', 'Marketa Vondrousova', 'Marta Kostyuk', 'Martina Trevisan', 'Mayar Sherif', 'Mirra Andreeva', 'Nadia Podoroska', 'Nao Hibino', 'Oceane Dodin', 'Ons Jabeur', 'Paula Badosa', 'Petra Kvitova', 'Petra Martic', 'Peyton Stearns', 'Qinwen Zheng', 'Rebeka Masarova', 'Sara Sorribes Tormo', 'Sloane Stephens', 'Sofia Kenin', 'Sorana Cirstea', 'Tamara Korpatsch', 'Tatjana Maria', 'Taylor Townsend', 'Varvara Gracheva', 'Veronika Kudermetova', 'Victoria Azarenka', 'Viktorija Golubic', 'Viktoriya Tomova', 'Xinyu Wang', 'Xiyu Wang', 'Yafan Wang', 'Yanina Wickmayer', 'Yue Yuan', 'Yulia Putintseva', 'Zhu Oxuanbai'))   

        # df_serve
        df_serve = pd.read_excel('6_WTA.xlsx', sheet_name= 'Servers Stats 2023')
        
        # df_return
        df_return = pd.read_excel('6_WTA.xlsx', sheet_name= 'Return Stats 2023')
        df_serve.columns = df_serve.columns.str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df_return.columns = df_return.columns.str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df_serve['player'] = df_serve['player'].apply(lambda x: x.title())
        df_return['player'] = df_return['player'].apply(lambda x: x.title())
        df_return.drop(['pos','rank', 'matches'], axis=1, inplace=True)
        df = pd.merge(df_serve, df_return, on='player', how='inner')
        df['aces_per_match'] = df['aces'] / df['matches']
        df['df_per_match'] = df['df'] / df['matches']
        df.drop(['matches', 'aces', 'df'], axis=1, inplace=True) # Since we already have more relevant information with the new created variables.
        min_ranking = df['rank'].min()
        max_ranking = df['rank'].max()
        df['rank'] = (df['rank'] - min_ranking) / (max_ranking - min_ranking)
        X = df.drop(columns=["pos", "rank"]).set_index("player")
        y = df["rank"]

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index = X.index)
        
        # Define the dictionary mapping short names to full names
        variable_names = {
        '1st_srv_perc': " 1st Serve (%)", 
        '1st_srv_pts_perc': "1st Serve Points Won (%)",
        '2nd_srv_perc': "2nd Serve (%)",
        'srv_pts_won_perc': "2nd Serve Points Won (%)",
        'bp_svd_perc':"Break Points (%)",
        'srv_gm_won_perc': "Service Games Won (%)",
        '1st_rtn_pts_perc': "1st Return Points Won (%)",
        '2nd_rtn_pts_perc': "2nd Return Points Won (%)",
        'rtn_gm_won_perc':"Return Games Won (%)",
        'rtn_pts_won_perc':"Return Points Won (%)",
        'bp_conv_perc': "Break Points Converted (%)",
        'aces_per_match': "Average Aces/Match",
        'df_per_match': "Average Double Faults/Match"}

        # Open a sidebar for a different feature option
        Tennis_female_list = list(variable_names.keys()) # Tennis_female_list = X.columns.tolist()
        Tennis_female_list_full = list(variable_names.values())
        Tennis_female_feature_full_name = st.sidebar.selectbox('Feature in focus:', Tennis_female_list_full)
        Tennis_female_feature = [key for key, value in variable_names.items() if value == Tennis_female_feature_full_name][0] # Get the corresponding short name from the dictionary
        
        # Open a sidebar for a different feature option
        Decil = st.sidebar.selectbox('Top Ranking (%) you desire to achieve (where 0,05 means top 5%):', ('0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                                                                                                          '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9'))

        #Decil_final = 1 - float(Decil)
        Decil_final = round(1 - float(Decil), 2)
        
        Player_2 = st.sidebar.selectbox('Select a WTA tennis player to compare:', ('Aliaksandra Sasnovich', 'Alycia Parks', 'Ana Bogdan', 'Anastasia Pavlyuchenkova', 'Anastasia Potapova', 'Anhelina Kalinina', 'Anna Blinkova', 'Anna Kalinskaya', 'Anna Karolina Schmiedlova', 'Arantxa Rus', 'Aryna Sabalenka', 'Ashlyn Krueger', 'Barbora Krejcikova', 'Beatriz Haddad Maia', 'Belinda Bencic', 'Bernarda Pera', 'Bianca Andreescu', 'Camila Giorgi', 'Camila Osorio', 'Caroline Dolehide', 'Caroline Garcia', 'Claire Liu', 'Clara Burel', 'Coco Gauff', 'Cristina Bucsa', 'Danielle Collins', 'Daria Kasatkina', 'Diana Shnaider', 'Diane Parry', 'Donna Vekic', 'Ekaterina Alexandrova', 'Elena Rybakina', 'Elina Avanesyan', 'Elina Svitolina', 'Elisabetta Cocciaretto', 'Elise Mertens', 'Emina Bektas', 'Emma Navarro', 'Greet Minnen', 'Iga Swiatek', 'Irina-Camelia Begu', 'Jaqueline Cristian', 'Jasmine Paolini', 'Jelena Ostapenko', 'Jessica Pegula', 'Jodie Burrage', 'Kamilla Rakhimova', 'Karolina Muchova', 'Karolina Pliskova', 'Katerina Siniakova', 'Katie Boulter', 'Kayla Day', 'Laura Siegemund', 'Lauren Davis', 'Lesia Tsurenko', 'Leylah Fernandez', 'Lin Zhu', 'Linda Fruhvirtova', 'Linda Noskova', 'Liudmila Samsonova', 'Lucia Bronzetti', 'Madison Keys', 'Magda Linette', 'Magdalena Frech', 'Maria Sakkari', 'Marie Bouzkova', 'Marketa Vondrousova', 'Marta Kostyuk', 'Martina Trevisan', 'Mayar Sherif', 'Mirra Andreeva', 'Nadia Podoroska', 'Nao Hibino', 'Oceane Dodin', 'Ons Jabeur', 'Paula Badosa', 'Petra Kvitova', 'Petra Martic', 'Peyton Stearns', 'Qinwen Zheng', 'Rebeka Masarova', 'Sara Sorribes Tormo', 'Sloane Stephens', 'Sofia Kenin', 'Sorana Cirstea', 'Tamara Korpatsch', 'Tatjana Maria', 'Taylor Townsend', 'Varvara Gracheva', 'Veronika Kudermetova', 'Victoria Azarenka', 'Viktorija Golubic', 'Viktoriya Tomova', 'Xinyu Wang', 'Xiyu Wang', 'Yafan Wang', 'Yanina Wickmayer', 'Yue Yuan', 'Yulia Putintseva', 'Zhu Oxuanbai'))

        # Opening our datasets
        cfs = pd.read_excel(f'cfs_6_{Decil_final}.xlsx')
        differences = pd.read_excel(f'differences_6_{Decil_final}.xlsx')
        st.write("<div style='height: 650px;'></div>", unsafe_allow_html=True)


        #if tabs == "1. General Sport Analysis":
        with tabs[0]:
            st.markdown("<h4 style='text-align: center;'>Dataset in focus</h1>", unsafe_allow_html=True)
            st.write(df)
            st.markdown("**Figure 1**: Representation of the DataFrame used. It aggregates all data used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # Concepts to take into account
            st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Player_differences = differences.loc[Player]

            # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # Create a histogram.
            plt.hist(differences_array, bins=20, edgecolor='black')
            plt.xlabel('Differences')
            plt.ylabel('Frequency')
            st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 4. Violin: Insights from SUGGESTED CHANGES
            differences_array = differences.values.flatten()
            # Create a violin plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.violinplot(y = differences_array, color='skyblue')
            plt.ylabel('Differences')
            st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 5. Density Plot: Insights from SUGGESTED CHANGES
            differences = differences.squeeze()  # Ensure it's a Series
            plt.figure(figsize=(10, 10))  
            sns.kdeplot(data=differences, shade=True)
            plt.xlabel('(CFS - X)')
            plt.ylabel('Density')
            st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 6. Radar Chart: Average SUGGESTED CHANGES per feature
            categories = list(differences.columns) # Setting categories as a list of all "differences" column.
            values = differences.mean().values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # Plot features. 
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown("<h4 style='text-align: center;'>Radar Chart: Average SUGGESTED CHANGES per feature</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 6**: Results from DICE. Another method to represent the differences obtained. **The axis defines each difference magnitude per feature.**")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 7. SWARM: Insights from SUGGESTED CHANGES
            # sns.swarmplot(data=differences, palette='coolwarm')
            # plt.xlabel('Features')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>SWARM: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # plt.xticks(rotation=90)  # Better adjusted the rotation angle so that we can better observe feature names.
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 7**: Results from DICE. Last representation of individual differences per feature, with a clear overview on which feature vary the most. **Each point represent a single instance of the dataset**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[1]:
            # 8. Bar Plot
            fig, ax = plt.subplots()
            ax.bar(Player_differences.index, Player_differences.values)
            plt.xticks(rotation=90)  # Adjusting the angle of my axis.
            plt.xlabel('Columns')
            plt.ylabel('Values')
            st.markdown(f"<h4 style='text-align: center;'>Bar Plot for {Player}</h1>", unsafe_allow_html=True)
            st.pyplot(fig) # Displaying plot in Streamlit
            st.markdown(f"**Figure 8**: Results from DICE for **{Player}**. As described in the previous tab, it provides 'what-if' explanations for the model output, by stating **which features would need to be altered in the counterfactual scenarios** compared to the original data to achieve the desired outcomes predicted by the model.  \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            # 9. KDE
            differences_array = differences[Tennis_female_feature].values
            # Create KDE plot
            plt.figure(figsize=(8, 6)) # Setting figure size.
            sns.kdeplot(differences_array, shade=True)
            plt.xlabel('Differences')
            plt.ylabel('Density')
            st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Tennis_female_feature_full_name}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Tennis_female_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Tennis_female_feature} is recommended to change**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            normalized_data_X = scaler.fit_transform(X)
            normalized_data_cfs = scaler.fit_transform(cfs)
            normalized_data_differences = scaler.fit_transform(differences)

            X_normalized = pd.DataFrame(normalized_data_X, columns=X.columns, index=X.index)
            cfs_normalized = pd.DataFrame(normalized_data_cfs, columns=cfs.columns, index=cfs.index)
            differences_normalized = pd.DataFrame(normalized_data_differences, columns=differences.columns, index=differences.index)

            # 10. Radar (per player) - INITIAL STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_X_normalized = X_normalized.loc[selected_player]    
            categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_X_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            values = player_differences.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences_normalized = differences_normalized.loc[selected_player]    
            categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 13. Radar (per player) - RECOMMENDED STATE
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_cfs_normalized = cfs_normalized.loc[selected_player]    
            categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            plt.figure(figsize=(8, 8)) # Setting figure size.
            plt.polar(angles, values) # Using polar coordinates.
            plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            plt.xticks(angles[:-1], categories) # Set the categories as labels.
            st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
            # Specify the name of the player.
            selected_player = Player
            # Filter the differences "DataFrame" for the selected player.
            player_cfs_normalized = cfs_normalized.loc[selected_player]
            player_values_cfs = player_cfs_normalized.values.tolist()
            player_values_cfs += player_values_cfs[:1]
            player_X_normalized = X_normalized.loc[selected_player]
            player_values_X = player_X_normalized.values.tolist()
            player_values_X += player_values_X[:1]
            # Changing angles and categories.
            categories = list(player_cfs_normalized.index)
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            # Plot for 'cfs', that represent the desired values.
            plt.figure(figsize=(8, 8))
            plt.polar(angles, player_values_cfs, label='recommended', color='blue')
            plt.fill(angles, player_values_cfs, alpha=0.25, color='blue')
            # Plot for 'X', that represent the initial values.
            plt.polar(angles, player_values_X, label='initial', color='green')
            plt.fill(angles, player_values_X, alpha=0.25, color='green')
            plt.xticks(angles[:-1], categories)
            st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE and RECOMMENDED STATE: for {selected_player} - NORMALIZED</h1>", unsafe_allow_html=True)
            plt.legend()
            st.pyplot() # Displaying plot in Streamlit
            st.markdown(f"**Figure 14**: To obtain clear insights, we overlapped previous **INITIAL** and **RECOMMENDADED STATES** visualizations. Recapping: \n - **Blue line** represent **DESIRED** feature values (Counterfactuals); \n - **Green line** represent **INITIAL** feature values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

                
        #else:
        with tabs[2]:
            # Concepts to take into account
            st.info("SHAP: (SHapley Additive exPlanations) can be defined as a game theoretic approach to explain the output of a machine learning model. It explains the impact and the importance of each feature on model output/predictions for a specific instance. \n It provides a more interpretable view of the model's behavior and  these values can be used to gain insights on which factors mostly influence specific predictions. \n Looks at the average value and give us information.")

            # 15. SHAP Bar Plot
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            explainer = shap.Explainer(lr, X)
            shap_values = explainer(X)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values, max_display=15)
            st.pyplot()  
            st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 16. SHAP Beeswarm Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.beeswarm(shap_values, max_display=15)
            st.pyplot()
            st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.r")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Tennis_female_index_feature = Tennis_female_list.index(Tennis_female_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Tennis_female_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.scatter(shap_values[:, Tennis_female_index_feature])
            st.pyplot()
            st.markdown(f"**Figure 17**: Scatter plot on feature **{Tennis_female_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Tennis_female_feature_full_name} feature, which means **'how much must {Tennis_female_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Tennis_female_feature_full_name}**.")
            st.markdown(f"This means that, for positive SHAP values, **{Tennis_female_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Tennis_female_feature_full_name} must impact negatively** the model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 18. SHAP Partial Dependence Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Tennis_female_feature_full_name}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.partial_dependence_plot(
                Tennis_female_feature, lr.predict, X, ice=False,
                model_expected_value=True, feature_expected_value=True) 
            st.pyplot()
            st.markdown(f"**Figure 18**: Model's dependence on the feature {Tennis_female_feature_full_name}, now in the new original feature space (X). It explains ***how SHAP values of {Tennis_female_feature_full_name} vary across a dataset** and how changes in the {Tennis_female_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Tennis_female_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Tennis_female_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Tennis_female_feature_full_name}** as a certain value;")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Tennis_female_index_player = X_indexes.index(Player)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[Tennis_female_index_player])
            st.pyplot()
            st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Player}, instead of, as in the previous two graphs, focusing on feature {Tennis_female_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # Concepts to take into account
            st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # 20. SHARP: Rank vs Score        
            import os
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            st.image("Rank_vs_Score_(6) WTA.png")
            st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**, meaning that: \n - **the higher the Player's Score, the better tends to be the its rank**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 21. SHARP: Top and Bottom 3 Individuals
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            st.image("Top_bottom_feature_importance_6.png")
            st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 22. SHARP: Feature Importance
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            st.image("Strata_boxplot_6.png", width=800)#, height=800)
            st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 23. Unary values in focus
            def scorer(dataset, columns=None):
                X, _ = check_inputs(dataset)                
                # Define weights for each column
                weights = [0.2, 0.2, 0.2, 0.2,  
                            0.2, 0.2, 0.2, 
                            0.2, 0.2, 0.2,
                            0.2, 0.5, -0.5]
                # Calculate the weighted sum for each row
                weighted_sum = np.sum(X * weights, axis=1)
                return weighted_sum

            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            xai = ShaRP(
                qoi="rank",
                target_function=scorer,
                measure="unary",
                sample_size=None, # sample_size=None,
                replace=False,
                random_state=42,
                verbose=1,
                n_jobs=-1)
            xai.fit(X_sharp)

            st.markdown(f"<h4 style='text-align: center;'>Unary values in focus</h1>", unsafe_allow_html=True)
            unary_values = pd.read_csv("cs_rankqoi_unary_values_(6)_wta.csv")
            unary_values.set_index(unary_values.columns[0], inplace=True)
            unary_values_player = unary_values.loc[Player].drop(["Score", "Ranking"])
            st.write(unary_values) #st.write(unary_values_player), if we want to filter by the player we chose.
            st.markdown("**Figure 23**: Representation of all Unary Values computed and used in our research.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 24. SHARP: Waterfall
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Waterfall Plot</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            rank_dict = {
            "upper_bounds": None,
            "lower_bounds": None,
            "features": None,
            "data": None,  
            "base_values": 0,
            "feature_names": unary_values_player.index.tolist(),
            "values": unary_values_player}

            _waterfall(rank_dict, max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} predictions. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            X_sharp = X
            X_sharp_np = X_sharp.values
            y = scorer(X_sharp_np)

            values = xai.pairwise(
                X_sharp.loc[Player].values, 
                X_sharp.loc[Player_2].values)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            pairwise_bars = xai.plot.bar(values, ax=ax)
            ax.set_ylabel("Contribution to Rank")
            ax.set_xlabel("")
            plt.xticks(rotation=90)
            st.pyplot()
            st.markdown(f"**Figure 25**: Pairwise comparison between {Player} and {Player_2}. It provides insights on which variables mostly contribute and which variables mostly harm each one. \n - **Positive values** for a certain feature, means that it **favors {Player} instead of {Player_2}**. \n - **Negative values** for a certain feature, means that it **favors {Player_2} instead of {Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

        #else:
        with tabs[5]:
            # Extracting values per feature
            all_values = []

            # Combine values from the main plots for each feature (DiCE, SHAP and SHARP).
            for feature in rank_dict["feature_names"]:
                feature_values = {"Feature": feature}
                if feature in Player_differences.index: # Get value from Plot 1 (DiCE: Player_differences)
                    feature_values["Player_differences"] = Player_differences[feature]
                else:
                    feature_values["Player_differences"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 2 (SHAP values)
                    shap_index = rank_dict["feature_names"].index(feature)
                    feature_values["SHAP_values"] = shap_values[Tennis_female_index_player].values[shap_index]
                else:
                    feature_values["SHAP_values"] = None
                if feature in rank_dict["feature_names"]: # Get value from Plot 3 (SHARP: rank_dict)
                    rank_index = rank_dict["feature_names"].index(feature)
                    feature_values["Rank_dict_values"] = rank_dict["values"][rank_index]
                else:
                    feature_values["Rank_dict_values"] = None
                
                # Append to the list of all values
                all_values.append(feature_values)

            # 26. DiCE vs SHAP vs SHARP: Comparing Methods
            # Convert to DataFrame and displaying the table.
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df_values_2 = pd.DataFrame(all_values)
            df_values_2.set_index('Feature', inplace=True)
            df_values_2.columns = ["DiCE Counterfactuals", "SHAP Values", "SHARP Values"] # Renaming columns. Replacing Rank Dict Values for SHARP Values.

            # Highlight the top largest and smallestvalues per column
            light_green = 'background-color: rgb(144, 238, 144)'  # Light green.
            light_red = 'background-color: rgba(255, 99, 71, 0.5)'  # Light red color (with transparency)

            # Highlight the top 3 values in a Series green.
            def highlight_top1(s):
                top1 = s.nlargest(1)
                bottom1 = s.nsmallest(1)
                is_top1 = s.isin(top1)
                is_bottom1 = s.isin(bottom1)
                colors = []
                #return [light_green if v else '' for v in is_top1]
                for v in s:
                    if v in top1.values:
                        colors.append(light_green)
                    elif v in bottom1.values:
                        colors.append(light_red)
                    else:
                        colors.append('')
                return colors

            # Apply the highlight_top3 function to the DataFrame and displaying it
            df_styled_2 = df_values_2.style.apply(highlight_top1)
            st.dataframe(df_styled_2, width=900)
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 28. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 28**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


# 6. Sidebar Part II
st.sidebar.header("Provide some feedback:")
st.sidebar.text_input("Mail Adress")
st.sidebar.text_input("Profession")
st.sidebar.radio("Professional Expert", ["Student", "Professor", "Other"])
st.sidebar.slider("How much did you find it relevant?", 0, 100)
st.sidebar.text_input("Additional Comments")
st.sidebar.button("Submit Feedback")
