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
#st.set_option('deprecation.showPyplotGlobalUse', False)

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
         "(3) Football Team 2023", 
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
            # # Create a violin plot
            # plt.figure(figsize=(8, 6))
            # sns.violinplot(y = differences_array, color='skyblue')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 5. Density Plot: Insights from SUGGESTED CHANGES
            # differences = differences.squeeze()  # Ensure it's a Series
            # plt.figure(figsize=(10, 10)) 
            # sns.kdeplot(data=differences, shade=True)
            # plt.xlabel('(CFS - X)')
            # plt.ylabel('Density')
            # st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 6. Radar Chart: Average SUGGESTED CHANGES per feature
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
            # # Filter "differences" DataFrame.
            # player_X_normalized = X_normalized.loc[selected_player]    
            # categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_X_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 11. Radar (per player) - SUGGESTED CHANGES
            # # Specify the name of the player
            # selected_player = Team
            # # Filter "differences" DataFrame.
            # player_differences = differences.loc[selected_player]    
            # categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            # values = player_differences.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # # Specify the name of the player
            # selected_player = Team
            # # Filter "differences" DataFrame.
            # player_differences_normalized = differences_normalized.loc[selected_player]    
            # categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 13. Radar (per player) - RECOMMENDED STATE
            # # Specify the name of the player
            # selected_player = Team
            # # Filter "differences" DataFrame.
            # player_cfs_normalized = cfs_normalized.loc[selected_player]
            # categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
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
            #st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            #shap.plots.bar(shap_values, max_display=15)
            #st.pyplot()  
            #st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            #st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 16. SHAP Beeswarm Plot
            st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.beeswarm(shap_values, max_display=15)
            st.pyplot()
            st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            basketball_team_index_feature = Basketball_team_list.index(Basketball_team_feature)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Basketball_team_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.scatter(shap_values[:, basketball_team_index_feature])
            # st.pyplot()
            # st.markdown(f"**Figure 17**: Scatter plot on feature **{Basketball_team_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Basketball_team_feature_full_name} feature, which means **'how much must {Basketball_team_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Basketball_team_feature_full_name}**.")
            # st.markdown(f"This means that, for positive SHAP values, **{Basketball_team_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Basketball_team_feature_full_name} must impact negatively** the model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 18. SHAP Partial Dependence Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Basketball_team_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.partial_dependence_plot(
            #     Basketball_team_feature, lr.predict, X, ice=False,
            #     model_expected_value=True, feature_expected_value=True) 
            # st.pyplot()
            # st.markdown(f"**Figure 18**: Model's dependence on the feature {Basketball_team_feature_full_name}, now in the new original feature space (X). It explains **how SHAP values of {Basketball_team_feature_full_name} vary across a dataset** and how changes in the {Basketball_team_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Basketball_team_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Basketball_team_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Basketball_team_feature_full_name}** as a certain value;")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            basketball_team_index_player = X_indexes.index(Team)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Team}</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[basketball_team_index_player], max_display=15)
            st.pyplot()
            st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Team}, instead of, as in the previous two graphs, focusing on feature {Basketball_team_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # # Concepts to take into account
            # st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # # 20. SHARP: Rank vs Score
            # import os
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            # st.image("Rank_vs_Score_(1) Basket Team.png")
            # st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 21. SHARP: Top and Bottom 3 Individuals
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            # st.image("Top_bottom_feature_importance_1.png")
            # st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 22. SHARP: Feature Importance
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            # st.image("Strata_boxplot_1.png")
            # st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            # st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Team}. Similarly to SHAP Waterfall, it attempts to explain {Team} ranking. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            
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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value. \n - Note that highest DiCE values does not necessary mean worse features. DiCE can impact either the best features or the worst features. But overall, the lowest the player ranking, the higher amount tend to be the player average DiCE values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            
            # 28. Create a Statistics DataFrame
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Statistics Comparison</h1>", unsafe_allow_html=True)
            average_abs_values = df_values_2.abs().mean() # Calculate the average of the absolute values for each column
            variance_values = df_values_2.var() # Calculate the variance for each column
            diff_max_min_values = df_values_2.max() - df_values_2.min() # Calculate the difference between the maximum and minimum values for each column
            df_stats = pd.DataFrame({
            'Average Absolute Value': average_abs_values,
            'Variance': variance_values,
            'Max-Min Difference (Amplitude)': diff_max_min_values})
            st.dataframe(df_stats, width=900)
            st.markdown(f"**Figure 28**: Table aggregating the average values, the variance and the amplitude from DiCE, SHAP and SHARP applied to {Team} and according to the selected decil.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 29. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 29**: Correlation matrix between DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


    # Check if the user selects the type as Player
    elif Team_vs_Player == 'Player':
        Player = st.sidebar.selectbox('Select the Player:', ('Aaron Gordon', 'Aaron Nesmith', 'Aaron Wiggins', 'AJ Griffin', 'Al Horford', 'Alec Burks', 'Aleksej Pokusevski', 'Alex Caruso', 'Alperen Þengün', 'Andrew Nembhard', 'Andrew Wiggins', 'Anfernee Simons', 'Anthony Davis', 'Anthony Edwards', 'Anthony Lamb', 'Austin Reaves', 'Austin Rivers', 'Ayo Dosunmu', 'Bam Adebayo', 'Ben Simmons', 'Bennedict Mathurin', 'Blake Wesley', 'Bobby Portis', 'Bogdan Bogdanovi?', 'Bojan Bogdanovi?', 'Bol Bol', 'Bones Hyland', 'Bradley Beal', 'Brandon Clarke', 'Brandon Ingram', 'Brook Lopez', 'Bruce Brown', 'Bryce McGowens', 'Buddy Hield', 'Cade Cunningham', 'Caleb Houstan', 'Caleb Martin', 'Cam Reddish', 'Cam Thomas', 'Cameron Johnson', 'Cameron Payne', 'Caris LeVert', 'Cedi Osman', 'Chance Comanche', 'Chris Boucher', 'Chris Duarte', 'Chris Paul', 'Christian Braun', 'Christian Wood', 'Chuma Okeke', 'CJ McCollum', 'Clint Capela', 'Coby White', 'Cody Martin', 'Cole Anthony', 'Collin Sexton', 'Corey Kispert', 'Cory Joseph', 'Daishen Nix', 'Damian Lillard', 'Damion Lee', 'Daniel Gafford', 'Daniel Theis', 'Darius Garland', 'David Roddy', 'Davion Mitchell', 'Dean Wade', 'Deandre Ayton', 'Dejounte Murray', 'Delon Wright', 'DeMar DeRozan', 'Deni Avdija', 'Dennis Schröder', 'Dennis Smith Jr.', 'Derrick White', 'Desmond Bane', 'Devin Booker', 'Devin Vassell', 'Dillon Brooks', 'Domantas Sabonis', 'Donovan Mitchell', 'Donte DiVincenzo', 'Dorian Finney-Smith', 'Doug McDermott', 'Draymond Green', 'Drew Eubanks', 'Duncan Robinson', 'Dwight Powell', 'Dyson Daniels', 'Eric Gordon', 'Eugene Omoruyi', 'Evan Fournier', 'Evan Mobley', 'Franz Wagner', 'Fred VanVleet', 'Gabe Vincent', 'Gabe York', 'Gary Harris', 'Gary Payton II', 'Gary Trent Jr.', 'George Hill', 'Georges Niang', 'Giannis Antetokounmpo', 'Gordon Hayward', 'Grant Williams', 'Grayson Allen', 'Hamidou Diallo', 'Harrison Barnes', 'Haywood Highsmith', 'Herbert Jones', 'Immanuel Quickley', 'Isaac Okoro', 'Isaiah Hartenstein', 'Isaiah Jackson', 'Isaiah Joe', 'Isaiah Livers', 'Isaiah Stewart', 'Ish Wainright', 'Ivica Zubac', 'Ja Morant', 'Jabari Smith Jr.', 'Jacob Gilyard', 'Jaden Ivey', 'Jaden McDaniels', 'Jae Crowder', 'Jakob Poeltl', 'Jalen Brunson', 'Jalen Duren', 'Jalen Green', 'Jalen McDaniels', 'Jalen Smith', 'Jalen Suggs', 'Jalen Williams', 'Jamal Murray', 'James Bouknight', 'James Harden', 'James Wiseman', 'Jaren Jackson Jr.', 'Jarred Vanderbilt', 'Jarrett Allen', 'Jaylen Brown', 'Jaylen Nowell', 'Jaylin Williams', 'Jayson Tatum', 'Jeenathan Williams', 'Jeff Green', 'Jerami Grant', 'Jeremiah Robinson-Earl', 'Jeremy Sochan', 'Jericho Sims', 'Jevon Carter', 'Jimmy Butler', 'Joe Harris', 'Joe Ingles', 'Joel Embiid', 'John Collins', 'John Konchar', 'John Wall', 'Johnny Davis', 'Jonas Valan?i?nas', 'Jonathan Kuminga', 'Jordan Clarkson', 'Jordan Goodwin', 'Jordan McLaughlin', 'Jordan Nwora', 'Jordan Poole', 'Jose Alvarado', 'Josh Giddey', 'Josh Green', 'Josh Hart', 'Josh Okogie', 'Josh Richardson', 'Joshua Primo', 'Jrue Holiday', 'Julius Randle', 'Justin Holiday', 'Justin Minaya', 'Justise Winslow', 'Jusuf Nurki?', 'Karl-Anthony Towns', 'Kawhi Leonard', 'Keegan Murray', 'Keita Bates-Diop', 'Keldon Johnson', 'Kelly Olynyk', 'Kelly Oubre Jr.', 'Kemba Walker', 'Kenrich Williams', 'Kentavious Caldwell-Pope', 'Kenyon Martin Jr.', 'Kevin Durant', 'Kevin Huerter', 'Kevin Knox', 'Kevin Love', 'Kevin Porter Jr.', 'Kevon Looney', 'Khris Middleton', 'Killian Hayes', 'Klay Thompson', 'Kris Dunn', 'Kristaps Porzi??is', 'Kyle Anderson', 'Kyle Kuzma', 'Kyle Lowry', 'Kyrie Irving', 'Lamar Stevens', 'LaMelo Ball', 'Landry Shamet', 'Larry Nance Jr.', 'Lauri Markkanen', 'LeBron James', 'Lonnie Walker IV', 'Louis King', 'Luguentz Dort', 'Luka Don?i?', 'Luka Šamani?', 'Luke Kennard', 'Mac McClung', 'Malaki Branham', 'Malcolm Brogdon', 'Malik Beasley', 'Malik Monk', 'Marcus Morris', 'Marcus Smart', 'Mark Williams', 'Markelle Fultz', 'Marvin Bagley III', 'Mason Plumlee', 'Matisse Thybulle', 'Max Strus', 'Maxi Kleber', 'Michael Porter Jr.', 'Mikal Bridges', 'Mike Conley', 'Mike Muscala', 'Mitchell Robinson', 'Monte Morris', 'Moritz Wagner', 'Myles Turner', 'Naji Marshall', 'Nassir Little', 'Naz Reid', 'Nic Claxton', 'Nick Richards', 'Nickeil Alexander-Walker', 'Nicolas Batum', 'Nikola Joki?', 'Nikola Vu?evi?', 'Norman Powell', 'Obi Toppin', 'Ochai Agbaji', 'OG Anunoby', 'Onyeka Okongwu', 'Oshae Brissett', 'Otto Porter Jr.', 'P.J. Tucker', 'P.J. Washington', 'Paolo Banchero', 'Pascal Siakam', 'Pat Connaughton', 'Patrick Beverley', 'Patrick Williams', 'Paul George', 'Precious Achiuwa', 'Quentin Grimes', 'R.J. Hampton', 'RaiQuan Gray', 'Reggie Bullock', 'Reggie Jackson', 'Ricky Rubio', 'RJ Barrett', 'Robert Covington', 'Robert Williams', 'Rodney McGruder', 'Romeo Langford', 'Rudy Gobert', 'Rui Hachimura', 'Russell Westbrook', 'Saddiq Bey', 'Sam Hauser', 'Sandro Mamukelashvili', 'Santi Aldama', 'Scottie Barnes', 'Seth Curry', 'Shaedon Sharpe', 'Shai Gilgeous-Alexander', 'Shake Milton', 'Shaquille Harrison', 'Skylar Mays', 'Spencer Dinwiddie', 'Stanley Johnson', 'Stephen Curry', 'Steven Adams', 'T.J. McConnell', 'T.J. Warren', 'Talen Horton-Tucker', 'Tari Eason', 'Taurean Prince', 'Terance Mann', 'Terrence Ross', 'Terry Rozier', 'Théo Maledon', 'Thomas Bryant', 'Tim Hardaway Jr.', 'Tobias Harris', 'Torrey Craig', 'Trae Young', 'Tre Jones', 'Tre Mann', 'Trendon Watford', 'Trey Lyles', 'Trey Murphy III', 'Troy Brown Jr.', 'Ty Jerome', 'Tyler Herro', 'Tyrese Haliburton', 'Tyrese Maxey', 'Tyus Jones', 'Victor Oladipo', 'Walker Kessler', 'Wendell Carter Jr.', 'Wenyen Gabriel', 'Wesley Matthews', 'Will Barton', 'Xavier Tillman Sr.', 'Yuta Watanabe', 'Zach Collins', 'Zach LaVine', 'Ziaire Williams', 'Zion Williamson'))
        
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
        
        Player_2 = st.sidebar.selectbox('Select a Team to compare:', ('Aaron Gordon', 'Aaron Nesmith', 'Aaron Wiggins', 'AJ Griffin', 'Al Horford', 'Alec Burks', 'Aleksej Pokusevski', 'Alex Caruso', 'Alperen Þengün', 'Andrew Nembhard', 'Andrew Wiggins', 'Anfernee Simons', 'Anthony Davis', 'Anthony Edwards', 'Anthony Lamb', 'Austin Reaves', 'Austin Rivers', 'Ayo Dosunmu', 'Bam Adebayo', 'Ben Simmons', 'Bennedict Mathurin', 'Blake Wesley', 'Bobby Portis', 'Bogdan Bogdanovi?', 'Bojan Bogdanovi?', 'Bol Bol', 'Bones Hyland', 'Bradley Beal', 'Brandon Clarke', 'Brandon Ingram', 'Brook Lopez', 'Bruce Brown', 'Bryce McGowens', 'Buddy Hield', 'Cade Cunningham', 'Caleb Houstan', 'Caleb Martin', 'Cam Reddish', 'Cam Thomas', 'Cameron Johnson', 'Cameron Payne', 'Caris LeVert', 'Cedi Osman', 'Chance Comanche', 'Chris Boucher', 'Chris Duarte', 'Chris Paul', 'Christian Braun', 'Christian Wood', 'Chuma Okeke', 'CJ McCollum', 'Clint Capela', 'Coby White', 'Cody Martin', 'Cole Anthony', 'Collin Sexton', 'Corey Kispert', 'Cory Joseph', 'Daishen Nix', 'Damian Lillard', 'Damion Lee', 'Daniel Gafford', 'Daniel Theis', 'Darius Garland', 'David Roddy', 'Davion Mitchell', 'Dean Wade', 'Deandre Ayton', 'Dejounte Murray', 'Delon Wright', 'DeMar DeRozan', 'Deni Avdija', 'Dennis Schröder', 'Dennis Smith Jr.', 'Derrick White', 'Desmond Bane', 'Devin Booker', 'Devin Vassell', 'Dillon Brooks', 'Domantas Sabonis', 'Donovan Mitchell', 'Donte DiVincenzo', 'Dorian Finney-Smith', 'Doug McDermott', 'Draymond Green', 'Drew Eubanks', 'Duncan Robinson', 'Dwight Powell', 'Dyson Daniels', 'Eric Gordon', 'Eugene Omoruyi', 'Evan Fournier', 'Evan Mobley', 'Franz Wagner', 'Fred VanVleet', 'Gabe Vincent', 'Gabe York', 'Gary Harris', 'Gary Payton II', 'Gary Trent Jr.', 'George Hill', 'Georges Niang', 'Giannis Antetokounmpo', 'Gordon Hayward', 'Grant Williams', 'Grayson Allen', 'Hamidou Diallo', 'Harrison Barnes', 'Haywood Highsmith', 'Herbert Jones', 'Immanuel Quickley', 'Isaac Okoro', 'Isaiah Hartenstein', 'Isaiah Jackson', 'Isaiah Joe', 'Isaiah Livers', 'Isaiah Stewart', 'Ish Wainright', 'Ivica Zubac', 'Ja Morant', 'Jabari Smith Jr.', 'Jacob Gilyard', 'Jaden Ivey', 'Jaden McDaniels', 'Jae Crowder', 'Jakob Poeltl', 'Jalen Brunson', 'Jalen Duren', 'Jalen Green', 'Jalen McDaniels', 'Jalen Smith', 'Jalen Suggs', 'Jalen Williams', 'Jamal Murray', 'James Bouknight', 'James Harden', 'James Wiseman', 'Jaren Jackson Jr.', 'Jarred Vanderbilt', 'Jarrett Allen', 'Jaylen Brown', 'Jaylen Nowell', 'Jaylin Williams', 'Jayson Tatum', 'Jeenathan Williams', 'Jeff Green', 'Jerami Grant', 'Jeremiah Robinson-Earl', 'Jeremy Sochan', 'Jericho Sims', 'Jevon Carter', 'Jimmy Butler', 'Joe Harris', 'Joe Ingles', 'Joel Embiid', 'John Collins', 'John Konchar', 'John Wall', 'Johnny Davis', 'Jonas Valan?i?nas', 'Jonathan Kuminga', 'Jordan Clarkson', 'Jordan Goodwin', 'Jordan McLaughlin', 'Jordan Nwora', 'Jordan Poole', 'Jose Alvarado', 'Josh Giddey', 'Josh Green', 'Josh Hart', 'Josh Okogie', 'Josh Richardson', 'Joshua Primo', 'Jrue Holiday', 'Julius Randle', 'Justin Holiday', 'Justin Minaya', 'Justise Winslow', 'Jusuf Nurki?', 'Karl-Anthony Towns', 'Kawhi Leonard', 'Keegan Murray', 'Keita Bates-Diop', 'Keldon Johnson', 'Kelly Olynyk', 'Kelly Oubre Jr.', 'Kemba Walker', 'Kenrich Williams', 'Kentavious Caldwell-Pope', 'Kenyon Martin Jr.', 'Kevin Durant', 'Kevin Huerter', 'Kevin Knox', 'Kevin Love', 'Kevin Porter Jr.', 'Kevon Looney', 'Khris Middleton', 'Killian Hayes', 'Klay Thompson', 'Kris Dunn', 'Kristaps Porzi??is', 'Kyle Anderson', 'Kyle Kuzma', 'Kyle Lowry', 'Kyrie Irving', 'Lamar Stevens', 'LaMelo Ball', 'Landry Shamet', 'Larry Nance Jr.', 'Lauri Markkanen', 'LeBron James', 'Lonnie Walker IV', 'Louis King', 'Luguentz Dort', 'Luka Don?i?', 'Luka Šamani?', 'Luke Kennard', 'Mac McClung', 'Malaki Branham', 'Malcolm Brogdon', 'Malik Beasley', 'Malik Monk', 'Marcus Morris', 'Marcus Smart', 'Mark Williams', 'Markelle Fultz', 'Marvin Bagley III', 'Mason Plumlee', 'Matisse Thybulle', 'Max Strus', 'Maxi Kleber', 'Michael Porter Jr.', 'Mikal Bridges', 'Mike Conley', 'Mike Muscala', 'Mitchell Robinson', 'Monte Morris', 'Moritz Wagner', 'Myles Turner', 'Naji Marshall', 'Nassir Little', 'Naz Reid', 'Nic Claxton', 'Nick Richards', 'Nickeil Alexander-Walker', 'Nicolas Batum', 'Nikola Joki?', 'Nikola Vu?evi?', 'Norman Powell', 'Obi Toppin', 'Ochai Agbaji', 'OG Anunoby', 'Onyeka Okongwu', 'Oshae Brissett', 'Otto Porter Jr.', 'P.J. Tucker', 'P.J. Washington', 'Paolo Banchero', 'Pascal Siakam', 'Pat Connaughton', 'Patrick Beverley', 'Patrick Williams', 'Paul George', 'Precious Achiuwa', 'Quentin Grimes', 'R.J. Hampton', 'RaiQuan Gray', 'Reggie Bullock', 'Reggie Jackson', 'Ricky Rubio', 'RJ Barrett', 'Robert Covington', 'Robert Williams', 'Rodney McGruder', 'Romeo Langford', 'Rudy Gobert', 'Rui Hachimura', 'Russell Westbrook', 'Saddiq Bey', 'Sam Hauser', 'Sandro Mamukelashvili', 'Santi Aldama', 'Scottie Barnes', 'Seth Curry', 'Shaedon Sharpe', 'Shai Gilgeous-Alexander', 'Shake Milton', 'Shaquille Harrison', 'Skylar Mays', 'Spencer Dinwiddie', 'Stanley Johnson', 'Stephen Curry', 'Steven Adams', 'T.J. McConnell', 'T.J. Warren', 'Talen Horton-Tucker', 'Tari Eason', 'Taurean Prince', 'Terance Mann', 'Terrence Ross', 'Terry Rozier', 'Théo Maledon', 'Thomas Bryant', 'Tim Hardaway Jr.', 'Tobias Harris', 'Torrey Craig', 'Trae Young', 'Tre Jones', 'Tre Mann', 'Trendon Watford', 'Trey Lyles', 'Trey Murphy III', 'Troy Brown Jr.', 'Ty Jerome', 'Tyler Herro', 'Tyrese Haliburton', 'Tyrese Maxey', 'Tyus Jones', 'Victor Oladipo', 'Walker Kessler', 'Wendell Carter Jr.', 'Wenyen Gabriel', 'Wesley Matthews', 'Will Barton', 'Xavier Tillman Sr.', 'Yuta Watanabe', 'Zach Collins', 'Zach LaVine', 'Ziaire Williams', 'Zion Williamson'))

        # Opening our datasets
        cfs = pd.read_excel(f'cfs_2_{Decil_final}.xlsx')
        differences = pd.read_excel(f'differences_2_{Decil_final}.xlsx')
        st.write("<div style='height: 650px;'></div>", unsafe_allow_html=True)


        #if tabs == "1. General Sport Analysis":
        with tabs[0]:
            # st.markdown("<h4 style='text-align: center;'>Dataset in focus</h1>", unsafe_allow_html=True)
            # st.write(df)
            # st.markdown("**Figure 1**: Representation of the DataFrame used. It aggregates all data used in our research.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # Concepts to take into account
            st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Player_differences = differences.loc[Player]

            # # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # # Create a histogram.
            # plt.hist(differences_array, bins=20, edgecolor='black')
            # plt.xlabel('Differences')
            # plt.ylabel('Frequency')
            # st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 4. Violin: Insights from SUGGESTED CHANGES
            # differences_array = differences.values.flatten()
            # # Create a violin plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.violinplot(y = differences_array, color='skyblue')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 5. Density Plot: Insights from SUGGESTED CHANGES
            # differences = differences.squeeze()  # Ensure it's a Series
            # plt.figure(figsize=(10, 10))
            # sns.kdeplot(data=differences, shade=True)
            # plt.xlabel('(CFS - X)')
            # plt.ylabel('Density')
            # st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 6. Radar Chart: Average SUGGESTED CHANGES per feature
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
            # # Create KDE plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.kdeplot(differences_array, shade=True)
            # plt.xlabel('Differences')
            # plt.ylabel('Density')
            # st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Basketball_player_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Basketball_player_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Basketball_player_feature} is recommended to change**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            normalized_data_X = scaler.fit_transform(X)
            normalized_data_cfs = scaler.fit_transform(cfs)
            normalized_data_differences = scaler.fit_transform(differences)

            X_normalized = pd.DataFrame(normalized_data_X, columns=X.columns, index=X.index)
            cfs_normalized = pd.DataFrame(normalized_data_cfs, columns=cfs.columns, index=cfs.index)
            differences_normalized = pd.DataFrame(normalized_data_differences, columns=differences.columns, index=differences.index)

            # # 10. Radar (per player) - INITIAL STATE
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_X_normalized = X_normalized.loc[selected_player]    
            # categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_X_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 11. Radar (per player) - SUGGESTED CHANGES
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_differences = differences.loc[selected_player]    
            # categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            # values = player_differences.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_differences_normalized = differences_normalized.loc[selected_player]    
            # categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 13. Radar (per player) - RECOMMENDED STATE
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_cfs_normalized = cfs_normalized.loc[selected_player]    
            # categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)            
            
            # # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
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
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.bar(shap_values, max_display=15)
            # st.pyplot()  
            # st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 16. SHAP Beeswarm Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.beeswarm(shap_values, max_display=15)
            # st.pyplot()
            # st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            # st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            basketball_player_index_feature = Basketball_player_list.index(Basketball_player_feature)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Basketball_player_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.scatter(shap_values[:, basketball_player_index_feature])
            # st.pyplot()
            # st.markdown(f"**Figure 17**: Scatter plot on feature **{Basketball_player_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Basketball_player_feature_full_name} feature, which means **'how much must {Basketball_player_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Basketball_player_feature_full_name}**.")
            # st.markdown(f"This means that, for positive SHAP values, **{Basketball_player_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Basketball_player_feature_full_name} must impact negatively** the model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 18. SHAP Partial Dependence Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Basketball_player_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.partial_dependence_plot(
            #     Basketball_player_feature, lr.predict, X, ice=False,
            #     model_expected_value=True, feature_expected_value=True) 
            # st.pyplot()
            # st.markdown(f"**Figure 18**: Model's dependence on the feature {Basketball_player_feature_full_name}, now in the new original feature space (X). It explains **how SHAP values of {Basketball_player_feature_full_name} vary across a dataset** and how changes in the {Basketball_player_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Basketball_player_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Basketball_player_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Basketball_player_feature_full_name}** as a certain value;")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            basketball_player_index_player = X_indexes.index(Player)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.waterfall(shap_values[basketball_player_index_player])
            # st.pyplot()
            # st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Player}, instead of, as in the previous two graphs, focusing on feature {Basketball_player_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # Concepts to take into account
            # st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # # 20. SHARP: Rank vs Score
            # import os
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            # st.image("Rank_vs_Score_(2) Basket Player.png")
            # st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 21. SHARP: Top and Bottom 3 Individuals
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            # st.image("Top_bottom_feature_importance_2.png")
            # st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 22. SHARP: Feature Importance
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            # st.image("Strata_boxplot_2.png")
            # st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            # st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} ranking. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            
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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value. \n - Note that highest DiCE values does not necessary mean worse features. DiCE can impact either the best features or the worst features. But overall, the lowest the player ranking, the higher amount tend to be the player average DiCE values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


            # 28. Create a Statistics DataFrame
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Statistics Comparison</h1>", unsafe_allow_html=True)
            average_abs_values = df_values_2.abs().mean() # Calculate the average of the absolute values for each column
            variance_values = df_values_2.var() # Calculate the variance for each column
            diff_max_min_values = df_values_2.max() - df_values_2.min() # Calculate the difference between the maximum and minimum values for each column
            df_stats = pd.DataFrame({
            'Average Absolute Value': average_abs_values,
            'Variance': variance_values,
            'Max-Min Difference (Amplitude)': diff_max_min_values})
            st.dataframe(df_stats, width=900)
            st.markdown(f"**Figure 28**: Table aggregating the average values, the variance and the amplitude from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


            
            # 29. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 29**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
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

            # # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Teams (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 3. Histograms: Insights from SUGGESTED CHANGES
            # # Transforming differences into an array.
            # differences_array = differences.values.flatten()
            # # Create a histogram.
            # plt.hist(differences_array, bins=20, edgecolor='black')
            # plt.xlabel('Differences')
            # plt.ylabel('Frequency')
            # st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 4. Violin: Insights from SUGGESTED CHANGES
            # differences_array = differences.values.flatten()
            # # Create a violin plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.violinplot(y = differences_array, color='skyblue')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 5. Density Plot: Insights from SUGGESTED CHANGES
            # differences = differences.squeeze()  # Ensure it's a Series
            # plt.figure(figsize=(10, 10))
            # sns.kdeplot(data=differences, shade=True)
            # plt.xlabel('(CFS - X)')
            # plt.ylabel('Density')
            # st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 6. Radar Chart: Average SUGGESTED CHANGES per feature
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
            # # Create KDE plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.kdeplot(differences_array, shade=True)
            # plt.xlabel('Differences')
            # plt.ylabel('Density')
            # st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Football_team_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Football_team_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Football_team_feature} is recommended to change**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
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
            # # Filter "differences" DataFrame.
            # player_X_normalized = X_normalized.loc[selected_player]    
            # categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_X_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Team
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            # categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            # values = player_differences.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.bar(shap_values, max_display=15)
            # st.pyplot()  
            # st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 16. SHAP Beeswarm Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.beeswarm(shap_values, max_display=15)
            # st.pyplot()
            # st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            # st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Football_team_index_feature = Football_team_list.index(Football_team_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Football_team_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.scatter(shap_values[:, Football_team_index_feature])
            # st.pyplot()
            # st.markdown(f"**Figure 17**: Scatter plot on feature **{Football_team_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Football_team_feature_full_name} feature, which means **'how much must {Football_team_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Football_team_feature_full_name}**.")
            # st.markdown(f"This means that, for positive SHAP values, **{Football_team_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Football_team_feature_full_name} must impact negatively** the model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 18. SHAP Partial Dependence Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Football_team_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.partial_dependence_plot(
            #     Football_team_feature, lr.predict, X, ice=False,
            #     model_expected_value=True, feature_expected_value=True) 
            # st.pyplot()
            # st.markdown(f"**Figure 18**: Model's dependence on the feature {Football_team_feature_full_name}, now in the new original feature space (X). It explains **how SHAP values of {Football_team_feature_full_name} vary across a dataset** and how changes in the {Football_team_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Football_team_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Football_team_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Football_team_feature_full_name}** as a certain value;")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Football_team_index_player = X_indexes.index(Team)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Team}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.waterfall(shap_values[Football_team_index_player])
            # st.pyplot()
            # st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Team}, instead of, as in the previous two graphs, focusing on feature {Football_team_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # # Concepts to take into account
            # st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # # 20. SHARP: Rank vs Score
            # import os
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            # st.image("Rank_vs_Score_(3) Football_Teams.png")
            # st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 21. SHARP: Top and Bottom 3 Individuals
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            # st.image("Top_bottom_feature_importance_3.png")
            # st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 22. SHARP: Feature Importance
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            # st.image("Strata_boxplot_3.png")
            # st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            # st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Team}. Similarly to SHAP Waterfall, it attempts to explain {Team} ranking. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Team}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Team}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            
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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value. \n - Note that highest DiCE values does not necessary mean worse features. DiCE can impact either the best features or the worst features. But overall, the lowest the player ranking, the higher amount tend to be the player average DiCE values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


            # 28. Create a Statistics DataFrame
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Statistics Comparison</h1>", unsafe_allow_html=True)
            average_abs_values = df_values_2.abs().mean() # Calculate the average of the absolute values for each column
            variance_values = df_values_2.var() # Calculate the variance for each column
            diff_max_min_values = df_values_2.max() - df_values_2.min() # Calculate the difference between the maximum and minimum values for each column
            df_stats = pd.DataFrame({
            'Average Absolute Value': average_abs_values,
            'Variance': variance_values,
            'Max-Min Difference (Amplitude)': diff_max_min_values})
            st.dataframe(df_stats, width=900)
            st.markdown(f"**Figure 28**: Table aggregating the average values, the variance and the amplitude from DiCE, SHAP and SHARP applied to {Team} and according to the selected decil.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            

            # 29. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 29**: Correlation matrix between DiCE, SHAP and SHARP applied to {Team} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


    # Check if the user selects the type as Player
    elif Team_vs_Player == 'Player':
        Player = st.sidebar.selectbox('Select the Player:', ('Abbas Ibrahim_252471', 'Abdoulaye Ba_204826', 'Abdu Cadri-Conte_247180', 'Abdul Mumin_235662', 'Abdul Wahab-Ibrahim_257995', 'Abel Issa-Camara_208005', 'Abel Ruiz-Ortega_239250', 'Abraham Ayomide-Marcus_264450', 'Achraf Lazaar_206161', 'Adel Taarabt_179605', 'Adilio Correia-Dos-Santos_262436', 'Adrian Marin-Gomez_224921', 'Afonso Gamelas-Sousa_258698', 'Alejandro Grimaldo-Garcia_210035', 'Alex Mendez_252285', 'Alex Nascimento_264268', 'Alexandre Penetra-Correia_264288', 'Alfa Semedo-Esteves_239641', 'Ali Alipour_259094', 'Ali Musrati_252308', 'Alioune Ndour_263366', 'Alisson Pelegrini-Safira_264463', 'Allano Brendon-De-Souza-Lima_229556', 'Anderson Oliveira-Da-Silva_254225', 'Andre Bukia_229889', 'Andre Castro-Pereira_184133', 'Andre Clovis-Silva-Filho_242406', 'Andre Filipe-Bras-Andre_199626', 'Andre Filipe-Cunha-Vidigal_243401', 'Andre Filipe-Luz-Horta_226370', 'Andre Filipe-Rio-Liberal_264307', 'Andre Filipe-Russo-Franco_262414', 'Andre Fonseca-Amaro_260121', 'Andre Gomes-Magalhaes-Almeida_194022', 'Andre Luis-Silva-De-Aguiar_246093', 'Andre Miguel-Lapa-Ricardo_264291', 'Andre Oliveira-Silva_242448', 'Andrei Chindris_248004', 'Andrija Lukovic_250962', 'Antoine Leautey_237198', 'Antonio Manuel-Pereira-Xavier_226278', 'Antonio Martinez-Lopez_234826', 'Antony Alves-Santos_263947', 'Arsenio Martins-Lafuente-Nunes_213746', 'Arthur Gomes-Lourenco_236782', 'Artur Jorge-Marques-Amorim_236505', 'Aylton Filipe-Boa-Morte_239186', 'Baptiste Aloe_208834', 'Bernardo Maria-Cardoso-Vital_263670', 'Bilel Aouacheria_240943', 'Boubacar Rafael-Neto-Hanne_240951', 'Brazao Teixeira-Pedro-David_248572', 'Bruno Alexandre-Vieira-Almeida_237670', 'Bruno Duarte-Da-Silva_252830', 'Bruno Goncalves-Do-Prado_262437', 'Bruno Miguel-Ponces-Lourenco_248811', 'Bruno Rafael-R-Do-Nascimento_238079', 'Bruno Ricardo-Valdez-Wilson_236924', 'Bruno Rodrigues_260167', 'Bruno Vinicius-Souza-Ramos_238858', 'Bruno Xavier-Almeida-Costa_244960', 'Carles Soria-Grau_262413', 'Carlos Vinicius-Santos-De-Jesus_209828', 'Cassiano Dias-Moreira_211561', 'Cesar Fernando-Simoes-Sousa_261777', 'Chancel Mbemba_210897', 'Charles Pickel_233884', 'Chima Akas_241765', 'Christian Neiva-Afonso_213795', 'Claudemir Domingues-De-Souza_186935', 'Claudio Winck-Neto_216580', 'Clecildo Rafael-Martins-Ladislau_200984', 'Clesio Bauque_218614', 'Cristian Gonzalez_236020', 'Cristian Parano_243469', 'Cryzan Queiroz-Barcelos_230601', 'Daniel Da-Silva-Dos-Anjos_232228', 'Daniel Santos-Braganca_252143', 'Danny Agostinho-Henriques_255524', 'Darwin Nunez_253072', 'David Dias-Resende-Bruno_205846', 'David Jose-Gomes-O-Tavares_243650', 'David Mota-Teixeira-Carmo_251616', 'Denilson Pereira-Junior_215557', 'Denis Will-Poha_230860', 'Derik Gean-Severino-Lacerda_258386', 'Diogo Alexandre-Almeida-Mendes_243918', 'Diogo Antonio-Cupido-Goncalves_231390', 'Diogo Da-Costa-Silva_259018', 'Diogo Dos-Santos-Cabral_246974', 'Diogo Filipe-Pinto-Leite_234575', 'Diogo Filipe-Rocha-Costa_222461', 'Diogo Jose-Figueiras_204401', 'Domingos Andre-Ribeiro-Almeida_247729', 'Douglas Willian-Da-Silva-Souza_220735', 'Dylan Batubinsika_239814', 'Eboue Kouassi_233840', 'Eduardo De-Sousa-Santos_251534', 'Eduardo Filipe-Quaresma_252144', 'Eduardo Gabriel-Aquino-Cossa_238896', 'Emmanuel Hackman_229890', 'Eugeni Valderrama-Domenech_209695', 'Eulanio Angelo-Chipela-Gomes_241981', 'Everton Sousa-Soares_222716', 'Ewerton Da-Silva-Pereira_207943', 'F Evanilson-De-Lima-Barbosa_256612', 'Fabiano Josue-De-Souza-Silva_257162', 'Fabio Daniel-Ferreira-Vieira_256958', 'Fabio Diogo-Agrela-Ferreira_234164', 'Fabio Jose-Ferreira-Pacheco_184606', 'Fabio Rafael-Rodrigues-Cardoso_213917', 'Fabio Samuel-Amorim-Silva_223302', 'Fabio Santos-Martins_224394', 'Fabricio Dos-Santos-Messias_202350', 'Fahd Moufi_232296', 'Falaye Sacko_230147', 'Fali Cande_256856', 'Felicio Mendes-Milson_256883', 'Felipe Augusto-Rodrigues-Pires_226706', 'Felipe Rodrigues-Da-Silva_256008', 'Fernando Manuel-Ferreira-Fonseca_238507', 'Filipe Bem-Relvas-V-Oliveira_263897', 'Filipe Couto-Cardoso_263072', 'Filipe Miguel-Barros-Soares_252139', 'Filipe Miguel-Neves-Ferreira_215665', 'Flavio B-J-Nazinho_261531', 'Flavio Da-Silva-Ramos_216815', 'Fode Konate_252045', 'Francis Cann_262428', 'Francisco F-Da-Conceicao_261050', 'Francisco Jorge-Tavares-Oliveira_262416', 'Francisco Jose-Coelho-Teixeira_258917', 'Francisco Jose-Navarro-Aliaga_247138', 'Francisco L-Lima-Silva-Machado_243686', 'Francisco Oliveira-Geraldes_228650', 'Francisco Reis-Ferreira_242236', 'Francisco Ribeiro-Tome_262430', 'Francisco Sampaio-Moura_251615', 'Gabriel Appelt-Pires_208141', 'Gaius Makouta_235570', 'Gaston Campi_220429', 'Gedson Carvalho-Fernandes_234568', 'Geovani Junior_264289', 'Giannelli Imbula_197681', 'Gil Bastiao-Dias_229453', 'Gilberto Moraes-Junior_211567', 'Gilson Benchimol-Tavares_263669', 'Giorgi Aburjania_232090', 'Gleison Wilson-Da-Silva-Moreira_234929', 'Godfried Frimpong_263255', 'Goncalo Baptista-Franco_258596', 'Goncalo Bernardo-Inacio_257179', 'Goncalo Domingues-Agrelos_239902', 'Goncalo Esteves_264022', 'Goncalo Matias-Ramos_256903', 'Guilherme Araujo-Soares_262268', 'Guilherme Borges-Guedes_263634', 'Guilherme Mantuan_258783', 'Guilherme Schettine-Guimaraes_236555', 'Gurkan Baskan_256931', 'Gustavo Affonso-Sauerbeck_230665', 'Hamidou Keyta_237380', 'Haris Seferovic_193408', 'Helder Jose-Castro-Ferreira_239446', 'Helder Jose-Oliveira-Sa_261689', 'Henrique Jocu_263144', 'Henrique Martins-Gomes_251450', 'Henrique Roberto-Rafael_262932', 'Herculano Bucancil-Nabian_263659', 'Heriberto Borges-Tavares_243270', 'Hernan De-La-Fuente_241696', 'Hidemasa Morita_242087', 'Hugo Silva-Oliveira_263623', 'Hugues Evrard-Zagbayou_262431', 'Iago Justen-Maidana-Martins_226151', 'Ibrahima Camara_247288', 'Igor De-Carvalho-Juliao_212367', 'Iker Undabarrena_229361', 'Ilija Vukotic_246192', 'Iuri Jose-Picanco-Medeiros_212230', 'Ivan Angulo_236800', 'Ivan Jaime-Pajuelo_246070', 'Ivan Marcano-Sierra_173521', 'Ivan Rossi_223495', 'Ivanildo Jorge-Mendes-Fernandes_244963', 'Ivo Tiago-Santos-Rodrigues_226877', 'Jackson Porozo_260740', 'Jan Vertonghen_172871', 'Javier Garcia-Fernandez_161754', 'Jean Carlos-De-Souza-Irmer_236303', 'Jean Gorby_264418', 'Jean Patric-Lima-Dos-Reis_258306', 'Jefferson F-Isidio_259325', 'Jefferson Pessanha-Agostinho_256893', 'Jeriel De-Santis_253919', 'Jesus Corona_193165', 'Jhon Murillo_228738', 'Joao Afonso-Crispim_220693', 'Joao Caiado-Vaz-Dias_245334', 'Joao Carlos-Fonseca-Silva_234935', 'Joao Carlos-Reis-Graca_204731', 'Joao Diogo-Fonseca-Ferreira_252577', 'Joao Jose-Pereira-Da-Costa_229473', 'Joao Maria-Palhinha-Goncalves_229391', 'Joao Mario-Naval-Costa-Eduardo_212814', 'Joao Mario-Neto-Lopes_257290', 'Joao Miguel-Ribeiro-Vigario_229147', 'Joao Othavio-Basso_236255', 'Joao Paulo-Dias-Fernandes_216531', 'Joao Paulo-Marques-Goncalves_252407', 'Joao Pedro-Almeida-Machado_213557', 'Joao Pedro-Da-Costa-Gamboa_228782', 'Joao Pedro-Sousa-Silva_252827', 'Joao Ricardo-Da-Silva-Afonso_223818', 'Jobson De-Brito-Gonzaga_251681', 'Joel Antonio-Soares-Ferreira_235486', 'Joel Tagueu_225456', 'Joeliton Lima-Santos_215496', 'Johan Mina_256830', 'Jordan Van-Der-Gaag_244600', 'Jorge Fernando-Dos-Santos-Silva_251232', 'Jorge Filipe-Oliveira-Fernandes_242359', 'Jorge Miguel-Lopes-Xavier_263645', 'Jorge Saenz-De-Miera_225106', 'Jose Carlos-Reis-Goncalves_258786', 'Jose Edgar-Andrade-Da-Costa_194956', 'Jose Manuel-Hernando-Riol_243155', 'Jose Manuel-Velazquez_196881', 'Jose Uilton-Silva-De-Jesus_248833', 'Joseph Amoah_228810', 'Jovane Eduardo-Borges-Cabral_244193', 'Juan Boselli_264427', 'Juan Delgado_214313', 'Juan Jose-Calero_226832', 'Julian Weigl_222028', 'Julio Rodrigues-Romao_258360', 'Kanya Fujimoto_257899', 'Kenji Gorre_213694', 'Kennedy Boateng_238987', 'Kepler Laveran-Lima-Ferreira_120533', 'Kevin Zohi_240730', 'Koffi Kouao_239919', 'Kuku Fidelis_261883', 'Lawrence Ofori_247258', 'Lazar Rosic_235002', 'Leandro Andrade-Silva_259771', 'Leandro Miguel-Pereira-Silva_230027', 'Leonardo Ruiz_231868', 'Lincoln Oliveira-Dos-Santos_233585', 'Loreintz Rosier_262412', 'Luca Van-Der-Gaag_253380', 'Lucas Da-Silva-De-Jesus_258350', 'Lucas Da-Silva-Izidoro_235782', 'Lucas De-Souza-Cunha_241829', 'Lucas Domingues-Piazon_203038', 'Lucas Fernandes-Da-Silva_234119', 'Lucas Henrique-Da-Silva_244006', 'Lucas Possignolo_238862', 'Lucas Queiroz-Canteiro_251990', 'Lucas Verissimo-Da-Silva_235688', 'Luis Carlos-Novo-Neto_204341', 'Luis Diaz_241084', 'Luis Fellipe-Rodrigues-Mota_263973', 'Luis Miguel-Afonso-Fernandes_197965', 'Luis Miguel-Castelo-Santos_257159', 'Luis Pedro-Alves-Bastos_258450', 'Luis Pedro-Pinto-Trabulo_235225', 'Luis Rafael-Soares-Alves_218936', 'Luiz Carlos-Martins-Moreira_148403', 'Luiz Gustavo-Benmuyal-Reis_262616', 'Luiz Phellype-Luciano-Silva_211461', 'Manconi Soriano-Mane_251504', 'Manuel Maria-Machado-C-Namora_260744', 'Manuel Ugarte_253306', 'Marcelo Amado-Djalo-Taritolay_220191', 'Marcelo Machado-Vilela_262433', 'Marcelo Ribeiro-Dos-Santos_263991', 'Marco Joao-Costa-Baixinho_229496', 'Marco Paulo-Silva-Soares_184142', 'Marcos Paulo-Costa-Do-Nascimento_255371', 'Marcos Paulo-Gelmini-Gomes_186568', 'Marcus Edwards_235619', 'Mario Gonzalez-Gutierrez_235970', 'Marko Grujic_232099', 'Matchoi Bobo-Djalo_252470', 'Mateus Quaresma-Correia_262438', 'Mateus Uribe_214047', 'Matheus De-Barros-Da-Silva_257265', 'Matheus De-Mello-Costa_262429', 'Matheus Luiz-Nunes_253124', 'Matheus Reis-De-Lima_230625', 'Mehdi Taremi_241788', 'Miguel Angelo-Moreira-Magalhaes_240882', 'Miguel Silva-Reisinho_252310', 'Mikel Agu_209984', 'Mikel Villanueva_233329', 'Miullen N-Felicio-Carva_258497', 'Modibo Sagnan_241708', 'Mohamed Aidara_262426', 'Mohamed Bouldini_262744', 'Mohamed Diaby_248831', 'Mohammad Mohebi_264425', 'Moises Mosquera_258919', 'Murilo De-Souza-Costa_225696', 'Nahuel Ferraresi_246388', 'Naoufel Khacef_258598', 'Nathan Santos-De-Araujo_258889', 'Nemanja Radonjic_243656', 'Nicolas Janvier_231266', 'Nicolas Otamendi_192366', 'Nikola Jambor_234635', 'Nilton Varela-Lopes_254258', 'Nuno Miguel-Gomes-Dos-Santos_227890', 'Nuno Miguel-Jeronimo-Sequeira_217546', 'Nuno Miguel-Reis-Lima_258642', 'Nuno Miguel-Valente-Santos_251438', 'Oday Dabbagh_264321', 'Or Dasa_263633', 'Oscar Estupinan_228761', 'Otavio Edmilson-Da-Silva-Monteiro_210411', 'Pablo Felipe-Pereira-De-Jesus_264290', 'Pablo Renan-Dos-Santos_239872', 'Pablo Sarabia-Garcia_198950', 'Patrick William-Sa-De-Oliveira_252150', 'Paulo Andre-Rodrigues-Oliveira_210679', 'Paulo Estrela-Moreira-Alves_255255', 'Paulo Henrique-Rodrigues-Cabral_229498', 'Paulo Sergio-Mota_211014', 'Pedro Alves-Correia_239915', 'Pedro Antonio-Pereira-Goncalves_240950', 'Pedro Antonio-Porro-Sauceda_243576', 'Pedro Augusto-Borges-Da-Costa_252291', 'Pedro David-Rosendo-Marques_236673', 'Pedro Filipe-Barbosa-Moreira_235210', 'Pedro Filipe-Rodrigues_239950', 'Pedro Henrique-De-Oliveira-Correia_258356', 'Pedro Henrique-Rocha-Pelagio_246866', 'Pedro Jorge-Goncalves-Malheiro_263810', 'Pedro Manuel-Da-Silva-Moreira_199848', 'Pedro Miguel-Cunha-Sa_238856', 'Pedro Miguel-Santos-Amador_256044', 'Pedro Nuno-Fernandes-Ferreira_224538', 'Petar Musa_244797', 'Pierre Sagna_201512', 'Racine Coly_222840', 'Rafael A-Ferreira-Silva_216547', 'Rafael Antonio-Figueiredo-Ramos_224891', 'Rafael Avelino-Pinto-Barbosa_242389', 'Rafael Euclides-Soares-Camacho_243055', 'Rafael Vicente-Ferreira-Santos_263643', 'Rafik Guitane_235955', 'Raphael Gregorio-Guzzo_228589', 'Raul Michel-Melo-Da-Silva_221540', 'Reggie Cannon_237000', 'Renat Dadashov_238736', 'Renato Barbosa-Dos-Santos-Junior_263239', 'Ricardo Andrade-Quaresma-Bernardo_20775', 'Ricardo Jorge-Da-Luz-Horta_213516', 'Ricardo Jorge-Oliveira-Antonio_263574', 'Ricardo Miguel-Martins-Alves_233870', 'Ricardo Sousa-Esgaio_212213', 'Ricardo Viana-Filho_263791', 'Riccieli Da-Silva-Junior_252095', 'Richard Ofori_208391', 'Roberto Jesus-Machado-Beto-Alves_239875', 'Rocha Moreira-Nuno-Goncalo_262985', 'Rodrigo Abascal_253276', 'Rodrigo Cunha-Pereira-De-Pinho_229683', 'Rodrigo F-Conceicao_263566', 'Rodrigo Marcos-Rodrigues-Andrade_264308', 'Rodrigo Martins-Gomes_259170', 'Rodrigo Ribeiro-Valente_264459', 'Rolando Jorge-Pires-Da-Fonseca_163083', 'Roman Yaremchuk_240702', 'Romario Manuel-Silva-Baro_252038', 'Ruben Alexandre-Gomes-Oliveira_213801', 'Ruben Alexandre-Rocha-Lima_192156', 'Ruben Barcelos-De-Sousa-Lameiras_225363', 'Ruben Daniel-Fonseca-Macedo_246834', 'Ruben Del-Campo_256852', 'Ruben Ismael-Valente-Ramos_252080', 'Ruben Miguel-Santos-Fernandes_18115', 'Ruben Miguel-Valente-Fonseca_252391', 'Ruben Nascimento-Vinagre_235172', 'Rui F-Da-Cunha-Correia_251486', 'Rui Filipe-Caetano-Moura_223297', 'Rui Miguel-Guerra-Pires_251669', 'Rui Pedro-Da-Rocha-Fonte_183518', 'Rui Pedro-Silva-Costa_239922', 'Ryotaro Meshino_237435', 'Salvador Jose-Milhazes-Agra_204737', 'Samuel Dias-Lino_251445', 'Sana Dafa-Gomes_263898', 'Sandro R-G-Cordeiro_190782', 'Sebastian Coates_197655', 'Sebastian Perez_214606', 'Sergio Miguel-Relvas-De-Oliveira_198031', 'Shoya Nakajima_232862', 'Shuhei Kawasaki_255829', 'Silvio Manuel-Ferreira-Sa-Pereira_189530', 'Simon Banza_231652', 'Simone Muratore_257195', 'Soualiho Meite_205391', 'Souleymane Aw_239074', 'Sphephelo Sithole_257293', 'Stefano Beltrame_208523', 'Stephen Antunes-Eustaquio_242380', 'Steven De-Sousa-Vitoria_192528', 'Telmo Arcanjo_257194', 'Thales Bento-Oleques_262435', 'Thibang Phete_231688', 'Tiago Alexandre-De-Sousa-Esgaio_251994', 'Tiago Almeida-Ilori_205185', 'Tiago B-De-Melo-Tomas_257073', 'Tiago Filipe-Alves-Araujo_264280', 'Tiago Filipe-Oliveira-Dantas_251436', 'Tiago Fontoura-F-Morais_259054', 'Tiago Melo-Almeida_256918', 'Tiago Rafael-Maia-Silva_212474', 'Tim Soderstrom_212693', 'Tomas Aresta-Machado-Ribeiro_253378', 'Tomas Costa-Silva_262986', 'Tomas Pais-Sarmento-Castro_255114', 'Tomas Reymao-Nogueira_242390', 'Tomas Romano-Dos-Santos-Handel_263059', 'Toni Borevkovic_244377', 'Trova Boni_241946', 'Valentino Lazaro_211147', 'Vitor Carvalho-Vieira_255471', 'Vitor Costa-De-Brito_232388', 'Vitor Machado-Ferreira_255253', 'Vitor Manuel-Carvalho-Oliveira_263829', 'Vitor Tormena-De-Farias_246098', 'Vitorino Pacheco-Antunes_178424', 'Vivaldo Neto_242181', 'Volnei Feltes_263342', 'Walterson Silva_236630', 'Wendell Nascimento-Borges_216466', 'Wenderson Nascimento-Galeno_239482', 'Wilinton Aponza_264595', 'Willyan Da-Silva-Rocha_251901', 'Wilson Migueis-Manafa-Janco_238857', 'Yan Bueno-Couto_259075', 'Yan Matheus-Santos-Souza_234627', 'Yanis Hamache_257234', 'Yaw Moses_262440', 'Yohan Tavares_199051', 'Yusupha Njie_239303', 'Zaidu Sanusi_251528', 'Zainadine Chavango-Junior_217235', 'Zouhair Feddal_205705'))        
        
        # df
        df = pd.read_excel('4_Football_Player_FIFA 2022.xlsx', sheet_name= 'PBC players_22')
        df = df[df['player_positions'] != 'GK']
        selected_leagues = ['Portuguese Liga ZON SAGRES']
        df = df[df['league_name'].isin(selected_leagues)]
        df.columns = df.columns.str.replace("%", "perc").str.replace("/", "_").str.replace(" ", "_").str.replace(".", "").str.lower()
        df = df.drop(columns=["player_url", # Not informative.
                              "long_name", # Non-printable information.
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
                            "nation_flag_url",
                            "nation_flag_url1"]).set_index("sofifa_id")
        df[['work_rate_attacking', 'work_rate_defensive']] = df['work_rate'].str.split('/', expand=True).replace({'Low': 1, 'Medium': 2, 'High': 3})
        df = df.drop(columns=["work_rate"]) # Not informative.
        df.isnull().sum()[df.isnull().sum() > 0]
        df.fillna(0, inplace=True)
        X = df.drop(columns=["overall"]).set_index("player_name")
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
        
        Player_2 = st.sidebar.selectbox('Select the Player to Compare:', ('Abbas Ibrahim_252471', 'Abdoulaye Ba_204826', 'Abdu Cadri-Conte_247180', 'Abdul Mumin_235662', 'Abdul Wahab-Ibrahim_257995', 'Abel Issa-Camara_208005', 'Abel Ruiz-Ortega_239250', 'Abraham Ayomide-Marcus_264450', 'Achraf Lazaar_206161', 'Adel Taarabt_179605', 'Adilio Correia-Dos-Santos_262436', 'Adrian Marin-Gomez_224921', 'Afonso Gamelas-Sousa_258698', 'Alejandro Grimaldo-Garcia_210035', 'Alex Mendez_252285', 'Alex Nascimento_264268', 'Alexandre Penetra-Correia_264288', 'Alfa Semedo-Esteves_239641', 'Ali Alipour_259094', 'Ali Musrati_252308', 'Alioune Ndour_263366', 'Alisson Pelegrini-Safira_264463', 'Allano Brendon-De-Souza-Lima_229556', 'Anderson Oliveira-Da-Silva_254225', 'Andre Bukia_229889', 'Andre Castro-Pereira_184133', 'Andre Clovis-Silva-Filho_242406', 'Andre Filipe-Bras-Andre_199626', 'Andre Filipe-Cunha-Vidigal_243401', 'Andre Filipe-Luz-Horta_226370', 'Andre Filipe-Rio-Liberal_264307', 'Andre Filipe-Russo-Franco_262414', 'Andre Fonseca-Amaro_260121', 'Andre Gomes-Magalhaes-Almeida_194022', 'Andre Luis-Silva-De-Aguiar_246093', 'Andre Miguel-Lapa-Ricardo_264291', 'Andre Oliveira-Silva_242448', 'Andrei Chindris_248004', 'Andrija Lukovic_250962', 'Antoine Leautey_237198', 'Antonio Manuel-Pereira-Xavier_226278', 'Antonio Martinez-Lopez_234826', 'Antony Alves-Santos_263947', 'Arsenio Martins-Lafuente-Nunes_213746', 'Arthur Gomes-Lourenco_236782', 'Artur Jorge-Marques-Amorim_236505', 'Aylton Filipe-Boa-Morte_239186', 'Baptiste Aloe_208834', 'Bernardo Maria-Cardoso-Vital_263670', 'Bilel Aouacheria_240943', 'Boubacar Rafael-Neto-Hanne_240951', 'Brazao Teixeira-Pedro-David_248572', 'Bruno Alexandre-Vieira-Almeida_237670', 'Bruno Duarte-Da-Silva_252830', 'Bruno Goncalves-Do-Prado_262437', 'Bruno Miguel-Ponces-Lourenco_248811', 'Bruno Rafael-R-Do-Nascimento_238079', 'Bruno Ricardo-Valdez-Wilson_236924', 'Bruno Rodrigues_260167', 'Bruno Vinicius-Souza-Ramos_238858', 'Bruno Xavier-Almeida-Costa_244960', 'Carles Soria-Grau_262413', 'Carlos Vinicius-Santos-De-Jesus_209828', 'Cassiano Dias-Moreira_211561', 'Cesar Fernando-Simoes-Sousa_261777', 'Chancel Mbemba_210897', 'Charles Pickel_233884', 'Chima Akas_241765', 'Christian Neiva-Afonso_213795', 'Claudemir Domingues-De-Souza_186935', 'Claudio Winck-Neto_216580', 'Clecildo Rafael-Martins-Ladislau_200984', 'Clesio Bauque_218614', 'Cristian Gonzalez_236020', 'Cristian Parano_243469', 'Cryzan Queiroz-Barcelos_230601', 'Daniel Da-Silva-Dos-Anjos_232228', 'Daniel Santos-Braganca_252143', 'Danny Agostinho-Henriques_255524', 'Darwin Nunez_253072', 'David Dias-Resende-Bruno_205846', 'David Jose-Gomes-O-Tavares_243650', 'David Mota-Teixeira-Carmo_251616', 'Denilson Pereira-Junior_215557', 'Denis Will-Poha_230860', 'Derik Gean-Severino-Lacerda_258386', 'Diogo Alexandre-Almeida-Mendes_243918', 'Diogo Antonio-Cupido-Goncalves_231390', 'Diogo Da-Costa-Silva_259018', 'Diogo Dos-Santos-Cabral_246974', 'Diogo Filipe-Pinto-Leite_234575', 'Diogo Filipe-Rocha-Costa_222461', 'Diogo Jose-Figueiras_204401', 'Domingos Andre-Ribeiro-Almeida_247729', 'Douglas Willian-Da-Silva-Souza_220735', 'Dylan Batubinsika_239814', 'Eboue Kouassi_233840', 'Eduardo De-Sousa-Santos_251534', 'Eduardo Filipe-Quaresma_252144', 'Eduardo Gabriel-Aquino-Cossa_238896', 'Emmanuel Hackman_229890', 'Eugeni Valderrama-Domenech_209695', 'Eulanio Angelo-Chipela-Gomes_241981', 'Everton Sousa-Soares_222716', 'Ewerton Da-Silva-Pereira_207943', 'F Evanilson-De-Lima-Barbosa_256612', 'Fabiano Josue-De-Souza-Silva_257162', 'Fabio Daniel-Ferreira-Vieira_256958', 'Fabio Diogo-Agrela-Ferreira_234164', 'Fabio Jose-Ferreira-Pacheco_184606', 'Fabio Rafael-Rodrigues-Cardoso_213917', 'Fabio Samuel-Amorim-Silva_223302', 'Fabio Santos-Martins_224394', 'Fabricio Dos-Santos-Messias_202350', 'Fahd Moufi_232296', 'Falaye Sacko_230147', 'Fali Cande_256856', 'Felicio Mendes-Milson_256883', 'Felipe Augusto-Rodrigues-Pires_226706', 'Felipe Rodrigues-Da-Silva_256008', 'Fernando Manuel-Ferreira-Fonseca_238507', 'Filipe Bem-Relvas-V-Oliveira_263897', 'Filipe Couto-Cardoso_263072', 'Filipe Miguel-Barros-Soares_252139', 'Filipe Miguel-Neves-Ferreira_215665', 'Flavio B-J-Nazinho_261531', 'Flavio Da-Silva-Ramos_216815', 'Fode Konate_252045', 'Francis Cann_262428', 'Francisco F-Da-Conceicao_261050', 'Francisco Jorge-Tavares-Oliveira_262416', 'Francisco Jose-Coelho-Teixeira_258917', 'Francisco Jose-Navarro-Aliaga_247138', 'Francisco L-Lima-Silva-Machado_243686', 'Francisco Oliveira-Geraldes_228650', 'Francisco Reis-Ferreira_242236', 'Francisco Ribeiro-Tome_262430', 'Francisco Sampaio-Moura_251615', 'Gabriel Appelt-Pires_208141', 'Gaius Makouta_235570', 'Gaston Campi_220429', 'Gedson Carvalho-Fernandes_234568', 'Geovani Junior_264289', 'Giannelli Imbula_197681', 'Gil Bastiao-Dias_229453', 'Gilberto Moraes-Junior_211567', 'Gilson Benchimol-Tavares_263669', 'Giorgi Aburjania_232090', 'Gleison Wilson-Da-Silva-Moreira_234929', 'Godfried Frimpong_263255', 'Goncalo Baptista-Franco_258596', 'Goncalo Bernardo-Inacio_257179', 'Goncalo Domingues-Agrelos_239902', 'Goncalo Esteves_264022', 'Goncalo Matias-Ramos_256903', 'Guilherme Araujo-Soares_262268', 'Guilherme Borges-Guedes_263634', 'Guilherme Mantuan_258783', 'Guilherme Schettine-Guimaraes_236555', 'Gurkan Baskan_256931', 'Gustavo Affonso-Sauerbeck_230665', 'Hamidou Keyta_237380', 'Haris Seferovic_193408', 'Helder Jose-Castro-Ferreira_239446', 'Helder Jose-Oliveira-Sa_261689', 'Henrique Jocu_263144', 'Henrique Martins-Gomes_251450', 'Henrique Roberto-Rafael_262932', 'Herculano Bucancil-Nabian_263659', 'Heriberto Borges-Tavares_243270', 'Hernan De-La-Fuente_241696', 'Hidemasa Morita_242087', 'Hugo Silva-Oliveira_263623', 'Hugues Evrard-Zagbayou_262431', 'Iago Justen-Maidana-Martins_226151', 'Ibrahima Camara_247288', 'Igor De-Carvalho-Juliao_212367', 'Iker Undabarrena_229361', 'Ilija Vukotic_246192', 'Iuri Jose-Picanco-Medeiros_212230', 'Ivan Angulo_236800', 'Ivan Jaime-Pajuelo_246070', 'Ivan Marcano-Sierra_173521', 'Ivan Rossi_223495', 'Ivanildo Jorge-Mendes-Fernandes_244963', 'Ivo Tiago-Santos-Rodrigues_226877', 'Jackson Porozo_260740', 'Jan Vertonghen_172871', 'Javier Garcia-Fernandez_161754', 'Jean Carlos-De-Souza-Irmer_236303', 'Jean Gorby_264418', 'Jean Patric-Lima-Dos-Reis_258306', 'Jefferson F-Isidio_259325', 'Jefferson Pessanha-Agostinho_256893', 'Jeriel De-Santis_253919', 'Jesus Corona_193165', 'Jhon Murillo_228738', 'Joao Afonso-Crispim_220693', 'Joao Caiado-Vaz-Dias_245334', 'Joao Carlos-Fonseca-Silva_234935', 'Joao Carlos-Reis-Graca_204731', 'Joao Diogo-Fonseca-Ferreira_252577', 'Joao Jose-Pereira-Da-Costa_229473', 'Joao Maria-Palhinha-Goncalves_229391', 'Joao Mario-Naval-Costa-Eduardo_212814', 'Joao Mario-Neto-Lopes_257290', 'Joao Miguel-Ribeiro-Vigario_229147', 'Joao Othavio-Basso_236255', 'Joao Paulo-Dias-Fernandes_216531', 'Joao Paulo-Marques-Goncalves_252407', 'Joao Pedro-Almeida-Machado_213557', 'Joao Pedro-Da-Costa-Gamboa_228782', 'Joao Pedro-Sousa-Silva_252827', 'Joao Ricardo-Da-Silva-Afonso_223818', 'Jobson De-Brito-Gonzaga_251681', 'Joel Antonio-Soares-Ferreira_235486', 'Joel Tagueu_225456', 'Joeliton Lima-Santos_215496', 'Johan Mina_256830', 'Jordan Van-Der-Gaag_244600', 'Jorge Fernando-Dos-Santos-Silva_251232', 'Jorge Filipe-Oliveira-Fernandes_242359', 'Jorge Miguel-Lopes-Xavier_263645', 'Jorge Saenz-De-Miera_225106', 'Jose Carlos-Reis-Goncalves_258786', 'Jose Edgar-Andrade-Da-Costa_194956', 'Jose Manuel-Hernando-Riol_243155', 'Jose Manuel-Velazquez_196881', 'Jose Uilton-Silva-De-Jesus_248833', 'Joseph Amoah_228810', 'Jovane Eduardo-Borges-Cabral_244193', 'Juan Boselli_264427', 'Juan Delgado_214313', 'Juan Jose-Calero_226832', 'Julian Weigl_222028', 'Julio Rodrigues-Romao_258360', 'Kanya Fujimoto_257899', 'Kenji Gorre_213694', 'Kennedy Boateng_238987', 'Kepler Laveran-Lima-Ferreira_120533', 'Kevin Zohi_240730', 'Koffi Kouao_239919', 'Kuku Fidelis_261883', 'Lawrence Ofori_247258', 'Lazar Rosic_235002', 'Leandro Andrade-Silva_259771', 'Leandro Miguel-Pereira-Silva_230027', 'Leonardo Ruiz_231868', 'Lincoln Oliveira-Dos-Santos_233585', 'Loreintz Rosier_262412', 'Luca Van-Der-Gaag_253380', 'Lucas Da-Silva-De-Jesus_258350', 'Lucas Da-Silva-Izidoro_235782', 'Lucas De-Souza-Cunha_241829', 'Lucas Domingues-Piazon_203038', 'Lucas Fernandes-Da-Silva_234119', 'Lucas Henrique-Da-Silva_244006', 'Lucas Possignolo_238862', 'Lucas Queiroz-Canteiro_251990', 'Lucas Verissimo-Da-Silva_235688', 'Luis Carlos-Novo-Neto_204341', 'Luis Diaz_241084', 'Luis Fellipe-Rodrigues-Mota_263973', 'Luis Miguel-Afonso-Fernandes_197965', 'Luis Miguel-Castelo-Santos_257159', 'Luis Pedro-Alves-Bastos_258450', 'Luis Pedro-Pinto-Trabulo_235225', 'Luis Rafael-Soares-Alves_218936', 'Luiz Carlos-Martins-Moreira_148403', 'Luiz Gustavo-Benmuyal-Reis_262616', 'Luiz Phellype-Luciano-Silva_211461', 'Manconi Soriano-Mane_251504', 'Manuel Maria-Machado-C-Namora_260744', 'Manuel Ugarte_253306', 'Marcelo Amado-Djalo-Taritolay_220191', 'Marcelo Machado-Vilela_262433', 'Marcelo Ribeiro-Dos-Santos_263991', 'Marco Joao-Costa-Baixinho_229496', 'Marco Paulo-Silva-Soares_184142', 'Marcos Paulo-Costa-Do-Nascimento_255371', 'Marcos Paulo-Gelmini-Gomes_186568', 'Marcus Edwards_235619', 'Mario Gonzalez-Gutierrez_235970', 'Marko Grujic_232099', 'Matchoi Bobo-Djalo_252470', 'Mateus Quaresma-Correia_262438', 'Mateus Uribe_214047', 'Matheus De-Barros-Da-Silva_257265', 'Matheus De-Mello-Costa_262429', 'Matheus Luiz-Nunes_253124', 'Matheus Reis-De-Lima_230625', 'Mehdi Taremi_241788', 'Miguel Angelo-Moreira-Magalhaes_240882', 'Miguel Silva-Reisinho_252310', 'Mikel Agu_209984', 'Mikel Villanueva_233329', 'Miullen N-Felicio-Carva_258497', 'Modibo Sagnan_241708', 'Mohamed Aidara_262426', 'Mohamed Bouldini_262744', 'Mohamed Diaby_248831', 'Mohammad Mohebi_264425', 'Moises Mosquera_258919', 'Murilo De-Souza-Costa_225696', 'Nahuel Ferraresi_246388', 'Naoufel Khacef_258598', 'Nathan Santos-De-Araujo_258889', 'Nemanja Radonjic_243656', 'Nicolas Janvier_231266', 'Nicolas Otamendi_192366', 'Nikola Jambor_234635', 'Nilton Varela-Lopes_254258', 'Nuno Miguel-Gomes-Dos-Santos_227890', 'Nuno Miguel-Jeronimo-Sequeira_217546', 'Nuno Miguel-Reis-Lima_258642', 'Nuno Miguel-Valente-Santos_251438', 'Oday Dabbagh_264321', 'Or Dasa_263633', 'Oscar Estupinan_228761', 'Otavio Edmilson-Da-Silva-Monteiro_210411', 'Pablo Felipe-Pereira-De-Jesus_264290', 'Pablo Renan-Dos-Santos_239872', 'Pablo Sarabia-Garcia_198950', 'Patrick William-Sa-De-Oliveira_252150', 'Paulo Andre-Rodrigues-Oliveira_210679', 'Paulo Estrela-Moreira-Alves_255255', 'Paulo Henrique-Rodrigues-Cabral_229498', 'Paulo Sergio-Mota_211014', 'Pedro Alves-Correia_239915', 'Pedro Antonio-Pereira-Goncalves_240950', 'Pedro Antonio-Porro-Sauceda_243576', 'Pedro Augusto-Borges-Da-Costa_252291', 'Pedro David-Rosendo-Marques_236673', 'Pedro Filipe-Barbosa-Moreira_235210', 'Pedro Filipe-Rodrigues_239950', 'Pedro Henrique-De-Oliveira-Correia_258356', 'Pedro Henrique-Rocha-Pelagio_246866', 'Pedro Jorge-Goncalves-Malheiro_263810', 'Pedro Manuel-Da-Silva-Moreira_199848', 'Pedro Miguel-Cunha-Sa_238856', 'Pedro Miguel-Santos-Amador_256044', 'Pedro Nuno-Fernandes-Ferreira_224538', 'Petar Musa_244797', 'Pierre Sagna_201512', 'Racine Coly_222840', 'Rafael A-Ferreira-Silva_216547', 'Rafael Antonio-Figueiredo-Ramos_224891', 'Rafael Avelino-Pinto-Barbosa_242389', 'Rafael Euclides-Soares-Camacho_243055', 'Rafael Vicente-Ferreira-Santos_263643', 'Rafik Guitane_235955', 'Raphael Gregorio-Guzzo_228589', 'Raul Michel-Melo-Da-Silva_221540', 'Reggie Cannon_237000', 'Renat Dadashov_238736', 'Renato Barbosa-Dos-Santos-Junior_263239', 'Ricardo Andrade-Quaresma-Bernardo_20775', 'Ricardo Jorge-Da-Luz-Horta_213516', 'Ricardo Jorge-Oliveira-Antonio_263574', 'Ricardo Miguel-Martins-Alves_233870', 'Ricardo Sousa-Esgaio_212213', 'Ricardo Viana-Filho_263791', 'Riccieli Da-Silva-Junior_252095', 'Richard Ofori_208391', 'Roberto Jesus-Machado-Beto-Alves_239875', 'Rocha Moreira-Nuno-Goncalo_262985', 'Rodrigo Abascal_253276', 'Rodrigo Cunha-Pereira-De-Pinho_229683', 'Rodrigo F-Conceicao_263566', 'Rodrigo Marcos-Rodrigues-Andrade_264308', 'Rodrigo Martins-Gomes_259170', 'Rodrigo Ribeiro-Valente_264459', 'Rolando Jorge-Pires-Da-Fonseca_163083', 'Roman Yaremchuk_240702', 'Romario Manuel-Silva-Baro_252038', 'Ruben Alexandre-Gomes-Oliveira_213801', 'Ruben Alexandre-Rocha-Lima_192156', 'Ruben Barcelos-De-Sousa-Lameiras_225363', 'Ruben Daniel-Fonseca-Macedo_246834', 'Ruben Del-Campo_256852', 'Ruben Ismael-Valente-Ramos_252080', 'Ruben Miguel-Santos-Fernandes_18115', 'Ruben Miguel-Valente-Fonseca_252391', 'Ruben Nascimento-Vinagre_235172', 'Rui F-Da-Cunha-Correia_251486', 'Rui Filipe-Caetano-Moura_223297', 'Rui Miguel-Guerra-Pires_251669', 'Rui Pedro-Da-Rocha-Fonte_183518', 'Rui Pedro-Silva-Costa_239922', 'Ryotaro Meshino_237435', 'Salvador Jose-Milhazes-Agra_204737', 'Samuel Dias-Lino_251445', 'Sana Dafa-Gomes_263898', 'Sandro R-G-Cordeiro_190782', 'Sebastian Coates_197655', 'Sebastian Perez_214606', 'Sergio Miguel-Relvas-De-Oliveira_198031', 'Shoya Nakajima_232862', 'Shuhei Kawasaki_255829', 'Silvio Manuel-Ferreira-Sa-Pereira_189530', 'Simon Banza_231652', 'Simone Muratore_257195', 'Soualiho Meite_205391', 'Souleymane Aw_239074', 'Sphephelo Sithole_257293', 'Stefano Beltrame_208523', 'Stephen Antunes-Eustaquio_242380', 'Steven De-Sousa-Vitoria_192528', 'Telmo Arcanjo_257194', 'Thales Bento-Oleques_262435', 'Thibang Phete_231688', 'Tiago Alexandre-De-Sousa-Esgaio_251994', 'Tiago Almeida-Ilori_205185', 'Tiago B-De-Melo-Tomas_257073', 'Tiago Filipe-Alves-Araujo_264280', 'Tiago Filipe-Oliveira-Dantas_251436', 'Tiago Fontoura-F-Morais_259054', 'Tiago Melo-Almeida_256918', 'Tiago Rafael-Maia-Silva_212474', 'Tim Soderstrom_212693', 'Tomas Aresta-Machado-Ribeiro_253378', 'Tomas Costa-Silva_262986', 'Tomas Pais-Sarmento-Castro_255114', 'Tomas Reymao-Nogueira_242390', 'Tomas Romano-Dos-Santos-Handel_263059', 'Toni Borevkovic_244377', 'Trova Boni_241946', 'Valentino Lazaro_211147', 'Vitor Carvalho-Vieira_255471', 'Vitor Costa-De-Brito_232388', 'Vitor Machado-Ferreira_255253', 'Vitor Manuel-Carvalho-Oliveira_263829', 'Vitor Tormena-De-Farias_246098', 'Vitorino Pacheco-Antunes_178424', 'Vivaldo Neto_242181', 'Volnei Feltes_263342', 'Walterson Silva_236630', 'Wendell Nascimento-Borges_216466', 'Wenderson Nascimento-Galeno_239482', 'Wilinton Aponza_264595', 'Willyan Da-Silva-Rocha_251901', 'Wilson Migueis-Manafa-Janco_238857', 'Yan Bueno-Couto_259075', 'Yan Matheus-Santos-Souza_234627', 'Yanis Hamache_257234', 'Yaw Moses_262440', 'Yohan Tavares_199051', 'Yusupha Njie_239303', 'Zaidu Sanusi_251528', 'Zainadine Chavango-Junior_217235', 'Zouhair Feddal_205705'))        

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

            # # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 3. Histograms: Insights from SUGGESTED CHANGES
            # # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # # Create a histogram.
            # plt.hist(differences_array, bins=20, edgecolor='black')
            # plt.xlabel('Differences')
            # plt.ylabel('Frequency')
            # st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 4. Violin: Insights from SUGGESTED CHANGES
            # differences_array = differences.values.flatten()
            # # Create a violin plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.violinplot(y = differences_array, color='skyblue')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 5. Density Plot: Insights from SUGGESTED CHANGES
            # differences = differences.squeeze()  # Ensure it's a Series
            # plt.figure(figsize=(10, 10)) 
            # sns.kdeplot(data=differences, shade=True)
            # plt.xlabel('(CFS - X)')
            # plt.ylabel('Density')
            # st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 6. Radar Chart: Average SUGGESTED CHANGES per feature
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
            # # Create KDE plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.kdeplot(differences_array, shade=True)
            # plt.xlabel('Differences')
            # plt.ylabel('Density')
            # st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Football_player_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Football_player_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Football_player_feature} is recommended to change**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
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
            # categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_X_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            # categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            # values = player_differences.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_differences_normalized = differences_normalized.loc[selected_player]    
            # categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 13. Radar (per player) - RECOMMENDED STATE
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_cfs_normalized = cfs_normalized.loc[selected_player]    
            # categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                        
            # # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
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
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.bar(shap_values, max_display=15)
            # st.pyplot()  
            # st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 16. SHAP Beeswarm Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.beeswarm(shap_values, max_display=15)
            # st.pyplot()
            # st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            # st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Football_player_index_feature = Football_player_list.index(Football_player_feature)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Football_player_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.scatter(shap_values[:, Football_player_index_feature])
            # st.pyplot()
            # st.markdown(f"**Figure 17**: Scatter plot on feature **{Football_player_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Football_player_feature_full_name} feature, which means **'how much must {Football_player_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Football_player_feature_full_name}**.")
            # st.markdown(f"This means that, for positive SHAP values, **{Football_player_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Football_player_feature_full_name} must impact negatively** the model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 18. SHAP Partial Dependence Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Football_player_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.partial_dependence_plot(
            #     Football_player_feature, lr.predict, X, ice=False,
            #     model_expected_value=True, feature_expected_value=True) 
            # st.pyplot()
            # st.markdown(f"**Figure 18**: Model's dependence on the feature {Football_player_feature_full_name}, now in the new original feature space (X). It explains **how SHAP values of {Football_player_feature_full_name} vary across a dataset** and how changes in the {Football_player_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Football_player_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Football_player_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Football_player_feature_full_name}** as a certain value;")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Football_player_index_player = X_indexes.index(Player)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.waterfall(shap_values[Football_player_index_player])
            # st.pyplot()
            # st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Player}, instead of, as in the previous two graphs, focusing on feature {Football_player_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # # Concepts to take into account
            # st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # # 20. SHARP: Rank vs Score
            # import os
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            # st.image("Rank_vs_Score_(4) Football Player.png")
            # st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 21. SHARP: Top and Bottom 3 Individuals
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            # st.image("Top_bottom_feature_importance_4.png")
            # st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 22. SHARP: Feature Importance
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            # st.image("Strata_boxplot_4.png")
            # st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            # st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} ranking. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            
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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value. \n - Note that highest DiCE values does not necessary mean worse features. DiCE can impact either the best features or the worst features. But overall, the lowest the player ranking, the higher amount tend to be the player average DiCE values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


            # 28. Create a Statistics DataFrame
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Statistics Comparison</h1>", unsafe_allow_html=True)
            average_abs_values = df_values_2.abs().mean() # Calculate the average of the absolute values for each column
            variance_values = df_values_2.var() # Calculate the variance for each column
            diff_max_min_values = df_values_2.max() - df_values_2.min() # Calculate the difference between the maximum and minimum values for each column
            df_stats = pd.DataFrame({
            'Average Absolute Value': average_abs_values,
            'Variance': variance_values,
            'Max-Min Difference (Amplitude)': diff_max_min_values})
            st.dataframe(df_stats, width=900)
            st.markdown(f"**Figure 28**: Table aggregating the average values, the variance and the amplitude from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 29. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 29**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


elif Sport == 'Tennis':     
    # Create a radio button for selecting the type (team or player)
    Male_vs_Female = st.sidebar.radio('Gender Preference:', ["Male", "Female"])

    # Check if the user selects the Male Player
    if Male_vs_Female == 'Male':
        Player = st.sidebar.selectbox('Select the Player:', ('Adrian Mannarino', 'Albert Ramos-Vinolas', 'Alejandro Davidovich Fokina', 'Alejandro Tabilo', 'Alex de Minaur', 'Alex Molcan', 'Alexander Bublik', 'Alexander Zverev', 'Andrey Rublev', 'Andy Murray', 'Arthur Rinderknech', 'Aslan Karatsev', 'Ben Shelton', 'Benjamin Bonzi', 'Bernabe Zapata Miralles', 'Borna Coric', 'Botic van de Zandschulp', 'Brandon Nakashima', 'Cameron Norrie', 'Carlos Alcaraz', 'Casper Ruud', 'Cristian Garin', 'Daniel Evans', 'Daniil Medvedev', 'David Goffin', 'Denis Shapovalov', 'Diego Schwartzman', 'Dominic Thiem', 'Dusan Lajovic', 'Emil Ruusuvuori', 'Fabio Fognini', 'Federico Coria', 'Federico Delbonis', 'Felix Auger-Aliassime', 'Filip Krajinovic', 'Frances Tiafoe', 'Francisco Cerundolo', 'Grigor Dimitrov', 'Holger Rune', 'Hubert Hurkacz', 'Ilya Ivashka', 'J.J. Wolf', 'Jack Draper', 'Jan-Lennard Struff', 'Jannik Sinner', 'Jaume Munar', 'Jenson Brooksby', 'Jiri Lehecka', 'Joao Sousa', 'John Isner', 'Karen Khachanov', 'Laslo Djere', 'Lorenzo Musetti', 'Lorenzo Sonego', 'Mackenzie McDonald', 'Marcos Giron', 'Marin Cilic', 'Marton Fucsovics', 'Matteo Berrettini', 'Maxime Cressy', 'Mikael Ymer', 'Miomir Kecmanovic', 'Nick Kyrgios', 'Nicolas Jarry', 'Nikoloz Basilashvili', 'Novak Djokovic', 'Oscar Otte', 'Pablo Carreno Busta', 'Pedro Martinez', 'Rafael Nadal', 'Reilly Opelka', 'Richard Gasquet', 'Roberto Bautista Agut', 'Sebastian Baez', 'Sebastian Korda', 'Soonwoo Kwon', 'Stefanos Tsitsipas', 'Steve Johnson', 'Tallon Griekspoor', 'Taro Daniel', 'Taylor Fritz', 'Thiago Monteiro', 'Tomas Martin Etcheverry', 'Tommy Paul', 'Ugo Humbert', 'Yoshihito Nishioka'))
        
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
        
        Player_2 = st.sidebar.selectbox('Select a ATP tennis player to compare:', ('Adrian Mannarino', 'Albert Ramos-Vinolas', 'Alejandro Davidovich Fokina', 'Alejandro Tabilo', 'Alex de Minaur', 'Alex Molcan', 'Alexander Bublik', 'Alexander Zverev', 'Andrey Rublev', 'Andy Murray', 'Arthur Rinderknech', 'Aslan Karatsev', 'Ben Shelton', 'Benjamin Bonzi', 'Bernabe Zapata Miralles', 'Borna Coric', 'Botic van de Zandschulp', 'Brandon Nakashima', 'Cameron Norrie', 'Carlos Alcaraz', 'Casper Ruud', 'Cristian Garin', 'Daniel Evans', 'Daniil Medvedev', 'David Goffin', 'Denis Shapovalov', 'Diego Schwartzman', 'Dominic Thiem', 'Dusan Lajovic', 'Emil Ruusuvuori', 'Fabio Fognini', 'Federico Coria', 'Federico Delbonis', 'Felix Auger-Aliassime', 'Filip Krajinovic', 'Frances Tiafoe', 'Francisco Cerundolo', 'Grigor Dimitrov', 'Holger Rune', 'Hubert Hurkacz', 'Ilya Ivashka', 'J.J. Wolf', 'Jack Draper', 'Jan-Lennard Struff', 'Jannik Sinner', 'Jaume Munar', 'Jenson Brooksby', 'Jiri Lehecka', 'Joao Sousa', 'John Isner', 'Karen Khachanov', 'Laslo Djere', 'Lorenzo Musetti', 'Lorenzo Sonego', 'Mackenzie McDonald', 'Marcos Giron', 'Marin Cilic', 'Marton Fucsovics', 'Matteo Berrettini', 'Maxime Cressy', 'Mikael Ymer', 'Miomir Kecmanovic', 'Nick Kyrgios', 'Nicolas Jarry', 'Nikoloz Basilashvili', 'Novak Djokovic', 'Oscar Otte', 'Pablo Carreno Busta', 'Pedro Martinez', 'Rafael Nadal', 'Reilly Opelka', 'Richard Gasquet', 'Roberto Bautista Agut', 'Sebastian Baez', 'Sebastian Korda', 'Soonwoo Kwon', 'Stefanos Tsitsipas', 'Steve Johnson', 'Tallon Griekspoor', 'Taro Daniel', 'Taylor Fritz', 'Thiago Monteiro', 'Tomas Martin Etcheverry', 'Tommy Paul', 'Ugo Humbert', 'Yoshihito Nishioka'))

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

            # # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 3. Histograms: Insights from SUGGESTED CHANGES
            # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # # Create a histogram.
            # plt.hist(differences_array, bins=20, edgecolor='black')
            # plt.xlabel('Differences')
            # plt.ylabel('Frequency')
            # st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 4. Violin: Insights from SUGGESTED CHANGES
            # differences_array = differences.values.flatten()
            # # Create a violin plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.violinplot(y = differences_array, color='skyblue')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
        
            # # 5. Density Plot: Insights from SUGGESTED CHANGES
            # differences = differences.squeeze()  # Ensure it's a Series
            # plt.figure(figsize=(10, 10)) 
            # sns.kdeplot(data=differences, shade=True)
            # plt.xlabel('(CFS - X)')
            # plt.ylabel('Density')
            # st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 6. Radar Chart: Average SUGGESTED CHANGES per feature
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
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.kdeplot(differences_array, shade=True)
            # plt.xlabel('Differences')
            # plt.ylabel('Density')
            # st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Tennis_male_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Tennis_male_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Tennis_male_feature} is recommended to change**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
                    
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
            # player_X_normalized = X_normalized.loc[selected_player]    
            # categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_X_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 11. Radar (per player) - SUGGESTED CHANGES
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_differences = differences.loc[selected_player]    
            # categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            # values = player_differences.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_differences_normalized = differences_normalized.loc[selected_player]    
            # categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 13. Radar (per player) - RECOMMENDED STATE
            # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_cfs_normalized = cfs_normalized.loc[selected_player]    
            # categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values, max_display=15)
            #st.pyplot()  
            #st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            #st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 16. SHAP Beeswarm Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.beeswarm(shap_values, max_display=15)
            # st.pyplot()
            # st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            # st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Tennis_male_index_feature = Tennis_male_list.index(Tennis_male_feature)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Tennis_male_feature_full_name}</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.scatter(shap_values[:, Tennis_male_index_feature])
            st.pyplot()
            st.markdown(f"**Figure 17**: Scatter plot on feature **{Tennis_male_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Tennis_male_feature_full_name} feature, which means **'how much must {Tennis_male_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Tennis_male_feature_full_name}**.")
            st.markdown(f"This means that, for positive SHAP values, **{Tennis_male_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Tennis_male_feature_full_name} must impact negatively** the model output.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 18. SHAP Partial Dependence Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Tennis_male_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.partial_dependence_plot(
            #     Tennis_male_feature, lr.predict, X, ice=False,
            #     model_expected_value=True, feature_expected_value=True) 
            # st.pyplot()
            # st.markdown(f"**Figure 18**: Model's dependence on the feature {Tennis_male_feature_full_name}, now in the new original feature space (X). It explains **how SHAP values of {Tennis_male_feature_full_name} vary across a dataset** and how changes in the {Tennis_male_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Tennis_male_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Tennis_male_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Tennis_male_feature_full_name}** as a certain value;")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Tennis_male_index_player = X_indexes.index(Player)
            st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 21. SHARP: Top and Bottom 3 Individuals
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            # st.image("Top_bottom_feature_importance_5.png")
            # st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 22. SHARP: Feature Importance
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            # st.image("Strata_boxplot_5.png")
            # st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            # st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} ranking. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            
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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value. \n - Note that highest DiCE values does not necessary mean worse features. DiCE can impact either the best features or the worst features. But overall, the lowest the player ranking, the higher amount tend to be the player average DiCE values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


            # 28. Create a Statistics DataFrame
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Statistics Comparison</h1>", unsafe_allow_html=True)
            average_abs_values = df_values_2.abs().mean() # Calculate the average of the absolute values for each column
            variance_values = df_values_2.var() # Calculate the variance for each column
            diff_max_min_values = df_values_2.max() - df_values_2.min() # Calculate the difference between the maximum and minimum values for each column
            df_stats = pd.DataFrame({
            'Average Absolute Value': average_abs_values,
            'Variance': variance_values,
            'Max-Min Difference (Amplitude)': diff_max_min_values})
            st.dataframe(df_stats, width=900)
            st.markdown(f"**Figure 28**: Table aggregating the average values, the variance and the amplitude from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # 29. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 29**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
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
        # min_ranking = df['rank'].min()
        # max_ranking = df['rank'].max()
        # df['rank'] = (df['rank'] - min_ranking) / (max_ranking - min_ranking)
        df['rank'] = 101 - df['rank']
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

            # # Concepts to take into account
            # st.info("DICE: method used to generate diverse counterfactual explanations for machine learning models. In simple words, it provides 'what-if' explanations for the model output. 'Counterfactuals' represent the desired values. 'X' represent the initial values. 'Differences' will be lead from now onwards, represent SUGGESTED CHANGES (recommendations) between the counterfactuals and the initial values.")

            # 1.1 Preparing future Histogram.
            cfs.set_index(cfs.columns[0], inplace=True)
            differences.set_index(differences.columns[0], inplace=True)
            # Plot bar
            Player_differences = differences.loc[Player]

            # # 2. Heatmap: Insights from SUGGESTED CHANGES
            plt.figure(figsize=(10, 10))
            sns.heatmap(differences, cmap='coolwarm')
            st.markdown("<h4 style='text-align: center;'>Heatmap: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            st.pyplot() # Displaying plot in Streamlit
            st.markdown("**Figure 2**: Results from DICE. Representation of all the differences obtained in our dataset, per instance. Visual representation of how the features would need to be altered in the counterfactual scenarios compared to the original data to achieve the desired outcomes predicted by the model. Players (in Y-axis) vs Features (in X-axis), with variations in absolute values: \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 3. Histograms: Insights from SUGGESTED CHANGES
            # # Transforming differences into an array.
            differences_array = differences.values.flatten()
            # # Create a histogram.
            # plt.hist(differences_array, bins=20, edgecolor='black')
            # plt.xlabel('Differences')
            # plt.ylabel('Frequency')
            # st.markdown("<h4 style='text-align: center;'>Histograms: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 3**: Results from DICE. It helps to understand the the overall pattern and where most of the differences are concentrated in. It indicates the frequency (in absolute values), per each difference value. \n - **Positive values** indicate an increase recommendation for that feature;  \n - **Negative values** indicate a decrease recommendation for that feature.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 4. Violin: Insights from SUGGESTED CHANGES
            # differences_array = differences.values.flatten()
            # # Create a violin plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.violinplot(y = differences_array, color='skyblue')
            # plt.ylabel('Differences')
            # st.markdown("<h4 style='text-align: center;'>Violin: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 4**: Results from DICE. Another simple method to interpret **where the majority of the differences** are concentrated. Mostly concentrated around < |0.1|. There is no feature on X-axis.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 5. Density Plot: Insights from SUGGESTED CHANGES
            # differences = differences.squeeze()  # Ensure it's a Series
            # plt.figure(figsize=(10, 10))  
            # sns.kdeplot(data=differences, shade=True)
            # plt.xlabel('(CFS - X)')
            # plt.ylabel('Density')
            # st.markdown("<h4 style='text-align: center;'>Density Plot: Insights from SUGGESTED CHANGES</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown("**Figure 5**: Results from DICE. Provides the distribution of **differences per feature**, indicating which ones vary the most and which one vary the least. The closer a feature is to zero, the less it varies.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 6. Radar Chart: Average SUGGESTED CHANGES per feature
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
            # # Create KDE plot
            # plt.figure(figsize=(8, 6)) # Setting figure size.
            # sns.kdeplot(differences_array, shade=True)
            # plt.xlabel('Differences')
            # plt.ylabel('Density')
            # st.markdown(f"<h4 style='text-align: center;'>KDE: Insights from SUGGESTED CHANGES for variable {Tennis_female_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 9**: Results from DICE regarding variable **{Tennis_female_feature}**. Provides the distribution of differences across all instances on this specific feature. In case the graph is empty, it means **{Tennis_female_feature} is recommended to change**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            # categories = list(player_X_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_X_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>INITIAL STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 10**: 'Radar' chart gives us a visual understanding of the current importance, per feature, **on {selected_player}**. Provides insights on which features are **currently contributing the most** for the actual model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 11. Radar (per player) - SUGGESTED CHANGES
            # Specify the name of the player
            selected_player = Player
            # Filter "differences" DataFrame.
            player_differences = differences.loc[selected_player]    
            # categories = list(player_differences.index) # Setting categories as a list of all "differences" column.
            # values = player_differences.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 11**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Provides insights on which features should **contribute more and less** in order to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 12. Radar (per player) - SUGGESTED CHANGES - Normalized.
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_differences_normalized = differences_normalized.loc[selected_player]    
            # categories = list(player_differences_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_differences_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>SUGGESTED CHANGES: Mean Differences for {selected_player} - Normalized</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 12**: 'Radar' chart gives us a closer look at the differences, per feature, **on {selected_player}**. Similar to the previous visualization, but with values normalized.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 13. Radar (per player) - RECOMMENDED STATE
            # # Specify the name of the player
            # selected_player = Player
            # # Filter "differences" DataFrame.
            # player_cfs_normalized = cfs_normalized.loc[selected_player]    
            # categories = list(player_cfs_normalized.index) # Setting categories as a list of all "differences" column.
            # values = player_cfs_normalized.values.tolist() # List of mean differences per feature.
            # values += values[:1]   # Connect the first and the last point of the radar, closing and creating a loop.
            # angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))] # Angles for each category.
            # angles += angles[:1] # Connect the first and the last point, closing creating a loop.
            # plt.figure(figsize=(8, 8)) # Setting figure size.
            # plt.polar(angles, values) # Using polar coordinates.
            # plt.fill(angles, values, alpha=0.25) # Fill the inside area with a semi-transparent color.
            # plt.xticks(angles[:-1], categories) # Set the categories as labels.
            # st.markdown(f"<h4 style='text-align: center;'>RECOMMENDED STATE: Values for {selected_player}</h1>", unsafe_allow_html=True)
            # st.pyplot() # Displaying plot in Streamlit
            # st.markdown(f"**Figure 13**: ''Radar' chart gives us a visual understanding of the desired importance, per feature, **on {selected_player}**. Provides insights on which features should **in the future contributing the most** to achieve the desired model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            # # 14. Radar (per player) - INITIAL and RECOMMENDED STATE overlapped
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
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Bar Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.bar(shap_values, max_display=15)
            # st.pyplot()  
            # st.markdown("**Figure 15**: Overview of the impact of **each feature on the model output/predictions**. It represents the **mean absolute value of each feature** for the overall dataset. \n - **The higher the SHAP Value mean**, the **higher its feature importance**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 16. SHAP Beeswarm Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Beeswarm Plot</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.beeswarm(shap_values, max_display=15)
            # st.pyplot()
            # st.markdown("**Figure 16**: Beeswarm Plot summarizes what are the **most relevant features** impact model output. Each instance is represented at the graph by a single point. The plot below sorts features by their SHAP value magnitudes. \n - In the X-axis, **positive SHAP values represent a positive impact** from the feature to the model output (positive SHAP values means that that feature contribute positively to its model outcome) (Features whose variance contribute positively to the player overall improvement have positive absolute values); \n - In the X-axis, **negative SHAP values represent a negative impact** from the feature to the model output (negative SHAP values means that that feature contributely negatively to its model outcome)(Features whose variance contribute negatively to the player overall improvement have negative absolute values); \n - **The red color code** for a specific instance, means that it a value above the dataset average for that specific feature; \n - **The blue color code** for a specific instance, means that it a value bellow the dataset average for that specific feature.")
            # st.markdown("For example, for features with mostly blue dot at the right side of the graph, it means that the lower the feature value, the higher it tends to be the outcome.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[3]:
            # 17. Scatter Plot
            Tennis_female_index_feature = Tennis_female_list.index(Tennis_female_feature)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Scatter Plot for feature {Tennis_female_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.scatter(shap_values[:, Tennis_female_index_feature])
            # st.pyplot()
            # st.markdown(f"**Figure 17**: Scatter plot on feature **{Tennis_female_feature_full_name}**, which shows its effect on model predictions. Each point represents an instance from the dataset. \n - **X-axis** represents the feature input value;  \n - **y-axis** represents the SHAP values for {Tennis_female_feature_full_name} feature, which means **'how much must {Tennis_female_feature_full_name} change the model output value'**; \n - **The gray area** represents, through an histogram, dataset distribution for **{Tennis_female_feature_full_name}**.")
            # st.markdown(f"This means that, for positive SHAP values, **{Tennis_female_feature_full_name} must impact positively** the model output, while for negative SHAP values, **{Tennis_female_feature_full_name} must impact negatively** the model output.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 18. SHAP Partial Dependence Plot
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Partial Dependence Plot for feature {Tennis_female_feature_full_name}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.partial_dependence_plot(
            #     Tennis_female_feature, lr.predict, X, ice=False,
            #     model_expected_value=True, feature_expected_value=True) 
            # st.pyplot()
            # st.markdown(f"**Figure 18**: Model's dependence on the feature {Tennis_female_feature_full_name}, now in the new original feature space (X). It explains **how SHAP values of {Tennis_female_feature_full_name} vary across a dataset** and how changes in the {Tennis_female_feature_full_name} values impact model's predictions. \n - **X-axis** represents SHAP values for the {Tennis_female_feature_full_name} feature; \n - **Y-axis** represents the variation per player; \n - **Gray horizontal line** represents the final expected value for the model; \n - **Gray vertical line** represents {Tennis_female_feature_full_name} average value; \n - **The blue line with positive slope** represents the model average value when we define **{Tennis_female_feature_full_name}** as a certain value;")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 19. SHAP Waterfall Plot
            X_indexes = X.index.tolist()
            Tennis_female_index_player = X_indexes.index(Player)
            # st.markdown(f"<h4 style='text-align: center;'>SHAP Waterfall Plot for {Player}</h1>", unsafe_allow_html=True)
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # shap.plots.waterfall(shap_values[Tennis_female_index_player])
            # st.pyplot()
            # st.markdown(f"**Figure 19**: Waterfall plot attempts to explain the predictions for {Player}, instead of, as in the previous two graphs, focusing on feature {Tennis_female_feature_full_name}. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


        #else:
        with tabs[4]:
            # # Concepts to take into account
            # st.info("SHARP: (SHapley for Rankings and Preferences), a framework that attemps to explain the contribution of features to different decils of an output in 'a ranking format' and can be base either on ShaPley or Unary values (we used the last one). According to recent studies, ShaRP claims that the weght of each feature does not correspond to its ShaPley value contribution (analyzed on tabs 3 and 4). Researches appoint that it depends on feature distribution (varying according to the decil in focus) and to local interactions between scoring features. ShaRP, derived from Quantitative Input Influence framework, can contribute to explain score-based and ranking type models.")

            # # 20. SHARP: Rank vs Score        
            # import os
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Rank vs Score</h1>", unsafe_allow_html=True)
            # st.image("Rank_vs_Score_(6) WTA.png")
            # st.markdown("**Figure 20**: Relationship between Score and Rank. Score function, which provides a certain weight to each variable in the dataset, was defined by us, acccording to our knowledge of the sport. We tend to see an **inverse relationship between Score and Rank**.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 21. SHARP: Top and Bottom 3 Individuals
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Top and Bottom 3 Individuals</h1>", unsafe_allow_html=True)
            # st.image("Top_bottom_feature_importance_6.png")
            # st.markdown("**Figure 21**: Top 3 and Bottom 3 instances with their respective aggregate feature importance, providing insights on which are the most and the **least relevant features for their ranking**. For example:  \n - Features with a **high positive values among the top 3**, means that it was a **key feature** for these instances to achieve this **high/good ranking**; \n - Features with a **considerable negative values among the bottom 3**, means that it was a **key feature** for these instances to achieve this **low/bad ranking;** ")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # # 22. SHARP: Feature Importance
            # st.markdown(f"<h4 style='text-align: center;'>SHARP: Feature Importance</h1>", unsafe_allow_html=True)
            # st.image("Strata_boxplot_6.png", width=800)#, height=800)
            # st.markdown("**Figure 22**: Visualization on how feature importance varies **across strata (different decil categories)**. \n - There are 5 decil categories, represented at the bottom of the graph. \n - All the features are legended at the top of the graph. \n - At the left side of the graph, we have indication of the importance magnitude. \n - Each feature importance is distributed thorugh a boxplot, indicating us Q1, Q2 (median) and Q3. The higher the position of the boxplot, **the higher the relevancy of that specific feature in that decil**. \n - **The longer the boxplot**, the **more different importances that feature acquire** in the dataset.")
            # st.markdown("We highly recommend you to open the figure (at the top right corner of the figure) and zoom it, so that you can have a better understanding of the main insights.")
            # st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 24**: Waterfall plot for the selected {Player}. Similarly to SHAP Waterfall, it attempts to explain {Player} ranking. In the X-axis, we have information of the entire model expected output value. The color code, along with its respective magnitude indication, inform if: \n - The **red features** are pushing the **prediction higher**; \n - The **blue features** are pushing the **prediction lower**; \n - The **gray values** before the feature name, indicate each feature value for **{Player}**; \n - The **gray value** on top of the graph, indicates the model prediction for **{Player}**.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            # 25. SHARP: Pairwise Comparison
            st.markdown(f"<h4 style='text-align: center;'>SHARP: Pairwise Comparison</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            
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
            #st.set_option('deprecation.showPyplotGlobalUse', False)
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
            st.markdown(f"**Figure 26**: Table aggregating the main insights from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **In green** is represent the highest positive value. \n - **In red** is represent the lowest negative value. \n - Note that highest DiCE values does not necessary mean worse features. DiCE can impact either the best features or the worst features. But overall, the lowest the player ranking, the higher amount tend to be the player average DiCE values.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)

            

            # 27. DiCE vs SHAP vs SHARP: Comparing Methods Graphically
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Comparing Methods Graphically</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            st.line_chart(df_values_2, width=800, height=600)
            st.markdown(f"**Figure 27**: Graphic representation of the previous table.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)
            
            
            # 28. Create a Statistics DataFrame
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Statistics Comparison</h1>", unsafe_allow_html=True)
            average_abs_values = df_values_2.abs().mean() # Calculate the average of the absolute values for each column
            variance_values = df_values_2.var() # Calculate the variance for each column
            diff_max_min_values = df_values_2.max() - df_values_2.min() # Calculate the difference between the maximum and minimum values for each column
            df_stats = pd.DataFrame({
            'Average Absolute Value': average_abs_values,
            'Variance': variance_values,
            'Max-Min Difference (Amplitude)': diff_max_min_values})
            st.dataframe(df_stats, width=900)
            st.markdown(f"**Figure 28**: Table aggregating the average values, the variance and the amplitude from DiCE, SHAP and SHARP applied to {Player} and according to the selected decil.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


            
            # 29. DiCE vs SHAP vs SHARP: Correlation Matrix
            st.markdown(f"<h4 style='text-align: center;'>DiCE vs SHAP vs SHARP: Correlation Matrix</h1>", unsafe_allow_html=True)
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            correlation_matrix = df_values_2.corr()
            st.write(correlation_matrix)    
            st.markdown(f"**Figure 29**: Correlation matrix between DiCE, SHAP and SHARP applied to {Player} and according to the selected decil. \n - **Positive values** represent a direct relationship, meaning that features increase and decrese together; \n - **Negative values** represent an indirect relationship, meaning when one of the methods increases, the other decreases; \n - **The highest the absolute value**, the most relevant the feature is.")
            st.write("<div style='height: 150px;'></div>", unsafe_allow_html=True)


# 6. Sidebar Part II
st.sidebar.header("Provide some feedback:")
st.sidebar.text_input("Mail Adress")
st.sidebar.text_input("Profession")
st.sidebar.radio("Professional Expert", ["Student", "Professor", "Other"])
st.sidebar.slider("How much did you find it relevant?", 0, 100)
st.sidebar.text_input("Additional Comments")
st.sidebar.button("Submit Feedback")