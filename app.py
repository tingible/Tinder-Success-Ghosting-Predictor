# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Predict Tinder Match Success/Ghosting', page_icon='\u2764\uFE0F', layout='wide', initial_sidebar_state='expanded')

# Set title of the app
st.title('\u2764\uFE0F Predict Tinder Match Success/Ghosting')

# Load data
df = pd.read_csv('merged_everything_overall_streamlit.csv')

# Set input widgets
st.sidebar.subheader('Select Tinder Conversation/Profile Features')
Num_messages = st.sidebar.slider('Number of messages exchanged', 1, 200, 50)
Min_age_filter = st.sidebar.slider('Minimum age filter', 18, 100, 30)
Max_age_filter = st.sidebar.slider('Maximum age filter', 18, 100, 30)
Num_of_likes_vs_swipes = st.sidebar.slider('% of right swipes', 1, 100, 50)
Positive_proportion = st.sidebar.slider('% of positive vibes', 1, 100, 50)
Negative_proportion = st.sidebar.slider('% of negative vibes', 1, 100, 50)
Num_of_matches_vs_likes = st.sidebar.slider('% of matches vs right swipes', 1, 100, 50)

# Features calculations
Age_range_filter = Max_age_filter - Min_age_filter
Pickiness_indicator = Num_of_likes_vs_swipes / 100
Neutral_proportion = (100 - Positive_proportion - Negative_proportion) / 100
Attractiveness_indicator = Num_of_matches_vs_likes / 100

# Separate to X and y
X = df.drop(['Label','User','Match_id'], axis=1)
y = df.Label

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=99)

# Build model
model = KNeighborsClassifier(n_neighbors=14)
model.fit(X_train, y_train)

# Generate prediction based on user selected attributes
y_pred = model.predict([[Num_messages, Age_range_filter, Pickiness_indicator, Neutral_proportion, Attractiveness_indicator]])


# Print input features
st.subheader('Tinder Conversation/Profile Features')
input_feature = pd.DataFrame([[Num_messages, Age_range_filter, Pickiness_indicator, Neutral_proportion, Attractiveness_indicator]],
                            columns=['Number of messages exchanged', 'Age range filter', 'Pickiness Indicator', '% of neutral vibes', 'Attractiveness Indicator'])
st.write(input_feature)

# Print predicted flower species
st.subheader('Tinder Match Prediction')
st.metric('Predicted Success/Ghosted is :', y_pred[0], '')
