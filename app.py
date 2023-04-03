# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Predict Tinder Match Success/Ghosting', page_icon='\u2764\uFE0F', layout='wide', initial_sidebar_state='expanded')

image = Image.open('Wordmark - Color1100.png')
st.image(image, width=600)

# Set title of the app
st.title('\u2764\uFE0F Tinder Match Predictor \u2764\uFE0F')

# Load data
df = pd.read_csv('merged_everything_overall_streamlit.csv')

#Set layout
st.header('Step 1: Enter Tinder Profile/Conversation Info')

with st.container():
    st.subheader(':orange[Profile-related info]')
    Num_of_matches_vs_likes = st.number_input('% of matches vs right swipes', min_value=1, max_value=100)

with st.container():
    st.subheader(':orange[Convo-related info]')
    col1, col2, col3 = st.columns(3)
    with col1:
        Num_messages = st.slider('Number of messages exchanged', 1, 3000,100)
    with col2:
        Message_word_count = st.slider('Estimated length of messages in word count', 1, 200, 5)
    with col3:
        Positive_percent = st.slider('% of positive vibes', 0, 100, 50)

# Features calculations
Attractiveness_indicator = Num_of_matches_vs_likes / 100
Positive_Neutral_proportion = Positive_percent / 100

# Separate to X and y
X = df.drop(['Label','User','Match_id'], axis=1)
y = df.Label

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=99)

# Build model
model = KNeighborsClassifier(n_neighbors=14)
model.fit(X_train, y_train)

# Generate prediction based on user selected attributes
y_pred = model.predict([[Num_messages, Message_word_count, Positive_Neutral_proportion, Attractiveness_indicator]])

# Generate label name based on y_pred results
if y_pred == 1:
    y_pred_label = "Success"
else:
    y_pred_label = "Ghosted"

# Print input features
st.subheader(':orange[Summary of Tinder Conversation/Profile Features]')
input_feature = pd.DataFrame([[Attractiveness_indicator, Num_messages, Message_word_count, Positive_propotion]],
                            columns=['Attractiveness Indicator','Number of messages exchanged',
                                     'Estimated length of messages in word count', '% of positive vibes'])
st.write(input_feature)

# Print predicted Tinder Match/Ghosted
st.header('Step 2: Generate Results')
st.subheader(':orange[Your Tinder Match is likely to be:]')
if st.button('Tell me the result of my match!'):
    st.metric('Predicted result is :', y_pred_label , '')
else:
    st.write('')
