import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob


st.set_page_config(page_title="NLP Sentimental Analysis", page_icon='https://img.icons8.com/office/16/twitter.png')
st.markdown("""
    <style>
        /* Set background image */
        .stApp {
            background-image: url("https://media.cnn.com/api/v1/images/stellar/prod/120127055441-twitter.jpg?q=x_0,y_109,h_765,w_1360,c_crop/h_720,w_1280");
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* Center the title */
        .title {
            text-align: center;
            border-radius: 50px;
            color: rgb(255 255 255);
            background: #043142e6;

        }
        
        /* Increase the size of text inputs */
        .stNumberInput label p{
            padding: 10px;
            font-size: 20px;
       }

       /* Increase the size of number inputs */
       .stNumberInput input[type="number"] {
           padding: 10px;
           font-size: 18px;
           }
       
       /* Prediction output  */ 
       .css-nahz7x p{
            font-size: 18px   
        }
       
       .st-b7 {
           color: black;
           font-size: 22px
       }

        h3, h1 {
            color: #094055;
            }
        
        .css-nahz7x{
            color: #094055;
            }
        
        .css-q8sbsg{
            color: #006e8f;;
            }
       
       .st-ch {
           background-color: rgb(255 255 255 / 70%) !important;
       }
       
       .uploadedFileName{
           color: #b98282;
           }
       
       .css-7oyrr6{
           color: #b98282;
           }
       
       
       /* Upload CSV design */
       
       .stMarkdown{
           margin-top: 35px;
           margin-bottom: 25px;

           }
            
       #upload-csv-file{
           text-align: center;
           color: #1c7a9a;
           font-size: 32px;
           background: #2fcefe;;
           border-radius: 50px;
           padding: 8px;
           margin-bottom: 25px;
       }
            
        .stAlert{
            background: rgb(255 255 255 / 65%);
            border-radius: 15px;
            }
            
     /* Input label color */
     .css-1qg05tj{
         color: rgb(164 209 234);
         }
    </style>
    <h1 class="title">NLP Sentimental Analysis</h1>
""", unsafe_allow_html=True)

# Load the trained Logistic Regression model
naive_bayes_model = joblib.load('naive_bayes_model.pkl')

# Load the TF-IDF Vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer3.pkl')

# Define class labels mapping
class_labels = {0: 'Irony', 1: 'Regular', 2: 'Sarcasm'}


# Input text box


def main():
    page = st.sidebar.radio("Navigate", ("Description","Tweet's Class Prediction", "Visualization"))
    
    if page == "Description":
        display_description_page()
    elif page == "Tweet's Class Prediction":
        tweets_class_prediction_page()
    elif page == "Visualization":
        display_visualization_page()
        
def tweets_class_prediction_page():
    input_text = st.text_area('Enter your text here:', '')
    if st.button('Predict'):
        if input_text:
            # Preprocess the input text
            input_text = [" ".join(input_text.split())]  # Clean and preprocess input
            input_tfidf = tfidf_vectorizer.transform(input_text)
            
            # Make prediction
            prediction = naive_bayes_model.predict(input_tfidf)
            prediction_label = class_labels.get(prediction[0], 'Unknown')
            
            # Display prediction
            st.write(f'Predicted Class: {prediction_label}')
        else:
            st.warning('Please enter text for prediction.')
            
    st.subheader('Upload CSV file')
     
     #csv input
     
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, usecols= ['tweets'])
         # Display the data frame
         
        predicted_values = []
        for index, row in df.iterrows():
            row_array = row['tweets']
            row_array = [" ".join(row_array.split())]
            input_tfidf = tfidf_vectorizer.transform(row_array)
            prediction = naive_bayes_model.predict(input_tfidf)
            prediction_label = class_labels.get(prediction[0], 'Unknown')
            predicted_values.append(prediction_label)
             

        # Display the updated DataFrame
        df['Prediction'] = predicted_values
        csv_file = df.to_csv(index=False)
        st.session_state['predicted_data'] = df;
        st.dataframe(df)
        st.download_button("Download CSV", data=csv_file, file_name = 'predicted_class_tweets.csv', mime='text/csv')
         
            
def display_description_page():
    st.subheader('Description')
    st.write('In this project, We embarked on a comprehensive analysis of textual data, specifically focusing on tweets extracted from various sources. The core objective was to unveil the underlying sentiments and nuanced expressions within these tweets. To achieve this, I employed state-of-the-art sentiment analysis techniques, harnessing the power of natural language processing and machine learning.')
    st.write('The dataset at the heart of this endeavor was structured with two essential columns: "tweets" and "classes," comprising figurative, irony, sarcasm, and regular categories. Leveraging the predictive capabilities of my model, I successfully assigned each tweet to its corresponding class, providing insights into the intricate emotions and rhetorical styles embedded within the Twitterverse.')
    st.write('This project delved into the intricate world of sentiment, discerning not only basic positive and negative sentiments but also the more sophisticated aspects of figurative language, irony, and sarcasm. By doing so, it offered a deeper understanding of the diverse and often playful nature of human expression on social media. The results of this analysis hold the potential for invaluable applications in fields such as social media marketing, opinion mining, and trend tracking, making it a significant contribution to the realm of sentiment analysis.')
            
def display_visualization_page():
    option = st.sidebar.radio("Select an option for Visualization", ("Countplot", "Sentiment Analysis"))

    # Display content based on the selected option
    if option == "Countplot":
        if st.session_state.get('predicted_data') is not None and not st.session_state['predicted_data'].empty:
            show_countplot()
        else:
            st.subheader('Please insert a CSV file to see the visualization');
    elif option == "Sentiment Analysis":
        if st.session_state.get('predicted_data') is not None and not st.session_state['predicted_data'].empty:
            show_sentiment_analysis()
        else:
            st.subheader('Please insert a CSV file to see the visualization');
            
def show_countplot():
    data = st.session_state['predicted_data']
    data['Prediction'] = data['Prediction'].astype('category')
    st.subheader("Countplot")
    sns.set(style="darkgrid")
    countplot = sns.countplot(x="Prediction", data=data)  # Replace 'Your_Column_Name' with the actual column name
    st.pyplot(countplot.figure)

def show_sentiment_analysis():
    data = st.session_state['predicted_data']
    data['Sentiment'] = data['Prediction'].apply(get_sentiment)
    st.write(data)
    st.title("Sentiment Analysis Chart")
    st.subheader("Sentiment Distribution")
    plt.hist(data['Sentiment'], bins=20, edgecolor='k')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
def get_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score < 0:
        return 'Negative'
    else:
        return 'Neutral'
    
   

        
if __name__=='__main__':
    main()
