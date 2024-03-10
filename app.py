import streamlit as st
import pandas as pd
import joblib


# streamlit run app.py

# read KMeans model
kmeans_model = joblib.load('kmeans_model.joblib')

# read edited listings data
cph_listings_df = pd.read_csv('cph_listings_df_edited.csv')

# method to filter listings based on number of reviews
def filter_boliger(min_review):

    # add each listing to a cluster
    listing_clusters = kmeans_model.predict(cph_listings_df[['price', 'number_of_reviews']])
    
    # save in df
    cph_listings_df['cluster'] = listing_clusters
    
    # Beregn det gennemsnitlige antal anmeldelser for hver klynge
    klynge_gns_anmeldelser = cph_listings_df.groupby('cluster')['number_of_reviews'].mean()
    
    # Filtrer boliger baseret på brugerens input for minimum antal anmeldelser
    filtrerede_boliger = cph_listings_df[cph_listings_df['number_of_reviews'] >= min_review]
    
    return filtrerede_boliger

# Streamlit-app
st.title('Boligsortering efter anmeldelser')

# user input
min_review = st.slider('Minimum Antal Anmeldelser:', min_value=0, max_value=int(cph_listings_df['number_of_reviews'].max()), value=0)

# Filtrer boliger baseret på brugerinput
filtrerede_data = filter_boliger(min_review)

# Vis filtrerede data
st.dataframe(filtrerede_data)
