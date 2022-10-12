import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import yaml
import spacy
import pickle
from sklearn.neighbors import NearestNeighbors
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import folium
from folium import plugins
import streamlit
import streamlit_folium
import geopandas as gpd

### Models
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
knn_model = pickle.load(open('../models/knn_10_cosine_brute.pkl', 'rb'))
w2v_model = pickle.load(open('../models/w2v.pkl', 'rb'))

### Files
weights = pickle.load(open('../data_output/weights.pkl', 'rb'))
coffee = pd.read_csv('../data_output/vectorized_coffee_limited.csv')
wheel = pd.read_csv('../data_output/sca_wheel_flavours.csv')
#coordinates = pd.read_csv('../data_output/coordinates_xy.csv')
coordinates = pd.read_csv('../data_output/coffee_origin_long_lat.csv')
combined_w_nan = pd.read_csv('../data_clean/combined_df_w_nan.csv')
combined_clean = pd.read_csv('../data_clean/cleaned_combined.csv')
df_word_freq = pd.read_csv('../data_output/descriptor1000.csv')



def read_yaml(filepath):
    with open(filepath, 'r') as f: 
        return yaml.safe_load(f)
    
    

def bs4(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup


def fill_list_with_nan(values, length):
    """
    function:  to match the rows of extracted values
    input: values = list of values | length = length of coffees        
    output: list of values filled in with Missing Values at not extracted idex position.
    """
    index = []
    text = []
    ml = []
    x = 0
    for i in values: 
        index.append(i[0])
        text.append(i[1])
        
    for idx, elem in enumerate(index):
        while x < length:
            if elem == x: 
                ml.append(text[idx])
                x = x + 1
                break
            else: 
                ml.append(np.NaN)
                x = x + 1
    return ml 


'''
def match_coffee_pick(coffee_to_match):
    
    vec = coffee.loc[coffee['coffee'] == coffee_to_match, 'coffee_vector'].tolist()[0]
    coffee_name_to_match = coffee.loc[coffee['coffee'] == coffee_to_match, 'coffee'].tolist()[0]
    flavours_to_match = coffee.loc[coffee['coffee'] == coffee_to_match, 'normalized_phrased_flavour'].tolist()[0]
    
    # vec = coffee_to_match['coffee_vector'].tolist()[0]
    # coffee_name_to_match = coffee_to_match['coffee'].tolist()[0]
    # flavours_to_match = coffee_to_match['normalized_phrased_flavour'].tolist()[0]
    distance, indice = knn_model.kneighbors(vec, return_distance=True, n_neighbors=11)
    distances = distance.tolist()[0][1:]
    indices = indice.tolist()[0][1:]
    
    print('The Coffee for which we want a recommendation is {}'.format(coffee_name_to_match))
    print('The normalized flavours of the coffee are: {}'.format(flavours_to_match))
    print()
    print('-----------')
    print()
    print('\n RECOMMENDED COFFEES: \n')
    
    recommended_coffee = []
    #recommended_coffee_flavour_profil = []
    #coffee_df_original = []
    #coffee_df_original_index = []
    
    n = 1
    for d, i in zip(distances, indices):
        #coffee_df_original.append(coffee['level_0'][i])
        #coffee_df_original_index.append(coffee['level_1'][i])
        suggested_coffee = coffee['coffee'][i]
        recommended_coffee.append(suggested_coffee)
        #suggested_coffee_flavours = coffee['normalized_phrased_flavour'][i]
        #recommendet_coffee_flavour_profil.append(suggested_coffee_flavours)
        print('Suggestion', str(n), ': {} ---- with a cosine distance of {:.3f}'.format(suggested_coffee, d))
        print('This coffee has the following flavour profile: {}'.format(suggested_coffee_flavours))
        print('')
        n+=1
        
    return recommended_coffee
'''

def match_coffee_pick(coffee_to_match):
    
    vec = coffee.loc[coffee['coffee'] == coffee_to_match, 'coffee_vector'].tolist()[0]
    coffee_name_to_match = coffee.loc[coffee['coffee'] == coffee_to_match, 'coffee'].tolist()[0]
    flavours_to_match = coffee.loc[coffee['coffee'] == coffee_to_match, 'normalized_phrased_flavour'].tolist()[0]
    
    distance, indice = knn_model.kneighbors(vec, return_distance=True, n_neighbors=11)
    distances = distance.tolist()[0][1:]
    indices = indice.tolist()[0][1:]
    
    print('The Coffee for which we want a recommendation is {}'.format(coffee_name_to_match))
    print('The normalized flavours of the coffee are: {}'.format(flavours_to_match))
    print()
    print('-----------')
    print()
    print('\n RECOMMENDED COFFEES: \n')
    
    recommended_coffee = []  
    
    n = 1
    for d, i in zip(distances, indices):
        #coffee_df_original.append(coffee['level_0'][i])
        #coffee_df_original_index.append(coffee['level_1'][i])
        suggested_coffee = coffee['coffee'][i]
        recommended_coffee.append(suggested_coffee)
        suggested_coffee_flavours = coffee['normalized_phrased_flavour'][i]
        #recommended_coffee_flavour_profil.append(suggested_coffee_flavours)
        print('Suggestion', str(n), ': {} ---- with a cosine distance of {:.3f}'.format(suggested_coffee, d))
        print('This coffee has the following flavour profile: {}'.format(suggested_coffee_flavours))
        print('')
        n+=1
        
    print(recommended_coffee[:3])
        
    return recommended_coffee



def similar_word_in_list(input_word):
    
    nlp = spacy.load("en_core_web_lg")
    similarity = 0
    descriptor = ''
    input_word = nlp(input_word)
    for term in weights.keys():
        term = nlp(term)
        if input_word.similarity(term) > similarity:
            similarity = input_word.similarity(term)
            descriptor = str(term)
    return descriptor, similarity
    
    
    
def match_flavour_pick(input_flavours, number_of_suggestion = 10):
    
    weighted_input_vectors = []
    
    for flavour_term in input_flavours:
        flavour_term = wordnet.lemmatize(flavour_term).lower()     
        
        if flavour_term not in weights:
            most_similar_descriptor, similarity = similar_word_in_list(flavour_term)
            print('This coffee characteristic {} is not matched. Instead we will use the similar term {}'.format(flavour_term, most_similar_descriptor))
            print('- The similarity of the words is the highes we could find with {:.2f}'.format(similarity))
            print()
            
            flav_idf = weights[most_similar_descriptor]
            flav_w2v_vector = w2v_model.wv.get_vector(most_similar_descriptor).reshape(1, 300)
            weighted_flav_vector = flav_idf * flav_w2v_vector
            weighted_input_vectors.append(weighted_flav_vector)
            continue

        else: 
            flav_idf = weights[flavour_term]
            flav_w2v_vector = w2v_model.wv.get_vector(flavour_term).reshape(1, 300)
            weighted_flav_vector = flav_idf * flav_w2v_vector
            weighted_input_vectors.append(weighted_flav_vector)

    input_vector = sum(weighted_input_vectors)

    distance, indice = knn_model.kneighbors(input_vector, return_distance=True, n_neighbors=number_of_suggestion+1)
    distances = distance.tolist()[0][1:]
    indices = indice.tolist()[0][1:]
        
    recommended_coffee = []
    recommended_coffee_flavour_profil = []
    coffee_df_original = []
    coffee_df_original_index = []
    
    print('\n RECOMMENDED COFFEES: \n')

    n = 1
    for d, i in zip(distances, indices):
        coffee_df_original.append(coffee['level_0'][i])
        coffee_df_original_index.append(coffee['level_1'][i])
        suggested_coffee = coffee['coffee'][i]
        recommended_coffee.append(suggested_coffee)
        suggested_coffee_flavours = coffee['normalized_phrased_flavour'][i]
        recommended_coffee_flavour_profil.append(suggested_coffee_flavours)
        print('Suggestion', str(n), ': {} ---- with a cosine distance of {:.3f}'.format(suggested_coffee, d))
        print('This coffee has the following flavour profile: {}'.format(suggested_coffee_flavours))
        print('')
        n+=1
    
    return recommended_coffee


def heatmap(coord):
    gdf = gpd.GeoDataFrame(coord, geometry=gpd.points_from_xy(coord.longitude, coord.latitude))
    gjson = gdf['geometry'].to_json()
    map = folium.Map(location = [15,30], tiles='Cartodb dark_matter', zoom_start = 2)
    heat_data = [[point.xy[1][0], point.xy[0][0]] for point in gdf.geometry ]
    plugins.HeatMap(heat_data, min_opacity=0.2, blur=18).add_to(map)
    return map

    
