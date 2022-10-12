import requests 
from bs4 import BeautifulSoup
import yaml
import spacy


def read_yaml(filepath):
    with open(filepath, 'r') as f: 
        return yaml.safe_load(f)
    
    

def bs4(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup



def rename(df):
    col = df.columns.tolist()
    for idx, i in enumerate(col):
        if i in ['coffee_title', 'title']:
            col[idx] = 'coffee'
        elif i in ['coffee_country', 'coffee_origin', 'land']:
            col[idx] = 'origin'
        elif i in ['flavor', 'review_text', 'review', 'flavours']:
            col[idx] = 'flavour'
        elif i =='roast_level':
            col[idx] = 'roast'
        elif i in ['aufbereitung', 'production', 'Production']:
            col[idx] = 'process'      
        elif i == 'coffee_variety':
            col[idx] = 'variety'
        else: 
            col[idx] = i           
            
        df.columns = col
    return df



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



def match_coffee_pick(coffee_to_match):
    
    vec = coffee_to_match['coffee_vector'].tolist()[0]
    coffee_name_to_match = coffee_to_match['coffee'].tolist()[0]
    flavours_to_match = coffee_to_match['normalized_phrased_flavour'].tolist()[0]
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
    recommended_coffee_flavour_profil = []
    coffee_df_original = []
    coffee_df_original_index = []    
    
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
        
    return indices, recommended_coffee, recommended_coffee_flavour_profil, coffee_df_original, coffee_df_original_index


def similar_word_in_list(input_word):
    
    nlp = spacy.load("en_core_web_lg")
    similarity = 0
    descriptor = ''
    input_word = nlp(input_word)
    for term in weights.keys():
        term = nlp(term)
        if input_word.similarity(term) > similarity:
            similarity = input_word.similarity(term)
            descriptor = term
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
    
    return indices, recommended_coffee, recommended_coffee_flavour_profil, coffee_df_original, coffee_df_original_index