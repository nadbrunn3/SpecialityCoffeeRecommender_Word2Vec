import streamlit as st
from function import *
from streamlit_option_menu import option_menu
import plotly.express as px
from streamlit_folium import st_folium
import folium


def main():

    with st.sidebar:
        choice = option_menu(
            menu_title='Main Menu',
            #options = ['Home', 'About', 'Flavour Based Recommender', 'Coffee Based Recommender', 'Heatmap Coffee Origin'])
            options = ['Home', 'About', 'Flavour Based Recommender','Heatmap Coffee Origin'])


    
    if choice == 'Home':
    
        st.title('SPECIALITY COFFEE RECOMMENDER')
        st.markdown('#')
        st.image('img/coffeeberry.jpg', caption=None, width=None , output_format="auto")
        st.markdown('#')
        
        st.header('What Makes a Good Cup of Coffe: ')
        
        st.markdown("`Speciality Coffee is a term for the heighest grade of coffee available.`")
        st.write("It can consistently exist through the dedication of the people who have made it their life's work to continually make quality their highest priority. This is not the work of only one person in the lifecycle of a coffee bean.")
        st.write("Specialty can only occur when all of those involved in the coffee value chain work in harmony and maintain a keen focus on standards and excellence from start to finish. This is no easy accomplishment, and yet because of these dedicated professionals, there are numerous specialty coffees available right now, across the globe, and likely right around the corner from you.")
        col1, col2, col3 = st.columns(3)
        with col3:
            st.caption('source: Speciality Coffee Association; Wikipedia')
            
        st.markdown('#')
        st.subheader('3 Key Factors impacting the Coffee Flavour')
        st.write('Klick to learn more')

        col1, col2, col3 = st.columns(3)
        with col1:
            with st.expander('Origin and Variety'):
            
                st.markdown('The big main varietys are `Arabica` and `Robusta`.')
                
                st.markdown('Arabica is often found in Speciality Coffee. Different Sub Varieties grow in different regions depending on the climate and the growing altitude.')
                
                st.write('People sometimes refere to the origin as a flavour type. Coffees from Kenyia for example have a winey note to them. Ethiopien coffee are more fruity and citric.')
                
                st.image('img/variety.jpeg', caption='source: sprudge')
                st.image('img/coffee_belt.jpeg', caption='source: melacoffe')
        

        with col2:
            with st.expander('Processing'):
                st.markdown('Many processing methods exist. The main ones are `Dry`, `Wet` and `Honey`.')
                st.image('img/process1.jpeg')
                st.image('img/processvs.webp')
            
        with col3:
            with st.expander('Roast Level'):
                st.markdown('Speciality Coffee in general will be roasted `Light` to `Medium`. Dark roasts are more often found in Italy where the Italian Espresso is served.')
                st.write('The darker the roast the more flavours are beeing covered.')
                st.image('img/roast.jpeg')
        
        
        
        
    if choice == 'About':
    
        
        st.subheader('Simply by replacing your morning coffee with green tea, you can lose up to 87% of what little joy you had left in your life')
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col5:
            st.caption('Goethe')
        st.markdown('#')
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            st.header('THE PROJECT')
        
        
        # HEATMAP

        
        st.map(coordinates)
        
        
        # TABS
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Process", "Data Gathering", "Data Cleaning", "The Model", "The Recommender", "Next Steps"])

        with tab1:
            st.header("The Process")
            
            st.markdown('1. Research in Scientific Articles and Reddit')
            st.markdown('2. Data Gathering - Web Scraping')
            st.markdown('3. Data Cleaning and Harmonizing')
            st.markdown('4. Vectorizing Flavour Profiles')
            st.markdown('5. Build Algorithm for Recommender')
            st.markdown('6. Continuous Improvement')
                            
            with st.expander('Gantt Diagram'):
                df = pd.DataFrame([
                    dict(Task="Data Gathering", Start='2022-08-01', Finish='2022-09-12', Completion_pct=30),
                    dict(Task="Data Processing", Start='2022-09-8', Finish='2022-09-14', Completion_pct=70),
                    dict(Task="Vectorizing Descriptors", Start='2022-09-10', Finish='2022-09-14', Completion_pct=100)
                    ])

                fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Completion_pct")

                st.plotly_chart(fig, use_container_width=True)
           

        with tab2:
           st.header("Data Gathering")
           
           st.subheader('Challenge: ')
           st.write(' - Only Speciality Coffee was used')
           st.write(' - Differences in Flavour Profiling depending on Cupper, Reviewer, Institution')
           st.write(' - Different Flavour Wheels exist (SCA, CCC)')
           st.write(' - European Coffee Roasters underrepresented in the data')
           st.write(' - Local Roasters need to be scrapped individually or contacted')
           st.write(' - Dependend Features: not disclosed from roaster/shop or not gathered in the web scraping process')
           st.subheader('Result: highly unharmonized data in different languages')
           
           with st.expander('Ressources'):
                st.write('Data for Web Scraping from: ')
                st.write('Coffee Review: https://www.coffeereview.com/')
                st.write('Fellows: https://www.coffee-fellows.com/')
                st.write('MistoBox: https://www.mistobox.com/')
                st.write('Kaffeezentrale: https://www.kaffeezentrale.at/')
                st.write('Special Thanks to Reddit User @nogg3r5 - https://airtable.com/appCJ1JZLBf0I7vmn/tblHUhm8lfPcw1nHC/viwQL1RtdUHyPWrNZ?blocks=hide')
                
                st.write('Kaffeothek: https://www.kaffeeothek.at/')
                
                st.markdown('#')
                st.write('Single Roasters: ')
                st.write('19 Grams: https://19grams.coffee/')
                st.write('Gardelli: https://shop.gardellicoffee.com/')
                
                st.markdown('#')
                st.write('Research Sources:')
                st.write('SCA: https://www.coffeereview.com/')
                st.write('CCC: https://counterculturecoffee.com/')
                st.write('CoffeeDiff: http://coffeediff.com/')
                st.write('Coffee Insurrection: https://www.coffeeinsurrection.com/')
                st.write('European Coffee Trip: https://europeancoffeetrip.com/')
                st.write('James Hoffmann: https://www.youtube.com/channel/UCMb0O2CdPBNi-QqPk5T3gsQ')
            
            
            
            
            

        with tab3:
           st.header("Data Cleaning")
           
           st.dataframe(combined_w_nan)
           
           fig1 = px.histogram(combined_clean, x="roast")
           st.plotly_chart(fig1)
           
           unique_process = combined_w_nan.process.unique()
           my_dict={'unique_processes': unique_process.tolist()}
           st.json(my_dict)
           
           fig2 = px.histogram(combined_clean, x="process")
           st.plotly_chart(fig2)
           
           
        with tab5:
            st.header("The Recommender")
            
            st.markdown('**Two Recommender Systems:**')
            st.write('1. Returns coffee beans based on input flavour(s)')
            st.write('2. Returns coffee beans that are similar to a specific coffee in the database')
            st.caption('The Application currently only supports Recommender 1. Recommender 2 will be deployed later in the year.')
            
            st.markdown('#')
            st.markdown('For the Recommendations and unsupervised Model - the `Nearest Neigbhor Algorithm` is used: ')
            
            code = '''knn = NearestNeighbors(n_neighbors = 10, metric='cosine', algorithm='brute')'''
            st.code(code)
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander('Flavour Based Recommender'):
                    st.image('img/flavourwheel.png')
                    st.markdown('STEP 1: To be able to match the flavours of the user the recommender provides a list of predefined flavours taken from the two most common flavour wheels. The `Spaciality Coffee Association Wheel` is the most frequenly used reference.')
                    st.markdown('STEP 2: Since not all flavours from the wheel are present in the coffee the model retrieves the most similar word in the database using the library `scapy.load("en_cor_web_lg")`.')
                    st.write('STEP 3: The model calculates the average weighted vector of all input features')
                    st.markdown('STEP 4: The Model returns the top 10 similar coffees by computing the `Kneighbors Cosine Distance`')
                
            with col2:
                with st.expander('Coffee Based Recommender'):
        
                    st.markdown('The model takes a Coffee Name as input and returns the top 10 similar coffees by computing the `Kneighbors Cosine Distance`')
                    
                

        with tab4:
            st.header("Algorithm Explained")
            st.write(' - Lemmatizing and Cleaning Stopwords')
            st.write(' - Bi_Gram and Tr_Gram')
            st.write(' - Word Embedding with W2V')
            st.write(' - Embedding Coffee Flavour Profil with Top 1000 Words')
            st.write(' - Weighing Flavours and avagering Coffee Profiles with TD-IDF')
            st.subheader('Result:')
            st.write('Each Coffee Review is now vectorized by the average of the weighted word vectors from the W2V Model!')
            
            with st.expander(label='Python Libaries'):
                code = '''import re
                import nltk
                from nltk.stem import WordNetLemmatizer
                from nltk.corpus import stopwords
                
                from gensim.models.phrases import Phrases, Phraser
                from gensim.models import Word2Vec
                
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                from sklearn.neighbors import NearestNeighbors
                '''
                st.code(code, language='python')
            
            with st.expander(label='nGram - Analyzing Tokens'):
                df = df_word_freq.reset_index()
                df = df.rename(columns={'Unnamed: 0': 'wfreq', '0': 'count'})
                st.json(df[['wfreq']].to_dict(), expanded=False)

                fig = px.line(df, x='wfreq', y='count', title = 'Top 1000 Words in Corpus')
                st.plotly_chart(fig)
            
        with tab6:
            st.header("Next Steps")
            st.write('Further Research:')
            st.markdown(' - Gather more `harmonized data` for `ORIGIN`, `VARIETY`, `ROAST LEVEL`, `PROCESS`')
            st.markdown(' - Increase the sample size from the `local/European region`')
            st.markdown(' - Include `flavour impacting factors` into the model recommendations')
            st.markdown(' - Improved `Parameter Setting` for NN and Kneighbor')
            
            st.write('Limitation:')
            st.markdown(' - For Finding the `Most Similar Descriptor the Word Vectores` that are used are not fitted to flavour descriptions. For example: For Oil it returns Water as the most similar word. This might be due to the fact that the two words are used frequentyl together in texts that are used to train the SpaCy Libary. In terms from flavour and mouthfeel they terms seem to be further apart.')
            
            
            st.write('Adding or Improving: ')
            st.markdown(' - `Deep EDA` on correlation between different aspects of the coffee and flavours')
            st.markdown(' - Creating Coffee Flavour Clusters with `Topic Clustering NLP`')
            st.markdown(' - `Increase word tokens` to > 1000')
            st.markdown(' - `Automate Coffee Scraper` and Cleaning Process')
            st.markdown(' - `Map Words in Corpus to a List of Words` - reducing unnecessary words like cup')
            st.markdown(' - Include a `Purchase Link` and Addditional Information to the output')
            
      
    if choice == 'Flavour Based Recommender':

        ### SELECT BOX FLAVOUR
        
        st.title('Search by FLAVOUR & AROMA')
        st.markdown('#')
        st.image('img/flavourwheel.png', caption=None, width=None , output_format="auto")
        st.markdown('#')

        st.subheader('pick the flavours you enjoy: ')
        form1 = st.form(key='Options')
        selected_flavours = form1.multiselect(label = '', options=wheel)
        
        button1 = form1.form_submit_button('Find Coffee Beans')

        if button1:
            recommended_coffee = match_flavour_pick(selected_flavours)
            df = coffee.loc[coffee.coffee.isin(recommended_coffee)]
            df.drop(['Unnamed: 0', 'index', 'level_1', 'flavour'], axis=1, inplace=True)
            df = df.rename(columns={'level_0': 'source'})
            st.dataframe(df)

          
    if choice == 'Coffee Based Recommender':
    
        st.title('Search by COFFEE')
        
        with st.expander('Choose from these coffees', expanded=True):
            st.dataframe(coffee)
            
        form2 = st.form(key='Recommend')
        coffee_list = coffee['coffee'].tolist()
        #coffee_choice = st.selectbox("",coffee_list)
        coffee_choice = form2.selectbox('', coffee_list)

        button2 = form2.form_submit_button('Find Coffee Beans')
        #button2 = st.button("Find Coffee Beans")
            
        ## Recommender
        st.write(type(coffee_choice),coffee_choice)
        recommended_coffee = match_coffee_pick(coffee_choice)
        
    if choice == 'Heatmap Coffee Origin':
    
        st.title('Where does our coffee come from: ')
                    
        map = heatmap(coordinates)
        st_folium(map, width= 725)
        
        
if __name__ == '__main__':
    main()
