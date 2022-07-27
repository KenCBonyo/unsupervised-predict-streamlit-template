"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from matplotlib.style import use
from nbformat import write
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
movies= pd.read_csv("resources/data/movies.csv")
imdb=pd.read_csv("resources/data/imdb_data.csv")
columns=["userId","movieId"]
ratings=pd.read_csv("resources/data/ratings.csv", usecols=columns)
movies["genres"] = movies["genres"].str.replace('|', ' ', regex=True)
movies["genres"] = movies["genres"].str.replace('(', ' ', regex=True)
movies["genres"] = movies["genres"].str.replace(')', ' ', regex=True)
movies["title"] = movies["title"].str.replace('(', ' ', regex=True)
movies["title"] = movies["title"].str.replace(')', ' ', regex=True)
movies["title"] = movies["title"].str.strip()
#insert into a new column named "movie_prod_year"
movies["movie_year"] = movies["title"].str[-4:]#extract last 4digits
movies["movie_year"] = movies["movie_year"].str.replace(r'[a-zA-Z]', '', regex=True)
movies["movie_year"] = movies["movie_year"].str.replace(r'[^a-zA-Z0-9]', '', regex= True)
movies["movie_year"] =movies["movie_year"].str.replace(r' ', '', regex=True)
movies["title"] = movies["title"].str[:-4]
movies["title"] = movies["title"].str.rstrip()
info_data= pd.merge(movies,imdb[['movieId','runtime','director', 'title_cast','budget']], on='movieId')
info_data["title_cast"] = info_data["title_cast"].str.replace('|', ',', regex=True)
movies_list=movies.sort_values(by=['genres', 'movie_year'], ascending=False)



# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","About Us", "Get movie by genre","snoop around", "Solution Overview", ]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.image('resources/imgs/Originals_Logo.png',width=400)
        
        st.header("Interesting Insights")
        st.markdown("Here are some insight that can help you decision making")

        st.title('Year')
        movie_prod_by_year= movies.groupby('movie_year')['movieId'].nunique().reset_index(name='movieIdCount')
        movie_prod_by_year.sort_values(by=['movieIdCount'], ascending=False)
        movie_prod_by_year= movie_prod_by_year.nlargest(columns='movieIdCount', n = 20) 
        data_year = pd.DataFrame({
            'index': movie_prod_by_year['movie_year'],
            'movie_count': movie_prod_by_year['movieIdCount'],}).set_index('index')
        with st.expander('See number of years data'):
            st.write(data_year)
        st.line_chart(data_year)
        with st.expander("See Explanation"):
            st.markdown("This chart shows the top ten years with the highest number of movies produced.\
                 It shows a trend in how the movie industry has grown over the years in the number of movies produced.\
                    We can see a consistent increase over the years, and a sharp decline in 2019, this, we assume can be due to the\
                        pandemic outbreak in the year 2019.")

        st.title('Genres')
        genre_count= movies.groupby('genres')['genres'].count().reset_index(name='movieIdCount')
        genre_count.sort_values(by=['movieIdCount'], ascending=False)
        genre_count = genre_count.nlargest(columns='movieIdCount', n = 10)
        data_genre = pd.DataFrame({'index':genre_count['genres'], 
        'movie_count':genre_count['movieIdCount'],}).set_index('index')
        #data_genre= data_genre.sort_values(by='movie_count', ascending=False)
        with st.expander('See top 10 genres data'):
            st.write(genre_count)
        st.bar_chart(data_genre)
        with st.expander("See Explanation"):
            st.markdown("This chart shows the top 10 most viewed and rated genre. We see Drama genre leading the way followed by \
                the comedy genre. This can help in budgeting for genre of movies to have on the platform.")
        
        st.title('Users')
        user_rating= ratings.groupby('userId')['movieId'].nunique().reset_index(name='movieIdCount')
        user_rating.sort_values(by=['movieIdCount'], ascending=False)
        user_rating = user_rating.nlargest(columns='movieIdCount', n = 10)   
        #plt.figure(figsize=(10,5))
        data = pd.DataFrame({
            'index': user_rating['userId'],
            'movie_count': user_rating['movieIdCount'],}).set_index('index')
        with st.expander('See top 10 users data'):
            st.write(user_rating)
        st.bar_chart(data)
        with st.expander("See Explanation"):
            st.markdown("This chart shows the top 10 movie viewers. User 547 has the highest number of movies viewed and rated\
                     with a huge gap and every other user. It can be hypothesised that this user is most likely\
                        an influencer. If he is, we can look out for movies that he rates highly and make sure such movies are also on our \
                            platform for other users to watch")
        
        #st.image('resources/imgs/image4.png',use_column_width=True)
        
        
        st.title('Modelling')
        st.image('resources/imgs/rec-systems.png',use_column_width=True)
        st.markdown("Gain a little insight into how we built our model.")
        with st.expander("See Explanation"):
            st.markdown("In building the application, we made use of both collaborative and content based filtering.")
            st.write('Collaborative Filtering')
            SVD_url = "https://analyticsindiamag.com/singular-value-decomposition-svd-application-recommender-system/"
            ricked_url= "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=43s"
            st.markdown('Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions\
                 by similar users. It works by searching a large group of people and finding a smaller set of users with tastes similar\
                     to a particular user. It looks at the items they like and combines them to create a ranked list of suggestions.\
                      There are many ways to decide which users are similar and combine their choices to create a list of recommendations.\
                        For this project, the dataset was trained using SVD model algorithm.\
                            You can read more on SVD model [here](%s)' % ricked_url)
#             st.markdown('oops, got ya!!! :satisfied:, meant [here](%s)' % SVD_url)
            st.write('Content-based Filtering')
            st.markdown("Content-based filtering is a type of recommender system that attempts to guess what a user may\
                 like based on that user’s activity. Content-based filtering makes recommendations by using keywords \
                    and attributes assigned to objects in a database (e.g., items in an online marketplace) and matching \
                        them to a user profile. The user profile is created based on data derived from a user’s actions, \
                            such as genres of movies watched, ratings (likes and dislikes), directors, cast of the movies e.t.c")


        
        
    if page_selection == "About Us":
        st.image('resources/imgs/Originals_Logo.png',width=400)
        
        st.title("About Us")
        st.write("The source, the core from where every creativity and ingenuity emanates, \
                  The Originals, we are a team of data scientist, sold out to solving \
                  real human problems with flavour and style. ")
        
        st.image('resources/imgs/pheel.png',use_column_width=True)
        
        st.title("Our Mission")
        st.write("Use data to optimize human possibilities of attaining excellence, one solution at a time.")
        
        st.title("Our Vision")
        st.write("Improve living conditions, championing new innovations powered by data")



    if page_selection == "Get movie by genre":
        st.image('resources/imgs/Originals_Logo.png',width=400)
        genres= movies_list.groupby(['genres'])['genres'].count().reset_index(name='Count').sort_values(['Count'], ascending=False)
        genre_list=list(genres['genres'])
        year_list=movies["movie_year"].sort_values()
        st.title("Get movies by Genre and Year")
		#st.subheader("Climate change tweet classification")
        genre= st.selectbox("What genre would you be interested in",genre_list)
        year= st.selectbox("What year would you be interested in",year_list.unique())
        fave=[genre, year]


        if st.button("recommend movies"):
            if year in movies_list['movie_year'].to_list():
            #for genre, year in fave:
                #if genre and year in fave:
                    #output= movies_list[movies_list['genres','movie_prod_year']== fave]
                output_year= movies_list[movies_list['movie_year']== year]
                output_year=pd.DataFrame(output_year)
            if genre in movies_list['genres'].to_list():
                output_genre=movies_list[movies_list['genres']== genre]
                output_genre=pd.DataFrame(output_genre)
            result=pd.merge(output_genre, output_year, on=['movie_year','movieId','genres','title'])
            result= result.drop("movieId", axis=1)
            if len(result)==0:
                st.write("oops :disappointed: .... sorry, no movie available for this category just yet :disappointed: ")
            elif len(result)<10:
                st.write("we are so sorry, we only have ", len(result), "movies for this category at this time")
                result=st.dataframe(result)
            else:
                result=result.sample(n=10)
                result=st.dataframe(result)

            #result=pd.DataFrame(result)
            
            #result= st.dataframe(result)
            #st.success(result)

    if page_selection == "snoop around":
        st.image('resources/imgs/Originals_Logo.png',width=400)
        st.title("Snoopy")
        st.markdown("want to get more information about your favorite movies, you are in the right place. :wink: All you have to do \
            is select or type the  title of the movie and voila!!!!")
        titles_list=info_data.sort_values(by='title')
        titles_list=titles_list["title"].unique()
        titles= st.selectbox("What movie would you like to snoop around",titles_list)
        if st.button("snoop"):
            if titles in info_data["title"].to_list():
                output= info_data[info_data['title']== titles]
                output=output.drop('movieId', axis=1)
                st.write('Here are some informations about the movie')
                output=st.table(output)
                #output=st.dataframe(output)
                #st.success(output)


                
                    




    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
