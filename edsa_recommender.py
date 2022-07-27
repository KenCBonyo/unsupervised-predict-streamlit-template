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
from nbformat import write
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    page_options = ["Recommender System","Solution Overview", "About Us", "Get movie by genre","snoop around"]

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
        
        st.title("Interesting Insights")
        st.markdown("Here are some insight that can help you decision making")

        st.write('Genres')
        st.image('resources/imgs/image2.png',use_column_width=True)
        with st.expander("See Explanation"):
            st.markdown("This chart shows the top 30 most rated movies. Shawshank Redemption got the most rating.\
                 Some movies got an average of 5 stars rating, but in the course of our analysis, we discovered\
                  that the number of ratings they got was very few, hence that was possible. What this chart shows is that,\
                     Shawshank Redemption got the most number of Users rating it >3.9 .")
        
        st.write('Users')
        user_rating= ratings.groupby('userId')['movieId'].nunique().reset_index(name='movieIdCount')
        user_rating.sort_values(by=['movieIdCount'], ascending=False)
        user_rating = user_rating.nlargest(columns='movieIdCount', n = 10)   
        plt.figure(figsize=(10,5))
        st.bar_chart(data=user_rating)
        #ax = sns.barplot(data=user_rating, x= 'userId', y = 'movieIdCount')
        #ax.set(ylabel = "count of movies rated", xlabel='UserId')
        plt.title("Top 10 Users")
        #st.pyplot(fig)
        #st.image('resources/imgs/image3.png',use_column_width=True)
        with st.expander("See Explanation"):
            st.markdown("This chart shows the top 30 movie viewers. This information is useful to identify the movie preference\
                of top customers. these top customers are most likely influencers and can make users watch a movie they probably\
                     wouldn't have considered watching.")
        
        st.image('resources/imgs/image4.png',use_column_width=True)
        st.write("This chart shows the years with the highest number of movies produced. This information will help you to visually explore how the movie industry has performed over the years")
        
        st.write('Modelling')
        st.image('resources/imgs/rec-systems.png',use_column_width=True)
        st.markdown("Gain a little insight into how we built our model.")
        
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
        titles= st.selectbox("What movie would you like to snoop :blush:",titles_list)
        if st.button("snoop"):
            if titles in info_data["title"].to_list():
                output= info_data[info_data['title']== titles]
                output=output.drop('movieId', axis=1)
                output=st.dataframe(output)
                #st.success(output)


                
                    




    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
