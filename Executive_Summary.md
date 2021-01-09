## Executive Summary

What's your favorite kind of music? Do you know what draws you to this genre -- the tempo, the instrumentation, the arrangements, the lyrics? For a lot of fans of country-western music, the lyrics and the stories are what draws them in. This might be different from fans of rock music, who are more turned on by the drum beats and the guitar solos.

In this analysis, I will look at a large corpuses of written discussion about country and rock to see what patterns emerge from the commentary. What do country music fans talk about that rock music fans eschew, and vice versa? Can we divine the topic of the discussion by looking only at the language used by the fans of the two distinct genres?

                        What type of model will be developed?
                        How will success be evaluated?

As certain patterns in the language emerge, can they then inform what is to come in these genres? I believe we can. Pulling oft-repeated words and phrases out a heated discussion amongst fans about current country music releases can easily inform what a country music songwriter might want to address in their next hit! If those words are highly distinct from the words use by fans of rock and roll, the trend is even more salient, as they are clearly nearest and dearest to hearts of the country music fanbase and not the music community at large.

To that end, I will a body of posts from both the country music subreddit and the rock music subreddit to show how the language of the posters itself can reveal the genre about which they are speaking. Beyond that, I will show that certain words lend themselves to this differentiation, and therefore can reveal the most important and interesting topics -- and thus words! -- to followers of each genre.

#### Why Do You Care?


Good question!

No, I didn't just waste many hours of work to tell country music artists to sing about love, trucks, country, and mama. This analysis digs deep to help musicians, songwriters, and fans alike better understand what themes at the heart of the music they love by dissecting the language they all use to talk about it. The data is drawn from the passionate commentary of the listeners themselves, so it tells an important story that is based on far more than conjecture: a data-driven report on what music fans care about the most.

Better yet, the breakdown is genre specific! Using robust classification algorithms, my analysis shows clearly that armchair rock music critics aren't saying the same thing as their country-western counterparts. While this report examines only two genres, I wholeheartedly believe that it can be applied across all types of music: classical, reggae,

 While there are many applications of this examination, my hope is that this helps artists, producers, venue owners, and music industry execs know what themes are currently resonating with their fan base. This will inform song composition, tour schedules, media market press packs, and so much more!  

If the fans are talking the talk, the musicians can walk the walk -- all supported by data!

The Solution!

With a clear analysis, I will help show the distinctive language being used by fans of different types of musics. In short, this will serve to unite fans and music professionals and help them find integrated ways to produce, market, and disseminate music, so that what the people want is what people get.

## Data Collection

*See notebook: [01_Web_Scraping_&_Data_Collection](projects/project_3/code/01_Web_Scraping_&_Data_Collection.ipynb)*

To perform the modeling, I collected actual posts from two very popular and prolific subreddits: [country music](https://www.reddit.com/r/country/) and [rock music](https://www.reddit.com/r/rock/) ("Rock music in all its forms."), which have 24.7K and 46.4K members, respectively. I used the 3000 most recent posts from each subreddit, and they were posted between June 25, 2020 and January 6, 2020. While the raw dataset had 77 features, I decided

## Data Cleaning & Pre-Processing

*See notebook: [02_Data_Cleaning](projects/project_3/code/02_Data_Cleaning.ipynb)*

Fortunately, the natural language data coming out of Reddit was fairly clean directly after scraping. I performed the following steps to get my data ready for Exploratory Data Analysis (EDA):

+ Put data into pandas DataFrame
+ Added a new column called `datetime` which converts the UTC time Reddit provides into a human-readable date
+ Added a new column called `merged` which joined the `title` and `selftext` columns that came out of Reddit. (`selftext` is thet body of the post.)
+ Added a column called `label` that would indicate the class of the post
+ Removed special characters, hyperlinks, and some numbers.
+ converted all letters to lowercase
+ Got rid of null values, though there were very few.
+ Created a new column called `tokens` by applying the `RegexpTokenizer(r'\w+')` to the `merged column`
+ Created a new column called  `stemmed` by applying the `SnowballStemmer(language='english')` to the `tokens` column, then joined these back together into a string for a new column called `combo`. I did end not end up using the `stemmed` or `combo` columns because the stemmed words proved to be detrimental to my analysis and visualizations, so eventually I removed these columns before pickling my DataFrames to be used in models.

Once the two DataFrames were nice and clean, with the right columns, I pickled both of them as `country_token_pickle.pkl` and `rock_token_pickle.pkl`, respectively.

## EDA

*See notebook: [03 - EDA_&_Model_Prep](projects/project_3/code/03_EDA_&_Model_Prep.ipynb)*

I first went back and looked at each of the two original DataFrames in their cleaned but not-yet-concatenated forms. Using my function `vectorize_and_plot`, I fit and transformed the `merged` column using `CountVectorizer` the used matplotlib to create plot showing the 15 most common words in each DataFrame. Not to be confused with the word importance plots that I build after modeling, the word frequency plots allowed me to see which word were most common and thereby informed my custom list of stop words for future use in modeling.

```python
country_rock_stop_words = ['country', 'countries', 'rock', 'roll', 'just','song', 'songs', 'music', 'album', 'band', 'bands', 'artist', 'artists', 've', 'don']
```
I chose to remove words that are most common in one or both of the subreddits because, while some of them might help improve accuracy, they weaken the storyline and impair the ability of models to detect important words and features by drawing heavy attention to themselves and thereby skewing coefficients and feature importance distribution. I removed "ve" and "don" because my pre-processing created these from contracted words, and they ended up showing up very frequently. In the future I would modify my pre-processing to fix this problem, but I ran out of time to do so.

The next EDA step was to inspect the lengths of the subreddit posts. For each of the DataFrames, I looked at the distribution, mean, minimum, and maximum of the post lengths by applying my post_length_distribution function.

Since there didn't appear to be any exhorbitantly long posts that would distort my modeling, and a scan of the shortest posts (<= 3 words) indicated that these very short ones were still valuable, I decided not to remove any posts based on length.  

I concatenated the till-now separate country and rock DataFrames into one DataFrame called `country_rock` and converted the values in the `label` column to binary numeric labels 0 and 1. The new DataFrame included 6000 rows -- 3000 from each subreddit -- because I had not had to drop any rows during preprocessing.

Finally, I saved my combined DataFrame as `country_rock.pkl` for use in modeling.

## Modeling

My first model was my baseline model, which actually included no machine learning. I took the value counts of my target variable `y`(`country_rock['label']`) by running `y.value_counts(normalize=True)`. The output show the 0.5 of the total have label 1 (rock subreddit) and 0.5 have label 0 (country subreddit). Thus the accuracy of my baseline model is 50%, i.e. if we were to predict that all of the posts are from the country music subreddit, we would be right 50% of the time.

#### Vectorization
In order to do any ML models, I would need to vectorize the natural language data first. As I tried different models, I paired them with either `CountVectorizer` or `Tf-Idf Vectorizer` to see which gave better results.

#### Choosing an Estimator


My ultimate goal was to find a classification model that:
1. Could identify the class of a subreddit with the greatest accuracy possible.
2. Minimized error due to variance i.e. is not too overfit.
3. Relates to my problem statement and could be used in viable analysis; ideally it would be interpretable and provide valuable insights.

From there, modeling occurred in two stages.

First, I performed a series of grid searches to identify the optimal estimator and hyperparameters for my final model. *(See notebook: [04_Model_Benchmarking](projects/project_3/code/04_Model_Benchmarking.ipynb)).* I ran grid searches on the following vectorizer + classifier pairings (check out the notebook for the hyperparameters I searched over to tune):

+ Countvectorizer + Logistic Regression
+ Tf-Idf + Logistic Regression
+ CountVectorizer + Multinomial Naïve Bayes
+ Tf-Idf + Multinomial Naïve Bayes
+ CountVectorizer + k-Nearest Neighbors
+ CountVectorizer + Extra Trees Classifier
+ Tf-Idf + Random Forest Classifier
+ CountVectorizer + Support Vector Classifier

For each of these, I performed an inital evaluation to know whether to tweak it or consider it for use in my final analysis. I found the best parameters, train test accuracy, test set accuracy, and overall accuracy (`.best_score_`) for each grid search.

I also plotted the confusion matrix to see where the false negatives and false positives were occurring. I thought this might inform some portion of my analysis, but in the end I used only as my evaluation metric. Since accuracy views all misclassifications as equal, the information in the confusion matrices is no longer relevant to my final analysis. However, they help me understand how my models are performing so I left them in my `04_Model_Benchmarking` notebook.

Given my initial objectives for my favorite model, above, I decided first to use my Tf-Idf + Logistic Regression combo for my final model and evaluation. This classification had reasonably good accuracy -- 82% on the testing set -- and, while being overfit, was not ostentatiously so. (Accuracy on the training set was 92%.) Moreover, accuracy was not my primary concern in this process because what I was hoping to do ultimately was to draw out the most important words in each of the models. So alas, the most significant reason for me choosing the Logistic Regression classifier is because it offers interpretable coefficients for my features -- words, in this case -- which was important given my stated goal of drawing out trends in the data via important words.

In the next notebook (see [05_Modeling_&_Evaluation_&_Visualizations](projects/project_3/code/05_Modeling_&_Evaluation_&_Visualizations.ipynb)), I executed the Tf-Idf-vectorized Logistic Regression model with the best hyperparameters selected by the grid search. I did so using `Pipeline` in two stages.

I then used this fit pipeline to execute my `find_most_important_words` and `plot_most_important_words` functions. Plotting the top 25 most important words from each subreddit yielded the following figures:

![importances](projects/project_3/code/importance.pdf)








Evaluation & Analysis
