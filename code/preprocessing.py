import pandas as pd
import numpy as np
import os
from sklearn import model_selection, metrics, preprocessing
from sklearn.model_selection import train_test_split
from torch_geometric.data import download_url, extract_zip
import argparse

parser = argparse.ArgumentParser(description="Preprocessing")
parser.add_argument('--sorted', action='store_true', default=False, help='whether to preprocess sorted dataset')
parser.add_argument('--threshold', type=float, default=3.5, help='rating threshold for positive entries')

args = parser.parse_args()

SORTED = args.sorted
RATING_THRESHOLD = args.threshold #default = 3.5

url_ml25 = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'

if not os.path.exists('../rawdata'):
    os.makedirs('../rawdata', exist_ok=True)

extract_zip(download_url(url_ml25, '../rawdata/'), '../rawdata/')

rating_path = '../rawdata/ml-25m/ratings.csv'

rating_df = pd.read_csv(rating_path)

if SORTED:
    if os.path.exists('../rawdata/ratings_sorted.csv'):
        rating_df = pd.read_csv('../rawdata/ratings_sorted.csv', index_col=0)
        rating_df.sort_values(by='userId', ascending=True, inplace=True)
    else:
        sort_df = pd.read_csv('../rawdata/user_sorted_rating_std.csv')
        rating_df['sort_index'] = rating_df['userId'].apply(lambda userId: sort_df.userId.eq(userId).idxmax())
        rating_df.drop(columns=['userId'], inplace=True)
        rating_df.rename(columns={"sort_index": "userId"}, inplace=True)
        rating_df.sort_values(by='userId', ascending=True, inplace=True)
        rating_df.to_csv('../rawdata/ratings_sorted.csv')

    pathTrain = os.sep.join(['../data', 'ml-25m_sorted', 'train.txt'])
    pathTest = os.sep.join(['../data', 'ml-25m_sorted', 'test.txt'])
else:
    lbl_user = preprocessing.LabelEncoder()
    rating_df.userId = lbl_user.fit_transform(rating_df.userId.values)
    pathTrain = os.sep.join(['../data', 'ml-25m', 'train.txt'])
    pathTest = os.sep.join(['../data', 'ml-25m', 'test.txt'])

lbl_movie = preprocessing.LabelEncoder()
rating_df.movieId = lbl_movie.fit_transform(rating_df.movieId.values)

done = 0
perc_done = 0
length = len(rating_df['userId'].unique())

with open(pathTrain, 'w') as train, open(pathTest, 'w') as test:
    for userId in rating_df['userId'].unique():
        user_df = rating_df[(rating_df['userId'] == userId) & (rating_df['rating'] > RATING_THRESHOLD)]
        user_df['movieId'] = user_df['movieId'].astype(str)
        if (len(user_df['movieId']) > 1):
            trainData, testData = train_test_split(user_df['movieId'], train_size=0.8, shuffle=False)
            trainLine = userId.__str__() + " " + ' '.join(trainData)
            testLine = userId.__str__() + " " + ' '.join(testData)
            
            train.write(trainLine)
            test.write(testLine)

            train.write("\n")
            test.write("\n")
        elif(len(user_df['movieId']) == 1):
            train.write(userId.__str__() + " " + user_df['movieId'].iloc[0])
            train.write("\n")
        done += 1
        perc_now = int((done / length) * 100)
        if (perc_now > perc_done):
            print("Done:", perc_now)
            perc_done = perc_now


