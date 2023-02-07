import pandas as pd
import numpy as np
import os
from sklearn import model_selection, metrics, preprocessing
from sklearn.model_selection import train_test_split
from torch_geometric.data import download_url, extract_zip
import argparse
from typing import Literal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description="Preprocessing")
parser.add_argument('--sorted', action='store_true', default=False, help='whether to preprocess sorted dataset')
parser.add_argument('--threshold', type=float, default=3.5, help='rating threshold for positive entries')
parser.add_argument('--dataset', type=str, default='ml-25m', help='which dataset to preprocess ["ml-25m", "ml-1m", "ml-latest-small"], default = "ml-25m"')

args = parser.parse_args()

SORTED = args.sorted
DATASET: Literal['ml-25m', 'ml-1m', 'ml-latest-small'] = args.dataset
RATING_THRESHOLD = args.threshold #default = 3.5

url_ml25 = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
url_ml1 = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
url_ml_latest_small = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

extract_zip(download_url(url_ml25, '../rawdata/'), '../rawdata/')
extract_zip(download_url(url_ml1, '../rawdata/'), '../rawdata/')
extract_zip(download_url(url_ml_latest_small, '../rawdata/'), '../rawdata/')

if DATASET == 'ml-25m':
    rating_path = '../rawdata/ml-25m/ratings.csv'
    rating_df = pd.read_csv(rating_path)
elif DATASET == 'ml-latest-small':
    rating_path = '../rawdata/ml-latest-small/ratings.csv'
    rating_df = pd.read_csv(rating_path)
elif DATASET == 'ml-1m':
    rating_path = '../rawdata/ml-1m/ratings.dat'

    rating_df = pd.read_csv(
        rating_path, 
        sep = '::', 
        header = None, 
        engine = 'python', 
        encoding = 'latin-1',
        names=["userId","movieId","rating","timestamp"]
    )
else:
    raise ValueError('Dataset error: ' + DATASET)

rating_df['realUserId'] = rating_df['userId']

print("Using Dataset:", DATASET)

sortedDir = os.sep.join(['../data', DATASET + '_sorted'])
unsortedDir = os.sep.join(['../data', DATASET])
os.makedirs(sortedDir, exist_ok=True)
os.makedirs(unsortedDir, exist_ok=True)

# size calculation
count_negatives = len(rating_df[rating_df['rating'] <= RATING_THRESHOLD])

count_positives = len(rating_df[rating_df['rating'] > RATING_THRESHOLD])

print("threshold", RATING_THRESHOLD)
print("Positive", count_positives / len(rating_df))
print("Negative", count_negatives / len(rating_df))

if SORTED:
    if os.path.exists('../rawdata/'+DATASET+'_ratings_sorted.csv'):
        rating_df = pd.read_csv('../rawdata/'+DATASET+'_ratings_sorted.csv', index_col=0)
        rating_df.sort_values(by='userId', ascending=True, inplace=True)
    else:
        sort_df = pd.read_csv('../rawdata/'+DATASET+'_user_sorted_rating_std.csv')
        rating_df['sort_index'] = rating_df['userId'].apply(lambda userId: sort_df.userId.eq(userId).idxmax())
        rating_df.drop(columns=['userId'], inplace=True)
        rating_df.rename(columns={"sort_index": "userId"}, inplace=True)
        rating_df.sort_values(by='userId', ascending=True, inplace=True)
        rating_df.to_csv('../rawdata/'+DATASET+'_ratings_sorted.csv')

    pathTrain = os.sep.join(['../data', DATASET + '_sorted', 'train.txt'])
    pathTest = os.sep.join(['../data', DATASET + '_sorted', 'test.txt'])
else:
    lbl_user = preprocessing.LabelEncoder()
    rating_df.userId = lbl_user.fit_transform(rating_df.userId.values)
    pathTrain = os.sep.join(['../data', DATASET, 'train.txt'])
    pathTest = os.sep.join(['../data', DATASET, 'test.txt'])

lbl_movie = preprocessing.LabelEncoder()
rating_df.movieId = lbl_movie.fit_transform(rating_df.movieId.values)

done = 0
perc_done = 0
length = len(rating_df['userId'].unique())

trainTestSplitPath = os.sep.join(['../rawdata', DATASET + '_traintestsplit.csv'])
existingTrainTestSplit = os.path.exists(trainTestSplitPath)

if existingTrainTestSplit:
    train_test_df = pd.read_csv(trainTestSplitPath)
else:
    train_test_df = pd.DataFrame(columns=['realUserId', 'trainLine', 'testLine'])

with open(pathTrain, 'w') as train, open(pathTest, 'w') as test:
    for userId in rating_df['userId'].unique():
        user_df = rating_df[(rating_df['userId'] == userId) & (rating_df['rating'] > RATING_THRESHOLD)]
        user_df['movieId'] = user_df['movieId'].astype(str)
        if (len(user_df['movieId']) > 1):
            realUserId = user_df['realUserId'].iloc[0]
            entry_df = train_test_df.loc[train_test_df['realUserId'] == realUserId]
            if not entry_df.empty:
                entry = entry_df.iloc[0]
                trainLine = entry['trainLine']
                testLine = entry['testLine']
            else:
                trainData, testData = train_test_split(user_df['movieId'], train_size=0.8, shuffle=True)
                trainLine = ' '.join(trainData)
                testLine = ' '.join(testData)
                new_entry = pd.Series({'realUserId': realUserId, 'trainLine': trainLine, 'testLine': testLine})
                train_test_df = pd.concat([train_test_df, new_entry.to_frame().T], ignore_index=True)
            trainUserLine = userId.__str__() + " " + trainLine
            testUserLine = userId.__str__() + " " + testLine
            
            train.write(trainUserLine)
            test.write(testUserLine)

            train.write("\n")
            test.write("\n")
        elif(len(user_df['movieId']) == 1):
            train.write(userId.__str__() + " " + user_df['movieId'].iloc[0])
            train.write("\n")
        done += 1
        perc_now = int((done / length) * 100)
        if (perc_now > perc_done):
            print("Done:", perc_now)
            train_test_df.to_csv(trainTestSplitPath, index=False)
            perc_done = perc_now


