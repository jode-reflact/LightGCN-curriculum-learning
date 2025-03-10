import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'ml-25m', 'ml-25m_small', 'ml-25m_sorted', 'ml-25m_sorted_small', 'ml-1m', 'ml-1m_sorted', 'ml-latest-small', 'ml-latest-small_sorted_rating_std', 'ml-latest-small_sorted_rating_only','ml-latest-small_sorted_rating_std_reversed', 'ml-latest-small_sorted_rating_only_reversed','ml-latest-small_sorted_rating_count', 'ml-1m_sorted_rating_std', 'ml-1m_sorted_random']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}