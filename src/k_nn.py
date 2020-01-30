# Implementation of k-nearest neighbours algorithm
# https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

# Path to Fruits 360 dataset
path = '/home/gino/Desktop/fruits-360_dataset/'


from fruit import Fruit

from skimage import io
from operator import itemgetter
import numpy as np



def set_label(nearest_neighbours, test_fruit):
# Receives a list of tuples containing the distances from the test_fruit
# to the n nearest neighbours and their label
    banana_vote = 0
    orange_vote = 0
    lemon_vote = 0
    
    for fruit in nearest_neighbours:
        if fruit[1] == 'banana':
            banana_vote += 1
        elif fruit[1] == 'orange':
            orange_vote += 1
        elif fruit[1] == 'lemon':
            lemon_vote += 1
        else:
            print("Error\n:")
            continue
        
    if banana_vote >= orange_vote and banana_vote >= lemon_vote:
        test_fruit.guessed_label = 'banana'
    elif orange_vote >= banana_vote and orange_vote >= lemon_vote:
        test_fruit.guessed_label = 'orange'
    elif lemon_vote >= banana_vote and lemon_vote >= orange_vote:
        test_fruit.guessed_label = 'lemon'
            
    return


def k_nn(neighbours, fruit_list, test_list):

    for test_fruit in test_list:

        distances = [np.sum(np.power(fruit.features - test_fruit.features, 2)) 
                     for fruit in fruit_list]
        labels = [fruit.known_label for fruit in fruit_list]

        train_dist_label = list(zip(distances, labels))
        # https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
        train_dist_label = sorted(train_dist_label, key=itemgetter(0))

        set_label(train_dist_label[0:neighbours], test_fruit)

    return


def main(feature_mode='hu_only', verbose=False):

    ##############################
    ########## Training ##########
    ##############################

    # We're not technically training, just locating fruits in feature space
    fruit_list = []

    banana_collection = io.ImageCollection([path 
                                            + 'fruits-360/Training/Banana/*.jpg', 
                                            path + 'fruits-360/Training/Banana Lady Finger/*.jpg'],
                                            load_func=Fruit.img_grayscale)
    orange_collection = io.ImageCollection(path 
                                           + 'fruits-360/Training/Orange/*.jpg',
                                           load_func=Fruit.img_grayscale)
    lemon_collection = io.ImageCollection(path 
                                          + 'fruits-360/Training/Lemon/*.jpg', 
                                          load_func=Fruit.img_grayscale)

    fruit_list = [Fruit(banana_collection.files[i], 'banana', feature_mode=feature_mode) 
                  for i in range(len(banana_collection))]
    fruit_list.extend([Fruit(orange_collection.files[i], 'orange', feature_mode=feature_mode) 
                       for i in range(len(orange_collection))])
    fruit_list.extend([Fruit(lemon_collection.files[i], 'lemon', feature_mode=feature_mode) 
                       for i in range(len(lemon_collection))])

    banana_size = len(banana_collection)
    orange_size = len(orange_collection)
    lemon_size = len(lemon_collection)

    ##########################
    ########## Test ##########
    ##########################

    banana_collection = io.ImageCollection([path 
                                            + 'fruits-360/Test/Banana/*.jpg', 
                                            path + 'fruits-360/Test/Banana Lady Finger/*.jpg'],
                                            load_func=Fruit.img_grayscale)
    orange_collection = io.ImageCollection(path 
                                           + 'fruits-360/Test/Orange/*.jpg',
                                           load_func=Fruit.img_grayscale)
    lemon_collection = io.ImageCollection(path 
                                          + 'fruits-360/Test/Lemon/*.jpg', 
                                          load_func=Fruit.img_grayscale)

    test_list = [Fruit(banana_collection.files[i], 'banana', feature_mode=feature_mode) 
                  for i in range(len(banana_collection))]
    test_list.extend([Fruit(orange_collection.files[i], 'orange', feature_mode=feature_mode) 
                       for i in range(len(orange_collection))])
    test_list.extend([Fruit(lemon_collection.files[i], 'lemon', feature_mode=feature_mode) 
                       for i in range(len(lemon_collection))])

    ##########################
    ########## k-nn ##########
    ##########################

    neighbours = 2
    k_nn(neighbours, fruit_list, test_list)

    if verbose:
        print(list(zip([fruit.known_label for fruit in test_list], 
        [fruit.guessed_label for fruit in test_list])))
        for fruit in test_list:
            if fruit.guessed_label != fruit.known_label:
                print("Error found!")

    return


if __name__ == '__main__':

    main(feature_mode='hu_only', verbose=True)


#
