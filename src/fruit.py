
from skimage import io, measure
import numpy as np


class Fruit:
    def __init__(self, path, debug=False):
        self.path = path
        self.hu_moments = []
        self.moment_ratio = 0
        self.known_label = None
        self.guessed_label = None
        self.threshold = 0.9
        
        self.calculate_features(debug)
        
        
    def calculate_features(self, debug=False):
        fruit_image = io.imread(self.path, as_gray=True)
        
        thresholded_fruit = fruit_image < self.threshold
        del fruit_image
        
        io.imshow(thresholded_fruit)
        
        fruit_central_moments = measure.moments_central(thresholded_fruit)
        
        hu_moments = measure.moments_hu(measure.moments_normalized(fruit_central_moments))
        # We only keep relevant hu moments, that is, components 1 and 3
        self.hu_moments = hu_moments[[1, 3]]
        self.hu_moments[:] = [-1*np.sign(j)*np.log10(np.abs(j)) for j in self.hu_moments[:]]
        
        fruit_eigvalues = measure.inertia_tensor_eigvals(thresholded_fruit, mu=fruit_central_moments)
        self.moment_ratio = max(fruit_eigvalues)/min(fruit_eigvalues)

        if debug == True:
            print("Central moments: \n", fruit_central_moments)
            print("Hu moments:\n", hu_moments)
            print("Inertia tensor eigenvalues:\n", fruit_eigvalues)


def main():
    
    # Path of a random image fruit
    path = '/home/gino/Desktop/fruits-360_dataset/fruits-360/Training/Banana Lady Finger/0_100.jpg'
    
    banana = Fruit(path)
    
    print(banana.hu_moments)
    print(banana.moment_ratio)
    
    # For matplotlib to be able to display the image install tk
    # In arch-like distros just do: sudo pacman -S tk
    io.imshow(path)
    io.show()
    
    
if __name__ == '__main__':
    main()
