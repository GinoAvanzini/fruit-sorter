
from skimage import io, measure, filters
import numpy as np


class Fruit:
    def __init__(self, path, label=None, debug=False, feature_mode='hu_plus_ratio'):
        self.path = path
        self.hu_moments = []
        self.moment_ratio = 0
        self.known_label = label
        self.guessed_label = None
        self.threshold = None
        self.feature_mode = feature_mode
        
        self.calculate_features(self.feature_mode, debug)
        
        
    def calculate_features(self, feature_mode, debug=False):

        fruit_image = io.imread(self.path, as_gray=True)
        
        sigma = 0.005*fruit_image.shape[0]
        filtered_fruit = filters.gaussian(fruit_image, sigma=sigma)

        # Apply triangle threshold to gaussian filtered image
        self.threshold = filters.threshold_triangle(filtered_fruit)
        thresholded_fruit = filtered_fruit < self.threshold
        
        fruit_central_moments = measure.moments_central(thresholded_fruit)
        hu_moments = measure.moments_hu(measure.moments_normalized(fruit_central_moments))
        # We only keep relevant hu moments, that is, components 1 and 3
        self.hu_moments = hu_moments[[1, 3]]
        
        # And we apply a log transform to them
        self.hu_moments = np.array([-1*np.sign(j)*np.log10(np.abs(j)) for j in self.hu_moments[:]])
        
        fruit_eigvalues = measure.inertia_tensor_eigvals(thresholded_fruit, mu=fruit_central_moments)
        self.moment_ratio = max(fruit_eigvalues)/min(fruit_eigvalues)
        
        if feature_mode == 'hu_plus_ratio':
            self.features = np.append(self.hu_moments, self.moment_ratio)
            self.feature_size = 3
        elif feature_mode == 'hu_only':
            self.features = np.array(self.hu_moments)
            self.feature_size = 2
            
        if debug:
            print("threshold: \n", self.threshold)
            print("Central moments: \n", fruit_central_moments)
            print("Hu moments:\n", hu_moments)
            print("Normalized Hu moments:\n",
                  [-1*np.sign(j)*np.log10(np.abs(j)) for j in hu_moments[:]])
            print("Inertia tensor eigenvalues:\n", fruit_eigvalues)
            print("Moment ratio:\n", self.moment_ratio)
            print("Features: \n", self.features)


    @staticmethod
    def img_grayscale(image):
        return io.imread(image, as_gray=True)


def main():
    
    # Path of a random image fruit
    path = '/home/gino/Desktop/fruits-360_dataset/fruits-360/Training/Banana Lady Finger/0_100.jpg'
    
    banana = Fruit(path, debug=True, feature_mode='hu_only')
    
    # For matplotlib to be able to display the image install tk
    # In arch-like distros just do: sudo pacman -S tk
    io.imshow(path)
    io.show()
    
    
if __name__ == '__main__':
    main()


#
