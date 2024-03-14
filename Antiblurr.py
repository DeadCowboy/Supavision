import numpy as np
from PIL import Image
from math import dist
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image

class Antiblurr():
    def __init__(self, filename: str, cof_radius: float = None, min_val = None) -> None:
        # Grayscale working image
        self.image = np.array(Image.open(filename).convert('L'))

        # Circle of confusion
        self.cof_radius = cof_radius if cof_radius else int(self.image.shape[0] * 0.05)

        #Create kernel array from image
        self.kernel, self.kernel_circle = self.create_kernel()

        self.min_val = min_val if min_val else 0.9
        self.kernel_fft = None
        self.image_fft = None
        self.antiblurr_image = None


    def image_SSD(image1, image2) -> float:
        """Compute the sum of elementwise squared differences between arrays"""
        return np.sum((image1 - image2) ** 2)


    def clamp_image_values(self,image):
        """Clamp all values with lower magnitude than min_val to min_val, conserving the sign.
        
        Every value in the array that is lower in magnitude than min_val is set to min_val with the same sign as the original value."""
        np.putmask(image, np.abs(image) < self.min_val, self.min_val * np.sign(image))
        return image

    def create_kernel(self):
        """Produce a numpy array with circle in the middle.
        
        Generate a numpy array of zeros with the shape of self.image and fills a circle in the middle with
        equal values, such that their sum is equal to 1. Also returns a numpy array with the bounding box of the nonzero circle"""
        kernel = np.zeros(self.image.shape)
        kernel_middle = ((kernel.shape[0]-1) // 2, (kernel.shape[1]-1) // 2)
        for x in range(kernel_middle[0]-self.cof_radius,kernel_middle[0]+self.cof_radius):
            for y in range(kernel_middle[1]-self.cof_radius,kernel_middle[1]+self.cof_radius):
                if dist((x,y),kernel_middle) < self.cof_radius:
                    kernel[x,y] = 1
        kernel /= np.sum(kernel)
        kernel_circle = kernel[kernel_middle[0] - self.cof_radius: kernel_middle[0] + self.cof_radius, kernel_middle[1] - self.cof_radius: kernel_middle[1] + self.cof_radius]
        return kernel, kernel_circle


    def blurr(self, image, pad: int = 0):
        return np.abs(fftconvolve(np.abs(image), self.kernel_circle, mode="same"))
    
    def antiblurr(self, image = None):
        """Antiblurr the image.
        
        Antiblurrs the image of the class, except if another image was parsed to be antiblurred instead."""
        target_image = image if image else self.image
        self.image_fft = np.fft.fft2(target_image)
        self.kernel_fft = np.fft.fft2(self.kernel)
        clamped_kernel_fft = self.clamp_image_values(self.kernel_fft)
        antiblurr_image_fft = self.image_fft / clamped_kernel_fft
        antiblurr_image = np.fft.ifft2(antiblurr_image_fft)
        # Take magnitude of complex number
        antiblurr_image = np.abs(antiblurr_image)
        antiblurr_image = np.fft.fftshift(antiblurr_image)
        self.antiblurr_image = antiblurr_image
        return antiblurr_image
    
    def show(self, image = None, norm_func = None):
        """Display the antiblurred picture in grayscale.
        
        Shows the antiblurred picture in grayscale. If it does not exist, antiblurr() is called.
        Also a custom image can be parsed, which is then displayed instead if provided.

        Parse a normalization function via 'norm_func'. This function is called on the image before 
        displaying."""
        target_image = None
        if image :
            target_image = image
        elif self.antiblurr_image:
            target_image = self.antiblurr_image
        else:
            target_image = self.antiblurr()
        
        if norm_func is None:
            plt.imshow(target_image, cmap="gray")
        else:
            plt.imshow(norm_func(target_image), cmap="gray")


        plt.show()



