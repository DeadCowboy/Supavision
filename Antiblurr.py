import numpy as np
from PIL import Image
from math import dist
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image

class Antiblurr():
    def __init__(self, filename: str, cof_radius) -> None:
        # Grayscale working image
        self.image = np.array(Image.open(filename).convert('L'))

        #Create kernel array from image
        self.kernel, self.kernel_circle = self.create_kernel()

        # Circle of confusion
        self.cof_radius = int(self.image.shape[0] * 0.05)

        self.kernel_fft = None
        self.image_fft = None
        self.antiblurr_image = None


    def image_SSD(image1, image2):
        """Compute the sum of elementwise squared differences between arrays"""
        return np.sum((image1 - image2) ** 2)


    def set_min_value(self,image: np.array, min_val=0.9):
        """Clamp all values with lower magnitude than min_val to min_val, conserving the sign.
        
        Every value in the array that is lower in magnitude than min_val is set to min_val with the same sign as the original value."""
        np.putmask(image, np.abs(image) < min_val, min_val * np.sign(image))
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


    def blurr(self, image: np.array, pad=0):
        return np.abs(fftconvolve(np.abs(image), self.kernel_circle, mode="same"))
    
    def antiblurr(self, min_val:float = 0.9, image = None):
        target_image = image if image else self.image
        self.image_fft = np.fft.fft2(target_image)
        self.kernel_fft = np.fft.fft2(self.kernel)
        clamped_kernel_fft = self.set_min_value(self.kernel_fft)
        antiblurr_image_fft = self.image_fft / clamped_kernel_fft
        antiblurr_image = np.fft.ifft2(antiblurr_image_fft)
        # antiblurr_image = np.fft.fftshift(antiblurr_image)
        self.antiblurr_image = antiblurr_image
        return antiblurr_image
    
    def show(self, image = None):
        """Display the antiblurred picture in grayscale.
        
        Shows the antiblurred picture in grayscale. If it does not exist, a custom image can be parsed, which is then
        displayed instead."""
        target_image = None
        if image :
            target_image = image
        elif self.antiblurr_image:
            target_image = self.antiblurr_image
        else:
            raise RuntimeError("You need to run the antiblurr() method or parse an image")
        
        plt.imshow(target_image, cmap="gray")
        plt.show()































# # Take Fourier of Image
# image_fourier = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
# # image_fourier = np.fft.fftshift(image)


# # Take Fourier of Kernel
# kernel_fourier = cv.dft(np.float32(kernel), flags=cv.DFT_COMPLEX_OUTPUT)
# # kernel_fourier = np.fft.fftshift(kernel)


# # Divide Fourier of image by Fourier of Kernel for inverse convolution
# filteredImage = np.multiply(image_fourier, kernel_fourier)
# # Inverse FFT of Kernel
# inverse_image = cv.idft(filteredImage)
# inverse_image = cv.magnitude(inverse_image[:,:,0], inverse_image[:,:,1])
# # kernel_img = kernel_img[:,:,0]

# # plt.imshow(kernel, cmap='gray')
# # plt.show()
# # plt.imshow(kernel_fft_magnitude)
# # plt.show()
# plt.imshow(inverse_image)
# plt.show()



