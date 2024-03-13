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

        self.image_fft = np.fft.fft2(self.image)
        self.kernel_fft = np.fft.fft2(self.kernel)


    def image_MSE(image1, image2):
        return np.sum((image1 - image2) ** 2)


    def set_min_value_np(self,image: np.array, min_value=1):
        np.putmask(image, np.abs(image) < min_value, min_value * np.sign(image))
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


    def blurr(self, image, pad=0):
        return np.abs(fftconvolve(np.abs(image), self.kernel_circle, mode="same"))


min_freq_amp_kernel = 0.9

# transform_kernel[transform_kernel < min_freq_amp_kernel] = min_freq_amp_kernel

min_freqs = [min_freq_amp_kernel * p for p in range(1,5)]
product_transform = transform_image / set_min_value_np(transform_kernel, min_value=0.9) # 0.9 is optimal by empirical experiment
# my_range = np.arange(0.1,10,0.1)
# product_transform = [transform_image / set_min_value_np(transform_kernel, min_value=min_val) for min_val in my_range]
# pyramid_images = [transform_image / set_min_value_np(transform_kernel, min_freq)  for min_freq in min_freqs]
# product_transform = np.sum(pyramid_images, axis=0)

# result = [np.fft.ifft2(product_transform) for product_transform in product_transform]
# result = [np.fft.fftshift(result) for result in result]

result = np.fft.ifft2(product_transform)
result = np.fft.fftshift(result)
# reblurred_image = reblurr(result)

# err = [image_MSE(image, reblurr(result)) for result in result]

# plt.plot(my_range, err)
# plt.show()

plt.imshow(reblurr(result), cmap='gray')
plt.show()
plt.imshow(np.abs(result), cmap='gray')
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



