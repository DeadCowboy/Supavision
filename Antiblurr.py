import numpy as np
from PIL import Image
from math import dist
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

def set_min_value(image: np.array, min_value=1):
    # image = np.copy(orig_image)
    for x,y in np.ndindex(image.shape):
        if np.abs(image[x,y]) < min_freq_amp_kernel:
            image[x,y] = min_freq_amp_kernel * np.sign(image[x,y])
    return image

def set_min_value_np(image: np.array, min_value=1):
    np.putmask(image, np.abs(image) < min_value, min_value * np.sign(image))
    return image

def reblurr(image, pad=0):
    return np.abs(fftconvolve(np.abs(image), kernel_circle, mode='same'))

def image_MSE(image1, image2):
    return np.sum((image1 - image2) ** 2)

image = np.array(Image.open(r'cat.png').convert('L'))
# Circle of confusion
radius = int(image.shape[0] * 0.05)

kernel = np.zeros(image.shape)
kernel_middle = ((kernel.shape[0]-1) // 2, (kernel.shape[1]-1) // 2)
for x in range(kernel_middle[0]-radius,kernel_middle[0]+radius):
    for y in range(kernel_middle[1]-radius,kernel_middle[1]+radius):
        if dist((x,y),kernel_middle) < radius:
            kernel[x,y] = 1

# Blurr pixel edges
blurr_size = 55
# kernel = cv.GaussianBlur(kernel,(blurr_size,blurr_size), 30)
kernel /= np.sum(kernel)
kernel_circle = kernel[kernel_middle[0] - radius: kernel_middle[0] + radius, kernel_middle[1] - radius: kernel_middle[1] + radius]



transform_kernel = np.fft.fft2(kernel)
transform_image = np.fft.fft2(image)

min_freq_amp_kernel = 0.5

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



