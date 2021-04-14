import cv2
import matplotlib.pyplot as plt
import numpy as np
import math



# get the image magnitude_spectrum 
def get_img_freq(img):

    # Covert the image in to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Compute the 2-dimensional discrete Fast Fourier Transform
    freq = np.fft.fft2(gray)

    # Shift the zero-frequency component to the center of the spectrum
    freq_shift = np.fft.fftshift(freq)
    #print(freq_shift)

    # Take absolute value and log to make pixels value to the log scale
    magnitude_spectrum = np.log(np.abs(freq_shift))

    return magnitude_spectrum


# input the image and the filter it want to use and do the fft and multiply with filter function then inverse fft and return the real part
def freq_filtering(img, img_filter):
    freq= np.fft.fft2(img)
    shifted_freq = np.fft.fftshift(freq)
    filtered_freq = shifted_freq * img_filter
    filtered_spectrum = np.log(np.abs(filtered_freq) + 1) # add 1 to avoid "divided by zero" exception
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_freq)).real

    return filtered_image, filtered_spectrum

def Ideal_filter(img, cut_off_ratio,low_or_high=True):


    height, width, channel= img.shape


    ideal_filter=np.zeros((height,width))
    shape = min(height,width)
    cut_off_freq = math.ceil(shape / 2 * cut_off_ratio)
    
    
    u_offset = (height-1) / 2
    v_offset = (width-1) / 2

    
    for u in range(height):
        shifted_u = u - u_offset

        for v in range(width):
            shifted_v = v - v_offset
            D = (shifted_u**2 + shifted_v **2)**(0.5)

            # according to the filtering function H
            if(D <= cut_off_freq):
                ideal_filter[u,v]=1
    if not low_or_high:
        ideal_filter= 1- ideal_filter

     # To hold some overflow value !!!!!!!! check theck the datatype
    filtered_image = np.zeros_like(img, dtype='float64')
    filtered_spectrum = np.zeros_like(img, dtype='float64')


    for c in range(channel):
         filtered_image[:,:,c], filtered_spectrum[:,:,c] = freq_filtering(img[:,:,c], ideal_filter)

    filtered_image = np.where(filtered_image>255, 255, filtered_image)
    filtered_image = np.where(filtered_image<0, 0, filtered_image)

    return filtered_image, filtered_spectrum



def Gaussian_filter(img, cut_off_ratio,low_or_high=True):


    height, width, channel= img.shape


    gaussian_filter=np.zeros((height,width))
    shape = min(height,width)
    cut_off_freq = math.ceil(shape / 2 * cut_off_ratio)

    u_offset = (height-1) / 2
    v_offset = (width-1) / 2


    for u in range(height):
        shifted_u = u - u_offset

        for v in range(width):
            shifted_v = v - v_offset
            D = (shifted_u**2 + shifted_v **2)**(0.5)

            # according to the Gaussian filtering function H
            gaussian_filter[u,v] = math.e**(-(D)**2 / (2*cut_off_freq**2))
    if not low_or_high:
        gaussian_filter= 1- gaussian_filter

    # To hold some overflow value !!!!!!!! check theck the datatype
    filtered_image = np.zeros_like(img, dtype='float64')
    filtered_spectrum = np.zeros_like(img, dtype='float64')


    for c in range(channel):
         filtered_image[:,:,c], filtered_spectrum[:,:,c] = freq_filtering(img[:,:,c], gaussian_filter)

    filtered_image = np.where(filtered_image>255, 255, filtered_image)
    filtered_image = np.where(filtered_image<0, 0, filtered_image)

    return filtered_image, filtered_spectrum





def Hybrid_image(img1, img2, cut_off_ratio, method):

    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])
    hybrid_img = np.zeros((min_height, min_width), dtype='float64')


    low_pass_img, low_pass_spectrum = method(img1, cut_off_ratio, True)

    high_pass_img, high_pass_spectrum = method(img2, cut_off_ratio, False)
    hybrid_img = low_pass_img[0:min_height, 0:min_width] + high_pass_img[0:min_height, 0:min_width]

    hybrid_img = np.where(hybrid_img>255, 255, hybrid_img)
    hybird_img = np.where(hybrid_img<0, 0, hybrid_img)
    return hybrid_img
