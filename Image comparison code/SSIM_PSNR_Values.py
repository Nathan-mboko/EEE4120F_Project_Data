import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Load the images
image1 = cv2.imread('4.jpg')
image2 = cv2.imread('3.jpg')

# Resize the images to the same dimensions (optional)
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Convert the images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calculate SSIM
ssim_index = ssim(gray_image1, gray_image2)

# Calculate PSNR
psnr_value = psnr(gray_image1, gray_image2)

# Display the results
print(f"SSIM: {ssim_index}")
print(f"PSNR: {psnr_value}")
