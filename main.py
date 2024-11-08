import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Đọc ảnh và chuyển sang ảnh mức xám
image = cv2.imread("anh_vt.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Toán tử Sobel
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# 2. Toán tử Prewitt
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
prewitt_x = cv2.filter2D(gray_image, cv2.CV_32F, kernelx)
prewitt_y = cv2.filter2D(gray_image, cv2.CV_32F, kernely)
prewitt_combined = cv2.magnitude(prewitt_x, prewitt_y)

# 3. Toán tử Robert
roberts_x_kernel = np.array([[1, 0], [0, -1]], dtype=int)
roberts_y_kernel = np.array([[0, 1], [-1, 0]], dtype=int)
roberts_x = cv2.filter2D(gray_image, cv2.CV_32F, roberts_x_kernel)
roberts_y = cv2.filter2D(gray_image, cv2.CV_32F, roberts_y_kernel)
roberts_combined = cv2.magnitude(roberts_x, roberts_y)

# 4. Toán tử Canny
canny_edges = cv2.Canny(gray_image, 100, 200)

# 5. Lọc Gaussian
gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Danh sách ảnh và tiêu đề
images = [gray_image, sobel_combined, prewitt_combined, roberts_combined, canny_edges, gaussian_blur]
titles = ["Ảnh gốc", "Sobel", "Prewitt", "Robert", "Canny", "Gaussian Blur"]
index = 0

# Hàm cập nhật ảnh
def update_image():
    img_display.set_data(images[index])
    plt.title(titles[index])
    plt.draw()

# Hàm xử lý khi nhấn nút Next
def next_image(event):
    global index
    index = (index + 1) % len(images)
    update_image()

# Hàm xử lý khi nhấn nút Previous
def prev_image(event):
    global index
    index = (index - 1) % len(images)
    update_image()

# Thiết lập figure và các nút
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
img_display = ax.imshow(images[index], cmap='gray')
plt.title(titles[index])
plt.axis('off')

# Tạo nút Previous
axprev = plt.axes([0.2, 0.05, 0.1, 0.075])
btn_prev = Button(axprev, 'Previous')
btn_prev.on_clicked(prev_image)

# Tạo nút Next
axnext = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_next = Button(axnext, 'Next')
btn_next.on_clicked(next_image)

plt.show()
