import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import data, img_as_float
from skimage import exposure

def procesar_y_guardar(image_path, carpeta_destino):
    # Leer la image de entrada
    image = cv2.imread(image_path)

    # Ecualización del histograma (YCrCb)
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    canales_yuv = list(cv2.split(image_yuv))  # Convertir la tupla a lista
    canales_yuv[0] = cv2.equalizeHist(canales_yuv[0])
    image_ecualizada_yuv = cv2.merge(canales_yuv)
    image_ecualizada_yuv = cv2.cvtColor(image_ecualizada_yuv, cv2.COLOR_YCrCb2BGR)

    # TRANSFORMACION LINEAL
    # Cambiar la intensidad del color (aumentar el brillo)
    brightened_image = cv2.addWeighted(image, 1, np.zeros_like(image), 0, 40)

    # Ajustar el contraste (usando alpha y beta)
    contrast = cv2.addWeighted(image, 10, np.zeros_like(image), 0, 0)

    # Ajustar el brillo y contraste (usando alpha y beta)
    adjusted_image = cv2.addWeighted(image, 10, np.zeros_like(image), 0, 40)

    # HISTOGRAMA
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    kernel_size = (5, 5)
    sigma = 0
    blurred_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)


    # LOGARÍTMICA
    # Aplicar la transformación logarítmica
    log_transformed_image = np.log1p(image.astype(float))

    # Normalizar la image transformada antes de mostrarla
    log_transformed_image_normalized = (255 * (log_transformed_image - np.min(log_transformed_image)) / (np.max(log_transformed_image) - np.min(log_transformed_image))).astype(np.uint8)

    # GAMMA
    # Corrección gamma 
    gamma = 1.5
    correccion_gamma = np.uint8(np.power(image / 255.0, gamma) * 255)

    # Carpeta de destino en el mismo directorio que el script
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Guardar las imágenes procesadas
    cv2.imwrite(os.path.join(carpeta_destino, "ecualizacion_yuv.jpg"), image_ecualizada_yuv)
    cv2.imwrite(os.path.join(carpeta_destino, "ecualizacion_skimage.jpg"), equalized_image)
    cv2.imwrite(os.path.join(carpeta_destino, "transformacion_logaritmica.jpg"), log_transformed_image_normalized)
    cv2.imwrite(os.path.join(carpeta_destino, "correccion_gamma.jpg"), correccion_gamma)
    cv2.imwrite(os.path.join(carpeta_destino, "correccion_gamma.jpg"), adjusted_image)


def apply_linear_transformation(image_path, alpha, beta, gris):
    # Leer la imagen desde un archivo
    image = cv2.imread(image_path)

    if gris == 'gris':
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
    # Cambiar la intensidad del color (aumentar el brillo)
    brightened_image = cv2.addWeighted(image, 1, np.zeros_like(image), 0, beta)

    # Ajustar el contraste (usando alpha y beta)
    contrast = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, 0)

    # Ajustar el brillo y contraste (usando alpha y beta)
    adjusted_image = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

    # Mostrar las imágenes originales y transformadas
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
    plt.subplot(1, 4, 2), plt.imshow(cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)), plt.title('Contraste Modificado')
    plt.subplot(1, 4, 3), plt.imshow(cv2.cvtColor(brightened_image, cv2.COLOR_BGR2RGB)), plt.title('Brillo Modificado')
    plt.subplot(1, 4, 4), plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)), plt.title('Brillo y Contraste Ajustados')
    plt.show()

    # Carpeta de destino
    carpeta_destino = "Imagenes_Doc/transformacion_lineal/"

    # Verificar si la carpeta existe, si no, crearla
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Guardar la imagen procesada en la carpeta de destino
    cv2.imwrite(os.path.join(carpeta_destino, "transformacion_linear.jpg"), contrast)

def apply_logarithmic_transformation(image_path, gris):
    # Leer la imagen desde un archivo
    image = cv2.imread(image_path)

    if gris == 'gris':
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Aplicar la transformación logarítmica
    log_transformed_image = np.log1p(image.astype(float))

    # Normalizar la imagen transformada antes de mostrarla
    log_transformed_image_normalized = (255 * (log_transformed_image - np.min(log_transformed_image)) / (np.max(log_transformed_image) - np.min(log_transformed_image))).astype(np.uint8)

    # Mostrar las imágenes originales y transformadas
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(log_transformed_image_normalized, cv2.COLOR_BGR2RGB)), plt.title('Transformación Logarítmica')
    plt.show()

def apply_exponential_transformation(image_path, gamma, gris):
    # Leer la imagen desde un archivo
    image = cv2.imread(image_path)

    if gris == 'gris':
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Aplicar la transformación exponencial
    exponential_transformed_image = np.power(image, gamma)

    # Normalizar la imagen transformada antes de mostrarla
    exponential_transformed_image_normalized = (255 * (exponential_transformed_image - np.min(exponential_transformed_image)) / (np.max(exponential_transformed_image) - np.min(exponential_transformed_image))).astype(np.uint8)

    # Mostrar las imágenes originales y transformadas
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(exponential_transformed_image_normalized, cv2.COLOR_BGR2RGB)), plt.title('Transformación Exponencial (gamma=' + str(gamma) + ')')
    plt.show()

def apply_arithmetic_operations(operation, image_path1, image_path2, gris):
    # Leer las imágenes desde archivos
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Ajustar el tamaño de las imágenes para que sean del mismo tamaño
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    # Convert to grayscale
    gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if gris == 'gris':
        if len(image1.shape) == 3:

            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur
    kernel_size = (5, 5)
    sigma = 0
    blurred_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)

    # Aplicar operadores aritméticos
    # Suma de imágenes
    if operation == 'suma':
        for i in range(3):
            added_image = cv2.add(image1, image2)

        # Mostrar las imágenes originales y transformadas
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
        plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)), plt.title('Imagen con filtro Gaussiano')
        plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB)), plt.title('Suma de Imágenes')
        plt.show()

    # Resta de imágenes
    if operation == 'resta':
        subtracted_image = cv2.subtract(image1, image2)
        
        # Mostrar las imágenes originales y transformadas
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
        plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)), plt.title('Imagen con filtro Gaussiano')
        plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(subtracted_image, cv2.COLOR_BGR2RGB)), plt.title('Resta de Imágenes')
        plt.show()

    if operation == 'mult':
        for i in range(3):
            added_image = cv2.multiply(image1, image2)

        # Mostrar las imágenes originales y transformadas
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
        plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)), plt.title('Imagen con filtro Gaussiano')
        plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB)), plt.title('Multiplicación de Imágenes')
        plt.show()

    if operation == 'div':
        for i in range(3):
            added_image = cv2.divide(image1, image2)

        # Mostrar las imágenes originales y transformadas
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
        plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)), plt.title('Imagen con filtro Gaussiano')
        plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB)), plt.title('División de Imágenes')
        plt.show()

def apply_gaussian_and_histogram_equalization(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    kernel_size = (5, 5)
    sigma = 0
    blurred_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)

    # Display the images
    plt.figure(figsize=(18, 6))
    plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(blurred_image, cmap='gray'), plt.title('Blurred Image')
    plt.subplot(133), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
    plt.show()


def apply_gaussian_and_histogram_equalization(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    kernel_size = (5, 5)
    sigma = 0
    blurred_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)

    # Calculate histogram of the original image
    histogram_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    histogram_blurred = cv2.calcHist([blurred_image], [0], None, [256], [0, 256])
    histogram_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])


    # Display the images and histogram
    plt.figure(figsize=(18, 9))

    plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(232), plt.imshow(blurred_image, cmap='gray'), plt.title('Blurred Image')
    plt.subplot(233), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')

    plt.subplot(234), plt.plot(histogram_original), plt.title('Histogram of Original Image')
    plt.subplot(235), plt.plot(histogram_blurred), plt.title('Histogram of Blurred Image')
    plt.subplot(236), plt.plot(histogram_equalized), plt.title('Histogram of equalized_image Image')


    plt.show()


def apply_mean_filter(image_path, kernel_size=(5, 5)):
    # Leer la imagen desde un archivo
    image = cv2.imread(image_path)

    # Aplicar el filtro de la media para reducir el ruido
    mean_filtered_image = cv2.blur(image, kernel_size)

    # Mostrar las imágenes originales y transformadas
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(mean_filtered_image, cv2.COLOR_BGR2RGB)), plt.title('Filtro de la Media')
    plt.show()

def apply_median_filter(image_path, kernel_size=5):
    # Leer la imagen desde un archivo
    image = cv2.imread(image_path)

    # Aplicar el filtro de la mediana para reducir el ruido
    median_filtered_image = cv2.medianBlur(image, kernel_size)

    # Mostrar las imágenes originales y transformadas
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB)), plt.title('Filtro de la Mediana')
    plt.show()

def apply_bilateral_filter(image_path, diameter=9, sigma_color=75, sigma_space=75):
    # Leer la imagen desde un archivo
    image = cv2.imread(image_path)

    # Aplicar el filtro bilateral para reducir el ruido
    bilateral_filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    # Mostrar las imágenes originales y transformadas
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB)), plt.title('Filtro de la Media')
    plt.show()