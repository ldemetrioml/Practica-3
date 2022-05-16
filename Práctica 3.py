import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2

def show_histograma(imagen, titulo):

    color = ('b','g','r')

    for channel,col in enumerate(color):
        histr = cv2.calcHist([imagen],[channel],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title(titulo)

def main():
    # Carga de imagenes
    imagen1 = cv2.imread('gaby1.0.jpg')
    imagen2 = cv2.imread('cielo.jpg')

    img1_np = np.array(imagen1)
    img2_np = np.array(imagen2)

    # Operaciones
    ancho = imagen2.shape[1]
    alto = imagen2.shape[0]

    traslacion = np.float32([[1,0,100],[0,1,150]])
    rotacion = cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)

    pts1 = np.float32([[100,400],[400,100],[100,100]])
    pts2 = np.float32([[50,300],[400,200],[80,150]])

    tras_fin = cv2.getAffineTransform(pts1,pts2)

    muestras = {
        'suma':             img1_np + img2_np,
        'resta':            img1_np - img2_np,
        'multiplicacion':   img1_np * img2_np,
        'division':         img1_np % img2_np,
        'potencia':         img1_np ** img2_np,
       
        'conjuncion':       img1_np | img2_np,
        'disyuncion':       img1_np & img2_np,
        'negativo':         255 - img1_np,
        'traslacion':       cv2.warpAffine(imagen2, traslacion, (ancho,alto)),
        'rotacion':         cv2.warpAffine(imagen1, rotacion, (ancho,alto)),
        'escalado':         cv2.resize(imagen2,(600,300), interpolation=cv2.INTER_CUBIC),
        'traslacion a fin': cv2.warpAffine(imagen1, tras_fin, (ancho,alto))
    }

    # Ecualizacion
    img_to_yuv = cv2.cvtColor(imagen1,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    imagen1_equ = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

    img_to_yuv = cv2.cvtColor(imagen2,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    imagen2_equ = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

    imagen_ecu = {}
    
    for muestra in muestras:
        img_to_yuv = cv2.cvtColor(muestras[muestra],cv2.COLOR_BGR2YUV)
        img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
        imagen_ecu[muestra] = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

    # Creacion de figura
    for muestra in muestras:
        fig = plt.figure(figsize=(13, 13))
        columns = 3
        rows = 4
    
        fig.add_subplot(rows, columns, 1)
        plt.imshow(imagen1)
        plt.title('Gaby')

        fig.add_subplot(rows, columns, 2)
        plt.imshow(muestras[muestra])
        plt.title(muestra)

        fig.add_subplot(rows, columns, 3)
        plt.imshow(imagen2)
        plt.title('Cielo')

        
        fig.add_subplot(rows, columns, 4)
        show_histograma(imagen1, 'Histrograma Gaby')

        fig.add_subplot(rows, columns, 5)
        show_histograma(muestras[muestra], 'Histograma '+ muestra)
        
        fig.add_subplot(rows, columns, 6)
        show_histograma(imagen2, 'Histograma Cielo')
        
        
        fig.add_subplot(rows, columns, 7)
        plt.imshow(imagen1_equ)
        plt.title('Cielo ecualizado')

        fig.add_subplot(rows, columns, 8)
        plt.imshow(imagen_ecu[muestra])
        plt.title(muestra +' ecualizado')

        fig.add_subplot(rows, columns, 9)
        plt.imshow(imagen2_equ)
        plt.title('Cielo ecualizado')
        
        
        fig.add_subplot(rows, columns, 10)
        show_histograma(imagen1_equ, 'Histrograma Gaby ecualizado')

        fig.add_subplot(rows, columns, 11)
        show_histograma(imagen_ecu[muestra], 'Histograma '+ muestra +' ecualizado')

        fig.add_subplot(rows, columns, 12)
        show_histograma(imagen2_equ, 'Histograma cielo ecualizado')

        plt.show()

if __name__ == "__main__":
    main()
        
