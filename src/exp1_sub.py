import cv2
import numpy as np
from scipy import fftpack
from tqdm import tqdm
from matplotlib import pyplot as plt

def dct_1D(a):
    N=a.shape[0]
    b=np.zeros(N,dtype=np.float32)
    index=np.arange(0,N,1,dtype=np.float32)
    for i in range(N):
        b[i]=np.sqrt(2/N)*sum(a*np.cos(np.pi*(2*index+1)*i/(2*N)))
    b[0]*=(1/np.sqrt(2))
    return b

def idct_1D(a):
    N=a.shape[0]
    b=np.zeros(N,dtype=np.float32)
    c=np.ones(N,dtype=np.float32)
    c[0]=1/np.sqrt(2)
    index=np.arange(0,N,1,dtype=np.float32)
    for i in range(N):
        b[i]=np.sqrt(2/N)*sum(c*a*np.cos(np.pi*(2*i+1)*index/(2*N)))
    return b

def dct_2D(a):
    N=a.shape[0]
    b=np.zeros(N,dtype=np.float32)
    index=np.arange(0,N,1,dtype=np.float32)
    for i in range(N):
        b[i]=np.sqrt(2/N)*sum(a*np.cos(np.pi*(2*index+1)*i/(2*N)))
    b[0]*=(1/np.sqrt(2))
    return b

def idct_2D(a):
    N=a.shape[0]
    b=np.zeros(N,dtype=np.float32)
    c=np.ones(N,dtype=np.float32)
    c[0]=1/np.sqrt(2)
    index=np.arange(0,N,1,dtype=np.float32)
    for i in range(N):
        b[i]=np.sqrt(2/N)*sum(c*a*np.cos(np.pi*(2*i+1)*index/(2*N)))
    return b

def calculate_PSNR(image_1, image_2):
    print(image_1)
    print(image_2)
    mse = np.mean((image_1 - image_2)**2)
    if mse == 0:
        return 1000
    PIXEL_MAX = 255.0
    return 10 * np.log10(PIXEL_MAX**2 / mse)

def main():
    gray_image = cv2.imread('../data/lena.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('../result/exp1/lena_gray.bmp', gray_image)
    # gray_image = np.float32(gray_image)
    # 1D-DCT
    # dct_1D_image = np.zeros(gray_image.shape, dtype=np.float32)
    # for i in tqdm(range(dct_1D_image.shape[0])):
    #     dct_1D_image[i] = dct_1D(gray_image[i])
    dct_1D_row=fftpack.dct(gray_image, axis=1, norm='ortho')
    cv2.imwrite('../result/exp1/dct_1D_row.bmp', dct_1D_row)
    # for i in tqdm(range(dct_1D_image.shape[1])):
    #     dct_1D_image[:,i] = dct_1D(dct_1D_image[:,i])
    dct_1D=fftpack.dct(dct_1D_row, axis=0, norm='ortho')
    cv2.imwrite('../result/exp1/dct_1D.bmp', dct_1D)
    # 1D-IDCT
    # idct_1D_image = np.zeros(gray_image.shape, dtype=np.float32)
    # for i in tqdm(range(idct_1D_image.shape[1])):
        # idct_1D_image[:,i] = idct_1D(dct_1D_image[:,i])
    idct_1D_row=fftpack.idct(dct_1D, axis=0, norm='ortho')
    cv2.imwrite('../result/exp1/idct_1D_row.bmp', idct_1D_row)
    # for i in tqdm(range(idct_1D_image.shape[0])):
    #     idct_1D_image[i] = idct_1D(idct_1D_image[i])
    idct_1D=fftpack.idct(idct_1D_row, axis=1, norm='ortho')
    cv2.imwrite('../result/exp1/idct_1D.bmp', idct_1D)
    # idct_1D_row=fftpack.idct(dct_1D, axis=0, norm='ortho')
    # cv2.imwrite('../result/exp1/idct_1D_row.bmp', idct_1D_row)
    # idct_1D=fftpack.idct(idct_1D_row, axis=1, norm='ortho')
    # cv2.imwrite('../result/exp1/idct_1D.bmp', idct_1D)
    psnr_1D=calculate_PSNR(gray_image, idct_1D)
    print('PSNR of 1D-DCT:', psnr_1D)
    # 2D-DCT
    dct_2D=fftpack.dctn(gray_image, norm='ortho')
    cv2.imwrite('../result/exp1/dct_2D.bmp', dct_2D)
    # 2D-IDCT
    idct_2D=fftpack.idctn(dct_2D, norm='ortho')
    cv2.imwrite('../result/exp1/idct_2D.bmp', idct_2D)
    psnr_2D=calculate_PSNR(gray_image, idct_2D)
    print('PSNR of 2D-DCT:', psnr_2D)



if __name__=='__main__':
    main()
    print("All is well!")