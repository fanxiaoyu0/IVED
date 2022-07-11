import cv2
import numpy as np
from scipy import fftpack
import time
from tqdm import tqdm

def calculate_PSNR(image_1, image_2):
    mse = np.mean((image_1 - image_2)**2)
    if mse == 0:
        return 1000
    return 10 * np.log10(255.0**2 / mse)

def main():
    gray_image = cv2.imread('../data/lena.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('../result/exp1/lena_gray.bmp', gray_image)
    
    start_time = time.time()
    # 1D-DCT
    dct_1D_row=fftpack.dct(gray_image, axis=1, norm='ortho')
    dct_1D=fftpack.dct(dct_1D_row, axis=0, norm='ortho')
    # 1D-IDCT
    idct_1D_row=fftpack.idct(dct_1D, axis=0, norm='ortho')
    idct_1D=fftpack.idct(idct_1D_row, axis=1, norm='ortho')
    
    print('Time of 1D-DCT:', time.time() - start_time)
    print('PSNR of 1D-DCT:', calculate_PSNR(gray_image, idct_1D))
    cv2.imwrite('../result/exp1/dct_1D_row.bmp', dct_1D_row)
    cv2.imwrite('../result/exp1/dct_1D.bmp', dct_1D)
    cv2.imwrite('../result/exp1/idct_1D_row.bmp', idct_1D_row)
    cv2.imwrite('../result/exp1/idct_1D.bmp', idct_1D)

    start_time = time.time()
    # 2D-DCT
    dct_2D=fftpack.dctn(gray_image, norm='ortho')
    # 2D-IDCT
    idct_2D=fftpack.idctn(dct_2D, norm='ortho')
    
    print('Time of 2D-DCT:', time.time() - start_time)
    print('PSNR of 2D-DCT:', calculate_PSNR(gray_image, idct_2D))
    cv2.imwrite('../result/exp1/dct_2D.bmp', dct_2D)
    cv2.imwrite('../result/exp1/idct_2D.bmp', idct_2D)

    start_time = time.time()
    # 2D-Block-DCT
    dct_block_2D=np.zeros(gray_image.shape)
    for i in range(0, dct_block_2D.shape[0], 8):
        for j in range(0, dct_block_2D.shape[1], 8):
            dct_block_2D[i:i+8, j:j+8]=fftpack.dctn(gray_image[i:i+8, j:j+8], norm='ortho')
    # 2D-Block-IDCT
    idct_block_2D=np.zeros(gray_image.shape)
    for i in range(0, idct_block_2D.shape[0], 8):
        for j in range(0, idct_block_2D.shape[1], 8):
            idct_block_2D[i:i+8, j:j+8]=fftpack.idctn(dct_block_2D[i:i+8, j:j+8], norm='ortho')

    print('Time of 2D-Block-DCT:', time.time() - start_time)
    print('PSNR of 2D-Block-DCT:', calculate_PSNR(gray_image, idct_block_2D))
    cv2.imwrite('../result/exp1/dct_block_2D.bmp', dct_block_2D)
    cv2.imwrite('../result/exp1/idct_block_2D.bmp', idct_block_2D)

def sub():
    gray_image = cv2.imread('../data/lena.bmp', cv2.IMREAD_GRAYSCALE)
    x=gray_image.shape[0]
    y=gray_image.shape[1]
    ratio_list=[4,16,64]
    for ratio in ratio_list:
        print('ratio:', ratio)

        start_time = time.time()
        # 1D-DCT
        dct_1D_row=fftpack.dct(gray_image, axis=1, norm='ortho')
        dct_1D_row[:,y//int(ratio**0.5):]=0
        dct_1D=fftpack.dct(dct_1D_row, axis=0, norm='ortho')
        dct_1D[x//int(ratio**0.5):,:]=0
        # 1D-IDCT
        idct_1D_row=fftpack.idct(dct_1D, axis=0, norm='ortho')
        idct_1D=fftpack.idct(idct_1D_row, axis=1, norm='ortho')

        print('Time of 1D-DCT:', time.time() - start_time)
        print('PSNR of 1D-DCT:', calculate_PSNR(gray_image, idct_1D))
        cv2.imwrite('../result/exp1/idct_1D_'+str(ratio)+'.bmp', idct_1D)

        start_time = time.time()
        # 2D-DCT
        dct_2D=fftpack.dctn(gray_image, norm='ortho')
        dct_2D[dct_2D.shape[0]//int(ratio**0.5):,:]=0
        dct_2D[:,dct_2D.shape[1]//int(ratio**0.5):]=0
        # 2D-IDCT
        idct_2D=fftpack.idctn(dct_2D, norm='ortho')

        print('Time of 2D-DCT:', time.time() - start_time)
        print('PSNR of 2D-DCT:', calculate_PSNR(gray_image, idct_2D))
        cv2.imwrite('../result/exp1/idct_2D_'+str(ratio)+'.bmp', idct_2D)
    
        start_time = time.time()
        # 2D-Block-DCT
        dct_block_2D=np.zeros(gray_image.shape)
        for i in range(0, dct_block_2D.shape[0], 8):
            for j in range(0, dct_block_2D.shape[1], 8):
                dct_block_2D[i:i+8, j:j+8]=fftpack.dctn(gray_image[i:i+8, j:j+8], norm='ortho')
                dct_block_2D[i:i+8, j:j+8][8//int(ratio**0.5):,:]=0
                dct_block_2D[i:i+8, j:j+8][:,8//int(ratio**0.5):]=0
        # 2D-Block-IDCT
        idct_block_2D=np.zeros(gray_image.shape)
        for i in range(0, idct_block_2D.shape[0], 8):
            for j in range(0, idct_block_2D.shape[1], 8):
                idct_block_2D[i:i+8, j:j+8]=fftpack.idctn(dct_block_2D[i:i+8, j:j+8], norm='ortho')
        
        print('Time of 2D-Block-DCT:', time.time() - start_time)
        print('PSNR of 2D-Block-DCT:', calculate_PSNR(gray_image, idct_block_2D))
        cv2.imwrite('../result/exp1/idct_block_2D_'+str(ratio)+'.bmp', idct_block_2D)

if __name__=='__main__':
    # main()
    sub()
    print("All is well!")