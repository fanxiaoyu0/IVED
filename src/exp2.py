import cv2
import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

def calculate_PSNR(image_1, image_2):
    mse = np.mean((image_1 - image_2)**2)
    if mse == 0:
        return 1000
    return 10 * np.log10(255.0**2 / mse)

def main():
    Cannon = np.array([[1, 1, 1, 2, 3, 6, 8, 10], [1, 1, 2, 3, 4, 8, 9, 8], [2, 2, 2, 3, 6, 8, 10, 8], [2, 2, 3, 4, 7, 12, 11, 9],\
        [3, 3, 8, 11, 10, 16, 15, 11], [3, 5, 8, 10, 12, 15, 16, 13], [7, 10, 11, 12, 15, 17, 17, 14], [14, 13, 13, 15, 15, 14, 14, 14]], dtype=np.float32)
    Nikon = np.array([[2, 1, 1, 2, 3, 5, 6, 7], [1, 1, 2, 2, 3, 7, 7, 7], [2, 2, 2, 3, 5, 7, 8, 7], [2, 2, 3, 3, 6, 10, 10, 7],\
        [2, 3, 4, 7, 8, 13, 12, 9], [3, 4, 7, 8, 10, 12, 14, 11], [6, 8, 9, 10, 12, 15, 14, 12], [9, 11, 11, 12, 13, 12, 12, 12]], dtype=np.float32)
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],\
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)
    My = np.array([[1, 1, 1, 2, 4, 6, 9, 12], [1, 1, 1, 2, 3, 5, 8, 11], [1, 1, 1, 2, 3, 5, 8, 11], [2, 2, 2, 2, 4, 6, 8, 11],\
        [4, 3, 3, 4, 5, 7, 9, 12], [6, 5, 5, 6, 7, 8, 10, 12], [9, 8, 8, 8, 9, 10, 11, 14], [12, 11, 11, 11, 12, 12, 14, 16]], dtype=np.float32)
    
    for i in range(8):
        for j in range(8):
            My[i][j] = max(int(0.4*(0.4*(i**2+j**2)+0.5*(0.2*min(i,j)+0.7*abs(i-j))**2)),1)  

    image_name_list=['lena','child','lake']
    for image_name in image_name_list:
        gray_image = cv2.imread('../data/'+image_name+'.bmp', cv2.IMREAD_GRAYSCALE)
        x=gray_image.shape[0]
        y=gray_image.shape[1]
        matirx_dict = {'Cannon': Cannon, 'Nikon': Nikon, 'Q': Q, 'My': My}
        PSNR_list_dict = {'Cannon': [], 'Nikon': [], 'Q': [], 'My': []}
        matrix_name_list=['Cannon', 'Nikon', 'Q', 'My']
        for matrix_name in matrix_name_list:
            matrix=matirx_dict[matrix_name]
            for a in range(1,20):
                dct_block_2D=np.zeros((x,y))
                for i in range(0, x, 8):
                    for j in range(0, y, 8):
                        dct_block_2D[i:i+8, j:j+8]=fftpack.dctn(gray_image[i:i+8, j:j+8], norm='ortho')
                        dct_block_2D[i:i+8, j:j+8]=np.round(dct_block_2D[i:i+8, j:j+8]/(a/10*matrix))*(a/10*matrix)
                idct_block_2D=np.zeros((x,y))
                for i in range(0, x, 8):
                    for j in range(0, y, 8):
                        idct_block_2D[i:i+8, j:j+8]=fftpack.idctn(dct_block_2D[i:i+8, j:j+8], norm='ortho')
                PSNR=calculate_PSNR(gray_image, idct_block_2D)
                PSNR_list_dict[matrix_name].append(PSNR)
                print('image name: '+image_name+',','matrix type: '+matrix_name+',','a=',a,'PSNR=', PSNR)
                cv2.imwrite('../result/exp2/'+image_name+'/'+matrix_name+'/'+str(a)+'.bmp', idct_block_2D)
        
        index=[i/10 for i in range(1,20)]
        for matrix_name in matrix_name_list:
            plt.plot(index,PSNR_list_dict[matrix_name],marker='o',label=matrix_name)
        plt.xlabel('a')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig('../result/exp2/PSNR_'+image_name+'.png')
        plt.close()

if __name__=='__main__':
    main()
    print("All is well!")