import cv2
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import os

def split_video_into_frames():
    vidcap = cv2.VideoCapture('../data/cars.avi')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("../result/exp3/frames/"+str(count)+".bmp", image)
        success,image = vidcap.read()
        count += 1
        if count >= 150:
            break

def calculate_MSE(image_1, image_2):
    return np.mean((image_1 - image_2)**2)

def main():
    first_frame=cv2.imread('../result/exp3/frames/20.bmp',cv2.IMREAD_GRAYSCALE)
    x=120
    y=316
    w=16
    target_block_pixel=first_frame[x:x+16,y:y+16]

    pixel_target_position_list=[]
    pixel_motion_vector_list=[]
    pixel_MSE_list=[]
    for frame_index in range(20,150):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp',cv2.IMREAD_GRAYSCALE)
        next_x=x
        next_y=y
        current_min_MSE=100000
        for i in range(max(0,x-w),min(x+w,frame.shape[0]-16)):
            for j in range(max(0,y-w),min(y+w,frame.shape[1]-16)):
                block_pixel=frame[i:i+16,j:j+16]
                MSE=calculate_MSE(target_block_pixel,block_pixel)
                if MSE<current_min_MSE:
                    current_min_MSE=MSE
                    next_x=i
                    next_y=j
        delta_x=next_x-x
        delta_y=next_y-y
        x=next_x
        y=next_y
        print("frame_index:",frame_index,"x:",x,"y:",y,"MSE:",current_min_MSE,"delta_x:",delta_x,"delta_y:",delta_y)
        pixel_target_position_list.append((x,y))
        pixel_motion_vector_list.append((delta_x,delta_y))
        pixel_MSE_list.append(current_min_MSE)

    plt.plot(pixel_MSE_list,label="pixel")
    # plt.xlabel('frame index')
    # plt.ylabel('pixel MSE')
    # plt.title('pixel MSE of motion vector')
    # plt.savefig('../result/exp3/MSE_pixel.png')
    # plt.close()

    new_frame_list=[]
    for frame_index in range(20,149,10):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp')
        index=frame_index-20
        start_point=pixel_target_position_list[index]
        end_point=pixel_target_position_list[min(index+10,len(pixel_target_position_list)-1)]
        start_point=(start_point[1],start_point[0])
        end_point=(end_point[1],end_point[0])
        new_frame_list.append(cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 1))

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
    videoWrite = cv2.VideoWriter('../result/exp3/motion_vector_pixel.avi', fourcc, 5, (352,288))
    for new_frame in new_frame_list:
        videoWrite.write(new_frame)
    videoWrite.release()

    x=120
    y=316
    target_block_dct=fftpack.dctn(target_block_pixel, norm='ortho')
    
    dct_target_position_list=[]
    dct_motion_vector_list=[]
    dct_MSE_list=[]
    for frame_index in range(20,150):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp',cv2.IMREAD_GRAYSCALE)
        next_x=x
        next_y=y
        current_min_MSE=100000
        pixel_MSE=100000
        for i in range(max(0,x-w),min(x+w,frame.shape[0]-16)):
            for j in range(max(0,y-w),min(y+w,frame.shape[1]-16)):
                block_pixel=frame[i:i+16,j:j+16]
                block_dct=fftpack.dctn(block_pixel, norm='ortho')
                dct_MSE=calculate_MSE(target_block_dct,block_dct)
                if dct_MSE<current_min_MSE:
                    current_min_MSE=dct_MSE
                    pixel_MSE=calculate_MSE(target_block_pixel,block_pixel)
                    next_x=i
                    next_y=j
        delta_x=next_x-x
        delta_y=next_y-y
        x=next_x
        y=next_y
        print("frame_index:",frame_index,"x:",x,"y:",y,"MSE:",current_min_MSE,"delta_x:",delta_x,"delta_y:",delta_y)
        dct_target_position_list.append((x,y))
        dct_motion_vector_list.append((delta_x,delta_y))
        dct_MSE_list.append(pixel_MSE)

    plt.plot(dct_MSE_list,label="dct")
    plt.xlabel('frame index')
    plt.ylabel('MSE')
    plt.title('MSE of motion vector')
    plt.legend()
    plt.savefig('../result/exp3/MSE.png')
    plt.close()

    new_frame_list=[]
    for frame_index in range(20,149,10):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp')
        index=frame_index-20
        start_point=dct_target_position_list[index]
        end_point=dct_target_position_list[min(index+10,len(dct_target_position_list)-1)]
        start_point=(start_point[1],start_point[0])
        end_point=(end_point[1],end_point[0])
        new_frame_list.append(cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 1))

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWrite = cv2.VideoWriter('../result/exp3/motion_vector_dct.avi', fourcc, 5, (352,288))
    for new_frame in new_frame_list:
        videoWrite.write(new_frame)
    videoWrite.release()

def sub_pixel():
    first_frame=cv2.imread('../result/exp3/frames/20.bmp',cv2.IMREAD_GRAYSCALE)
    x=120
    y=316
    w=16
    target_block_pixel=first_frame[x:x+16,y:y+16]

    pixel_target_position_list=[]
    pixel_motion_vector_list=[]
    pixel_MSE_list=[]
    for frame_index in range(20,150):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp',cv2.IMREAD_GRAYSCALE)
        next_x=x
        next_y=y
        current_min_MSE=100000
        for i in range(max(0,x-w),min(x+w,frame.shape[0]-16)):
            for j in range(max(0,y-w),min(y+w,frame.shape[1]-16)):
                block_pixel=frame[i:i+16,j:j+16]
                MSE=calculate_MSE(target_block_pixel,block_pixel)
                if MSE<current_min_MSE:
                    current_min_MSE=MSE
                    next_x=i
                    next_y=j
        delta_x=next_x-x
        delta_y=next_y-y
        x=next_x
        y=next_y
        print("frame_index:",frame_index,"x:",x,"y:",y,"MSE:",current_min_MSE,"delta_x:",delta_x,"delta_y:",delta_y)
        pixel_target_position_list.append((x,y))
        pixel_motion_vector_list.append((delta_x,delta_y))
        pixel_MSE_list.append(current_min_MSE)
    plt.plot(pixel_MSE_list,label="all_pixel")

    target_block_even_pixel=np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            if i%2==0 and j%2==0:
                target_block_even_pixel[i][j]=target_block_pixel[i][j]

    pixel_target_position_list=[]
    pixel_motion_vector_list=[]
    pixel_MSE_list=[]
    for frame_index in range(20,150):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp',cv2.IMREAD_GRAYSCALE)
        next_x=x
        next_y=y
        current_min_MSE=100000
        pixel_MSE=100000
        for i in range(max(0,x-w),min(x+w,frame.shape[0]-16)):
            for j in range(max(0,y-w),min(y+w,frame.shape[1]-16)):
                block_pixel=frame[i:i+16,j:j+16]
                block_even_pixel=np.zeros((16,16))
                for i in range(16):
                    for j in range(16):
                        if i%2==0 and j%2==0:
                            block_even_pixel[i][j]=block_pixel[i][j]
                MSE=calculate_MSE(target_block_even_pixel,block_even_pixel)
                if MSE<current_min_MSE:
                    current_min_MSE=MSE
                    pixel_MSE=calculate_MSE(target_block_pixel,block_pixel)
                    next_x=i
                    next_y=j
        delta_x=next_x-x
        delta_y=next_y-y
        x=next_x
        y=next_y
        print("frame_index:",frame_index,"x:",x,"y:",y,"MSE:",current_min_MSE,"delta_x:",delta_x,"delta_y:",delta_y)
        pixel_target_position_list.append((x,y))
        pixel_motion_vector_list.append((delta_x,delta_y))
        pixel_MSE_list.append(pixel_MSE)
    plt.plot(pixel_MSE_list,label="even_pixel")
    plt.xlabel('frame index')
    plt.ylabel('MSE')
    plt.title('MSE of motion vector')
    plt.legend()
    plt.savefig('../result/exp3/MSE_pixel.png')
    plt.close()

def sub_dct():
    first_frame=cv2.imread('../result/exp3/frames/20.bmp',cv2.IMREAD_GRAYSCALE)
    x=120
    y=316
    w=16
    target_block_pixel=first_frame[x:x+16,y:y+16]
    target_block_dct=fftpack.dctn(target_block_pixel, norm='ortho')

    dct_target_position_list=[]
    dct_motion_vector_list=[]
    pixel_MSE_list=[]
    for frame_index in range(20,150):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp',cv2.IMREAD_GRAYSCALE)
        next_x=x
        next_y=y
        current_min_MSE=100000
        for i in range(max(0,x-w),min(x+w,frame.shape[0]-16)):
            for j in range(max(0,y-w),min(y+w,frame.shape[1]-16)):
                block_pixel=frame[i:i+16,j:j+16]
                block_dct=fftpack.dctn(block_pixel, norm='ortho')
                dct_MSE=calculate_MSE(target_block_dct,block_dct)
                if dct_MSE<current_min_MSE:
                    current_min_MSE=dct_MSE
                    pixel_MSE=calculate_MSE(target_block_pixel,block_pixel)
                    next_x=i
                    next_y=j
        delta_x=next_x-x
        delta_y=next_y-y
        x=next_x
        y=next_y
        print("frame_index:",frame_index,"x:",x,"y:",y,"MSE:",current_min_MSE,"delta_x:",delta_x,"delta_y:",delta_y)
        dct_target_position_list.append((x,y))
        dct_motion_vector_list.append((delta_x,delta_y))
        pixel_MSE_list.append(pixel_MSE)
    plt.plot(pixel_MSE_list,label="all_dct")

    dct_target_position_list=[]
    dct_motion_vector_list=[]
    pixel_MSE_list=[]
    for frame_index in range(20,150):
        frame=cv2.imread('../result/exp3/frames/'+str(frame_index)+'.bmp',cv2.IMREAD_GRAYSCALE)
        next_x=x
        next_y=y
        current_min_MSE=100000
        for i in range(max(0,x-w),min(x+w,frame.shape[0]-16)):
            for j in range(max(0,y-w),min(y+w,frame.shape[1]-16)):
                block_pixel=frame[i:i+16,j:j+16]
                block_dct=fftpack.dctn(block_pixel, norm='ortho')
                block_dct[8:,:]=0
                block_dct[:,8:]=0
                dct_MSE=calculate_MSE(target_block_dct,block_dct)
                if dct_MSE<current_min_MSE:
                    current_min_MSE=dct_MSE
                    pixel_MSE=calculate_MSE(target_block_pixel,block_pixel)
                    next_x=i
                    next_y=j
        delta_x=next_x-x
        delta_y=next_y-y
        x=next_x
        y=next_y
        print("frame_index:",frame_index,"x:",x,"y:",y,"MSE:",current_min_MSE,"delta_x:",delta_x,"delta_y:",delta_y)
        dct_target_position_list.append((x,y))
        dct_motion_vector_list.append((delta_x,delta_y))
        pixel_MSE_list.append(pixel_MSE)
    plt.plot(pixel_MSE_list,label="part_of_dct")
    plt.xlabel('frame index')
    plt.ylabel('MSE')
    plt.title('MSE of motion vector')
    plt.legend()
    plt.savefig('../result/exp3/MSE_dct.png')
    plt.close()

if __name__=='__main__':
    main()
    sub_pixel()
    sub_dct()
    print("All is well!")