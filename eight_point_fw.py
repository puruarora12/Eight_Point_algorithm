import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import random

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))
    print(pts1.shape , pts2.shape)
    # print(pts1)

    #todo: Normalize the points
    mean1  = np.mean(pts1, axis =0)
    mean2 = np.mean(pts2, axis = 0)
    distance1 = []
    distance2 = []

    for i in range(pts1.shape[0]):
        distance1.append(np.linalg.norm(pts1[i] - mean1))
        distance2.append(np.linalg.norm(pts2[i] - mean2))
    distance1 = np.array(distance1)
    distance2 = np.array(distance2)
    # print(distance1.shape , distance2.shape)
    
    std1 = np.mean(distance1 , axis=0)
    std2= np.mean(distance2 , axis=0)
    # print(std1 , std2)
    # raise
    # print(f"{mean1} {mean2} {std1} {std2} {pts1.shape}  {pts2.shape}")
    
    #todo: Form the matrix A
    one_mat  = np.ones((pts1.shape[0],1 ) , dtype = np.int32)
    pts1 = np.hstack((pts1 , one_mat))
    pts2 = np.hstack((pts2 , one_mat))
    transform_mat1 = np.array([[math.sqrt(2)/std1  , 0 , 0], [0 , math.sqrt(2)/std1 ,0], [-math.sqrt(2)/std1*mean1[0] , -math.sqrt(2)/std1*mean1[1] , 1 ]])
    transform_mat2 = np.array([[math.sqrt(2)/std2  , 0 , 0], [0 , math.sqrt(2)/std2 ,0], [-math.sqrt(2)/std2*mean2[0] , -math.sqrt(2)/std2*mean2[1] , 1 ]])
    # print(transform_mat1.T)
    # print(transform_mat2.T)
    
    new_pts1 = (pts1 @ transform_mat1).T
    new_pts2 = (pts2 @ transform_mat2).T
    # print(new_pts1)
    
    print(new_pts1.shape , new_pts2.shape)
    
    A = np.zeros((new_pts1.shape[1] , 9))
    for i in range(new_pts1.shape[1]):
        A[i] = np.reshape((np.array([new_pts2[: , i]]).reshape((3,1))@np.array([new_pts1[: , i]]).reshape((3,1)).T) , (1,9))
    u, sigma, v = np.linalg.svd(A)
    v=v.T
    
    fundamental_matrix = np.reshape(v[: , -1] ,(3,3))
   
    fu , fsigma, fv = np.linalg.svd(fundamental_matrix)
    fsigma[2]=0
    fundamental_matrix = transform_mat2 @fu@ np.diag(fsigma)@fv@transform_mat1.T
    print(fundamental_matrix.shape)
    
    print(fundamental_matrix.shape)
    # print(fundamental_matrix)
    
    return fundamental_matrix


    #todo: Find the fundamental matrix
    # raise NotImplementedError

def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1500, threshold = 0.00001):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))
    counter = -1
    best_fm = None

    for i in range(num_trials):
        indexes = random.sample(range(pts1.shape[0]),8)
        fundamental_matrix = FindFundamentalMatrix(pts1[indexes] ,pts2[indexes] )
        # test  = np.setdiff1d(np.arrange(pts1.shape[0]) , np.asarray(indexes))

        pts1_test = np.hstack((pts1 , np.ones((pts1.shape[0],1) , dtype=np.int32)))
        pts2_test = np.hstack((pts2 , np.ones((pts2.shape[0],1) , dtype=np.int32)))
        inlier = 0
        for x in range(pts1_test.shape[0]):
            error = abs(pts2_test[x]@fundamental_matrix@pts1_test[x])
            if error<threshold:
                inlier+=1
                # print(error)
        if inlier>counter:
            best_fm = fundamental_matrix
            counter= inlier
        # print(best_fm)
        # raise
        return best_fm
        

   
if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = True

    #Load images
    images1 = ["mount_rushmore_1.jpg" , "myleft.jpg"  ,"notredam_1.jpg"]
    images2 = ["mount_rushmore_2.jpg", "myright.jpg"  ,"notredam2.jpg"]
    # threshold = [1070 ,  1050 , 1050]
    # images1 = ["notredam_1.jpg"]
    # images2 = ["notredam2.jpg"]
    for i in range(len(images1)):
        image1_path = os.path.join(data_path, images1[i])
        image2_path = os.path.join(data_path, images2[i])
        image1 = np.array(Image.open(image1_path).convert('L'))
        image2 = np.array(Image.open(image2_path).convert('L'))


        #Find matching keypoints
        pts1, pts2 = find_matching_keypoints(image1, image2)

        #Builtin opencv function for comparison
        F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

        #todo: FindFundamentalMatrix
        used =""
        if use_ransac:
            F = FindFundamentalMatrixRansac(pts1, pts2 , num_trials=1500,threshold=0.0001)
            F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)[0]
            used = "with RANSAC"
        else:
            F = FindFundamentalMatrix(pts1, pts2)
            F /=F[2,2]
            F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
            used = "without RANSAC"
        print("\n")
        print(f"{images1[i].split('.')[0] } Calculated Fundamental matrix {used} \n {F}" )
        print("\n")
        print(f"{images1[i].split('.')[0] } OpenCV Fundamental matrix {used} \n {F_true}" )
        print("\n")

        # print(F)
        # Find epilines corresponding to points in second image,  and draw the lines on first image
        fig, axis = plt.subplots(2,2)

        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        # print(lines1)
        img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)


        axis[0,0].imshow(img1)
        axis[0,0].set_title(f'Calculated1 {used}')
        axis[0,0].axis('off')

        axis[0,1].imshow(img2)
        axis[0,1].set_title(f'Calculated2 {used}')
        axis[0,1].axis('off')

    

        # Plot the results of CV2 Fundamental matrix
        # Find epilines corresponding to points in second image,  and draw the lines on first image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_true)
        lines1 = lines1.reshape(-1, 3)
        img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
        # fig, axis = plt.subplots(1, 2)

        axis[1,0].imshow(img1)
        axis[1,0].set_title(f'CV2 Image1 {used}')
        axis[1,0].axis('off')

        axis[1,1].imshow(img2)
        axis[1,1].set_title(f'CV2 Image2 {used}')
        axis[1,1].axis('off')
        fig.savefig(f"{(images1[i].split('.'))[0]}1_{used}.png")
        plt.show()

        # Plot the results of my Fundamental matrix
        # Find epilines corresponding to points in first image, and draw the lines on second image
        fig, axis = plt.subplots(2, 2 )

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
        
        axis[0,0].imshow(img1)
        axis[0,0].set_title(f'Calculated1 {used}')
        axis[0,0].axis('off')

        axis[0,1].imshow(img2)
        axis[0,1].set_title(f'Calculated2 {used}')
        axis[0,1].axis('off')

        
        # Plot the results of CV2 Fundamental matrix 
        # Find epilines corresponding to points in first image, and draw the lines on second image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_true)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
        # fig, axis = plt.subplots(1, 2)

        axis[1,0].imshow(img1)
        axis[1,0].set_title(f'CV2 Image1 {used}')
        axis[1,0].axis('off')

        axis[1,1].imshow(img2)
        axis[1,1].set_title(f'CV2 Image2 {used}')
        axis[1,1].axis('off')
        fig.savefig(f"{(images1[i].split('.'))[0]}2_{used}.png")
        plt.show()