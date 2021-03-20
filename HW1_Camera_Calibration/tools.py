import numpy as np
import cv2

def calibration(objpoints,imgpoints):
    All_H = []
    All_V = []
    for i in range(len(objpoints)):
        temp_h = Find_h(objpoints[i],imgpoints[i])
        All_H.append(temp_h)
        
        temp_v = Find_v(temp_h)
        All_V.append(temp_v)

    All_V = np.array(All_V).reshape(-1,6)


    b = Find_b(All_V)
    B = Find_B(b)
    K = Find_K(B)
    K = K/K[2][2]
    Intrinsic = K
    


    All_extrinsic = []
    for i in range(len(objpoints)):
        temp_extrinsic = Find_Extrinsic(All_H[i],K)
        All_extrinsic.append(temp_extrinsic)
    All_extrinsic = np.array(All_extrinsic).reshape(-1,6)
    return Intrinsic , All_extrinsic


    
    





def Find_h(objpoints,imgpoints):
    All_p = np.zeros((2*len(objpoints),9))
    for i in range(len(objpoints)):
        temp_p = Find_p(objpoints[i],imgpoints[i])
        All_p[2*i:2*i+2,:] = temp_p
    _ , _ , V_T = np.linalg.svd(All_p)
    hi = V_T.T[:,-1]
    hi = hi.reshape(3,3)
    hi = hi/hi[2,2]
    return hi


def Find_p(objpoints,imgpoints):
    p = np.zeros((2,9))
    x1 = objpoints[0]
    y1 = objpoints[1]
    x2 = imgpoints[0]
    y2 = imgpoints[1]
    objpoints[2] = 1
    p[0,0:3] = objpoints
    p[1,3:6] = objpoints
    p[0,6] = -1*x2*x1
    p[0,7] = -1*x2*y1
    p[0,8] = -1*x2
    p[1,6] = -y2*x1
    p[1,7] = -y2*y1
    p[1,8] = -y2
    return p

def Find_v(h):
    V = np.zeros((2,6))
    h11,h12,h13 = h[0]
    h21,h22,h23 = h[1]
    h31,h32,h33 = h[2]
    V[0,0] = h11*h12
    V[0,1] = h21*h12+h11*h22
    V[0,2] = h21*h22
    V[0,3] = h31*h12+h11*h32
    V[0,4] = h21*h32 + h31*h22
    V[0,5] = h31*h32
    V[1,0] = h11*h11-h12*h12
    V[1,1] = h21*h11+ h11*h21-h22*h12-h12*h22 
    V[1,2] = h21*h21-h22*h22
    V[1,3] = h31*h11+h11*h31 - h32*h12-h12*h32
    V[1,4] = h21*h31+h31*h21 - h22*h32 - h32*h22
    V[1,5] = h31*h31 - h32*h32
    return V

def Find_b(All_V):
    _ , _ , v_T = np.linalg.svd(All_V)
    b = v_T.T[:,-1]
    return b    


def Find_B(b):
    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,0] = b[3]
    B[2,1] = b[4]
    B[2,2] = b[5]
    return B

def Find_K(B):
    eigvalsB = np.linalg.eigvals(B)
    if np.all(eigvalsB>0):
        print("Calculating SVD")
    else:
        B= -B
    G = np.linalg.cholesky(B)
    k_inv = G.T
    K = np.linalg.inv(k_inv) 
    return K
    
    
def Find_Extrinsic(H,K):
    h1,h2,h3 = H.T
    K_inv = np.linalg.inv(K)
    s = 1/np.linalg.norm((np.dot(K_inv,h1)))
    r1 = s * np.dot(K_inv , h1)
    r2 = s * np.dot(K_inv , h2)
    r3 = np.cross(r1,r2)
    t = s * np.dot(K_inv , h3 )
    
    Extrinsic_temp = np.zeros((3,3))
    Extrinsic_temp[:,0] = r1
    Extrinsic_temp[:,1] = r2
    Extrinsic_temp[:,2] = r3
    rotation_vector , _ = cv2.Rodrigues(Extrinsic_temp)
    Extrinsic = np.concatenate((rotation_vector,t.reshape(-1,1)),axis = 0)
    Extrinsic = Extrinsic.reshape(6)
    return Extrinsic
