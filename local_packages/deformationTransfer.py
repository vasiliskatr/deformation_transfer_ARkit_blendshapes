import numpy as np
from math import sqrt, acos, cos, sin
import scipy.sparse as sp
import numba
from numba import jit



def find_adjsacent_faces (faces):
    """ For every triangle/face in a mesh, it retuns the adjacent triangles/faces """

    print ("\nFinding adjacent faces/triangles...")
    adjacent_faces = [[] for row in range(len(faces))] 
    flist = faces[:,[0, 1, 2]].tolist()
    for i in range(len(flist)):
        set1 = set(flist[i])
        for j in range(i+1, len(flist)):
            set2 = set(flist[j])
            if(len(set1 & set2)==2): #i.e. if 2 same vertex indices are found between set1 and set2 ->  set1 and set2 are adjacent
                adjacent_faces[i] += [j]
                adjacent_faces[j] += [i]
            if(len(adjacent_faces[i])>=3): break
    return adjacent_faces


@jit(nopython=True, parallel=True)
def compute_v4 (vertices, faces):
    v4 = np.zeros((len(faces), 3), dtype = np.float32)
    for k in numba.prange(len(faces)):
        cross = np.cross(vertices[faces[k, 1],0:3] - vertices[faces[k, 0],0:3], vertices[faces[k, 2],0:3]-vertices[faces[k, 0],0:3])
        v4[k] = vertices[faces[k, 0],0:3] + cross / sqrt(np.linalg.norm(cross))
        
    return v4


@jit(nopython=True, parallel=True)
def compute_face_normals(vertices, faces):
    face_normals = np.zeros((len(faces), 3), dtype = np.float32)
    for k in numba.prange(len(faces)):
        cross = np.cross(vertices[faces[k, 1],0:3] - vertices[faces[k, 0],0:3], vertices[faces[k, 2],0:3] - vertices[faces[k, 0],0:3])  
        face_normals[k] = cross / np.linalg.norm(cross)
    
    return face_normals




def compute_V_inverse_local (vertices, face, v4):
    
    a = np.array([vertices[face[1],0:3] - vertices[face[0],0:3]])
    b = np.array([vertices[face[2],0:3] - vertices[face[0],0:3]])
    c = np.array([v4 - vertices[face[0],0:3]])
    V_inverse_local = np.linalg.solve(np.hstack([a.T, b.T, c.T]), np.identity(3))
    
    return V_inverse_local



def compute_V_inverse (vertices, faces, v4):
    V_inverse = np.zeros((len(faces), 3, 3), dtype = np.float32)  #  inverse  matrix of vertices, including v4 
    for i in range(len(faces)):
        V_inverse[i] = compute_V_inverse_local (vertices, faces[i], v4[i])
    
    return V_inverse




#make large matrices and vectors
def makeEs_ATA_ATc(vertices, faces, Vinv):
    
    Adjacent = find_adjsacent_faces(faces)
    
    adj_num = 0
    for i in range(len(Adjacent)):
        for j in range(len(Adjacent[i])):
            adj_num+=1

    Es_cVector = sp.csr_matrix((1, adj_num*9), dtype=np.float32)
    Es_cVector = Es_cVector.T
    Es_A = sp.lil_matrix((adj_num*9, len(vertices)*3 + len(faces)*3), dtype=np.float32)

    row = 0
    Vinv_list = Vinv.tolist()
    Faces = faces.tolist()
    len_vertices = len(vertices)

    for i in range(len(Adjacent)):
        for j in Adjacent[i]:

            e1 = np.sum(Vinv[i,0:3,0])
            e2 = np.sum(Vinv[i,0:3,1])
            e3 = np.sum(Vinv[i,0:3,2])

            idx_v1 = Faces[i][0]
            Es_A[row*9, idx_v1*3]     = -e1
            Es_A[row*9+1, idx_v1*3+1] = -e1
            Es_A[row*9+2, idx_v1*3+2] = -e1
            Es_A[row*9+3, idx_v1*3]   = -e2
            Es_A[row*9+4, idx_v1*3+1] = -e2
            Es_A[row*9+5, idx_v1*3+2] = -e2
            Es_A[row*9+6, idx_v1*3]   = -e3
            Es_A[row*9+7, idx_v1*3+1] = -e3
            Es_A[row*9+8, idx_v1*3+2] = -e3

            idx_v2 = Faces[i][1]
            Es_A[row*9, idx_v2*3]     = Vinv_list[i][0][0]
            Es_A[row*9+1, idx_v2*3+1] = Vinv_list[i][0][0]
            Es_A[row*9+2, idx_v2*3+2] = Vinv_list[i][0][0]
            Es_A[row*9+3, idx_v2*3]   = Vinv_list[i][0][1]
            Es_A[row*9+4, idx_v2*3+1] = Vinv_list[i][0][1]
            Es_A[row*9+5, idx_v2*3+2] = Vinv_list[i][0][1]
            Es_A[row*9+6, idx_v2*3]   = Vinv_list[i][0][2]
            Es_A[row*9+7, idx_v2*3+1] = Vinv_list[i][0][2]
            Es_A[row*9+8, idx_v2*3+2] = Vinv_list[i][0][2]

            idx_v3 = Faces[i][2]
            Es_A[row*9, idx_v3*3]     = Vinv_list[i][1][0]
            Es_A[row*9+1, idx_v3*3+1] = Vinv_list[i][1][0]
            Es_A[row*9+2, idx_v3*3+2] = Vinv_list[i][1][0]
            Es_A[row*9+3, idx_v3*3]   = Vinv_list[i][1][1]
            Es_A[row*9+4, idx_v3*3+1] = Vinv_list[i][1][1]
            Es_A[row*9+5, idx_v3*3+2] = Vinv_list[i][1][1]
            Es_A[row*9+6, idx_v3*3]   = Vinv_list[i][1][2]
            Es_A[row*9+7, idx_v3*3+1] = Vinv_list[i][1][2]
            Es_A[row*9+8, idx_v3*3+2] = Vinv_list[i][1][2]

            Es_A[row*9, i*3+len_vertices*3]     = Vinv_list[i][2][0]
            Es_A[row*9+1, i*3+len_vertices*3+1] = Vinv_list[i][2][0]
            Es_A[row*9+2, i*3+len_vertices*3+2] = Vinv_list[i][2][0]
            Es_A[row*9+3, i*3+len_vertices*3]   = Vinv_list[i][2][1]
            Es_A[row*9+4, i*3+len_vertices*3+1] = Vinv_list[i][2][1]
            Es_A[row*9+5, i*3+len_vertices*3+2] = Vinv_list[i][2][1]
            Es_A[row*9+6, i*3+len_vertices*3]   = Vinv_list[i][2][2]
            Es_A[row*9+7, i*3+len_vertices*3+1] = Vinv_list[i][2][2]
            Es_A[row*9+8, i*3+len_vertices*3+2] = Vinv_list[i][2][2]

            e1 = np.sum(Vinv[j,0:3,0])
            e2 = np.sum(Vinv[j,0:3,1])
            e3 = np.sum(Vinv[j,0:3,2])

            idx_v1 = Faces[j][0]
            Es_A[row*9, idx_v1*3]     += e1
            Es_A[row*9+1, idx_v1*3+1] += e1
            Es_A[row*9+2, idx_v1*3+2] += e1
            Es_A[row*9+3, idx_v1*3]   += e2
            Es_A[row*9+4, idx_v1*3+1] += e2
            Es_A[row*9+5, idx_v1*3+2] += e2
            Es_A[row*9+6, idx_v1*3]   += e3
            Es_A[row*9+7, idx_v1*3+1] += e3
            Es_A[row*9+8, idx_v1*3+2] += e3

            idx_v2 = Faces[j][1]
            Es_A[row*9, idx_v2*3]     += -Vinv_list[j][0][0]
            Es_A[row*9+1, idx_v2*3+1] += -Vinv_list[j][0][0]
            Es_A[row*9+2, idx_v2*3+2] += -Vinv_list[j][0][0]
            Es_A[row*9+3, idx_v2*3]   += -Vinv_list[j][0][1]
            Es_A[row*9+4, idx_v2*3+1] += -Vinv_list[j][0][1]
            Es_A[row*9+5, idx_v2*3+2] += -Vinv_list[j][0][1]
            Es_A[row*9+6, idx_v2*3]   += -Vinv_list[j][0][2]
            Es_A[row*9+7, idx_v2*3+1] += -Vinv_list[j][0][2]
            Es_A[row*9+8, idx_v2*3+2] += -Vinv_list[j][0][2]

            idx_v3 = Faces[j][2]
            Es_A[row*9, idx_v3*3]     += -Vinv_list[j][1][0]
            Es_A[row*9+1, idx_v3*3+1] += -Vinv_list[j][1][0]
            Es_A[row*9+2, idx_v3*3+2] += -Vinv_list[j][1][0]
            Es_A[row*9+3, idx_v3*3]   += -Vinv_list[j][1][1]
            Es_A[row*9+4, idx_v3*3+1] += -Vinv_list[j][1][1]
            Es_A[row*9+5, idx_v3*3+2] += -Vinv_list[j][1][1]
            Es_A[row*9+6, idx_v3*3]   += -Vinv_list[j][1][2]
            Es_A[row*9+7, idx_v3*3+1] += -Vinv_list[j][1][2]
            Es_A[row*9+8, idx_v3*3+2] += -Vinv_list[j][1][2]

            Es_A[row*9, j*3+len_vertices*3]     += -Vinv_list[j][2][0]
            Es_A[row*9+1, j*3+len_vertices*3+1] += -Vinv_list[j][2][0]
            Es_A[row*9+2, j*3+len_vertices*3+2] += -Vinv_list[j][2][0]
            Es_A[row*9+3, j*3+len_vertices*3]   += -Vinv_list[j][2][1]
            Es_A[row*9+4, j*3+len_vertices*3+1] += -Vinv_list[j][2][1]
            Es_A[row*9+5, j*3+len_vertices*3+2] += -Vinv_list[j][2][1]
            Es_A[row*9+6, j*3+len_vertices*3]   += -Vinv_list[j][2][2]
            Es_A[row*9+7, j*3+len_vertices*3+1] += -Vinv_list[j][2][2]
            Es_A[row*9+8, j*3+len_vertices*3+2] += -Vinv_list[j][2][2]

            row += 1

    return np.dot(Es_A.T, Es_A), np.dot(Es_A.T, Es_cVector)



def makeEi_ATA_ATc(vertices, faces, Vinv):
    Flat_identity = sp.csc_matrix((0, 1), dtype=np.float32)
    for i in range(len(faces)):
        Flat_identity = sp.vstack((Flat_identity, sp.identity(3, format="lil").reshape((9, 1))), format="csc", dtype=np.float32)

    Ei_cVector = Flat_identity
    Ei_A = sp.lil_matrix((len(faces)*9, len(vertices)*3 + len(faces)*3), dtype=np.float32)

    Vinv_list = Vinv.tolist()
    Faces = faces.tolist()
    len_vertices = len(vertices)
    for i in range(len(faces)):
        idx_f = i
        e1 = np.sum(Vinv[idx_f,0:3,0])
        e2 = np.sum(Vinv[idx_f,0:3,1])
        e3 = np.sum(Vinv[idx_f,0:3,2])
                
        idx_v1 = Faces[idx_f][0]
        Ei_A[i*9, idx_v1*3]     = -e1
        Ei_A[i*9+1, idx_v1*3+1] = -e1
        Ei_A[i*9+2, idx_v1*3+2] = -e1
        Ei_A[i*9+3, idx_v1*3]   = -e2
        Ei_A[i*9+4, idx_v1*3+1] = -e2
        Ei_A[i*9+5, idx_v1*3+2] = -e2
        Ei_A[i*9+6, idx_v1*3]   = -e3
        Ei_A[i*9+7, idx_v1*3+1] = -e3
        Ei_A[i*9+8, idx_v1*3+2] = -e3

        idx_v2 = Faces[idx_f][1]
        Ei_A[i*9, idx_v2*3]     = Vinv_list[idx_f][0][0]
        Ei_A[i*9+1, idx_v2*3+1] = Vinv_list[idx_f][0][0]
        Ei_A[i*9+2, idx_v2*3+2] = Vinv_list[idx_f][0][0]
        Ei_A[i*9+3, idx_v2*3]   = Vinv_list[idx_f][0][1]
        Ei_A[i*9+4, idx_v2*3+1] = Vinv_list[idx_f][0][1]
        Ei_A[i*9+5, idx_v2*3+2] = Vinv_list[idx_f][0][1]
        Ei_A[i*9+6, idx_v2*3]   = Vinv_list[idx_f][0][2]
        Ei_A[i*9+7, idx_v2*3+1] = Vinv_list[idx_f][0][2]
        Ei_A[i*9+8, idx_v2*3+2] = Vinv_list[idx_f][0][2]
                
        idx_v3 = Faces[idx_f][2]
        Ei_A[i*9, idx_v3*3]     = Vinv_list[idx_f][1][0]
        Ei_A[i*9+1, idx_v3*3+1] = Vinv_list[idx_f][1][0]
        Ei_A[i*9+2, idx_v3*3+2] = Vinv_list[idx_f][1][0]
        Ei_A[i*9+3, idx_v3*3]   = Vinv_list[idx_f][1][1]
        Ei_A[i*9+4, idx_v3*3+1] = Vinv_list[idx_f][1][1]
        Ei_A[i*9+5, idx_v3*3+2] = Vinv_list[idx_f][1][1]
        Ei_A[i*9+6, idx_v3*3]   = Vinv_list[idx_f][1][2]
        Ei_A[i*9+7, idx_v3*3+1] = Vinv_list[idx_f][1][2]
        Ei_A[i*9+8, idx_v3*3+2] = Vinv_list[idx_f][1][2]

        Ei_A[i*9, idx_f*3+len_vertices*3]    = Vinv_list[idx_f][2][0]
        Ei_A[i*9+1, idx_f*3+len_vertices*3+1] = Vinv_list[idx_f][2][0]
        Ei_A[i*9+2, idx_f*3+len_vertices*3+2] = Vinv_list[idx_f][2][0]
        Ei_A[i*9+3, idx_f*3+len_vertices*3]   = Vinv_list[idx_f][2][1]
        Ei_A[i*9+4, idx_f*3+len_vertices*3+1] = Vinv_list[idx_f][2][1]
        Ei_A[i*9+5, idx_f*3+len_vertices*3+2] = Vinv_list[idx_f][2][1]
        Ei_A[i*9+6, idx_f*3+len_vertices*3]   = Vinv_list[idx_f][2][2]
        Ei_A[i*9+7, idx_f*3+len_vertices*3+1] = Vinv_list[idx_f][2][2]
        Ei_A[i*9+8, idx_f*3+len_vertices*3+2] = Vinv_list[idx_f][2][2]

    return np.dot(Ei_A.T, Ei_A), np.dot(Ei_A.T, Ei_cVector)


@jit(nopython=True, parallel=True)
def compute_face_cenetroids (vertices,faces):
    centeroids = np.zeros((len(faces), 3), dtype=np.float32)
    for i in numba.prange(len(faces)):
        centeroids[i] = (vertices[faces[i, 0]] + vertices[faces[i, 1]] + vertices[faces[i, 2]])/3
    return centeroids



def get_compatible_faces (d_target_centroids, target_faces, d_target_face_normals, source_centroids, source_faces, source_face_normals, threshold = 10):

    """ 
    Gets the compatible faces(triangles) between two objects of different topology.
    The corresponding faces must be close to each other (centroid proximity)
    and the respective face(triangle) normals should be less than 90deg.
    
    For a pair of objects of different topology, this funtion is called twice. 
    target mesh -> source mesh
    and 
    source mesh -> target mesh
    ---------------------------------------
    d_target_centroids: ndarray with the centroids for each face(triangle) of the deforrmed target mesh [:,3]
    target_faces: ndarray with the faces(triangles) of target mesh [:,3]
    d_target_face_normals: ndarray with the normals for every face(triangle) of target mesh [:,3]
    source_centroids: ndarray with the centroids for each face(triangle) of source mesh [:,3]
    source_faces: ndarray with the faces(triangles) of source mesh [:,3]
    source_face_normals: ndarray with the normals for every face(triangle) of the source mesh [:,3]
    threshold: scalar, consider matches which are closer than this
    ---------------------------------------
     """

    threshold = 2
    corr = []
    ##for all target Faces
    for i in range(len(target_faces)):
        validindex = -1
        norm = np.linalg.norm(d_target_centroids[i] - source_centroids, axis=1)
        while(np.min(norm) < threshold):
            idx = np.argmin(norm)
            if(np.dot(d_target_face_normals[i], source_face_normals[idx])>0): # check that the normals of close faces are < 90 deg.
                validindex = idx
                break
            else:
                norm[idx] = threshold
        if(validindex >= 0):
            corr += [[validindex, i]]
    
    print (len(corr), " correspondeces found out of ", len(target_faces), " faces on the target mesh")
    corr_step1 = len(corr)
    
        ##for all Source Faces
    for i in range(len(source_faces)):
        validindex = -1
        norm = np.linalg.norm(source_centroids[i] - d_target_centroids, axis=1)
        while(np.min(norm) < threshold):
            idx = np.argmin(norm)
            if(np.dot(source_face_normals[i], d_target_face_normals[idx])>0):
                validindex = idx
                break
            else:
                norm[idx] = threshold
        if(validindex >= 0):
            corr += [[i, validindex]]
            
    print (len(corr) - corr_step1, " correspondeces found out of ", len(source_faces), "faces on the source mesh")  
    
    return corr



def get_correspondece_faces (source_vertices, source_faces, deformed_target_vertices, target_faces):

    """ 
    Uses [def get_compatible_faces] to compute all the  face correspondeces between the source object and target object
    ---------------------------------------
    source_vertices: np.ndarray source_vertices neutral face [:,3]
    source_faces: np.ndarray source_faces neutral face (object 1) [:,3]
    deformed_target_vertices: np.ndarray target_vertices neutral face which is deformed using NRICP to mactch the shape of the source neutral face [:,3]
    target_faces: np.ndarray target_faces neutral face (object 1) [:,3] (does not change with NRICP since the topology of the deformed_target is the same as the target)
    ---------------------------------------
    """
    
    print ("\nComputing triangle correspondences...")
    
    source_face_normals = compute_face_normals(source_vertices, source_faces)
    source_centroids = compute_face_cenetroids(source_vertices, source_faces)

    deformed_target_face_normals = compute_face_normals(deformed_target_vertices, target_faces)
    deformed_target_centroids = compute_face_cenetroids(deformed_target_vertices, target_faces)
    
    correspondences = get_compatible_faces(deformed_target_centroids, target_faces, deformed_target_face_normals, source_centroids, source_faces, source_face_normals)
    
    return correspondences




def make_source_rotation_matrix (vertices_1, faces_1, v4_1, vertices_2, faces_2):
    """ 
    Computes the matrix of affine transformations for each face(triangle) of a 3D object which goes from frame A(vertices-1) to frame B(vertices_2)
    We use this to extract the deformation for each pair of the source blend shapes (generic blend shapes).
    ---------------------------------------
    vertices_1: np.ndarray source_vertices neutral face (object 1) [:,3]
    vertices_2: np.ndarray source_vertices some expression  (object 2) [:,3]
    faces_1 = faces_2: np.ndarray with the faces(triangles) [:,3]. The rotation matrix is calculated for objects of the same topology 
    - this could be reduced to take only faces_1 as input [:,3]
    v4_1: the 4th vertic for every face(triangle) of the source_vertices neutral face (i.e. vertices_1), see [def compute_v4]  [:,3]
    ---------------------------------------
    """
    print ("Computing source rotation...")
    source_rotation = np.zeros((len(faces_1), 3, 3), dtype = np.float32)
        
    for i in range(len(faces_1)):
        
        Vinv = compute_V_inverse_local (vertices_1, faces_1[i], v4_1[i])#Source Neutral Vinv
    
        a = np.array([vertices_2[faces_2[i,1],0:3] - vertices_2[faces_2[i,0],0:3]]) #v2-v1
        b = np.array([vertices_2[faces_2[i,2],0:3] - vertices_2[faces_2[i,0],0:3]]) #v3-v1
        cross = np.cross(a, b)
        c = cross / sqrt(np.linalg.norm(cross)) #v4-v1 
      
        source_rotation[i] = np.dot(np.hstack([a.T, b.T, c.T]), Vinv) # S = np.dot(SE_Vtil, SN_Vinv)
    
    return source_rotation




#Make Deformation Transfer matrix
def makeEd_A(corr, AN_Vertices, AN_Faces, AN_Vinv):
    Ed_A = sp.lil_matrix((len(corr)*9, len(AN_Vertices)*3 + len(AN_Faces)*3), dtype=np.float32)

    for i in range(len(corr)):
        idx_f = corr[i][1]
        e1 = np.sum(AN_Vinv[idx_f,0:3,0])
        e2 = np.sum(AN_Vinv[idx_f,0:3,1])
        e3 = np.sum(AN_Vinv[idx_f,0:3,2])

        Ed_A[i*9, AN_Faces[idx_f,0]*3]     = -e1
        Ed_A[i*9+1, AN_Faces[idx_f,0]*3+1] = -e1
        Ed_A[i*9+2, AN_Faces[idx_f,0]*3+2] = -e1
        Ed_A[i*9+3, AN_Faces[idx_f,0]*3]   = -e2
        Ed_A[i*9+4, AN_Faces[idx_f,0]*3+1] = -e2
        Ed_A[i*9+5, AN_Faces[idx_f,0]*3+2] = -e2
        Ed_A[i*9+6, AN_Faces[idx_f,0]*3]   = -e3
        Ed_A[i*9+7, AN_Faces[idx_f,0]*3+1] = -e3
        Ed_A[i*9+8, AN_Faces[idx_f,0]*3+2] = -e3

        Ed_A[i*9, AN_Faces[idx_f,1]*3]     = AN_Vinv[idx_f,0,0]
        Ed_A[i*9+1, AN_Faces[idx_f,1]*3+1] = AN_Vinv[idx_f,0,0]
        Ed_A[i*9+2, AN_Faces[idx_f,1]*3+2] = AN_Vinv[idx_f,0,0]
        Ed_A[i*9+3, AN_Faces[idx_f,1]*3]   = AN_Vinv[idx_f,0,1]
        Ed_A[i*9+4, AN_Faces[idx_f,1]*3+1] = AN_Vinv[idx_f,0,1]
        Ed_A[i*9+5, AN_Faces[idx_f,1]*3+2] = AN_Vinv[idx_f,0,1]
        Ed_A[i*9+6, AN_Faces[idx_f,1]*3]   = AN_Vinv[idx_f,0,2]
        Ed_A[i*9+7, AN_Faces[idx_f,1]*3+1] = AN_Vinv[idx_f,0,2]
        Ed_A[i*9+8, AN_Faces[idx_f,1]*3+2] = AN_Vinv[idx_f,0,2]

        Ed_A[i*9, AN_Faces[idx_f,2]*3]     = AN_Vinv[idx_f,1,0]
        Ed_A[i*9+1, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,1,0]
        Ed_A[i*9+2, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,1,0]
        Ed_A[i*9+3, AN_Faces[idx_f,2]*3]   = AN_Vinv[idx_f,1,1]
        Ed_A[i*9+4, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,1,1]
        Ed_A[i*9+5, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,1,1]
        Ed_A[i*9+6, AN_Faces[idx_f,2]*3]   = AN_Vinv[idx_f,1,2]
        Ed_A[i*9+7, AN_Faces[idx_f,2]*3+1] = AN_Vinv[idx_f,1,2]
        Ed_A[i*9+8, AN_Faces[idx_f,2]*3+2] = AN_Vinv[idx_f,1,2]

        Ed_A[i*9, idx_f*3+len(AN_Vertices)*3]     = AN_Vinv[idx_f,2,0]
        Ed_A[i*9+1, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,0]
        Ed_A[i*9+2, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,0]
        Ed_A[i*9+3, idx_f*3+len(AN_Vertices)*3]   = AN_Vinv[idx_f,2,1]
        Ed_A[i*9+4, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,1]
        Ed_A[i*9+5, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,1]
        Ed_A[i*9+6, idx_f*3+len(AN_Vertices)*3]   = AN_Vinv[idx_f,2,2]
        Ed_A[i*9+7, idx_f*3+len(AN_Vertices)*3+1] = AN_Vinv[idx_f,2,2]
        Ed_A[i*9+8, idx_f*3+len(AN_Vertices)*3+2] = AN_Vinv[idx_f,2,2]
            
    return Ed_A


def makeEd_ATc(corr, source_rotation, Ed_A):
    Ed_cVector = sp.lil_matrix((1, len(corr)*9), dtype=np.float32)            
    for i in range(len(corr)):
        Ed_cVector[0, i*9:i*9+9] = (source_rotation[corr[i][0]].T).flatten()
    Ed_cVector = Ed_cVector.T

    return np.dot(Ed_A.T, Ed_cVector)