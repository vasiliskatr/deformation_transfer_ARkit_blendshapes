import numpy as np
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
#from sklearn.neighbors import NearestNeighbors

from plotly.io import write_image


# Reads an obj file and returns vertex coordinates [x,y,z], number of vertices n and mesh triangle coordinates [i,j,k]
def Read(filename, flipZY = False, QuadMode = False):
    
    # Vertices
    x = []
    y = []
    z = []
    
    # Triangle Indices
    i = []
    j = []
    k = []
    
    # Texture coordinates
    u = []
    v = []
    
    if QuadMode==True:
        # Quad Indices
        A = []
        B = []
        C = []
        D = []
    
    
    vertex_read_done = 0
    
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content] 
    
  
    for line in content:
        temp = line.split()

        if len(temp)>0:
            if temp[0] == 'v':
                
                if flipZY==False:
                    x.append(float(temp[1]))
                    y.append(float(temp[2]))
                    z.append(float(temp[3]))
                else:
                    x.append(float(temp[1]))
                    y.append(float(temp[3]))
                    z.append(float(temp[2]))
                    
            
            elif temp[0] == 'f':
                
                if vertex_read_done == 0:
                    vertex_texture_correspondence = np.zeros(len(x))
                    vertex_read_done = 1
                
                i.append(int(temp[1].split('/')[0]) - 1)
                j.append(int(temp[2].split('/')[0]) - 1)
                k.append(int(temp[3].split('/')[0]) - 1)
                
                vertex_texture_correspondence[i[-1]] = int(temp[1].split('/')[1]) - 1
                vertex_texture_correspondence[j[-1]] = int(temp[2].split('/')[1]) - 1
                vertex_texture_correspondence[k[-1]] = int(temp[3].split('/')[1]) - 1
                
                if len(temp) == 5:
                    i.append(int(temp[1].split('/')[0]) - 1)
                    j.append(int(temp[3].split('/')[0]) - 1)
                    k.append(int(temp[4].split('/')[0]) - 1)
                    
                    vertex_texture_correspondence[k[-1]] = int(temp[4].split('/')[1]) - 1
                    
                    if QuadMode==True:
                        A.append(int(temp[1].split('/')[0]) - 1)
                        B.append(int(temp[2].split('/')[0]) - 1)
                        C.append(int(temp[3].split('/')[0]) - 1)
                        D.append(int(temp[4].split('/')[0]) - 1)
            
            elif temp[0] == 'vt':
                u.append(float(temp[1]))
                v.append(float(temp[2]))
                
                
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    i = np.asarray(i)
    j = np.asarray(j)
    k = np.asarray(k)
    
    u = np.asarray(u)
    v = np.asarray(v)
    
    if QuadMode == True:
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        nq = A.shape[0]
        A = A.reshape(1, nq)
        B = B.reshape(1, nq)
        C = C.reshape(1, nq)
        D = D.reshape(1, nq)
        quad = np.concatenate((A, B, C, D), axis=0)
        
    
    
    n = x.shape[0]
    nt = i.shape[0]
    nuv = u.shape[0]
    
    x = x.reshape(1, n)
    y = y.reshape(1, n)
    z = z.reshape(1, n)
    
    vertices = np.concatenate((x, y, z), axis=0)
    
    i = i.reshape(1, nt)
    j = j.reshape(1, nt)
    k = k.reshape(1, nt)
    
    tri = np.concatenate((i, j, k), axis=0)
    
    u = u.reshape(nuv, 1)
    v = v.reshape(nuv, 1)
    uv_coord = np.concatenate((u, v), axis=1)
    
    if QuadMode == False:
        return vertices, tri, uv_coord, vertex_texture_correspondence.astype(int), n
    else:
        return vertices, tri, quad, n

# Plots a 3D scatter plot of the mesh using x,y,z
def ShowScatter(vertices):
    
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    df = pd.DataFrame({'x':x, 'y':y, 'z':z})
    n = x.shape[0]
    index = np.linspace(0,n-1,n)
    index= index.astype(int)
    index= index.astype(str)
    
    #fig = px.scatter_3d(df, x='x', y='y', z='z')
    fig=go.Figure(data=[go.Scatter3d(x=df['x'],y=df['y'],z=df['z'],mode='markers',marker=dict(size=2), hovertext=index)])
    fig.show()

# Plots a 3D mesh using x,y,z and i,j,k
def ShowMesh(vertices, triangles):
    
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    i = triangles[0, :].reshape(triangles.shape[1])
    j = triangles[1, :].reshape(triangles.shape[1])
    k = triangles[2, :].reshape(triangles.shape[1])
    
    n = x.shape[0]
    index = np.linspace(0,n-1,n)
    index= index.astype(int)
    index= index.astype(str)
    
    fig = go.Figure(data=[
    go.Mesh3d(
        x=x,y=y,z=z,i=i,j=j,k=k,showscale=True, hovertext = index)])

    fig.show()

    
# Show two meshes on the same figure with 2 different colours
def Show2Meshes(vertices1, triangles1, vertices2, triangles2):
    
    x = np.concatenate( (vertices1[0, :], vertices2[0, :]), axis=0 ).reshape(vertices1.shape[1] + vertices2.shape[1])
    y = np.concatenate( (vertices1[1, :], vertices2[1, :]), axis=0 ).reshape(vertices1.shape[1] + vertices2.shape[1])
    z = np.concatenate( (vertices1[2, :], vertices2[2, :]), axis=0 ).reshape(vertices1.shape[1] + vertices2.shape[1])
    
    triangles2 = triangles2 + vertices1.shape[1]
    
    i = np.concatenate( (triangles1[0, :], triangles2[0, :]), axis=0 ).reshape(triangles1.shape[1] + triangles2.shape[1])
    j = np.concatenate( (triangles1[1, :], triangles2[1, :]), axis=0 ).reshape(triangles1.shape[1] + triangles2.shape[1])
    k = np.concatenate( (triangles1[2, :], triangles2[2, :]), axis=0 ).reshape(triangles1.shape[1] + triangles2.shape[1])
    
    n = x.shape[0]
    index = np.linspace(0,n-1,n)
    index= index.astype(int)
    index= index.astype(str)
    
    fig = go.Figure(data=[
    go.Mesh3d(
        x=x,y=y,z=z,i=i,j=j,k=k,showscale=True, hovertext = index,
        colorscale=[[0, 'gold'],
                    [1, 'cyan']],
        intensity = np.concatenate( (np.zeros(triangles1.shape[1]), np.ones(triangles2.shape[1])), axis =0),
        opacity = 0.7,
        intensitymode='cell')])

    fig.show()







def Rotate_Mesh_X(Template_vert, theta):
       
    R = np.zeros((3, 3)) 
    R[1,1] = np.cos(theta)
    R[1,2] = np.sin(theta)
    R[2,1] = -np.sin(theta)
    R[2,2] = np.cos(theta)
    R[0,0] = 1
        
    rotated_vertices = np.matmul(R, Template_vert)
        
    return rotated_vertices

def Rotate_Mesh_Y(Template_vert, theta):
       
    R = np.zeros((3, 3)) 
    R[0,0] = np.cos(theta)
    R[0,2] = -np.sin(theta)
    R[2,0] = np.sin(theta)
    R[2,2] = np.cos(theta)
    R[1,1] = 1
        
    rotated_vertices = np.matmul(R, Template_vert)
        
    return rotated_vertices

def Rotate_Mesh_Z(Template_vert, theta):
       
    R = np.zeros((3, 3)) 
    R[0,0] = np.cos(theta)
    R[0,1] = np.sin(theta)
    R[1,0] = -np.sin(theta)
    R[1,1] = np.cos(theta)
    R[2,2] = 1
        
    rotated_vertices = np.matmul(R, Template_vert)
        
    return rotated_vertices

def WireFrameMesh(vertices, triangles, quadMode = True):
    
    vertices = vertices.transpose()
    triangles = triangles.transpose()
    
    trace1 = go.Scatter3d(x=vertices[:, 0].flatten(), y=vertices[:, 1].flatten(), z=vertices[:, 2].flatten(), mode='markers', marker=dict(size=0.0001), name='markers')
    
    x_lines = list()
    y_lines = list()
    z_lines = list()
    
    if quadMode == False:


        for i in range(triangles.shape[0]):

            A_index = triangles[i][0]
            B_index = triangles[i][1]
            C_index = triangles[i][2]

            A = vertices[A_index]
            B = vertices[B_index]
            C = vertices[C_index]

            ## Line AB ##
            # Point A
            x_lines.append(A[0])
            y_lines.append(A[1])
            z_lines.append(A[2])

            # Point B
            x_lines.append(B[0])
            y_lines.append(B[1])
            z_lines.append(B[2])

            # Point C
            x_lines.append(C[0])
            y_lines.append(C[1])
            z_lines.append(C[2])

            # Point A
            x_lines.append(A[0])
            y_lines.append(A[1])
            z_lines.append(A[2])

            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)
              
    else:
        
        for i in range(triangles.shape[0]):

            A_index = triangles[i][0]
            B_index = triangles[i][1]
            C_index = triangles[i][2]
            D_index = triangles[i][3]

            A = vertices[A_index]
            B = vertices[B_index]
            C = vertices[C_index]
            D = vertices[D_index]

            ## Line AB ##
            # Point A
            x_lines.append(A[0])
            y_lines.append(A[1])
            z_lines.append(A[2])

            # Point B
            x_lines.append(B[0])
            y_lines.append(B[1])
            z_lines.append(B[2])

            # Point C
            x_lines.append(C[0])
            y_lines.append(C[1])
            z_lines.append(C[2])
            
            # Point D
            x_lines.append(D[0])
            y_lines.append(D[1])
            z_lines.append(D[2])
            
            # Point A
            x_lines.append(A[0])
            y_lines.append(A[1])
            z_lines.append(A[2])

            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)
            
        
    trace2 = go.Scatter3d(x=x_lines,y=y_lines,z=z_lines,mode='lines',name='lines')    
    fig = go.Figure(data=[trace1, trace2])
    fig.show()
    

def ShowMeshAndWireFrame(vertices, triangles, quad):
    
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    i = triangles[0, :].reshape(triangles.shape[1])
    j = triangles[1, :].reshape(triangles.shape[1])
    k = triangles[2, :].reshape(triangles.shape[1])
    
    n = x.shape[0]
    index = np.linspace(0,n-1,n)
    index= index.astype(int)
    index= index.astype(str)
    
    trace1 = go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,showscale=True, hovertext = index)
    
    ## Plotting the wireframe
    x_lines = list()
    y_lines = list()
    z_lines = list()
    vertices = vertices.transpose()
    quad = quad.transpose()
    
    for i in range(quad.shape[0]):
        
        A_index = quad[i][0]
        B_index = quad[i][1]
        C_index = quad[i][2]
        D_index = quad[i][3]

        A = vertices[A_index]
        B = vertices[B_index]
        C = vertices[C_index]
        D = vertices[D_index]

        ## Line AB ##
        # Point A
        x_lines.append(A[0])
        y_lines.append(A[1])
        z_lines.append(A[2])

        # Point B
        x_lines.append(B[0])
        y_lines.append(B[1])
        z_lines.append(B[2])

        # Point C
        x_lines.append(C[0])
        y_lines.append(C[1])
        z_lines.append(C[2])

        # Point D
        x_lines.append(D[0])
        y_lines.append(D[1])
        z_lines.append(D[2])

        # Point A
        x_lines.append(A[0])
        y_lines.append(A[1])
        z_lines.append(A[2])

        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
            
      
    trace2 = go.Scatter3d(x=x_lines,y=y_lines,z=z_lines,mode='lines',name='lines') 
    fig = go.Figure(data=[trace1, trace2])
    
    fig.show()





    vertices = Landmark_vertices
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    df = pd.DataFrame({'x':x, 'y':y, 'z':z})
    
    #fig = px.scatter_3d(df, x='x', y='y', z='z')
    trace3 = go.Scatter3d(x=df['x'],y=df['y'],z=df['z'],mode='markers',marker=dict(size=marker_size))



    fig = go.Figure(data=[trace1, trace2, trace3])
    
    fig.show()
    



# Plots a 3D mesh using x,y,z and i,j,k
def ShowMeshAndLandMarks(vertices, triangles, Landmark_vertices, marker_size = 4):
    
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    i = triangles[0, :].reshape(triangles.shape[1])
    j = triangles[1, :].reshape(triangles.shape[1])
    k = triangles[2, :].reshape(triangles.shape[1])
    
    n = x.shape[0]
    index = np.linspace(0,n-1,n)
    index= index.astype(int)
    index= index.astype(str)
    
    trace1 = go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,showscale=True, hovertext = index)
    
    vertices = Landmark_vertices
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    df = pd.DataFrame({'x':x, 'y':y, 'z':z})
    
    #fig = px.scatter_3d(df, x='x', y='y', z='z')
    trace2 = go.Scatter3d(x=df['x'],y=df['y'],z=df['z'],mode='markers',marker=dict(size=marker_size))
    
    fig = go.Figure(data=[trace1, trace2])
    
    fig.show()
    
def SaveObj(vert, tri, template_destination, save_destination, save_normals = False, Head_Mode = True, CM = False):
    
    vertices = vert.copy()
    vertices = vertices.transpose()
    triangles = tri.copy()
    triangles = triangles.transpose()
    #normals = create_nomrals_for_vertices(vertices, triangles)
    #normals = -normals
    ## Reference to output file
    output = open(save_destination ,"w+")
    
    ## Reference to template obj file
    with open(template_destination) as f:
        content = f.readlines()

    content = [x.strip() for x in content] 
    
    
    vertex_counter = 0
    vn_counter = 0
    
    decimal_points = 6
    
    output.write('# Generated by CARV3D. It is illegal to use this identity without the approval of CARV3D.' + '\n')
    
    for line in content:
        temp = line.split()

        if len(temp)>0:
            
            if temp[0] == 'v':
                new_line = 'v  ' + str(round(vertices[vertex_counter][0], decimal_points)) + ' ' + str(round(vertices[vertex_counter][1], decimal_points)) + ' ' + str(round(vertices[vertex_counter][2], decimal_points)) + '\n'
                output.write(new_line)
                vertex_counter = vertex_counter + 1
            
            elif temp[0] == 'vn' and vn_counter<vertices.shape[0]:
                
                if save_normals==True:
                    new_line = 'vn ' + str(round(normals[vn_counter][0], decimal_points)) + ' ' + str(round(normals[vn_counter][1], decimal_points)) + ' ' + str(round(normals[vn_counter][2], decimal_points)) + '\n'
                    output.write(new_line)
                    vn_counter = vn_counter + 1
            
            elif temp[0] == 'vt':
                output.write(line + '\n')
            
            elif temp[0] == 'f':
                
                if CM == False:
                
                    if Head_Mode == True:
                        new_line = 'f ' 

                        if save_normals==True:
                            for pp in range(len(temp)-1,0, -1):
                                new_line = new_line + temp[pp].split('/')[0] + '/' + temp[pp].split('/')[1] + '/'+ temp[pp].split('/')[0] + ' ' 
                        else:

                            for pp in range(len(temp)-1,0, -1):
                                new_line = new_line + temp[pp].split('/')[0] + '/' + temp[pp].split('/')[1] + ' '

                        output.write(new_line + '\n')

                    else:
                        output.write(line + '\n')
                else:
                    
                    new_line = 'f '
                    
                    for pp in range(1, len(temp)):
                        new_line = new_line + temp[pp].split('/')[0] + '/' + temp[pp].split('/')[1] + ' '
                        
                    
                    output.write(new_line + '\n')
                    

            #else:
                #output.write(line + '\n')
            
    output.close()
    
    


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr
    
def create_nomrals_for_vertices(vertices, faces):
    
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
    
    return normalize_v3(norm)

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr
    
def ShowDeltaGradAbs(Base_vert, Exp_vert, triangles, mi, ma):
    
    vertices = Exp_vert.copy()
    Deltas = np.linalg.norm((Exp_vert.transpose() - Base_vert.transpose()), axis=1)
    #Deltas = Deltas/max(Deltas)
    
    print (Deltas)
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    i = triangles[0, :].reshape(triangles.shape[1])
    j = triangles[1, :].reshape(triangles.shape[1])
    k = triangles[2, :].reshape(triangles.shape[1])
    
    n = x.shape[0]
    index = np.linspace(0,n-1,n)
    index= index.astype(int)
    index= index.astype(str)
    
    fig = go.Figure(data=[
    go.Mesh3d(
        x=x,y=y,z=z,i=i,j=j,k=k,colorscale=[[0, 'cyan'],[0.125,'blue'], [0.25, 'purple'],
                                            [0.375, 'green'], [0.5, 'yellow'],[0.625,'orange'],
                                            [0.75, 'magenta'], [0.875, 'red'], [1.0, 'darkred']],
                                            cmin = mi, cmax = ma,
        intensity=Deltas, 
        showscale=True, hovertext = index)])

    fig.show()


def ShowDeltaGrad(Base_vert, Exp_vert, triangles, saveimage = False, path = ' ', format=None,
                      scale=None, width=None, height=None):


  

    vertices = Exp_vert.copy()
    Deltas = np.linalg.norm((Exp_vert.transpose() - Base_vert.transpose()), axis=1)
    Deltas = Deltas/max(Deltas)

    
    x = vertices[0, :].reshape(vertices.shape[1])
    y = vertices[1, :].reshape(vertices.shape[1])
    z = vertices[2, :].reshape(vertices.shape[1])
    
    i = triangles[0, :].reshape(triangles.shape[1])
    j = triangles[1, :].reshape(triangles.shape[1])
    k = triangles[2, :].reshape(triangles.shape[1])
    
    n = x.shape[0]
    index = np.linspace(0,n-1,n)
    index= index.astype(int)
    index= index.astype(str)
    
    fig = go.Figure(data=[
    go.Mesh3d(
        x=x,y=y,z=z,i=i,j=j,k=k,colorscale=[[0, 'cyan'],[0.125,'blue'], [0.25, 'purple'],
                                            [0.375, 'green'], [0.5, 'yellow'],[0.625,'orange'],
                                            [0.75, 'magenta'], [0.875, 'red'], [1.0, 'darkred']],
        intensity=Deltas, 
        showscale=True, hovertext = index)])

    if saveimage == False:
        fig.show()
    else:
        write_image(fig, path, format,
                      scale, width, height)





def center (vertices):
    vertices[0,:] = vertices[0,:] - vertices[0,:].mean()
    vertices[1,:] = vertices[1,:] - vertices[1,:].mean()
    vertices[2,:] = vertices[2,:] - vertices[2,:].mean()
    return vertices

def normalise (vertices, mi, ma):
    span = [mi, ma]
    vertices = (span[1]-span[0]) * (vertices - vertices.min())/(vertices.max() - vertices.min()) + span[0] # around origin
    return vertices

def align_target_to_source(target_vertices, target_faces, target_landmarks, source_vertices, source_faces, source_landmarks, verbose = False):

    
    source_landmark_vertces = source_vertices[:,source_landmarks[:]].transpose()
    target_landmark_vertices = target_vertices[:,target_landmarks[:]].transpose()

    c, R, t = align_rigid_scale(target_landmark_vertices, source_landmark_vertces)
    
    #print ("R =\n", R)
    #print ("c =", c)
    #print ("t =\n", t)

    #print ("Check:  Target face *cR + t = Source face  is", np.allclose(target_landmark_vertices.dot(c*R) + t, source_landmark_vertces))
    err = ((target_landmark_vertices.dot(c * R) + t - source_landmark_vertces) ** 2).sum()
    print ("Residual error", err)
    target_vertices = (target_vertices.T.dot(c * R) + t).T

    #T = (T.dot(c * R) + t - S)
    #TT = T.transpose()
   # TT = (TT.dot(c * R) + t)
    #TA = TT.transpose()
    
    #ReadOBJ.Show2Meshes(S, FS, TA, FT)
    if verbose == True:
        return target_vertices, c, R, t
    else:
        return target_vertices


def align_rigid_scale(P, Q):
    # Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
    # with least-squares error.
    # Returns (scale factor c, rotation matrix R, translation vector t) such that
    #   Q = P*cR + t
    # if they align perfectly, or such that
    #   SUM over point i ( | P_i*cR + t - Q_i |^2 )
    # is minimised if they don't align perfectly.
    
    #Q = Q[:, LS.LM[:]].transpose()
    #P = P[:, LT.LM[:]].transpose()
    
    assert P.shape == Q.shape
    n, _ = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(centeredP.T, centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


def Rotate_Mesh_X(Template_vert, theta):
       
    R = np.zeros((3, 3))
    R[1,1] = np.cos(theta)
    R[1,2] = np.sin(theta)
    R[2,1] = -np.sin(theta)
    R[2,2] = np.cos(theta)
    R[0,0] = 1
        
    rotated_vertices = np.matmul(R, Template_vert)
        
    return rotated_vertices


def Rotate_Mesh_Y(Template_vert, theta):
       
    R = np.zeros((3, 3))
    R[0,0] = np.cos(theta)
    R[0,2] = np.sin(theta)
    R[2,0] = -np.sin(theta)
    R[2,2] = np.cos(theta)
    R[1,1] = 1
        
    rotated_vertices = np.matmul(R, Template_vert)
        

    return rotated_vertices


def Rotate_Mesh_Z(Template_vert, theta):
       
    R = np.zeros((3, 3))
    R[0,0] = np.cos(theta)
    R[0,1] = np.sin(theta)
    R[1,0] = -np.sin(theta)
    R[1,1] = np.cos(theta)
    R[2,2] = 1
        
    rotated_vertices = np.matmul(R, Template_vert)
        

    return rotated_vertices


def filter_def (v,vb,c):
    delta = vb - v
    norm_delta = np.linalg.norm(delta.T,axis=1,keepdims=True)
    cut_off = c*norm_delta.max()
    #mask_cut_off = [1 if element <= cut_off else 0 for element in norm_delta]
    #mask_cut_off = np.asarray(mask_cut_off, dtype=np.int)
    mask = [1-((element-norm_delta.min())/(cut_off-norm_delta.min())) if element <= cut_off else 0 for element in norm_delta]
    mask = np.asarray(mask, dtype=np.float32)
    delta_cut_smooth = delta*mask.T
    vb2 = vb - delta_cut_smooth
    
    return vb2



def compute_vertex_normals(vertices, faces, face_normals):
    """ For every vertex, consider the normals of all neighbouring faces (can be more than 3) """

    vertex_normals = np.zeros((len(vertices), 3), dtype = np.float32)
    face_list = [[] for row in range(len(vertices))] #Triangle index list that shares the vertex
    for i in range(len(faces)):
        face_list[faces[i,0]] += [i]
        face_list[faces[i,1]] += [i]
        face_list[faces[i,2]] += [i]
        
    for j in range(len(vertices)):
        normal_sum = np.sum(face_normals[face_list[j]], axis=0)
        vertex_normals[j] = normal_sum / np.linalg.norm(normal_sum)
    
    return vertex_normals
