import numpy as np

def change_vars(MG):
    """
    MG: np array of shape (3,...) containing 3D cartesian coordinates.
    returns spherical coordinates theta and phi (could return rho if needed)
    """
    rho = np.sqrt(np.sum(np.square(MG), axis=0))
    phi = np.squeeze(np.arctan2(MG[1, ...], MG[0, ...])) 
    theta = np.squeeze(np.arccos(MG[2, ...] / rho))
    # The center value is Nan due to the 0/0. So make it 0.
    theta[np.isnan(theta)] = 0
    rho = np.squeeze(rho)

    return theta, phi

def rev_chane_vars(phi, theta, rho=1):
    
    x = rho*np.sin(phi)*np.cos(theta)
    y = rho*np.sin(phi)*np.sin(theta)
    z = rho*np.cos(phi)
    DP = np.stack([x,y,z],axis=1)
    return DP
    
def arsNorm(A):
    # vectorized norm() function
    rez = A[:, 0] ** 2 + A[:, 1] ** 2 + A[:, 2] ** 2
    rez = np.sqrt(rez)
    return rez


def arsUnit(A, radius):
    # vectorized unit() functon
    normOfA = arsNorm(A)
    rez = A / np.stack((normOfA, normOfA, normOfA), 1)
    rez = rez * radius
    return rez

def sphereTriangulation(M, n_gamma):
    """
    Defines points on the sphere that we use for alpha (z) and beta (y') Euler angles sampling. We can have 24 points (numIterations=0), 72 (numIterations=1), 384 (numIterations=2) etc.
    Copied from the matlab function https://ch.mathworks.com/matlabcentral/fileexchange/38909-parametrized-uniform-triangulation-of-3d-circle-sphere
    M is the number total of orientation, i.e. number of points on the sphere * number of angles for the gamma angle (n_gamma).

    """
    
    F0 = 8 # number of faces of a octahedra
    V0 = 6 # number of vertices
    
    # the number of point after each iterations is given by V+n = 1.5*F0/3(4**n-1)+ V0 
    # numIter0 = int(np.log(((M/n_gamma)-V0)*(2/F0)+1)/(np.log(4)))
    # numIter1 = int(np.ceil(np.clip((np.log(M/(24))/np.log(n_gamma)+1)-1, a_min=0, a_max=None)))
    
    numIter = int(np.ceil(np.log(M-8)/np.log(n_gamma)-2))

    # function returns stlPoints fromat and ABC format if its needed,if not - just delete it and adapt to your needs
    radius = 1
    
    
    # basic Octahedron reff:http://en.wikipedia.org/wiki/Octahedron
    # ( ?1, 0, 0 )
    # ( 0, ?1, 0 )
    # ( 0, 0, ?1 )
    A = np.asarray([1, 0, 0]) * radius
    B = np.asarray([0, 1, 0]) * radius
    C = np.asarray([0, 0, 1]) * radius
    # from +-ABC create initial triangles which define oxahedron
    triangles = np.asarray([A, B, C,
                            A, B, -C,
                            # -x, +y, +-Z quadrant
                            -A, B, C,
                            -A, B, -C,
                            # -x, -y, +-Z quadrant
                            -A, -B, C,
                            -A, -B, -C,
                            # +x, -y, +-Z quadrant
                            A, -B, C,
                            A, -B, -C])  # -----STL-similar format
    # for simplicity lets break into ABC points...
    #triangles = np.unique(triangles,axis=0)
    selector = np.arange(0, len(triangles[:, 1]) - 2, 3)
    Apoints = triangles[selector, :]
    Bpoints = triangles[selector + 1, :]
    Cpoints = triangles[selector + 2, :]
    # in every of numIterations
    for iteration in range(numIter):
        # devide every of triangle on three new
        #        ^ C
        #       / \
        # AC/2 /_4_\CB/2
        #     /\ 3 /\
        #    / 1\ /2 \
        # A /____V____\B           1st              2nd              3rd               4th
        #        AB/2
        # new triangleSteck is [ A AB/2 AC/2;     AB/2 B CB/2;     AC/2 AB/2 CB/2    AC/2 CB/2 C]
        AB_2 = (Apoints + Bpoints) / 2
        # do normalization of vector
        AB_2 = arsUnit(AB_2, radius)  # same for next 2 lines
        AC_2 = (Apoints + Cpoints) / 2
        AC_2 = arsUnit(AC_2, radius)
        CB_2 = (Cpoints + Bpoints) / 2
        CB_2 = arsUnit(CB_2, radius)
        Apoints = np.concatenate((Apoints,  # A point from 1st triangle
                                  AB_2,  # A point from 2nd triangle
                                  AC_2,  # A point from 3rd triangle
                                  AC_2))  # A point from 4th triangle..same for B and C
        Bpoints = np.concatenate((AB_2, Bpoints, AB_2, CB_2))
        Cpoints = np.concatenate((AC_2, CB_2, CB_2, Cpoints))
    # now tur points back to STL-like format....
    numPoints = np.shape(Apoints)[0]

    selector = np.arange(numPoints)
    selector = np.stack((selector, selector + numPoints, selector + 2 * numPoints))

    selector = np.swapaxes(selector, 0, 1)
    selector = np.concatenate(selector)
    stlPoints = np.concatenate((Apoints, Bpoints, Cpoints))
    stlPoints = stlPoints[selector, :]

    return stlPoints, Apoints, Bpoints, Cpoints


def get_euler_angles(M, n_gamma=4):
    '''
    Returns the zyz Euler angles with shape (M, 3) for the defined number of orientations M.
    (intrinsic Euler angles in the zyz convention)
    '''
    if M == 1:
        zyz = np.array([[0, 0, 0]])
    elif M == 2:
        zyz = np.array([[0, 0, 0], [180, 0, 0]])
    elif M == 4:  # Implement Klein's four group see Worrall and Brostow 2018
        zyz = np.array([[0, 0, 0], [180, 0, 0], [0, 180, 0], [180, 180, 0]])
    elif M == 8:  # Test of theta and phi.
        zyz = np.array(
            [[0, 0, 0], [0, 45, 315], [0, 90, 270], [0, 135, 225], [0, 180, 180], [0, 225, 135], [0, 270, 90],
             [0, 315, 45]])
    elif M == 24:  # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz = np.array([[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270],
                        [0, 90, 0], [0, 90, 90], [0, 90, 180], [0, 90, 270],
                        [0, 180, 0], [0, 180, 90], [0, 180, 180], [0, 180, 270],
                        [0, 270, 0], [0, 270, 90], [0, 270, 180], [0, 270, 270],
                        [90, 90, 0], [90, 90, 90], [90, 90, 180], [90, 90, 270],
                        [90, 270, 0], [90, 270, 90], [90, 270, 180], [90, 270, 270]
                        ])

    else:
        # Parametrized uniform triangulation of 3D circle/sphere:

        i = np.log(M-8)/np.log(n_gamma)-2
        # No need for stlPoints AND A, B, C
        stlPoints, _, _, _ = sphereTriangulation(M,n_gamma)
        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.

        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.
        alpha, beta = change_vars(np.swapaxes(stlPoints,0,1)) # The Euler angles alpha and beta are respectively theta and phi in spherical coord.



        n_gamma = min(n_gamma, max(M//np.unique(stlPoints, axis=0).shape[0],1))


        step_gamma = 2*np.pi/n_gamma
        gamma2 = np.tile(np.linspace(0,2*np.pi-step_gamma,n_gamma),alpha.shape[0])
        alpha2 = np.repeat(alpha,n_gamma)
        beta2 = np.repeat(beta,n_gamma)
        zyz2 = np.stack((alpha2,beta2,gamma2),axis=1)*180/np.pi
        zyz2[zyz2<0] = zyz2[zyz2<0]+360
        zyz2 = np.where(np.unique(zyz2,axis=0)==360., 0., np.unique(zyz2,axis=0))
        zyz = np.unique(zyz2,axis=0)
    

    return zyz
