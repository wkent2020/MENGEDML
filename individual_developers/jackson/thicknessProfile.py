import numpy as np 
from scipy.spatial.transform import Rotation as R
import scipy.integrate as integrate
import matplotlib.pyplot as plt 


'''
This framework should work for shapes that are not ellipsoids, so long as 
it is posssible to write a close form boolean that checks whether coordinate lies inside the shape
'''

#Ellipsoid parameters
semiaxes = np.array([4, 2, 3])
translation = np.array([0, 0, 1])
rotationAxis = np.array([1, -1, -1])
r = R.from_rotvec(np.pi/2 * rotationAxis)

def ellipsoid(x, y, z, semiaxes):
    '''
    Characterisitic function of an ellipsoid with semiaxes given and aligned along x,y,z axes
    '''
    #Avoid divide by zero
    if semiaxes.min() == 0:
        print("Invalid parameters")
        return None
    #Check if point lies inside ellipsoid
    if np.sum(np.square(np.array([x,y,z])/semiaxes)) <= 1:
        return 1
    else:
        return 0

#Coordinate transformations
def transform(x, y, z, rotation, translation):
    '''
    Transform the given coordinate by applying the rotation matrix from the rotation object followed by the translation
    '''
    transform = r.apply(np.array([x,y,z])) + translation
    return transform[0], transform[1], transform[2]

def inverseTransform(x, y, z, rotation, translation):
    '''
    Apply the reverse transformation on the given coordinate by undoing the translation and then undoing the rotation
    '''
    transform = r.inv().apply(np.array([x,y,z]) - translation)
    return transform[0], transform[1], transform[2]

def generalEllipsoid(x, y, z, semiaxes, rotation, translation):
    '''
    Characteristic function for a general ellipsoid
    First transform back to principal axes then check if coordinate lies inside untransformed ellipsoid
    '''
    xNew, yNew, zNew = inverseTransform(x,y,z, rotation, translation)
    return ellipsoid(xNew, yNew, zNew, semiaxes)

# Set up coordinate grid

projNum = 50
maxVal = np.sqrt(np.sum(np.square(semiaxes)))
maxCoordinate = translation + maxVal
minCoordinate = translation - maxVal 

#xCoordinates = np.linspace(minCoordinate[0], maxCoordinate[0], projNum)
yCoordinates = np.linspace(minCoordinate[1], maxCoordinate[1], projNum)
zCoordinates = np.linspace(minCoordinate[2], maxCoordinate[2], projNum)
yy, zz = np.meshgrid(yCoordinates, zCoordinates)
yzCoordinates = np.vstack([yy.ravel(), zz.ravel()]).T
#totalCoordinates = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# Projection along x axis
thicknesses = []
for yz in yzCoordinates:
    #Integration at each grid point along x-axis
    projectedThickness, err = integrate.quad(generalEllipsoid, minCoordinate[0], maxCoordinate[0], args=(yz[0], yz[1], semiaxes, r, translation))
    thicknesses.append(projectedThickness)

thicknesses = np.reshape(np.array(thicknesses), yy.shape)

#Plotting script
fig, ax = plt.subplots()
h = ax.contourf(yCoordinates, zCoordinates, thicknesses)
cbar = fig.colorbar(h, ax=ax)
cbar.set_label("Projected x Thickness")
ax.set_title("Ellipsoid with axes " + str(semiaxes[0]) + "," + str(semiaxes[1]) + "," + str(semiaxes[2]))
ax.set_xlabel("y Coordinate")
ax.set_ylabel("z Coordinate")
fig.suptitle("Rotation about (" + str(rotationAxis[0]) + "," + str(rotationAxis[1]) + "," + str(rotationAxis[2]) + 
                "); Translation by " + str(translation[0]) + "," + str(translation[1]) + "," + str(translation[2]))
plt.show()