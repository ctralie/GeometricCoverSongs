import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gf1d
import scipy.io as sio
import scipy.misc
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.image import Image

#Get smoothed curvature vectors up to a particular order
def getCurvVectors(X, MaxOrder, sigma):
    XSmooth = gf1d(X, sigma, axis=0, order = 0)
    Vel = gf1d(X, sigma, axis=0, order = 1, mode = 'nearest')
    VelNorm = np.sqrt(np.sum(Vel**2, 1))
    VelNorm[VelNorm == 0] = 1
    Curvs = [XSmooth, Vel]
    for order in range(2, MaxOrder+1):
        Tors = gf1d(X, sigma, axis=0, order=order, mode = 'nearest')
        for j in range(1, order):
            #Project away other components
            NormsDenom = np.sum(Curvs[j]**2, 1)
            NormsDenom[NormsDenom == 0] = 1
            Norms = np.sum(Tors*Curvs[j], 1)/NormsDenom
            Tors = Tors - Curvs[j]*Norms[:, None]
        Tors = Tors/(VelNorm[:, None]**order)
        Curvs.append(Tors)
    return Curvs

#Get zero crossings estimates from all curvature/torsion
#measurements by using the dot product
def getZeroCrossings(Curvs):
    Crossings = []
    for C in Curvs:
        dots = np.sum(C[0:-1, :]*C[1::, :], 1)
        cross = np.arange(len(dots))
        cross = cross[dots < 0]
        Crossings.append(cross)
    return Crossings

def getScaleSpaceImages(X, MaxOrder, sigmas):
    print("TODO")

#A class for doing animation of curvature scale space images
#for 2D/3D curves
class CSSAnimation(animation.TimedAnimation):
    def __init__(self, X, sigmas):
        self.X = X
        self.sigmas = sigmas
        self.SSImage = np.zeros((len(sigmas), X.shape[0]))
        
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        #Original curve
        self.origCurve = Line2D([], [], color='black')
        ax1.add_line(self.origCurve)
        self.origCurve.set_data(X[:, 0], X[:, 1])
        ax1.set_title('Original Curve')
        
        #Smoothed Curve
        self.smoothCurve = ax2.scatter([], [], [], c=[])
        ax2.hold(True)
        self.smoothCurveInfl = ax2.scatter([], [], 20, 'r')
        
        #Curvature magnitude plot
        self.curvMagPlot = Line2D([], [], color='black')
        
        #Scale space image
        
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)
    
    def _draw_frame(self, framedata):
        i = framedata
        print("i = ", i)
        
        Curvs = getCurvVectors(self.X, 2, self.sigmas[i])
        Crossings = getZeroCrossings(Curvs)
        XSmooth = Curvs[0]
        Curv = Curvs[2]
        CurvMag = np.sqrt(np.sum(Curv**2, 1)).flatten()
        
        self.smoothCurve.set_offsets(XSmooth[:, 0:2])
        self.smoothCurve.set_array(CurvMag)
        XInflection = XSmooth[Crossings[2], :]
        self.smoothCurveInfl.set_offsets(XInflection[:, 0:2])
        
        curvMagPlot.set_data(np.arange(len(CurvMag)), CurvMag)
        
        SSImage[-i-1, Crossings[2]] = 1
        SSImageRes = scipy.misc.imresize(SSImage, [300, 300])
        SSImageRes[SSImageRes > 0] = 1
        SSImageRes = 1-SSImageRes
        ssImagePlot.set_data(SSImageRes)
    
    def new_frame_seq(self):
        print("len(self.sigmas) = ", len(self.sigmas))
        return iter(range(len(self.sigmas)))
    
    def _init_draw(self):
        print("TODO")

if __name__ == '__main__':
    X = sio.loadmat('hmm_gpd/plane_data/Class1_Sample2.mat')['x']
    X = np.array(X, dtype=np.float32)
    sigmas = np.linspace(1, 200, 200)
    
    ani = CSSAnimation(X, sigmas)
    plt.show()
