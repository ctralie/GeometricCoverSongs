import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gf1d
import scipy.io as sio
import scipy.misc
import matplotlib.animation as animation

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
class CSSAnimator(animation.FuncAnimation):
    def __init__(self, fig, X, sigmas):
        self.fig = fig
        self.X = X
        self.sigmas = sigmas
        self.SSImage = np.zeros((len(sigmas), X.shape[0]))
        
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        [self.ax1, self.ax2, self.ax3, self.ax4] = [ax1, ax2, ax3, ax4]
        
        #Original curve
        self.origCurve, = ax1.plot(X[:, 0], X[:, 1])
        ax1.set_title('Original Curve')
        
        #Smoothed Curve
        self.smoothCurve = ax2.scatter([], [], [], c=[])
        ax2.hold(True)
        self.smoothCurveInfl = ax2.scatter([], [], 20, 'r')
        ax2.set_xlim([np.min(X[:, 0]), np.max(X[:, 0])])
        ax2.set_ylim([np.min(X[:, 1]), np.max(X[:, 1])])
        
        #Curvature magnitude plot
        self.curvMagPlot, = ax3.plot([])
        
        #Scale space image plot
        self.ssImagePlot = ax4.imshow(self.SSImage, interpolation = 'none', aspect = 'auto')
        
        #Setup animation thread
        animation.FuncAnimation.__init__(self, fig, func = self._draw_frame, frames = len(sigmas), interval = 50)
    
    def _draw_frame(self, i):
        print i
        
        Curvs = getCurvVectors(self.X, 2, self.sigmas[i])
        Crossings = getZeroCrossings(Curvs)
        XSmooth = Curvs[0]
        Curv = Curvs[2]
        
        #Draw Curvature Magnitude Plot
        CurvMag = np.sqrt(np.sum(Curv**2, 1)).flatten()
        self.curvMagPlot.set_xdata(np.arange(len(CurvMag)))
        self.curvMagPlot.set_ydata(CurvMag)
        if i == 0:
            self.ax3.set_xlim([0, len(CurvMag)])
            self.ax3.set_ylim([0, np.max(CurvMag)])
        
        #Draw smoothed curve with inflection points
        self.smoothCurve.set_offsets(XSmooth[:, 0:2])
        self.smoothCurve.set_array(CurvMag)
        self.smoothCurve.__sizes = 20*np.ones(XSmooth.shape[0])
        XInflection = XSmooth[Crossings[2], :]
        self.smoothCurveInfl.set_offsets(XInflection[:, 0:2])
        
        self.SSImage[-i-1, Crossings[2]] = 1
        #Do some rudimentary anti-aliasing
        SSImageRes = scipy.misc.imresize(self.SSImage, [300, 300])
        SSImageRes[SSImageRes > 0] = 1
        SSImageRes = 1-SSImageRes
        self.ssImagePlot.set_array(SSImageRes)

if __name__ == '__main__':
    X = sio.loadmat('hmm_gpd/plane_data/Class1_Sample2.mat')['x']
    X = np.array(X, dtype=np.float32)
    sigmas = np.linspace(1, 200, 200)
    
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    
    fig = plt.figure()
    ani = CSSAnimator(fig, X, sigmas)
    ani.save("out.mp4", writer = writer)
