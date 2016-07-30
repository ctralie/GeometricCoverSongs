#Show how much raw C saves time wise with a double for loop
import SequenceAlignment as SA
import _SequenceAlignment as SAC
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    D = np.random.rand(1000, 1000)
    D = D < 0.1
    D = np.array(D, dtype='double')
    
    start = time.time()
    ans = SAC.swalignimpconstrained(D)
    end = time.time()
    print "Time elapsed C: %g seconds, ans = %g"%(end-start, ans)
    
    start = time.time()
    ans = SA.swalignimpconstrained(D)[0]
    end = time.time()
    print "Time elapsed raw python: %g seconds, ans = %g"%(end - start, ans)
