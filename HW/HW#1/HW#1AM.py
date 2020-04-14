#import module
import matplotlib.pyplot as plt
import numpy as np
#frequency
f_c = 50 #50Hz
f_1 = 2 #2Hz
f_2 = 5 #2Hz
#time
t = np.linspace(0,1,1000)
#carrier
carrier = np.cos(2*np.pi*f_c*t)
#signal
S1 = np.cos(2*np.pi*f_1*t + 0.5*np.pi) #degree 90
S2 = 0.5*np.cos(2*np.pi*f_2*t + 0.25*np.pi)  #degree 45
signal = S1 + S2
#am = signal*carrier
am = (signal+2)*carrier
am = am + 0.8*np.random.rand(1000)
#plot
plt.subplot(311)
plt.plot(t,carrier,label=str('carrier'))
plt.legend(loc="lower right")  
plt.subplot(312)
plt.plot(t,signal,label=str('signal'))
plt.legend(loc="lower right")  
plt.subplot(313)
plt.plot(t,am,label=str('AM'))
plt.legend(loc="lower right")  
plt.show()
