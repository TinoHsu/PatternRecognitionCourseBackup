import matplotlib.pyplot as plt
import numpy as np

# given condition
bit_stream = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
f_low = 2 #2Hz
f_high = 5 #5Hz
#bit_stream_next = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

# decide by yourself
total_point = 1000

# modulation function
def FSKmodulation(bit_stream, f_low, f_high, total_point):
    # basic setting
    bit_stream_length = bit_stream.shape[0]
    pt_per_bit = int(total_point/bit_stream_length)
    t = np.linspace(0, bit_stream_length, total_point)
    # generate two carriers
    carrier_high = np.cos(2*np.pi*f_high*t)
    carrier_low = np.cos(2*np.pi*f_low*t)
    # multiplexer
    for i in range(bit_stream_length):
        # according to bit_stream state choose carrier
        if bit_stream[i] == 0:
            fsk_now = carrier_low[i*pt_per_bit:(i+1)*pt_per_bit]
            signal_now = np.zeros(pt_per_bit)
        if bit_stream[i] == 1:
            fsk_now = carrier_high[i*pt_per_bit:(i+1)*pt_per_bit]
            signal_now = np.ones(pt_per_bit)
        # link sequence
        if (i == 0):
            fsk_past = fsk_now.copy()
            signal_past = signal_now.copy()
        else:    
            fsk_past = np.concatenate((fsk_past, fsk_now),axis=0)
            signal_past = np.concatenate((signal_past, signal_now),axis=0)
    
    return signal_past, carrier_high, carrier_low, fsk_past, t

# execution
signal, carrier_high, carrier_low, fsk, t = FSKmodulation(bit_stream, f_low, f_high, total_point)

# plot results
plt.subplot(411)
plt.plot(t,signal,label=str('signal'))
plt.legend(loc="lower right")  
plt.subplot(412)
plt.plot(t,carrier_high,label=str('carrier high'))
plt.legend(loc="lower right")  
plt.subplot(413)
plt.plot(t,carrier_low,label=str('carrier low'))
plt.legend(loc="lower right")  
plt.subplot(414)
plt.plot(t,fsk,label=str('FSK'))
plt.legend(loc="lower right")  
plt.show()