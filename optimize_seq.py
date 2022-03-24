from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from threading import Thread
from multiprocessing import Process


"""
Steps for optimization:
1. For n bit sequence, we take n/4 bits and then create the full sin wave by muxing it
2. To generate the sequence: create bin(2**(n/4)) and then subtract 1 from it in each iteration until it goes to 0. This new sequence will be what we need for operation
"""
bit_length = 64
min_ = 0
max_ = int(2**(bit_length/4)-1)
thd_min = np.zeros(int(2**(bit_length/4)),dtype=np.float16)
# with open('data.txt', 'w') as f:
# 	f.write("Data")

class optimize_sequence:

    def __init__(self, n, init_n=0, end_n=1):
        self.bits = int(n/4)
        self.init_n = init_n
        self.end_n = end_n

    def exec_thread(self):
        for i in range(self.init_n, self.end_n):
            self.num_seq = i
            seq = self.complete_sequence()
            self.get_thd(seq)

    def generate_nth_quarter(self):
        seq = np.zeros(self.bits, dtype = np.int16)
        binary_seq = bin((2**self.bits)-1-self.num_seq)[2:]
        for i in range(len(binary_seq)):
            seq[len(seq)-len(binary_seq)+i] = binary_seq[i]
        return seq

    def complete_sequence(self):
        seq = self.generate_nth_quarter()
        seq_first = np.concatenate((seq,seq[::-1]), axis=None)
        seq_inv = []
        for i in seq_first:
            if i ==1:
                seq_inv.append(0)
            else:
                seq_inv.append(1)
        return np.concatenate((seq_first,seq_inv), axis=None)

    def find_fft(self, seq):
        return np.fft.fft(seq)

    def get_thd(self, seq):
        fft = self.find_fft(seq)
        # print(fft)
        thd_num = np.sum(np.square(np.abs(fft[2:15])))
        thd_den = np.abs(fft[1])
        thd = thd_den/np.sqrt(thd_num) ## Note that this is the inverse of thd
        # print(self.num_seq)
        data = str(self.num_seq)+" "+str(thd)
        print(data)
        with open('data.txt', 'a') as f:
            f.write(data+"\n")
        # thd_min[self.num_seq] = thd

if __name__ == "__main__":

    for i in range(min_, max_, 100000):
        print(i)
        threads = []
        for k in range(i,i+100000,1000):
            if k+1000<max_:
                o = optimize_sequence(bit_length,k,k+1000)
                threads.append(Process(target=o.exec_thread, args=()))
            else:
                o = optimize_sequence(bit_length,k,max_)
                threads.append(Process(target=o.exec_thread, args=()))
        for th in range(0,len(threads),8):
            if th+8<len(threads):
                end = th+8
            else:
                end = len(threads)
            for j in range(th, end):
                threads[j].start()
            for j in range(th,end):
                threads[j].join()
    print("Waiting here")
    print(np.argmax(thd_min))
    obj = optimize_sequence(bit_length, min_, max_)
    # print(obj.complete_sequence())
    arr = obj.complete_sequence()
    plt.stem(np.abs(np.fft.fft(arr)/np.max(np.fft.fft(arr)))[1:])