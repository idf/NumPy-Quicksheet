import math
class Plotter(object):
    def plot(self, L_lst):
        plt.figure("Log_likelihood")
        plt.ylabel("L: Log-likelihood")
        plt.xlabel("N: number of iterations")
        plt.plot([i+1 for i in xrange(len(L_lst))], L_lst)
        plt.show()
