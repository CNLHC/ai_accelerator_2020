import matplotlib.pyplot as plt
import matplotlib


def main():
    def toFloat(x):
        try:
            return float(x)
        except:
            return None
    li = [toFloat(x) for x in open("./assets/loss",'r').readlines()]
    li = [x for x in li if x is not None]
    plt.figure(figsize=(35,20))
    matplotlib.rcParams.update({'font.size': 32})
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Samples")
    plt.plot(li[::5],linewidth=3,color="k")
    plt.savefig("loss.eps")
    
main()