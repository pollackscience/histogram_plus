import numpy as np
import matplotlib.pyplot as plt
from histogram_plus import hist


def test_simple_hist1(cmdopt, data_gen1):
    output = hist(data_gen1[0])
    if cmdopt == "generate":
        with open('answers/answers_simple_hist1.npz', 'wb') as f:
            np.savez(f, bc=output[0], be=output[1])
        plt.show()
    elif cmdopt == "test":
        answers = np.load('answers/answers_simple_hist1.npz')
        assert(np.all(output[0] == answers['bc']))
        assert(np.all(output[1] == answers['be']))


def test_simple_hist2(cmdopt, data_gen1):
    output = hist(data_gen1[0], weights=data_gen1[2], bins=20, normed=True, color='red')
    if cmdopt == "generate":
        with open('answers/answers_simple_hist2.npz', 'wb') as f:
            np.savez(f, bc=output[0], be=output[1])
        plt.show()
    elif cmdopt == "test":
        answers = np.load('answers/answers_simple_hist2.npz')
        assert(np.all(output[0] == answers['bc']))
        assert(np.all(output[1] == answers['be']))


def test_blocks_hist(cmdopt, data_gen1):
    output = hist(data_gen1[0], bins='blocks', scale='binwidth', color='green')
    if cmdopt == "generate":
        with open('answers/answers_blocks_hist.npz', 'wb') as f:
            np.savez(f, bc=output[0], be=output[1])
        plt.show()
    elif cmdopt == "test":
        answers = np.load('answers/answers_blocks_hist.npz')
        assert(np.all(output[0] == answers['bc']))
        assert(np.all(output[1] == answers['be']))
