import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd


def plot_train(gen_num, train_time):
    plt.plot(gen_num, train_time, '-*')
    plt.xlabel('Generation number')
    plt.ylabel('Training time')
    plt.title('Generation over training time')
    plt.margins(0.05)
    plt.grid(True)
    plt.show()


def plot_avgFitness(gen_num, avg_fitness):
    plt.plot(gen_num, avg_fitness, '-*')
    plt.xlabel('Generation number')
    plt.ylabel('Average fitness')
    plt.title('Generation over average fitness')
    plt.margins(0.05)
    plt.grid(True)
    plt.show()


def plot_maxFitness(gen_num, max_fitness):
    plt.plot(gen_num, max_fitness, '-*')
    plt.xlabel('Generation number')
    plt.ylabel('Max fitness')
    plt.title('Generation over average fitness')
    plt.margins(0.05)
    plt.grid(True)
    plt.show()

def main_plot():
    df = pd.read_csv('stats.txt', header=None)
    raw_data = df[0].values #use this method to parse data since the csv file has empty header
    print raw_data
    raw_data = [i.split(' ') for i in raw_data]
    gen_num, train_time, avg_fitness, max_fitness = [], [], [], []
    plt.close('all')
    for each_gen_data in raw_data:
        gen_num.append(int(each_gen_data[0]))
        train_time.append(float(each_gen_data[1]))
        avg_fitness.append(float(each_gen_data[2]))
        max_fitness.append(float(each_gen_data[3]))

    plot_train(gen_num, train_time)
    plot_avgFitness(gen_num, avg_fitness)
    plot_maxFitness(gen_num, max_fitness)

if __name__ == '__main__':
    main_plot()
