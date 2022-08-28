import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("whitegrid") #横坐标有标线，纵坐标没有标线，背景白色
sns.set_style("darkgrid") #默认，横纵坐标都有标线，组成一个一个格子，背景稍微深色
sns.set_style("dark")#背景稍微深色，没有标线线
sns.set_style("white")#背景白色，没有标线线
sns.set_style("ticks") #xy轴都有非常短的小刻度
sns.despine(offset=30,left=True) #去掉上边和右边的轴线，offset=30表示距离轴线（x轴）的距离,left=True表示左边的轴保留



def sinplot(flip=1): #自定义一个函数
	x = np.linspace(0,14,100) #0-14取100个点

	for i in range(1,7): #画了7条线
		plt.plot(x,np.sin(x + i *0.5) * (7 - i) * flip) #sin函数
	plt.show()

sinplot()
