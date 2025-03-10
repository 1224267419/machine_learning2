import numpy as np
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def demo0():
    # 创建一个SVM分类器
    cls=svm.SVC()
    x=[[0,0],[1,1]]
    y=[-1,1]
    cls.fit(x,y)
    print(cls.predict([[-2,-2]])) #属于类别1
    print(cls.get_params()) # 获取参数
def mnist_demo():
    # 查看具体图像
    def show_num(num):
        num = num.reshape(28, 28)
        plt.imshow(num)
        plt.axis('off')  # 把x轴关掉
        plt.show()
    #特征降维
    import time
    from sklearn.decomposition import PCA
    # 多次使用pca，确定最后的最优模型
    def n_components_analysis(n, x_train, y_train, x_val, y_val):
        # 记录开始时间
        start = time.time()

        # pca降维实现
        pca = PCA(n_components=n) #实例化转换器
        #整数为特征数量,浮点数为方差占比,'mle'将自动选取主成分个数n，使得满足所要求的方差百分比
        print('特征降维，传递的参数为：{}'.format(n))
        pca.fit(x_train)

        # 在训练集和测试集进行降维
        x_train_pca = pca.transform(x_train)
        x_val_pca = pca.transform(x_val)
        print('降维后的训练集维度：{}'.format(x_train_pca.shape))

        # 利用svc进行训练
        print('开始使用svc进行训练')
        ss = svm.SVC()
        ss.fit(x_train_pca, y_train)

        # 获取accuracy
        accuracy = ss.score(x_val_pca, y_val)

        # 记录结束时间
        end = time.time()
        print('准确率是:{}，消耗时间是:{}s'.format(accuracy, int(end - start)))

        return accuracy

    train=pd.read_csv('./mnist_dataset/train.csv')
    # print(train.head())
    #0列为label，1-785列为特征
    x=train.iloc[:,1:].values #取1-785列(特征)
    y=train.iloc[:,0].values #取0列（标签）
    # print(x.shape,y.shape)

    # show_num(x[999]) #查看第n个图像
    x=x/255.0 #归一化

    #数据集分割
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



    # print(n_components_analysis('mle', x_train, y_train, x_test, y_test))
    #mle保留了694个特征,消耗266s
    # print(n_components_analysis(0.95, x_train, y_train, x_test, y_test))
    # score=[]
    # for i in np.linspace(0.7,0.9, 5):
    #     acc=n_components_analysis(i, x_train, y_train, x_test, y_test)
    #     print("保留{}的数据量时,svm的准确率为{}".format(i,acc))
    #     score.append(acc)
    #     #使用的特征越多,准确率不一定越高,但时间会变长
    # plt.plot(np.linspace(0.7,0.9, 5),np.array(score))
    # plt.show()
    #根据图片找出训练时间和acc较为合适的值即可,0.85
    print(n_components_analysis(0.85, x_train, y_train, x_test, y_test))


if __name__ == '__main__':
    # demo0()
    mnist_demo()