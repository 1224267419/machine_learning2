from sklearn import svm
def demo0():
    # 创建一个SVM分类器
    cls=svm.SVC()
    x=[[0,0],[1,1]]
    y=[-1,1]
    cls.fit(x,y)
    print(cls.predict([[-2,-2]])) #属于类别1
    print(cls.get_params()) # 获取参数


if __name__ == '__main__':
    demo0()