




实现手写numpy实现线性回归
2023/12/8
```python
import random
import matplotlib.pyplot as plt
import numpy as np

# generate the data
def synthetic_data(w,b,num_examples):
    X = np.random.normal(scale=1,size=(num_examples,len(w)))
    y = np.dot(X,w) + b
    y += np.random.normal(scale=0.001,size=y.shape)
    return X,y.reshape((-1,1))

# iterate on the data
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=indices[i:min(i+batch_size,num_examples)]
        yield features[batch_indices],labels[batch_indices]


class LinearRegression:
    def __init__(self):
        self.w=None
        self.b=None

    def init_wb(self,num_of_features):
        self.w=np.random.normal(0,0.1,[num_of_features, 1])
        self.b=np.zeros(1)
    def train(self,X,y,batch_size=10,method='mbgd',learning_rate=0.0001,num_epoches=1000):
        self.init_wb(X.shape[1])
        loss_history=[]
        if method=='mbgd':#小批量梯度下降
            for epoch in range(num_epoches):
                for x_train,y_train in data_iter(batch_size,X,y):
                    loss,gra_w,gra_b=self.loss_and_gradient(x_train,y_train,batch_size)
                    self.w-=learning_rate*gra_w
                    self.b-=learning_rate*gra_b
                    loss_history.append(loss)
        elif method=='sgd':#随机梯度下降
            for epoch in range(num_epoches):
                indices = list(range(len(X)))
                random.shuffle(indices)
                for i in indices:
                    loss, gra_w, gra_b = self.loss_and_gradient(X[i], y[i], 1)

                    self.w -= learning_rate * gra_w
                    self.b -= learning_rate * gra_b
                    loss_history.append(loss)

        else: #普通梯度下降
            for epoch in range(num_epoches):
                loss,gra_w,gra_b=self.loss_and_gradient(X,y,batch_size)

                self.w-=learning_rate*gra_w
                self.b-=learning_rate*gra_b
                loss_history.append(loss)
        return loss_history


    def loss_and_gradient(self,x_train,y_train,batch_size):
        y1=x_train.dot(self.w)+self.b
        diff=y1-y_train
        loss=1.0/2*np.sum(diff * diff)
        if batch_size==1:
            gra_w=x_train.reshape(-1, 1)*diff
            gra_b=diff

        else:
            gra_w=np.dot(x_train.T,diff)/batch_size
            gra_b=np.sum(diff)/batch_size
        return loss,gra_w,gra_b


    def predict(self,X):
        return np.dot(X,self.w)+self.b


if __name__ == '__main__':
    W=np.array([2,-3.4]).T
    b=4.2
    num_ex=1000
    features,labels=synthetic_data(W,b,num_ex)
    model=LinearRegression()
    loss_history=model.train(features,labels)
    print(loss_history[-1])
    print(model.w,model.b)
    #画图
    plt.scatter(features[:,1],labels)
    plt.plot(features[:,1],model.predict(features))
    plt.show()



```

遇到的问题：
1. 我把一个*data_iter*写错了，导致在小批量梯度下降的时候出现问题，下次要看仔细。
2. 在随机梯度下降的时候，发现x_train.T和x_train是一样的，并没有实现转置效果，是由于x_train是一维函数，所以只能将其拉伸成二维*x_train.reshape(-1, 1)*，不能转置。