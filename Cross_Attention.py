import numpy as np

class CrossAttention:
    def __init__(self,X,dimension):
        self.row,self.col=X.shape
        self.head_dim=dimension
        self.weight_K=self.define_weights(self.col,self.head_dim)
        self.weight_Q=self.define_weights(self.col,self.head_dim)
        self.weight_V=self.define_weights(self.col,self.head_dim)
       
    def define_weights(self,x,y):
        lmt=np.sqrt(6/(x+y))
        arr=np.random.uniform(-lmt,lmt,size=(x,y))
        return arr
   
    def Cross_Softmax(self,K,Q,V):
        self.K=K
        self.Q=Q
        self.V=V
        K_transpose=np.transpose(K)
        self.QKT=np.dot(Q,K_transpose)
        self.QKT_scaled=(self.QKT/np.sqrt(self.head_dim))
       
        self.soft=np.zeros_like(self.QKT_scaled)
        for i in range(self.QKT_scaled.shape[0]):
            each_row=self.QKT_scaled[i]
            exp_row=np.exp(each_row-np.max(each_row)) #-np.max(each_row)
            sum_ex=np.sum(exp_row)
            self.soft[i]=exp_row/(sum_ex+1e-8)
           
        self.finalProduct=np.dot(self.soft,V)
        return self.finalProduct

    def intializeCrossAttention(self,Key,Query):
        self.Key=Key
        self.Query=Query
        K=np.dot(self.Key,self.weight_K)
        Q=np.dot(self.Query,self.weight_Q)
        V=np.dot(self.Key,self.weight_V)
        result_att=self.Cross_Softmax(K,Q,V)
        return result_att

    def backpropagate(self,d_output):
        self.d_finalProduct=d_output
        self.d_soft=np.dot(self.d_finalProduct,self.V.T)
        self.d_V=np.dot(self.soft.T,self.d_finalProduct)
        a,b=self.soft.shape
        self.d_QKT_scaled=np.zeros_like(self.soft)
        for i in range(a):
            for j in range(b):
                self.d_QKT_scaled[i,j]=self.soft[i,j]*(self.d_soft[i,j]-np.sum(self.d_soft[i,:]*self.soft[i,:]))
       
        self.d_QKT=self.d_QKT_scaled/np.sqrt(self.head_dim)
       
        self.d_Q=np.dot(self.d_QKT,self.K)
        self.d_K=np.dot(self.d_QKT.T,self.Q)
       
        self.d_weight_Q=np.dot(self.Query.T,self.d_Q)
        self.d_weight_K=np.dot(self.Key.T,self.d_K)
        self.d_weight_V=np.dot(self.Key.T,self.d_V)
       
        self.d_Query=np.dot(self.d_Q,self.weight_Q.T)
        self.d_Key=(np.dot(self.d_K,self.weight_K.T)+np.dot(self.d_V,self.weight_V.T))
       
        return self.d_Key,self.d_Query
   
    def update(self,lr):
        self.weight_Q=self.weight_Q-lr*(self.d_weight_Q)
        self.weight_K=self.weight_K-lr*(self.d_weight_K)
        self.weight_V=self.weight_V-lr*(self.d_weight_V)

       
             
       
                   
