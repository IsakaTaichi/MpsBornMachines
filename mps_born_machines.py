import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import copy
import matplotlib.pyplot as plt

torch.manual_seed(100)
np.random.seed(100)
'''
mps born machines

'''
class mps_born_machines(nn.Module):
    def __init__(self, x_num=9, b=20, lr=0.01):
        super().__init__()
        self.x_num= x_num
        self.b = b
        self.lr=lr
        #テンソル
        self.register_parameter(name='AL', param=nn.Parameter(torch.Tensor(np.random.randn(2, b)*1))) # 左テンソル
        #self.register_parameter(name='AL', param=nn.Parameter(torch.Tensor(np.random.uniform(0,0.01,(2, b)))))
        #3足テンソル
        for i in range(2,x_num):
            self.register_parameter(name=f'A{i}', param=nn.Parameter(torch.Tensor(np.random.randn(b, 2, b))))
            #self.register_parameter(name=f'A{i}', param=nn.Parameter(torch.Tensor(np.random.uniform(0,0.01,(b, 2, b)))))
        #右テンソル
        self.register_parameter(name='AR', param=nn.Parameter(torch.Tensor(np.random.randn(b, 2))))
        #self.register_parameter(name='AR', param=nn.Parameter(torch.Tensor(np.random.uniform(0,0.01,(b, 2)))))
        #print([*self.parameters()])
        
        #optimizer設定
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
    
    '''
    入力データの生成確率を計算
    '''
    def psi(self, x):
        I0 = torch.tensor([[1, 0]], dtype=torch.float32)
        I1 = torch.tensor([[0, 1]], dtype=torch.float32)
        #縮約
        I = torch.where((x[:, 0] == 0)[:, None], I0, I1)
        p = torch.einsum("ia,xi->xa",self.get_parameter('AL'),I)
        for i in range(2,self.x_num):
            I = torch.where((x[:, i-1] == 0)[:, None], I0, I1)
            p = torch.einsum("xa,aib,xi->xb",p,self.get_parameter(f'A{i}'),I)
        I = torch.where((x[:, self.x_num-1] == 0)[:, None], I0, I1)
        p = torch.einsum("xa,ai,xi->x",p,self.get_parameter('AR'),I)
        
        return p
    '''
    データの生成
    step数だけサンプリングを行う
    '''
    def generate(self,batch_x0,step):
        for j in range(step): #サンプリング数           
            #乱数
            r = np.random.randint(0, batch_x0.shape[1], batch_x0.shape[0]) #データの1ビットだけを反転
            #次の状態の候補
            batch_x1 = copy.deepcopy(batch_x0) # 次の候補にコピー
            
            for i in range(batch_x0.shape[0]): #バッチデータ数
                if batch_x1[i,r[i]] == 0:
                    batch_x1[i,r[i]] = 1
                    
                elif batch_x1[i,r[i]] == 1:
                    batch_x1[i,r[i]] = 0
            
            # 状態の確率計算
            psi0 = self.psi(batch_x0)**2 # 入力データの確率
            psi1=  self.psi(batch_x1)**2 # 生成候補の確率
            
            
            #状態遷移
            for i in range(batch_x0.shape[0]): #バッチデータ数
                if psi1[i]/psi0[i] < 1: #確率の比が1未満なら確率rで遷移
                    r = torch.rand(1)
                    if psi1[i]/psi0[i] >= r:
                        batch_x0[i,:] = batch_x1[i,:] #遷移
                    else:
                        batch_x0[i,:] = batch_x0[i,:] #遷移しない
                        
                elif psi1[i]/psi0[i] >= 1:
                    batch_x0[i,:] = batch_x1[i,:] #遷移
       
        return batch_x0
    '''
    学習
    '''
    def learn(self,batch_x0):
        batch = batch_x0.shape[0] #バッチ数
        self.optimizer.zero_grad()
        
        #確率計算
        psi = self.psi(batch_x0)**2
            
        #分配関数計算
        z = torch.einsum("ij,ik->jk",self.get_parameter('AL'),self.get_parameter('AL'))
        for i in range(2,self.x_num):
            z = torch.einsum("jk,jil,kim->lm",z,self.get_parameter(f'A{i}'),self.get_parameter(f'A{i}'))
        z = torch.einsum("ij,ik,jk->...",z,self.get_parameter('AR'),self.get_parameter('AR'))
        
        # loss 計算 負の対数尤度
        loss = -torch.mean(torch.log(psi/z)) 
        #print(loss)
        
        # optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().numpy()