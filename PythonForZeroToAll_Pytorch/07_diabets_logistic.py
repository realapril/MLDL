from torch import nn, optim, from_numpy
import numpy as np
import torch.nn.functional as F

xy = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])#맨뒤만 y 값
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        3레이어- deep하게 해보기 
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6) 
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        #이 예제는 relu로하든 시그모이드로하든 비슷. 보통 10레이어 이상부터 vanising gradient문제가 생겨서 그런듯.
        # out1 = F.relu(self.l1(x))
        # out2 = F.relu(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
       
        return y_pred


# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')#BCELoss 가 크로스 엔트로피
optimizer = optim.SGD(model.parameters(), lr=0.1)#SGD 알고리즘- 확률적경사하강법이라는 좀더 발전된 경사하강법. batch데이터에 대응하기 더좋음.  모두 optimization의 한종류
 
# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data) #메뉴얼 데이터 피드->나중엔 배치쓸거임

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()