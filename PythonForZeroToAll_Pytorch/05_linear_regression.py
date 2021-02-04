from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])

#1단계: 모델설계(forward)
class Model(nn.Module): #Module의 subclass 인 Model
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

#2단계: loss function and an Optimizer
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') #Mean Square Error 평균제곱오차
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#러닝레이트 0.01  #SGD 알고리즘- 확률적경사하강법이라는 좀더 발전된 경사하강법. batch데이터에 대응하기 더좋음. 모두 optimization의 한종류
#요새는 아담 많이씀  torch.optim.Adam

#3단계 포워드, 로스,백워드, 스텝
# Training loop 
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # 3) Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    #4) 스텝
    optimizer.step()  #torch.optim.SGD(model.parameters()의 모델.파라미터 를 step 에넣어 업데이트함


# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item()) #모델테스트할땐 model에 바로넣거나 model.forward(hour_var)이런식으로 넣음