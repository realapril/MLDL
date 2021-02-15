import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage


## 1. Hyper Parameters #########################################################
learning_rate = 0.001
training_epochs = 5 #15
batch_size = 100

tf.random.set_seed(777)

## 2. Data Augmentation #########################################################
def data_augmentation(images, labels):
    aug_images = []
    aug_labels = []    
    
    for x, y in zip(images, labels):        
        #원본데이터 저장
        aug_images.append(x) 
        aug_labels.append(y)
        
        #이미지 중간값 저장해두는데 rotation, shift 등 하고나면 빈공간채우기위해
        bg_value = np.median(x)
        for _ in range(4):#데이터 4배만들어서 원래+4배=총 5배 데이터됨
            angle = np.random.randint(-15, 15, 1)
            rot_img = ndimage.rotate(x, int(angle), reshape=False, cval=bg_value)
            
            shift = np.random.randint(-2, 2, 2)
            shift_img = ndimage.shift(rot_img, shift, cval=bg_value)            
            
            aug_images.append(shift_img)
            aug_labels.append(y)
    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels)
    return aug_images, aug_labels

## 3. MNIST Dataset, data pipelining #########################################################
mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, train_labels = data_augmentation(train_images, train_labels)

train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.
#빠진 채널 넣는부분. 디멘션 하나 끝에(-1) 추가해주세요
train_images = np.expand_dims(train_images, axis=-1) 
test_images = np.expand_dims(test_images, axis=-1)

#원핫 인코딩
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)    

#데이터 잘라서 공급하는부분
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
                buffer_size=500000).batch(batch_size) #데이터 늘어서 버퍼도 키워줌- 데이터셋 보통 몰려있기때문에 충분히 섞기위해
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
##########################################################################

## 4. NN Model 만들기 #########################################################
class ConvBNRelu(tf.keras.Model): #배치놈 하는 과정넣느라 추가함
    #conv->배치놈->렐루
    def __init__(self, filters, kernel_size=3, strides=1, padding='SAME'):
        super(ConvBNRelu, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                                        padding=padding, kernel_initializer='glorot_normal')
                                        #kernel_initializer 바꾸나 안바꾸나 비슷하긴함 다 자비에
        self.batchnorm = tf.keras.layers.BatchNormalization()
    def call(self, inputs, training=False):
        layer = self.conv(inputs)
        layer = self.batchnorm(layer)
        layer = tf.nn.relu(layer)
        return layer
class DenseBNRelu(tf.keras.Model):
    def __init__(self, units):
        super(DenseBNRelu, self).__init__()
        self.dense = keras.layers.Dense(units=units, kernel_initializer='glorot_normal')
        self.batchnorm = tf.keras.layers.BatchNormalization()
    def call(self, inputs, training=False):
        layer = self.dense(inputs)
        layer = self.batchnorm(layer)
        layer = tf.nn.relu(layer)
        return layer

class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = ConvBNRelu(filters=32, kernel_size=[3, 3], padding='SAME')        
        self.pool1 = keras.layers.MaxPool2D(padding='SAME')
        self.conv2 = ConvBNRelu(filters=64, kernel_size=[3, 3], padding='SAME')
        self.pool2 = keras.layers.MaxPool2D(padding='SAME')
        self.conv3 = ConvBNRelu(filters=128, kernel_size=[3, 3], padding='SAME')
        self.pool3 = keras.layers.MaxPool2D(padding='SAME')
        self.pool3_flat = keras.layers.Flatten()
        self.dense4 = DenseBNRelu(units=256)
        self.drop4 = keras.layers.Dropout(rate=0.4)
        self.dense5 = keras.layers.Dense(units=10, kernel_initializer='glorot_normal')
    def call(self, inputs, training=False):
        net = self.conv1(inputs)        
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net

#이제 모델이 여러개
models = []
num_models = 5 #앙상블 몇개할건지
for m in range(num_models):
    models.append(MNISTModel())

## 5 los #########################################################
def loss_fn(model, images, labels):
    logits = model(images, training=True) #training=True 면 레이어에서 드랍아웃한게 실행됨
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))    
    return loss
## 6. gradient #########################################################
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)#백 프로퍼게이션
    #로스를 뭘로 미분하냐면 모델에 있는 모든 파라미터로 계산해주세요

## 7. Optimizer #########################################################
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
# global_step = tf.train.get_or_create_global_step()
global_step=tf.compat.v1.train.get_or_create_global_step()
# global_step = tf.Variable(1, name="global_step")
lr_decay = tf.compat.v1.train.exponential_decay(learning_rate, global_step, 
                                      train_images.shape[0]/batch_size*num_models*5, 0.5, staircase=True)
# lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 
#                                       train_images.shape[0]/batch_size*num_models*5, 0.5, staircase=True)
#                                         #3번째 조건: 5번 에폭이지나면(num_models*5) 4번째 인자인 0.5* 만큼 러닝레이트 변경
#                                         #staircase 갑자기 한번에 러닝레이트 변경하는 옵션
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)

## 8. Accuracy #########################################################
def evaluate(models, images, labels):
    predictions = np.zeros_like(labels)#모델 3개에서 나온 결과 종합한것
    for model in models:
        logits = model(images, training=False)
        predictions += logits #아웃풋 계속 더함
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

## 9. checkpoint #########################################################
cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'minst_cnn_seq'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

checkpoints = []
for m in range(num_models):
    checkpoints.append(tf.train.Checkpoint(cnn=models[m]))

## 10. train and validate #########################################################
# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0    
    
    for images, labels in train_dataset:
        for model in models: #3중포문: 모델따라 다르게 학습
            grads = grad(model, images, labels)                
            optimizer.apply_gradients(zip(grads, model.variables))
            loss = loss_fn(model, images, labels)
            avg_loss += loss / num_models
        acc = evaluate(models, images, labels)
        avg_train_acc += acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step
    
    for images, labels in test_dataset:        
        acc = evaluate(models, images, labels)        
        avg_test_acc += acc
        test_step += 1    
    avg_test_acc = avg_test_acc / test_step    

    print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss), 
          'train accuracy = ', '{:.4f}'.format(avg_train_acc), 
          'test accuracy = ', '{:.4f}'.format(avg_test_acc))
    
    
    for idx, checkpoint in enumerate(checkpoints):
        checkpoint.save(file_prefix=checkpoint_prefix+'-{}'.format(idx))

print('Learning Finished!')
