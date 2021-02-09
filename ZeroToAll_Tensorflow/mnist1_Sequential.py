import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os


## 1. Hyper Parameters #########################################################
learning_rate = 0.001
training_epochs = 15
batch_size = 100

tf.random.set_seed(777)


## 2. MNIST Dataset, data pipelining #########################################################
mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    
#(60000, 28, 28)  (60000)       (10000,28,28)  (10000)   channel 빠져있음
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
                buffer_size=100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
##########################################################################

## 3. NN Model 만들기 #########################################################
def create_model():
    model = keras.Sequential() #sequencial API - 순서대로 레이어 쌓음
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, padding='SAME', 
                                  input_shape=(28, 28, 1)))#첫 레이어엔 input_shape써야함
    model.add(keras.layers.MaxPool2D(padding='SAME')) #2x2 size &stride 2 is default
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    #fully connected Layer
    model.add(keras.layers.Flatten()) #vector 피는부분
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.4)) #dense layer파라미터가 너무 많아서 조금 떨어트림
    model.add(keras.layers.Dense(10))
    return model



model = create_model()
model.summary()

## 4 los #########################################################
def loss_fn(model, images, labels):
    logits = model(images, training=True) #training=True 면 레이어에서 드랍아웃한게 실행됨
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))    
    return loss
## 5. gradient #########################################################
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)#백 프로퍼게이션
    #로스를 뭘로 미분하냐면 모델에 있는 모든 파라미터로 계산해주세요

## 6. Optimizer #########################################################
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

## 7. Accuracy #########################################################
def evaluate(model, images, labels):
    logits = model(images, training=False)#트레이닝이 아니라 실제라서. dropout 안일어남
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


## 8. checkpoint #########################################################
cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'minst_cnn_seq'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

checkpoint = tf.train.Checkpoint(cnn=model)


## 9. train and validate #########################################################
# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs): 
    #1epoch 마다 일어나는일
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0
    
    for images, labels in train_dataset: #배치만큼 일어나는일
        grads = grad(model, images, labels)      
        optimizer.apply_gradients(zip(grads, model.variables)) #weight 가 업데이트됨
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)
        avg_loss = avg_loss + loss
        avg_train_acc = avg_train_acc + acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step
    
    #한 에폭마다 평가해봄
    for images, labels in test_dataset:        
        acc = evaluate(model, images, labels)        
        avg_test_acc = avg_test_acc + acc
        test_step += 1    
    avg_test_acc = avg_test_acc / test_step    

    print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss), 
          'train accuracy = ', '{:.4f}'.format(avg_train_acc), 
          'test accuracy = ', '{:.4f}'.format(avg_test_acc))
    
    #한에폭마다 저장해봄
    checkpoint.save(file_prefix=checkpoint_prefix)

print('Learning Finished!')
