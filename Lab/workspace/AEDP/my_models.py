import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Input, Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import resnet_v2, VGG16, VGG19, InceptionV3

from keras.backend import int_shape
from tensorflow.keras import initializers

# resnetv2 = 50 101 152
##############################################################################################################

def cnn_cla(input_shape, n_class):
    
    initializer = initializers.HeNormal()
    
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.15)(model)
    
    model = Conv2D(4, kernel_size=(3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2,2))(model)
    #model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu')(model)   
    model = BatchNormalization()(model)
    #model = MaxPooling2D(pool_size=(2, 2))(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization()(model)
    #model = MaxPooling2D(pool_size=(2, 2))(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization()(model)
    #model = MaxPooling2D(pool_size=(2, 2))(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization()(model)

    model = Flatten()(model)
    
    model = Dropout(0.5)(model)
    model = Dense(units=32, activation='relu')(model)
    model = Dropout(0.5)(model)
    output = Dense(units=n_class, activation='softmax')(model)
    model = Model(inputs=input, outputs=output)

    return model

##############################################################################################################

def rgb_resnet_model(input_shape, n_class):
    model_resnet = resnet_v2.ResNet101V2(include_top=False, weights='imagenet', input_shape= input_shape)
    model_resnet.trainable = False
    for layer in model_resnet.layers:
        layer.trainable = False
    model = Flatten()(model_resnet.output)
    #model = GlobalAveragePooling2D()(model_resnet.output)
    model = Dropout(0.5)(model)
    model = Dense(units=1000, activation='relu')(model)
    model = Dropout(0.5 )(model)
    output = Dense(units=n_class, activation='softmax')(model)
    model = Model(inputs=model_resnet.input, outputs=output)

    return model

##############################################################################################################

def g_resnet_model(input_shape, n_class):
    model_resnet = resnet_v2.ResNet50V2(include_top=False, weights=None, input_shape= input_shape)
    model_resnet.trainable = False
    for layer in model_resnet.layers:
        layer.trainable = False
    model = Flatten()(model_resnet.output)
    #model = GlobalAveragePooling2D()(model_resnet.output)
    #model = Dropout(0.2)(model)
    model = Dense(units=100, activation='relu')(model)
    model = Dropout(0.5)(model)
    output = Dense(units=n_class, activation='softmax')(model)
    model = Model(inputs=model_resnet.input, outputs=output)

    return model

##############################################################################################################

def se_block(block_input , num_filters, ratio = 8):
    
    pool1 = GlobalAveragePooling2D()(block_input)
    flat = Reshape((1,1,num_filters))(pool1)
    dense1 = Dense(num_filters//ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])
    
    return scale


def resnet_block(block_input, num_filters):
    
    if int_shape(block_input)[3] != num_filters:
        block_input = Conv2D(num_filters, kernel_size=(1, 1))(block_input)
    
    conv1 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(norm1)
    conv2 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(relu1)
    norm2 = BatchNormalization()(conv2)
    
    se = se_block(norm2, num_filters=num_filters)
    
    sum = Add()([block_input, se])
    relu2 = Activation('relu')(sum)
    
    return relu2



def se_resnet14(input_shape, n_class):
    
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    
    model = Conv2D(8, kernel_size = (3,3), activation='relu', padding='same')(model)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,16)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,32) 
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,64)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,128)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = Flatten()(model)
    
    model = Dropout(0.5)(model)
    model = Dense(units=300, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=300, activation='relu')(model)
    model = Dropout(0.5)(model)
    
    output = Dense(units=n_class, activation='softmax')(model)
        
    model = Model(inputs=input, outputs=output)
    return model
               

##############################################################################################################

def get_model_seq_regress(input_shape):
    initializer = initializers.HeNormal()

    input_d = Input(shape=input_shape)
    model_p = experimental.preprocessing.RandomFlip('horizontal')(input_d)
    model_p = experimental.preprocessing.RandomRotation(0.2)(model_p)
    model_p = Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu')(model_p)
    model_p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model_p)
    model_p = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(model_p)
    model_p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model_p)
    model_p = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(model_p)
    model_p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model_p)
    model_p = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(model_p)
    model_p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model_p)

    model_p = Flatten()(model_p)
    model_p = Dropout(0.2)(model_p)
    model_p = Dense(units=500, activation='relu')(model_p)
    model_p = Dropout(0.2)(model_p)
    model_p = Dense(units=500, activation='relu')(model_p)
    model_p = Dropout(0.2)(model_p)
    model_p = Dense(units=1, activation='sigmoid')(model_p) # 0~1 
    model_p = Model(inputs=input_d, outputs=model_p)

    return model_p

##############################################################################################################

def g_resnet50v2(input_shape):
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    input_tensor = model 
    model_resnet = resnet_v2.ResNet50V2(include_top=False, weights=None, input_tensor= input_tensor)
    model_resnet.trainable = False
    for layer in model_resnet.layers:
        layer.trainable = False

    model = Flatten()(model_resnet.output)
    model = Dropout(0.5)(model)
    model = Dense(units=16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=model_resnet.input, outputs=model)

    return model

def g_resnet101v2(input_shape):
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    input_tensor = model 
    model_resnet = resnet_v2.ResNet101V2(include_top=False, weights=None, input_tensor= input_tensor)
    model_resnet.trainable = False
    for layer in model_resnet.layers:
        layer.trainable = False

    model = Flatten()(model_resnet.output)
    model = Dropout(0.5)(model)
    model = Dense(units=16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=model_resnet.input, outputs=model)

    return model


def g_resnet152v2(input_shape):
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    input_tensor = model 
    model_resnet = resnet_v2.ResNet152V2(include_top=False, weights=None, input_tensor= input_tensor)
    model_resnet.trainable = False
    for layer in model_resnet.layers:
        layer.trainable = False

    model = Flatten()(model_resnet.output)
    model = Dropout(0.5)(model)
    model = Dense(units=16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=model_resnet.input, outputs=model)

    return model

def g_VGG16(input_shape):
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    input_tensor = model 
    model_VGG = VGG16(include_top=False, weights=None, input_tensor= input_tensor)
    model_VGG.trainable = False
    for layer in model_VGG.layers:
        layer.trainable = False

    model = Flatten()(model_VGG.output)
    model = Dropout(0.5)(model)
    model = Dense(units=16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=model_VGG.input, outputs=model)

    return model

def g_VGG19(input_shape):
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    input_tensor = model 
    model_VGG = VGG19(include_top=False, weights=None, input_tensor= input_tensor)
    model_VGG.trainable = False
    for layer in model_VGG.layers:
        layer.trainable = False

    model = Flatten()(model_VGG.output)
    model = Dropout(0.5)(model)
    model = Dense(units=16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=model_VGG.input, outputs=model)

    return model

def g_InceptionV3(input_shape):
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    input_tensor = model 
    model_Inception = InceptionV3(include_top=False, weights=None, input_tensor= input_tensor)
    model_Inception.trainable = False
    for layer in model_Inception.layers:
        layer.trainable = False

    model = Flatten()(model_Inception.output)
    model = Dropout(0.5)(model)
    model = Dense(units=16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=model_Inception.input, outputs=model)

    return model


##############################################################################################################

###92프로 달성 모델
def cnn_regress(input_shape):
    
    act = 'relu'
    initializer = initializers.HeNormal()
    
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    #model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    model = Conv2D(16, kernel_size=(3, 3), padding='same', activation=act)(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2,2))(model)
    #model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(16, kernel_size=(3, 3), padding='same', activation=act)(model)   
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    #model = AveragePooling2D(pool_size=(2, 2))(model)

    model = Conv2D(32, kernel_size=(3, 3), padding='same', activation=act)(model)
    model = BatchNormalization()(model)
    #model = MaxPooling2D(pool_size=(2, 2))(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)

    
    model = Conv2D(32, kernel_size=(3, 3), padding='same', activation=act)(model)
    model = BatchNormalization()(model)
    #model = MaxPooling2D(pool_size=(2, 2))(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)

    model = Conv2D(64, kernel_size=(3, 3), padding='same', activation=act)(model)
    model = BatchNormalization()(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Flatten()(model)
    
    model = Dropout(0.5)(model)
    model = Dense(units=16, activation=act)(model)
    model = Dropout(0.5)(model)

    output = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=input, outputs=output)

    return model

##############################################################################################################

def cnn_attention_regress(input_shape):
    
    input = Input(shape=input_shape)
    model = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    model = experimental.preprocessing.RandomRotation(0.2)(model)
    
    model = Conv2D(8, kernel_size = (3,3), activation='relu', padding='same')(model)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,16)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,32) 
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,64)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,64)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = resnet_block(model,128)
    model = MaxPooling2D((2,2),strides=2)(model)
    
    model = Flatten()(model)
    
    model = Dropout(0.5)(model)
    model = Dense(units=300, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=300, activation='relu')(model)
    model = Dropout(0.5)(model)

    output = Dense(units=1, activation='sigmoid')(model)
        
    model = Model(inputs=input, outputs=output)
    return model

##############################################################################################################

def rgb_resnet_regress(input_shape):
    
    model_resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape= input_shape)
    model_resnet.trainable = False
    for layer in model_resnet.layers:
        layer.trainable = False

    model = Flatten()(model_resnet.output)
    model = Dropout(0.5)(model)
    model = Dense(units=500, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=500, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(units=1, activation='sigmoid')(model) # 0~1 
    model = Model(inputs=model_resnet.input, outputs=model)

    return model

##############################################################################################################\



def cnn_block(input):
    model = Conv2D(4, kernel_size=(3, 3), padding='same', activation='relu')(input)
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2,2))(model)
    #model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu')(model)   
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    #model = AveragePooling2D(pool_size=(2, 2))(model)
    
    model = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization()(model)
    #model = MaxPooling2D(pool_size=(2, 2))(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)

    model = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization()(model)
    #model = MaxPooling2D(pool_size=(2, 2))(model)
    model = AveragePooling2D(pool_size=(2, 2))(model)

    
    return model


def output_block(input):
    
    model = Dropout(0.5)(input)
    model = Dense(units=16, activation='relu')(model)
    model = Dropout(0.5)(model)
    return model


def bangdoon(input_shape):
    initializer = initializers.HeNormal()
    
    input = Input(shape=input_shape)
    input_1 = experimental.preprocessing.RandomFlip('horizontal_and_vertical')(input)
    input_2 = experimental.preprocessing.RandomRotation(0.15)(input)
    
    model_1 = cnn_block(input_1)
    
    model_2 = cnn_block(input_2)
    
    model = Concatenate(axis=1)([model_1,model_2])
    
    model = Flatten()(model)
    
    model = output_block(model)

    
    output = Dense(units=1, activation = 'sigmoid')(model)
    model = Model(inputs=input, outputs=output)

    return model