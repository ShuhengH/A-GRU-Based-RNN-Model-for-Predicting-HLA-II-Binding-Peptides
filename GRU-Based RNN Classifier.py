
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import matthews_corrcoef,roc_curve,auc
from keras import regularizers

#***********************************************************File into*******************************************************
df_train=pd.read_excel(r'/home/DL/train.xlsx',Sheetname='训练集')
df_test=pd.read_excel(r'/home/DL/test.xlsx',Sheetname='测试集')

#******************************************************************************************************************
#The Y value
lables_list = df_train['标签'].unique().tolist()
dig_lables = dict(enumerate(lables_list))
lable_dig = dict((lable,dig) for dig, lable in dig_lables.items())
df_train['标签_数字'] = df_train['标签'].apply(lambda lable: lable_dig[lable])
num_classes = len(dig_lables)
train_lables = to_categorical(df_train['标签_数字'],num_classes=num_classes)
#The X value
num_words = 21   #The total number of words
max_len = 99  
tokenizer = Tokenizer(num_words=num_words)
df_all=pd.concat([df_train['文本'],df_test['文本']])
tokenizer.fit_on_texts(df_all)
train_sequences = tokenizer.texts_to_sequences(df_train['文本'])
train_data = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_sequences = tokenizer.texts_to_sequences(df_test['文本'])
test_data = pad_sequences(test_sequences, maxlen=max_len, padding='post')
#A random seed
seed=7
np.random.seed(seed)
#The training set and the verification set are divided
DATE_train, x_val, LABLES_train, y_val = train_test_split(train_data,df_train['标签_数字'],test_size=0.4, random_state=0,shuffle=True)

#***********************************************************model framework*******************************************************
model = tf.keras.Sequential()
vocab_size=len(tokenizer.word_index)+1
model.add(layers.Embedding(input_dim=vocab_size,output_dim=128,mask_zero=True,embeddings_initializer='uniform'))
model.add(layers.BatchNormalization())
model.add(layers.SpatialDropout1D(0.5))
model.add(layers.GRU(128,return_sequences=False,dropout=0.2, recurrent_dropout=0.2,kernel_regularizer=regularizers.l2(0.0000001))) 
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.0000001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#*******************************************************************************************************************************
#early_stopping
early_stopping = EarlyStopping(
    monitor='val_acc', 
    patience=10, 
    verbose=1, 
    mode='max')
Checkpoint = ModelCheckpoint(         
         '/home/DL'+'/model-{epoch:02d}.hdf5',
    monitor='val_loss', 
    verbose=1,
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto',
    period=50)
#*********************************************************************************************************************************
history = model.fit(DATE_train,LABLES_train,batch_size=128,shuffle=True,epochs=200,validation_data=(x_val,y_val),callbacks=[early_stopping,Checkpoint])

#******************************************************************Training set and validation set evaluation***************************************************************
score=model.evaluate(DATE_train,LABLES_train,verbose=0)
predict_val = model.predict(DATE_train)
#print('val-predict_val:',predict_val)
predict_classes=model.predict_classes(DATE_train)
#print('train-predict_classes:',predict_classes)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
tn, fp, fn, tp = metrics.confusion_matrix(LABLES_train,predict_classes).ravel()
spe = tn/(tn+fp)
print('train-spe:',spe)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
sen=metrics.recall_score(LABLES_train,predict_classes)
print('train-sen:',sen)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
auc1 = metrics.roc_auc_score(LABLES_train,predict_val[:,1])
print('train-auc:',auc1) 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
mcc = matthews_corrcoef(LABLES_train,predict_classes)
print('train-mcc:',mcc) 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
F1=metrics.f1_score(LABLES_train,predict_classes)
print('train-F:',F1)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
score=model.evaluate(x_val,y_val,verbose=0)
print('Val Loss:',score[0])
print('Val-acc:',score[1])
#-----------------------------------------------------------------------------------------------------------------------------------------------------
predict_val = model.predict(x_val)
#print('val-predict_val:',predict_val)
predict_classes=model.predict_classes(x_val)
print('val-predict_classes:',predict_classes)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
tn, fp, fn, tp = metrics.confusion_matrix(y_val,predict_classes).ravel()
spe = tn/(tn+fp)
print('val-spe:',spe)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
sen=metrics.recall_score(y_val,predict_classes)
print('val-sen:',sen)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
auc2 = metrics.roc_auc_score(y_val,predict_val[:,1])
print('val-auc:',auc2) 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
mcc = matthews_corrcoef(y_val,predict_classes)
print('val-mcc:',mcc) 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
F=metrics.f1_score(y_val,predict_classes)
print('val-F1:',F)
#**********************************************************************Model save and load**************************************************************************
#Save the model
model.save('GRU-1.h5')
#Load model
model= tf.keras.models.load_model('GRU-1.h5')
model.summary()

#**********************************************************************Test set evaluation**************************************************************************
pred_ = [model.predict(vec.reshape(1,max_len)).argmax() for vec in test_data]
#print('pred_')
#print(pred_)
df_test['分类结果_预测'] = [dig_lables[dig] for dig in pred_]
print('test-acc:',metrics.accuracy_score(df_test['标签'],df_test['分类结果_预测']))
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#Confusion_matrix
tn, fp, fn, tp = metrics.confusion_matrix(df_test['标签'],df_test['分类结果_预测']).ravel()
spe = tn/(tn+fp)
print('test-spe:',spe)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
sen=metrics.recall_score(df_test['标签'],df_test['分类结果_预测'])
print('test-sen:',sen)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
y_pred = model.predict_proba(test_data)[:,1]
#print('y_pred',y_pred)
fpr, tpr, thresholds = roc_curve(df_test['标签'], y_pred)
print('test-auc:',auc(fpr, tpr))
#-----------------------------------------------------------------------------------------------------------------------------------------------------
mcc = matthews_corrcoef(df_test['标签'], df_test['分类结果_预测'])
print('test-mcc:',mcc) 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
f1_score=metrics.f1_score(df_test['标签'],df_test['分类结果_预测'], labels=None, pos_label=1, average='weighted', sample_weight=None)
print('test-F1:',f1_score)

#**************************************************************************figure**********************************************************************
plt.figure(1)
print('训练集验证集图形')
epoch = np.arange(1,len(history.history['loss'])+1,1)
font2 = {#'family' : 'Times New Roman',
'size'   : 14,}
#plt.title('Loss')
plt.plot(epoch,history.history['loss'], label='Training set')
plt.plot(epoch,history.history['val_loss'], label='Validation set')
plt.xlabel("Epoch",font2)
plt.ylabel("Loss",font2)
plt.ylim((0.1, 0.8))
plt.legend(loc='best',frameon=False)
plt.legend(prop=font2)
#plt.show()
plt.savefig('GRU1.png',dpi=300)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(2)
epoch = np.arange(1,len(history.history['acc'])+1,1)
font2 = {#'family' : 'Times New Roman',
'size'   : 14,}
#plt.title('Accuracy')
plt.plot(epoch,history.history['acc'], label='Training set')
plt.plot(epoch,history.history['val_acc'], label='Validation set')
plt.xlabel("Epoch",font2)
plt.ylabel("Accuracy",font2)
plt.ylim((0.5, 1))
plt.legend(loc='best',frameon=False)
plt.legend(prop=font2)
#plt.show()#
plt.savefig('GRU2.png',dpi=300)


