#!/usr/bin/env python
# coding: utf-8

# Sonja Kittl CAS ADS Module 3 project 12.03.2021

# # Bacterial taxonomy Model
# 
# Bacterial taxonomy is  largely based on 16SrRNA sequence clustering.
# The idea of this project is to see if the neural network can learn to classify bacteria into orders based on 16SrRNA sequences.
# As a test case I used 4 orders of actinobacteria:
# - actinomycetales
# - corynebacteriales (= mycobacteriales)
# - micrococcales
# - propionibacteriales
# 
# The network should be able to distinguish between these related bacterial orders based on 16SrRNA sequences.
# Plublicly available data from genbank was used (including only RefSeq data): 
# 
# to get the data the following NCBI queries were used and data downloaded as fasta:
# 
#     ((33175[BioProject] OR 33317[BioProject])) AND txid2037[Organism] 
# 
#     ((33175[BioProject] OR 33317[BioProject])) AND txid85007[Organism] 
# 
#     ((33175[BioProject] OR 33317[BioProject])) AND txid85006[Organism]
#     
#     ((33175[BioProject] OR 33317[BioProject])) AND txid85009[Organism]
# 

# In[1]:


#from IPython.display import Image
#Image(filename = "M3_project_data/trees/tree.png", width = 1200, height = 600)


# To replace the unique identifiers with uniform labels the following bash commands were used
# 
# - sed "/>/c>actinomycetales" txid2037.fasta > actinomycetales_taxid_la.fasta
# - sed "/>/c>corynebacteriales" txid85007.fasta > corynebacteriales_taxid_la.fasta
# - sed "/>/c>micrococcales" txid85006.fasta > micrococcales_taxid_la.fasta
# - sed "/>/c>propionibacteriales" txid85009.fasta > propionibacteriales_taxid_la.fasta

# merge into one dataset 
# - cat *la.fasta > actinobacteria_data4.fasta

# In[2]:


import sys

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipyd
import tensorflow as tf
import re



#from IPython.core.display import HTML
#HTML("""<style> .rendered_html code { 
#    padding: 2px 5px;
#   color: #0000aa;
#   background-color: #cccccc;
#} </style>""")


# In[3]:


#funcion to modify the input data, takes a string of DNA and returns one hot encoded version thereof

def one_hot_dna(myDNA):
    """this function takes as input a string of DNA and returns an array using one-hot encoding
        non acgt bases are encoded with [0,0,0,0]
    """
    #print(myDNA[0:10])
    myDNA=myDNA.lower()
    i=0
    myDNA_encoded=[]
    for base in myDNA:
        #print(base)
        if base=='a':
            myDNA_encoded+=[[1,0,0,0]]
        elif base=='c':
            myDNA_encoded+=[[0,1,0,0]]
        elif base=='g':
            myDNA_encoded+=[[0,0,1,0]]
        elif base=='t':
            myDNA_encoded+=[[0,0,0,1]]
        else:
            myDNA_encoded+=[[0,0,0,0]]
        #print(i,base,myDNA_encoded[i])
        i+=1
        
    return myDNA_encoded
        


# In[4]:


#funcion to reverse the encoding of the DNA sequence

def decode_dna(my_one_hot_DNA):
    """this function takes as input an array of one-hot encoded DNA and returns a string
        non acgt bases are encoded with [0,0,0,0]
    """
    myDNA_encoded=[]
    i=0
    myDNA_string=''
    for base in my_one_hot_DNA:
        #print(base)
        if base==[1,0,0,0]:
            myDNA_string+='a'
        elif base==[0,1,0,0]:
            myDNA_string+='c'
        elif base==[0,0,1,0]:
            myDNA_string+='g'
        elif base==[0,0,0,1]:
            myDNA_string+='t'
        else:
            myDNA_string+='n'
        #print(i,base,myDNA_encoded[i])
        i+=1
        
    return myDNA_string


# In[5]:


#to reshape sequences from x data to be used for decoding
def reshape_sequence(sequence):
    new_sequence=[]
    for base in sequence:
        #print(list(base.T[0]))
        new_sequence+=[list(base.T[0])]
    return new_sequence
        
        


# In[6]:

#to test the functions to encode and decode DNA
#myDNA='aacgttcnta'
#encoded_DNA=one_hot_dna(myDNA)
#decoded_DNA=decode_dna(encoded_DNA)
#print(encoded_DNA)
#print(decoded_DNA)


# In[7]:


#function to transform labels to integers
def int_label(mylabels):
    """
    this function takes a list of labels and returns a list of integers encoding the lables
    it also returns a dictionary of the lables and corresponding integers
    
    """
    #to get unique valuses
    label_set=set(mylabels)
    #print(label_set)
    labels_unique=list(label_set)
    #print(labels_unique)
    label_count=len(labels_unique)
    #print(label_count)
    #make a dictionary for unique labels and their encoding
    i=0
    newlabel=0
    labels_dict={}
    for label in labels_unique:
        newlabel=i
        labels_dict[label]=newlabel
        i+=1
    #print(labels_dict)
    labels_int=[]
    for label in mylabels:
        labels_int+=[labels_dict[label]]
    #print(labels_int)
    return labels_int, labels_dict
    


# In[8]:


#function to read the input file

def read_input(myfile):
    """
    this function reads a fasta file
    following the > must be the label(must be alphanumeric)
    returns an array of one-hot encoded sequences,
            an array of integer encoded corresponding labels,
            a dictionary with the label encoding
    """
    labels=[]
    sequences=[]
    sequences_one_hot=[]
    #regex to extract genus name
    regex=re.compile(r'>([a-zA-Z0-9]+)')
    
    with open(myfile) as fobj:
        
        i=0
        for line in fobj:
            line=line.rstrip()
                #to ignore empty lines
            if line:
                    #read names and sequences in separate arrays

                if line[0]=='>':    
                    mo=regex.search(line)
                    labels+=[mo.group(1)]
                    sequences+=[[]]
                    #print(len(labels))
                    #print(len(sequences))
                    i+=1
                if line[0]!='>':
                    sequences[i-1]+=line

            
                
    #print(sequences)
    
    #make all sequences equal length, by padding at the end
    my_lenghts=[]
    my_equal_seq=[]
    for sequence in sequences:
        my_lenghts+=[len(sequence)]
    #    if len(sequence)>1500:
    #        print(''.join(sequence))
    max_length=max(my_lenghts)
    print(max_length)
    #max_length=1600 #to test 
    i=0
    for sequence in sequences:
        sequence=''.join(sequence)
        
        if len(sequence)<max_length:
            sequence=sequence+'n'*(max_length-len(sequence))
            #print(sequence[-10:-1])
        my_equal_seq+=[sequence]
    
    
    #encode sequences
    for sequence in my_equal_seq:
        #print(i,len(sequence))
        i+=1
        encoded=one_hot_dna(sequence)
        sequences_one_hot+=[encoded]
    #print(sequences[10])
    #print(sequences_one_hot[10])
    #encode labels
    #labels_one_hot=one_hot_label(labels)
    labels_integer, labels_dict=int_label(labels)
    return sequences_one_hot, labels_integer, labels_dict


# In[9]:


#read the data
mydata=input("Enter the path to your data")
x_data, y_data, labels_dict=read_input(mydata)
#x_data, y_data, labels_dict=read_input("actinobacteria_data/acti_data3.fasta")
print(labels_dict)


# In[10]:


print(np.array(x_data).shape)


# In[11]:


number_of_labels=max(y_data)+1
print(number_of_labels)


# In[12]:


#split the data into groups according to their labels, one group for each label

sequences=[]
labels=[]
for label in range(0,number_of_labels):
    sequences+=[[]]
    labels+=[[]]
#print(sequences)

for i in range(0,len(y_data)):
    for label in range(0,number_of_labels):
        if y_data[i]==label:
            sequences[label]+=[x_data[i]]
            labels[label]+=[label]
#print(labels)
#print(sequences)


# In[13]:


group_sizes={}
for label in range(0,number_of_labels):
    group_sizes[label]=len(labels[label])


# In[14]:


#find the index with the most and least samples
print(group_sizes)
smallest_group=min(group_sizes, key=group_sizes.get)
largest_group=max(group_sizes, key=group_sizes.get)
print(smallest_group, largest_group)


# In[15]:


#augment data to the size of half the largest group by randomly replacing 2 bases with n
data_size=int(group_sizes[largest_group]/2)
for label in range(0,number_of_labels):
    if group_sizes[label]<data_size:
        to_impute=data_size-group_sizes[label]
        print(label, to_impute)
        for i in range(0,to_impute):
            mypool=np.copy(sequences[label])
            replace=np.random.randint(0,1500)
            replace2=np.random.randint(0,1500)
            draw=np.random.randint(0,group_sizes[label])
            #print(replace,draw)
            my_sequence=mypool[draw]
            #print(my_sequence)
            #print(my_sequence[replace])
            my_sequence[replace]=[0,0,0,0]
            #print(my_sequence[replace])
            my_sequence[replace2]=[0,0,0,0]
            sequences[label]+=[my_sequence]
            labels[label]+=[label]
    

            


# In[16]:



        
#randomly permutate the data in each group
group_sizes={}
for label in range(0,number_of_labels):
    group_sizes[label]=len(labels[label])
    idx = np.random.permutation(len(labels[label]))
    labels[label] = np.array(labels[label])[idx]
    sequences[label] = np.array(sequences[label])[idx]

    print(label, ': sequences shape', sequences[label].shape, 'label shape', labels[label].shape)


# In[17]:


#find the index with the least samples
print(group_sizes)
smallest_group=min(group_sizes, key=group_sizes.get)
largest_group=max(group_sizes, key=group_sizes.get)
print(smallest_group, largest_group)


# In[18]:




#We will take 70% from each for training and 20% for validation and 10% for prediction

n_group = sequences[smallest_group].shape[0] 
n_train_group = n_group*70//100
n_valid_group = n_group*90//100

x_train = sequences[0][:n_train_group]
y_train = labels[0][:n_train_group]

x_valid = sequences[0][n_train_group:n_valid_group]
y_valid = labels[0][n_train_group:n_valid_group]

x_predict = sequences[0][n_valid_group:group_sizes[smallest_group]]
y_predict = labels[0][n_valid_group:group_sizes[smallest_group]]

for group in range(1,number_of_labels):

    x_train = np.concatenate([x_train, sequences[group][:n_train_group]])
    y_train = np.concatenate([y_train, labels[group][:n_train_group]])
    x_valid = np.concatenate([x_valid, sequences[group][n_train_group:n_valid_group]])
    y_valid = np.concatenate([y_valid, labels[group][n_train_group:n_valid_group]])
    x_predict = np.concatenate([x_predict, sequences[group][n_valid_group:group_sizes[smallest_group]]])
    y_predict = np.concatenate([y_predict, labels[group][n_valid_group:group_sizes[smallest_group]]])  


# In[19]:


print('x_train shape', x_train.shape, 'y_train shape', y_train.shape)
print('x_valid shape', x_valid.shape, 'y_valid shape', y_valid.shape)
print('x_predict shape', x_predict.shape, 'y_predict shape', y_predict.shape)


# In[20]:


#print(y_valid)


# In[21]:


#randomly permutate validation data
idx = np.random.permutation(len(y_valid))
y_valid = y_valid[idx]
x_valid = x_valid[idx]


# In[22]:


#randomly permutate training data
idx = np.random.permutation(len(y_train))
y_train = y_train[idx]
x_train = x_train[idx]


# In[23]:


#randomly permutate the prediction data
idx = np.random.permutation(len(y_predict))
y_predict = y_predict[idx]
x_predict = x_predict[idx]


# In[ ]:





# In[24]:


#to expand the dimensions
x_train = np.expand_dims(x_train, -1)
x_valid = np.expand_dims(x_valid, -1)
x_predict = np.expand_dims(x_predict, -1)


# In[25]:


input_shape=x_train.shape[1:]
print(input_shape)


# In[26]:


#taken from https://keras.io/examples/vision/mnist_convnet/ modified
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, kernel_size=(4, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 1)),
        tf.keras.layers.Conv2D(16, kernel_size=(4, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 1)),
        tf.keras.layers.Conv2D(8, kernel_size=(4, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(number_of_labels, activation="softmax"),
    ]
)


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.003) ,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


# In[27]:


model.summary()


# In[28]:


save_path = 'save/model_{epoch}.ckpt'
save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True)

hist = model.fit(x=x_train, y=y_train,
                 epochs=15, batch_size=5, 
                 validation_data=(x_valid, y_valid),
                 callbacks=[save_callback])


# In[29]:


fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()


# In[30]:


#get the prediction data in DNA form to use for alignment
print(labels_dict)
#invert the dictionalry
inv_labels_dict = dict(zip(labels_dict.values(), labels_dict.keys()))
print(inv_labels_dict)
with open('predict_data.fasta', 'w') as file:
    for i in range(len(y_predict)):
        label=y_predict[i]
        #print(label)
        file.write('>'+str(i)+str(inv_labels_dict[label])+'\n')
        my_encoded=reshape_sequence(x_predict[i])
        my_sequence=decode_dna(my_encoded)
        file.write(str(my_sequence)+'\n')


# In[31]:



# Making predictions.
print(labels_dict)
predictions = model.predict(x_predict)
predictions_true=np.zeros(number_of_labels)
predictions_false=np.zeros(number_of_labels)

predictions_true_rv=np.zeros(number_of_labels)
predictions_false_rv=np.zeros(number_of_labels)
with np.printoptions(precision=3, suppress=True):
    for i in range(len(y_predict)):
        predicted_label=list(predictions[i]).index(max(predictions[i]))
        true_label=y_predict[i]
        if predicted_label==true_label:
            predictions_true[true_label]+=1
            predictions_true_rv[predicted_label]+=1
        else:
            predictions_false[true_label]+=1
            predictions_false_rv[predicted_label]+=1

            
        
        print(i,'Prediction is ', y_predict[i]==predicted_label, ': true label: ', y_predict[i], 'predicted label:', predicted_label, 'predictions: ', predictions[i])
    #print(predictions_true, predictions_false)
    for label in range(0,number_of_labels):
        print(inv_labels_dict[label],':','%sensitivity=',               round(100*predictions_true[label]/(predictions_true[label]+predictions_false[label])),                  '%PPV=', round(100*predictions_true_rv[label]/(predictions_true_rv[label]+predictions_false_rv[label])),               )


# In[ ]:




