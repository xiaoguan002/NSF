from __future__ import print_function
import numpy as np


import cPickle
import theano
import scipy.spatial.distance as Dis


X = cPickle.load(open("C:/Users/guan/Desktop/ff.p","rb"))
Y = cPickle.load(open("C:/Users/guan/Desktop/model.p","rb"))
raw_data,vector = Y[0],Y[1]
Y_label,soft_out= X[0], X[3]

print (len(Y_label))
labels = np.array([9493,8635,10159,2293])

 # 4 label nums
total_num = len(Y_label)    #30580
label_count = 4
index_cos = []
Y_predict = []
recall_step = 0.1
recall = 0.1
precision = []

test_count = 100
arr = np.arange(total_num)
np.random.shuffle(arr)
test_index = arr[:test_count]
#print (test_index)


vector = np.asarray(vector)
vector = vector.reshape(total_num,300)
print (vector.shape)

vector_cos = Dis.pdist( vector,'cosine')

print (vector_cos.shape)

#test
'''
tempi=100
tempj=5
i =min(tempi,tempj)
j=max(tempi,tempj)
print (i,j)
index = i*total_num + j - i*(i+1)/2 - i - 1
print(index)
print(vector_cos[index])
print(Dis.cosine(vector[i,:],vector[j,:]))
exit()


for i in range(total_num):
    Y_predict.append(np.argmax(soft_out[i][0]))

print ("Y_predict : " , len(Y_predict))
'''

for i in test_index:
    labels[Y_label[i]] -= 1
    per_cos =[]
    for j in range(total_num):
        if j in test_index:
            continue
        temp_i = min(i,j)
        temp_j = max(i,j)
        index = temp_i*total_num + temp_j - temp_i*(temp_i+1)/2 - temp_i - 1
        per_cos.append([j, vector_cos[index]])

    per_cos.sort(key = lambda x : x[1])
    index_cos.append([per_cos, Y_label[i]] )
    print ("index :", i, "cos, sort : ok")

print ("index_cos : " , len(index_cos))
print ("labels : " , labels)


while recall <=  1 :

        label_num = np.array(labels * recall,dtype=int)
        print (" recall : ", recall, " label_num : ",label_num)
        index_pre = 0
        for i in range(test_count):
            TP = 0
            FP = 0

            for k in range(total_num-test_count):

                if Y_label[index_cos[i][0][k][0]] == index_cos[i][1]:
                    TP +=1
                else:
                    FP +=1


                if TP == label_num[index_cos[i][1]]:
                    break
            print ("i : ", i, "index : ", test_index[i],"TP, FP: ", TP,FP, "label : ", index_cos[i][1]," label_num : ", label_num[index_cos[i][1]])
            index_pre += (float(TP) /float(TP+FP))

        precision.append( index_pre/test_count)
        print ( [recall, index_pre/test_count])
        recall += recall_step

print (precision)










