import numpy as np
import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os



# def getDataSet():
#     group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
#     labels = ['A', 'A',
#               'B', 'B']
#     return group, labels

def classifyFunc(Xin, group, labels, k):# classify X (a vector) with a corresponding label
        '''
        # we take the closest k examples and find the majority vote among them, then
        # we choose the lable of this one to be the label of Xin
        '''

        nod = group.shape[0] #number_of_dataset
        diffMat = np.tile(Xin, (nod, 1)) - group    #copy the Xin nod times and make it as a matrix, then calculation
                                                    #to get the distance from Xin to the group examples
                                                    #but we need them to be array so to get the ||diffMat||^2
        distanceMat_pre = diffMat**2
        distanceMat_pre = distanceMat_pre.sum(axis=1)   #we add all the dimensions together
        distanceMat = np.sqrt(distanceMat_pre)          #get the real distance
        #argsort 返回用来排序的下标
        sortedDicies = distanceMat.argsort()
        classCount = {}

        for i in range(k):
            voteLabel = labels[sortedDicies[i]]

            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        sorted_classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        #sorted_classCount belongs to list type, each one in it is a tuple, can not be changed
        #operator.itemgetter is to get the number in the dictionary tyep variable classCount
        #to find the k largest votes belong to which label
        return  sorted_classCount[0][0]




def getDatingSet(file_path):#read the file and get dataset and label matrix
    with open(file_path, 'r') as file:
        number_of_records = len(file.readlines())
        returnMat = np.zeros((number_of_records, 3))
        labelVector = []

    index = 0

    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            records = line.split('\t')
            returnMat[index, :] = records[0:3]
            labelVector.append(records[-1])
            index += 1

    return returnMat, labelVector

def normDataSet(dataSet):# normalize the dataset
    max_value = dataSet.max(axis=0)
    min_value = dataSet.min(axis=0)
    range = max_value - min_value
    number_of_data = dataSet.shape[0]
    norm_data = np.zeros(dataSet.shape)


    norm_data = dataSet - np.tile(min_value, (number_of_data, 1))
    norm_data = norm_data/np.tile(range, (number_of_data,1))
    return norm_data, min_value, range





# label_int = []
# for item in labels:
#     label_int.append(ord(item[0]))
#
#
#
# index = 0
#
# didntLike = []
# smallDoses = []
# largeDoses = []
#
#
# for data in labels:
#     if data == 'didntLike':
#         didntLike.append((dataset[index, 0], dataset[index, 1], dataset[index, 2]))
#     elif data == 'smallDoses':
#         smallDoses.append((dataset[index, 0], dataset[index, 1], dataset[index, 2]))
#     elif data == 'largeDoses':
#         largeDoses.append((dataset[index, 0], dataset[index, 1], dataset[index, 2]))
#     index += 1
#
#
# didntLike, mind, ranged = normDataSet(np.array(didntLike))
# smallDoses, mins, ranges = normDataSet(np.array(smallDoses))
# largeDoses, minl, rangel = normDataSet(np.array(largeDoses))
#
# fig = plt.figure()
# # ax = fig.add_subplot(1, 1, 1)#total 1*1=1 graph
# ax = Axes3D(fig)
#
# s1 = ax.scatter(didntLike[:, 0], didntLike[:, 1], didntLike[:, 2], c = '#66FF00')
# s2 = ax.scatter(smallDoses[:, 0], smallDoses[:, 1], smallDoses[:, 2], c = '#EE0066')
# s3 = ax.scatter(largeDoses[:, 0], largeDoses[:, 1], largeDoses[:, 2], c = '#FFFF00')
# plt.legend((s1, s2, s3), ('didntLike', 'smallDoses', 'largeDoses'), loc=1)
# plt.show()





def data_seperate(dataset, labels, validation_rate, test_number, number_of_groups): # we seperate the data into
                                                                                    #test_set, test_labels, training_set, training_labels

    t = test_number
    m = int(validation_rate * dataset.shape[0])
    test_set = np.array(dataset[t*m:(t+1)*m, :])
    test_labels = np.array(labels[t*m:(t+1)*m])
    if t == 0:
        training_set = np.array(dataset[m:, :])
        training_labels = np.array(labels[m:])
    elif t == number_of_groups-1:
        training_set = np.array(dataset[0:t*m, :])
        training_labels = np.array(labels[0:t * m])
    else:
        training_set = np.append(dataset[0:t*m, :], dataset[(t+1)*m:, :], axis=0)
        training_labels = np.append(labels[0:t*m], labels[(t+1)*m:], axis=0)

    return test_set, test_labels, training_set, training_labels





def errorChecking(dataset, labels, validation_rate): #check the error rate for different validation rate
    n = dataset.shape[0]
    m = int(validation_rate*n)
    number_of_groups = 1/validation_rate
    error_rate = 0
    for i in range(int(number_of_groups)):
        print('round: ', i+1)
        error = 0
        test_set, test_labels, training_set, training_labels = data_seperate(dataset, labels, validation_rate, i, number_of_groups)

        for j in range(m):
            type = classifyFunc(test_set[j, :], training_set, training_labels, 3)
            if type != test_labels[j]:
                error += 1
        print('error is : ', error)
        error = error / n
        error_rate += error

    print('error_rate is : ', error_rate*validation_rate)




# #test 1
# dataset, labels = getDatingSet('datingTestSet.txt')
# dataset, minvalue, range_of_data = normDataSet(dataset)

def convert_into_vector(filename):
    result = np.zeros((1, 1024))
    # print(result.shape)
    with open(filename, 'r') as file:
        for i in range(32):
            line = file.readline()
            for j in range(32):
                result[0, i*32+j] = int(line[j])
    return result


def parepareDataSet(filefolder):
    files = os.listdir(filefolder)
    number_of_files = len(files)
    dataset = np.zeros((number_of_files, 1024))
    labels = np.zeros(number_of_files)
    index = 0
    for item in files:
        type = int(item.split('.')[0].split('_')[0])
        labels[index] = int(type)
        dataset[index, :] = convert_into_vector('KNN/trainingDigits/%s' % item)
        index += 1
    return dataset, labels

def handwrittingTest():
    trainingdataset, traininglabels = parepareDataSet('KNN/trainingDigits')
    testdataset, testlabels = parepareDataSet('KNN/testDigits')

    n1 = trainingdataset.shape[0]
    n2 = testdataset.shape[0]

    error = 0
    with open('KNN/log.txt', 'w') as file:
        for i in range(int(n2)):
            type = classifyFunc(testdataset[i, :], trainingdataset, traininglabels, 1)
            if type == testlabels[i]:
                pass
            else:
                error += 1
                s = 'predict answer is %s while the true number is %s \n'%(type, testlabels[i])
                file.write(s)
                print(s)
        error_rate = error/n2
        print(error_rate)
        file.write(str(error_rate)+'\n')
        file.flush()



handwrittingTest()


#
# error = 0
# times = 1/validation_rate
# error_rate = 0
# for i in range(m):
#     type = classifyFunc(dataset[i], dataset[m:, :], labels[m:], 4)
#     if type != labels[i]:
#         # print(type, 'original one is ', labels[i])
#         error += 1





