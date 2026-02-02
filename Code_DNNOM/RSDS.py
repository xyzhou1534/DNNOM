import numpy as np
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

minNumSample = 10


class BinaryTree:
    """An Special BinaryTree.

        Construct a special binary tree, store the data in the nodes of the tree,
        node labels, left and right subtree positions


    """

    def __init__(self, labels=np.array([]), datas=np.array([])):
        self.label = labels
        self.data = datas
        self.leftChild = None
        self.rightChild = None

    def set_rightChild(self, rightObj):
        self.rightChild = rightObj

    def set_leftChild(self, leftObj):
        self.leftChild = leftObj

    def get_rightChild(self):
        return self.rightChild

    def get_leftChild(self):
        return self.leftChild

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label


def RSDS(train_data, tree_num=100):
    """Handling data noise using completely random forest judgment.

        Establish a tree_num completely random tree. The data label in each leaf node
        of the tree is compared with the parent node label to obtain the noise judgment
        label of each data in the case of a tree, and all the completely random tree noise
        judgment labels are combined to vote to determine the noise data. Denoised data
        set after processingEstablish a tree_num completely random tree. The data label
        in each leaf node of the tree is compared with the parent node label to obtain
        the noise judgment label of each data in the case of a tree, and all the completely
        random tree noise judgment labels are combined to vote to determine the noise data.
        Denoised data set after processing

        Parameters
        ----------
        train_data :Numpy type data set.

        tree_num :Total number of random trees.

    """

    m, n = train_data.shape
    forest = np.array([])
    for i in range(10):
        tree = CRT(train_data)
        visiTree = visitCRT(tree)   #shape(2,n)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]       #get labels
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    noiseForest = np.sum(forest, axis=1)
    nn = 0.5 * tree_num
    #原始算法中只保留边界点，去掉噪声点和安全点
    noiseForest = np.array(list(map(lambda x: 1 if x >= nn or x == 0 else 0, noiseForest)))
    denoiseTraindata = deleteNoiseData(train_data, noiseForest)
    return denoiseTraindata





def CRT(data):
    """Build A Completely Random Tree.

        Add a column at the end of the data, store the initial sequence
        number of each piece of data, call the function ‘generateTree’
        spanning tree

         Parameters
         ----------
         data :Numpy type data set

     """
    numberSample = data.shape[0]
    orderAttribute = np.arange(numberSample).reshape(numberSample, 1)  # (862, 1)
    data = np.hstack((data, orderAttribute))
    completeRandomTree = generateTree(data)
    return completeRandomTree


def generateTree(data, uplabels=[]):
    """Iteratively Generating A Completely Random Tree.

         Complete random tree by random partitioning of random attributes

         Parameters
         ----------
         data :Numpy type data set

         uplabels :rootlabel

     """
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size

    if numberAttribute == 0:
        return None

    numberAttribute = numberAttribute - 2  # Subtract the added serial number and label

    # The category of the current data, also called the node category
    labelNumKey = []  # todo
    if numberSample == 1:  # Only one sample left
        labelvalue = data[0][0]
        rootdata = data[0][numberAttribute + 1]
    else:
        labelNum = Counter(data[:, 0])
        labelNumKey = list(labelNum.keys())  # Key (label)
        labelNumValue = list(labelNum.values())  # Value (quantity)
        labelvalue = labelNumKey[labelNumValue.index(max(labelNumValue))]  # Vote to find the label
        rootdata = data[:, numberAttribute + 1]
    rootlabel = np.hstack((labelvalue, uplabels))  # todo

    # Call the class 'BinaryTree', passing in tags and data
    CRTree = BinaryTree(rootlabel, rootdata)
    '''
    The 'rootlabel' and 'rootdata' are obtained above, the 'rootlabel' is a label (derived by voting), 
    the 'rootdata' is a series of serial numbers, and finally the class BinaryTree is called.
    '''
    # There are at least two conditions for the tree to stop growing:
    # 1 the number of samples is limited;
    # 2 the first column is all equal
    if numberSample < minNumSample or len(labelNumKey) < 2:
        # minNumSample defaults to 10 or only 1 of the label types are left.
        return CRTree
    else:
        maxCycles = 1.5 * numberAttribute  # Maximum number of cycles
        # maxCycles = 2
        i = 0
        while True:
            # Once a data exception occurs: except for the above two exceptions that
            # stop the tree growth condition, that is, the error data, the loop here will not stop.
            i += 1
            splitAttribute = np.random.randint(1, numberAttribute)  # Randomly select a list of attributes
            if splitAttribute > 0 and splitAttribute < numberAttribute + 1:
                dataSplit = data[:, splitAttribute]
                uniquedata = list(set(dataSplit))
                if len(uniquedata) > 1:
                    break
            if i > maxCycles:  # Tree caused by data anomaly stops growing
                return CRTree
        sv1 = np.random.choice(uniquedata)
        i = 0
        while True:
            i += 1
            sv2 = np.random.choice(uniquedata)
            if sv2 != sv1:
                break
            if i > maxCycles:
                return CRTree
        splitValue = np.mean([sv1, sv2])
        '''
        The above randomly selected rows and columns are obtained, and the final 'splitValue' is an average
        '''

        # Call split function
        leftdata, rightdata = splitData(data, splitAttribute, splitValue)

        # Set the left subtree, the right subtree
        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree




def visitCRT(tree):
    """
    Traversing the tree to get the relationship between the data and the node label.

         The traversal tree stores the data number and node label stored in each node of the
         completely random tree.

         Parameters
         ----------
         tree :Root node of the tree.

        returns a matrix of two rows and N columns, the first row is the index of the sample, 
        and the second row is the threshold of the label noise.
        e.g.
        [[ 36. 499. 547. 557. 563. 587.]
        [  0.   0.   0.   0.   0.   0.]]

    """
    if not tree.get_leftChild() and not tree.get_rightChild():  # If the left and right subtrees are empty
        data = tree.get_data()  # data is the serial number of the sample
        labels = checkLabelSequence(tree.get_label())  # Existing tag sequence
        try:
            labels = np.zeros(len(data)) + labels
        except TypeError:
            pass
        result = np.vstack((data, labels))
        return result
    else:
        resultLeft = visitCRT(tree.get_leftChild())
        resultRight = visitCRT(tree.get_rightChild())
        result = np.hstack((resultLeft, resultRight))
        return result


def deleteNoiseData(data, noiseOrder):
    """Delete noise points in the training set.

         Delete the noise points in the training set according to the noise
         judgment result of each data in noiseOrder.

         Parameters
         ----------
         data :Numpy type data set.

         noiseOrder :Determine if each piece of data is a list of noise.

     """
    m, n = data.shape
    data = np.hstack((data, noiseOrder.reshape(m, 1)))
    redata = np.array(list(filter(lambda x: x[n] == 0, data[:, ])))
    redata = np.delete(redata, n, axis=1)
    return redata


"""check whether the label of the parent node and the leaf node are consistent."""


def checkLabelSequence(labels):
    """Check label sequence.

         Check if the leaf node is the same as the parent node.

         Parameters
         ----------
         labels :label sequence.

     """
    return 1 if labels[0] != labels[1] else 0



def splitData(data, splitAttribute, splitValue):
    """Dividing data sets.

         Divide the data into two parts, leftData and rightData, based on the splitValue
         of the split attribute column element.

         Parameters
         ----------
         data:Numpy type data set.

         splitAttribute:Randomly selected attributes when dividing.

         splitValue:Dividing the value obtained by dividing the selected attribute.

     """
    rightData = np.array(list(filter(lambda x: x[splitAttribute] > splitValue, data[:, ])))
    leftData = np.array(list(filter(lambda x: x[splitAttribute] <= splitValue, data[:, ])))
    return leftData, rightData




def RSDS_smote(train_data, tree_num=100):
    '''
    description: 
                RSDS针对多分类不平衡数据集的改进方法,
                1.先对所有样本去噪，
                2.对剩下数据集进行压缩
                    (1)挑选数量最多的类A，去掉A类的内部点
                    (2)if 压缩后的A仍然最多:直接return
                        else 有其他类>压缩后的A,压缩其他类,递归
                3.每个类最多压缩一次。如果压缩次数=类别数，直接返回数据
    param {
        train_data:  ndarray  [label,feture1,feture2,feture3,......]
        tree_num:
    }
    return {
        resutl: the data after rsds_smote
                ndarray,[label,feture1,feture2,feture3,......]
    }
    '''

    m, n = train_data.shape
    forest = np.array([])
    for i in range(tree_num):
        tree = CRT(train_data)
        visiTree = visitCRT(tree)   #shape(2,n)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]       #get labels
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    noiseForest = np.sum(forest, axis=1)
    print(noiseForest)


    '''step1:先统一去噪，剩下内部点和边界点'''
    nn = 0.5 * tree_num     #阈值0.5
    innerForest = np.array(list(map(lambda x: 1 if x == 0 else 0, noiseForest))) #内部点
    noiseForest = np.array(list(map(lambda x: 1 if x >=nn else 0, noiseForest)))#噪声点
    print(  
            Counter(innerForest),'\n',
            Counter(noiseForest)
        )

    denoiseTraindata = deleteNoiseData(train_data, noiseForest)#把noiseForest中0的留下
    innerForest_denoise = innerForest[np.where(noiseForest==0)]   #去噪之后的边界点矩阵[0,1]
    print('统一去噪后的数量:\t',len(denoiseTraindata),innerForest_denoise.shape,'\n')


    '''step2:压缩多数类,递归删除多数类中的内部点'''
    time = 0    #压缩次数
    num_dict = Counter(denoiseTraindata[:,0])     #dict:   去噪后数据的   标签：数量
    max_times = len(num_dict)       #最大压缩次数

    def step2(denoiseTraindata,times:int,num_dict:dict,max_times:int):
        '''
            description: 
            param {
                denoiseTraindata:   去噪后的数据集,
                                    ndarray  [label,feture1,feture2,feture3,......]
                times:压缩次数
            }
            return {
                train_data: 递归压缩后的数据集,ndarray  
                            [label,feture1,feture2,feture3,......]
            }
        '''
        major_labels = max(num_dict,key=num_dict.get)   #去噪后数据中最多的那一类的标签
        major_inner_Forest = innerForest_denoise[np.where(denoiseTraindata[:,0]==major_labels)]#多数类边界点矩阵

        # print(num_dict)

        major_data = denoiseTraindata[np.where(denoiseTraindata[:,0]==major_labels)]  #去噪后的多数类
        # print('压缩前的多数类样本数:\t',len(major_data),'\t\t标签是:\t',major_labels)
        major_data_deinner = major_data[np.where(major_inner_Forest==0)]
        # print('压索后的多数类样本数:\t',len(major_data_deinner))

        train_data = np.vstack([
            major_data_deinner,
            denoiseTraindata[np.where(denoiseTraindata[:,0]!=major_labels)]
        ])      #合并数据集
        times+=1    


        # print('压索后的数据总数:\t',len(train_data),'\n')
        #压缩后的数据集的  新多数类
        new_major_labels = max(Counter(train_data[:,0]),key=Counter(train_data[:,0]).get)


        if new_major_labels == major_labels:
            # print('多数类压缩后仍然是最多的，直接返回结果,压缩次数:\t',times)
            return train_data
        elif  new_major_labels != major_labels and times < max_times:
            # print('多数类压缩后少于其他类,每个类最多压缩一次,递归压缩新的多数类,压缩次数:\t',times)
            del num_dict[major_labels]
            # print('删除字典中的已经压缩过的键:\t',num_dict)
            return step2(train_data,times,num_dict,max_times)  #递归
        elif times == max_times:
            # print('压缩次数已满，直接返回数据集:\t',times,len(train_data))
            return train_data


    result = step2(denoiseTraindata,time,num_dict,max_times)
    return result



def RSDS_zhou(train_data, tree_num=100):
    """ Handling data noise using completely random forest judgment.
        Establish a tree_num completely random tree. The data label in each leaf node
        of the tree is compared with the parent node label to obtain the noise judgment
        label of each data in the case of a tree, and all the completely random tree noise
        judgment labels are combined to vote to determine the noise data. Denoised data
        set after processingEstablish a tree_num completely random tree. The data label
        in each leaf node of the tree is compared with the parent node label to obtain
        the noise judgment label of each data in the case of a tree, and all the completely
        random tree noise judgment labels are combined to vote to determine the noise data.
        Denoised data set after processing
        Parameters
        ----------
        train_data :Numpy type data set.
        tree_num :Total number of random trees.
    """

    m, n = train_data.shape
    forest = np.array([])
    for i in range(10):
        tree = CRT(train_data)
        visiTree = visitCRT(tree)   #shape(2,n)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]       #get labels
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    orignal_Forest = np.sum(forest, axis=1)     #没有去噪的权重
    nn = 0.5 * tree_num     #阈值
    
    #只需要去除噪声点，保留边界点和内部点。 1:true, 0:false
    noiseForest = np.array(list(map(lambda x: 1 if x >=nn else 0, orignal_Forest)))#噪声点矩阵
    weight_denoise = orignal_Forest[np.where(noiseForest == 0)]#去噪后的权重矩阵

    denoiseTraindata = deleteNoiseData(train_data, noiseForest)#把noiseForest中0的留下
    return denoiseTraindata,weight_denoise  #去噪后的