import numpy as np



def dataLoad():
    bbs_train = cleanFileData('data/bbs-train.txt')
    imgs_train = cleanFileData('data/imgs-train.txt')
    label_train = cleanFileData('data/label-train.txt')
    labels = np.array([int(x[1]) for x in label_train])
    return bbs_train,imgs_train,labels


def cleanFileData(fileName):
    data = []
    finalData = []
    file = open(fileName, 'r')
    for line in file.readlines():
        data.append(line.rstrip('\n').split(','))
    for x in data : finalData.append(x[0].split())
    return np.array(finalData).astype(np.float32)

if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    print("done")


