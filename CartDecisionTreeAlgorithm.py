from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support,roc_curve,auc,roc_auc_score
from sklearn.metrics import confusion_matrix,roc_curve
import StringIO
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,VarianceThreshold
import time
import sys
import pylab as pl
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np

#My Code here
import neurolab as nl
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,ClassificationDataSet
from sklearn.metrics import accuracy_score
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

import numpy as np
class CartDecisionTreeAlgorithm:



    def __init__(self,train_file,test_file):
        self.train_file = train_file
        self.test_file = test_file

    def classify_random_forest(self,X,Y):
       rf = RandomForestClassifier(n_estimators=100,max_features='auto')

       return rf.fit(X,Y)

    def classify(self,X,Y):
        return tree.DecisionTreeClassifier().fit(X,Y)

    def classify_SVM(self,X,Y):
        svmmodel = svm.SVC()
        return svmmodel.fit(X,Y)

    def loadData(self,file_name):
        with open(file_name) as f:
            data = []
            for line in f:
                line = line.strip().split(",")
                data.append([x for x in line])

        return  data

#My code starts

    def neuralNetworksTrain(self):
        alldata = ClassificationDataSet( 23, 1, nb_classes=2)
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        for i in range(0,len(features)):
            alldata.addSample(features[i], target[i])        
            
        tstdata, trndata = alldata.splitWithProportion(0.25)
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        
        INPUT_FEATURES = 23
        CLASSES = 2
        HIDDEN_NEURONS = 200
        WEIGHTDECAY = 0.1
        MOMENTUM = 0.1
        fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,outclass=LinearLayer)
        trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY)
        
        for k in range(0,5):
            trainer.trainEpochs(1)
            pred = trainer.testOnClassData(dataset=tstdata)
            actual = tstdata['class']
            self.computeAccuracy(actual,pred)
        #trnresult = percentError(trainer.testOnClassData(),
        #                         trndata['class'])
        #tstresult = percentError(trainer.testOnClassData(
        #                         dataset=tstdata), tstdata['class'])

        #    print("epoch: %4d" % trainer.totalepochs,
        #      "  train error: %5.2f%%" % trnresult,
        #      "  test error: %5.2f%%" % tstresult)
            #out = fnn.activateOnDataset(griddata)
            # the highest output activation gives the class
            #out = out.argmax(axis=1)
            #out = out.reshape(X.shape)
            print "Precision recall F score support metrics for Neural Networks "
            print precision_recall_fscore_support(actual,pred)
            print "confusion matrix"
            print confusion_matrix(actual,pred)


        
        
        
        
#My code ends

    def learnSVM(self):
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        model = self.classify_SVM(features,target)

        test_input_data = self.loadData(self.test_file)
        actualOutput = [x[1] for x in test_input_data]
        actualOutput = actualOutput[1:]
        features = [x[2:] for x in test_input_data]
        features = features[1:]

        predictedOutput = model.predict(features)
        #print predictedOutput
        #print actualOutput
        self.computeAccuracy(predictedOutput,actualOutput)
        print "Precision recall F score support metrics for SVM "
        print precision_recall_fscore_support(actualOutput,predictedOutput)
        print "confusion matrix"
        print confusion_matrix(actualOutput,predictedOutput)





    def learnRF(self):
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        # feature selection
        #features_new = self.doFeatureSelection(features,target,10)
        model = self.classify_random_forest(features,target)

        test_input_data = self.loadData(self.test_file)
        actualOutput = [x[1] for x in test_input_data]
        actualOutput = actualOutput[1:]
        features = [x[2:] for x in test_input_data]
        features = features[1:]

        predictedOutput = model.predict(features)
        #print predictedOutput
        #print actualOutput
        self.computeAccuracy(predictedOutput,actualOutput)
        print "Precision recall F score support metrics for RF "
        print precision_recall_fscore_support(actualOutput,predictedOutput)
        print "confusion matrix"
        print confusion_matrix(actualOutput,predictedOutput)
        predictedOutput = np.array(predictedOutput)
        actualOutput = np.array(actualOutput)
        X= []
        Y=[]
        for a in predictedOutput:
            X.append(int(a))
        for a in actualOutput:
            Y.append(int(a))
       # self.plotROC(Y,X)
        result = zip(Y,X)
        self.write_To_File(result,"RF-predictions.csv")

    def write_To_File(self,result,filename):
        f = open(filename,'w')
        for pair in result:
            f.write(str(pair[0]) + "\t" + str(pair[1]) )
            f.write("\n")
        f.close()

    def learnCART(self):
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        # feature selection
        #features_new = self.doFeatureSelection(features,target)
        model = self.classify(features,target)

        test_input_data = self.loadData(self.test_file)
        actualOutput = [x[1] for x in test_input_data]
        actualOutput = actualOutput[1:]
        features = [x[2:] for x in test_input_data]
        features = features[1:]

        predictedOutput = model.predict(features)
        #print predictedOutput
        #print actualOutput
        self.computeAccuracy(predictedOutput,actualOutput)
        print "Precision recall Fscore support metrics for CART "
        print precision_recall_fscore_support(actualOutput,predictedOutput)
        print "\nconfusion matrix\n"
        print confusion_matrix(actualOutput,predictedOutput)
        self.printDTRules(model)
        X= []
        Y=[]
        for a in predictedOutput:
            X.append(int(a))
        for a in actualOutput:
            Y.append(int(a))
        self.plotROC(Y,X)
        result = zip(Y,X)
        self.write_To_File(result,"cart-predictions.csv")



    def printDTRules(self,model):
        dot_data = StringIO.StringIO()
        #with open("rules_1L.dot","w") as output_file:
        out = tree.export_graphviz(model, out_file="rules_1L.dot")


    def plotROC(self,actualOutput,predictedOutput):
        fpr, tpr, thresholds = roc_curve(predictedOutput,actualOutput)
        roc_auc = auc(fpr,tpr)
        print "Area under the ROC curve : %f" % roc_auc
        pl.clf()
        pl.plot(fpr,tpr,label="ROC Curve (area = %0.2f)" % roc_auc)
        pl.plot([0,1],[0,1],'k--')
        pl.xlim(0.0,1.0)
        pl.ylim(0.0,1.0)
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example')
        pl.legend(loc="lower right")
        pl.show()

    def computeAccuracy(self,predictedOutput,actualOutput):
        count = 0
        for i in range(len(predictedOutput)):
            if predictedOutput[i] == actualOutput[i]:
                count = count +1
        print "Accuracy for model is "
        print float(count)/float(len(predictedOutput))

    def doFeatureSelection(self,features,target,k):
        features_int = np.array(features,dtype=float)
        target_int = np.array(target,dtype=float)
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        features_new = sel.fit_transform(features_int)
        #features_new = SelectKBest(chi2,k=10).fit_transform(features_int,target_int)
        return features_new

'''start_time = time.time()
print "Decision Tree\n"
obj = CartDecisionTreeAlgorithm('../clean_data/clean_train_1L.csv','../clean_data/clean_test_1k.csv')
#obj.learnCART()


print "\nRandom Forests\n"

obj.learnRF()
time_elapsed = time.time() - start_time
print "Time taken " + str(time_elapsed)


print "\n SVM \n"
#obj.learnSVM()'''


print 'Neural Networks'
obj = CartDecisionTreeAlgorithm('../clean_data/clean_test_1k.csv','../clean_data/clean_test_1L.csv')
obj.neuralNetworksTrain()