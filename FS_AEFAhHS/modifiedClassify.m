function [accuracy] = modifiedClassify(trainFeatures, trainLabels, testFeatures, testLabels, agent, classifierName, paramValue)
accuracy = 0.00;

if classifierName == "svm"
    accuracy = SVC(trainFeatures, trainLabels, testFeatures, testLabels, agent, paramValue);
elseif classifierName == "knn"
    accuracy = knnClassifier(trainFeatures, trainLabels, testFeatures, testLabels, agent, paramValue);
elseif classifierName == "rf"
    accuracy = RandomForest(trainFeatures, trainLabels, testFeatures, testLabels, agent, paramValue);
else
   fprintf("Wrong classifier name!\n"); 
end

end