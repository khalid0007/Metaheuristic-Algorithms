clear;
clc;
featureSet = csvread('LTrP50.csv');
initPop = size(featureSet, 2);

[trainSet, testSet] = splitTT(featureSet, 0.80);

% [trainFeatures, trainLabels] = SMOTE(trainSet(:, 1:size(trainSet, 2) - 1), trainSet(:, size(trainSet, 2)));

trainFeatures = trainSet(:, 1:size(trainSet, 2) - 1);
trainLabels = trainSet(:, size(trainSet, 2));

testFeatures = testSet(:, 1:size(testSet, 2) - 1);
testLabels = testSet(:, size(testSet, 2));

[selectedFeatures,initialPerformance, finalPerformance] = AEFA(trainFeatures, trainLabels, testFeatures, testLabels);
finalSelectedFeatures = logical([selectedFeatures, 1]);

fprintf("Initial Population: %d, Initial Accuracy: %f\n", initPop, initialPerformance);
fprintf("Final Population: %d, Final Accuracy: %f\n", sum(finalSelectedFeatures) - 1, finalPerformance);
csvwrite("AEFA_LTrP50.csv", featureSet(:, finalSelectedFeatures));