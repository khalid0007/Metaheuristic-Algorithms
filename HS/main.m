clc;
featureSet = csvread('LTrP50.csv');

[trainSet, testSet] = splitTT(featureSet, 0.80);

trainFeatures = trainSet(:, 1:size(trainSet, 2) - 1);
trainLabels = trainSet(:, size(trainSet, 2));

testFeatures = testSet(:, 1:size(testSet, 2) - 1);
testLabels = testSet(:, size(testSet, 2));

x = [];
y = [];

i = 20;
while i <= 100
    x = [x i];
    y = [y FS_HS(trainFeatures, trainLabels, testFeatures, testLabels, i, 50)];
    i = i + 10;
end

% fprintf("Initial Accuracy : %f\n", initialPerformance);
% fprintf("Final Accuracy : %f\n", finalPerformance);

ln = plot(x, y);
xlabel('Population');
ylabel('Accuracy %');
title('Accuracy % vs Population Size');

ln.LineWidth = 2;
ln.Color = [0 0.5 0.5];
ln.Marker = '*';
ln.MarkerEdgeColor = 'b';

% finalSelectedFeatures = logical([x, 1]);
% % 
% disp(sum(finalSelectedFeatures))
% % 
% csvwrite("MI_PCC_LTrP50_3.csv", featureSet(:, finalSelectedFeatures));
