clear;
clc;
global trainFeatures trainLabels testFeatures testLabels classifierName paramValue;

% Classifier Details
classifierName = "knn";
paramValue = 5;

addpath('D:\Project\UCI_datasets');
DATASET_NAMES = {'BreastCancer', 'BreastEW', 'CongressEW', 'Exactly', 'Exactly2', 'HeartEW', 'Ionosphere', 'Lymphography', 'M-of-n', 'PenglungEW', 'sonar', 'SpectEW', 'Tic-tac-toe', 'Vote', 'Wine', 'Zoo'};

details = zeros([16, 4]);

for dataset = 3:3
    featureSet = csvread([DATASET_NAMES{dataset} '.csv']);
    initPop = size(featureSet, 2);
    [trainSet, testSet] = splitTT(featureSet, 0.80);

    trainFeatures = trainSet(:, 1:size(trainSet, 2) - 1);
    trainLabels = trainSet(:, size(trainSet, 2));

    testFeatures = testSet(:, 1:size(testSet, 2) - 1);
    testLabels = testSet(:, size(testSet, 2));
    
    for iter = 1:1
        clc;
        [ift, fft, iacc, fcc] = hybrid(20, 30);
        
        if details(dataset, 4) < fcc
           details(dataset, 1) = ift;
           details(dataset, 2) = fft;
           details(dataset, 3) = iacc;
           details(dataset, 4) = fcc;
        end
    end
end


% % Creating result CSV
% csv = fopen('UCC_Results_rf.csv', 'w');
% 
% for dataset = 1:16
%    for i = 1:4
%        fprintf(csv, '%d,',details(dataset, i));
%    end
%    fprintf(csv, '%s\n',DATASET_NAMES{dataset});
% end
% 
% fclose(csv);

