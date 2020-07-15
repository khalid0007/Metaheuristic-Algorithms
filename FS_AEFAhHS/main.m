clear;
clc;
global trainFeatures trainLabels testFeatures testLabels classifierName paramValue;

% Classifier Details
classifierName = "knn";
paramValue = 5;

addpath('D:\Project\UCI_datasets');
DATASET_NAMES = {'BreastCancer', 'BreastEW', 'CongressEW', 'Exactly', 'Exactly2', 'HeartEW', 'Ionosphere', 'Lymphography', 'M-of-n', 'PenglungEW', 'sonar', 'SpectEW', 'Tic-tac-toe', 'Vote', 'Wine', 'Zoo', 'KrVsKpEW', 'WaveformEW'};

details = zeros([18, 4]);

for dataset = 1:18
    name_data = DATASET_NAMES{dataset};
    featureSet = csvread([name_data '.csv']);
    initPop = size(featureSet, 2);
    [trainSet, testSet] = splitTT(featureSet, 0.80);

    trainFeatures = trainSet(:, 1:size(trainSet, 2) - 1);
    trainLabels = trainSet(:, size(trainSet, 2));

    testFeatures = testSet(:, 1:size(testSet, 2) - 1);
    testLabels = testSet(:, size(testSet, 2));
    
    popSize = [10, 20, 30, 40, 50];
    AEHS = zeros(1, 5);
    HS = zeros(1, 5);
    AEFA = zeros(1, 5);
    
    for p = 1:5
        for it = 1:5
           clc;
           [ift, fft, iacc, fcc] = hybrid(p*10, 30, 1);
           
           if fcc > AEHS(p)
               AEHS(p) = fcc;
           end
        end
        
        for it = 1:1
           clc;
           [ift, fft, iacc, fcc] = hybrid(p*10, 30, 2);
           
           if fcc > HS(p)
               HS(p) = fcc;
           end
        end
        
        for it = 1:1
           clc;
           [ift, fft, iacc, fcc] = hybrid(p*10, 30, 3);
           
           if fcc > AEFA(p)
               AEFA(p) = fcc;
           end
        end
    end
    
    h = figure;
    plot(popSize, AEHS, 'b', popSize, HS, 'm', popSize, AEFA, 'g');
    title(name_data);
    xlabel('Population Size');
    ylabel('Accuracy');
    legend('AEHS','HS', 'AEFA');

    savefig(h, ['Pop\' name_data '.fig']);
    close all;
end
