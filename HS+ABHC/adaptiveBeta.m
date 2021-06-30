function [harmonyMemory] =  adaptiveBeta(harmonyMemory)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

global trainFeatures trainLabels testFeatures testLabels classifierName paramValue;

countHarmony = size(harmonyMemory, 1);
featureLen = size(harmonyMemory, 2) - 1;
percent = 0.30;
bmin = 0.15; %parameter: (can be made 0.01)
bmax = 1;
maxIter = 10; % parameter: (can be increased )

fprintf("Implementing local search!!\n");

for iterNum = 1:maxIter
    for harmonyNum = 1:countHarmony
        neighbour = harmonyMemory(harmonyNum, 1:featureLen);
        s = RandStream('mlfg6331_64');
        y = randsample(s,featureLen,floor(percent*featureLen),true);
        
        for i = 1:floor(percent*featureLen)
            neighbour(1,y(i)) = 1 - neighbour(1,y(i));
        end
        
        beta = bmin + (iterNum / maxIter)*(bmax - bmin);
        
        for fN = 1:featureLen
           rr = rand();
           if rr <= beta
               neighbour(1, fN) = harmonyMemory(harmonyNum, fN);
           end
        end
        
        acc = modifiedClassify(trainFeatures, trainLabels, testFeatures, testLabels, neighbour, classifierName, paramValue);
        
        if acc >= harmonyMemory(harmonyNum, featureLen + 1)
            harmonyMemory(harmonyNum, :) = [neighbour acc];
        end
        
    end
end

harmonyMemory = specialSort(harmonyMemory);

disp(harmonyMemory(:, featureLen + 1));

fprintf("\nPress enter to continue:: ");
pause();

end

