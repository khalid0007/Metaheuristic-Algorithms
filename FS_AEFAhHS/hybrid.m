function [initFeatures, initialPerformance, finalFeatures, finalPerformance, finalFeaturesSelected] = hybrid(populationSize, maxIteration, mode)
global trainFeatures trainLabels testFeatures testLabels classifierName paramValue;
featureLength = size(trainFeatures, 2);
countHarmony = populationSize;
harmonyMemorySize = [countHarmony, featureLength + 1];

harmonyMemory = zeros(harmonyMemorySize);

for indx = 1:featureLength
    harmonyMemory(1, indx) = 1;
end

% First half is one
for indx = 1:floor((featureLength + 1)/2)
    harmonyMemory(2, indx) = 1;
end

% Second half is one
for indx = floor((featureLength + 1)/2):featureLength
    harmonyMemory(3, indx) = 1;
end

% Add random chromosomes
for i = 4:countHarmony
   for j = 1:ceil(0.50*featureLength)
      randomIndex = mod(randi(1000000), featureLength) + 1;
      harmonyMemory(i, randomIndex) = 1;
   end
   
   % if not a single feature is selected
   % roughly 10% features are flipped to compendate the scenario
   if sum(harmonyMemory(i, 1:featureLength)) == 0
      for k = 1:ceil(0.1*featureLength)
         randIndexX =  randi(featureLength);
         harmonyMemory(i, randIndexX) = 1;
      end
   end
end

for i = 1:countHarmony
    agents = harmonyMemory(i, 1:featureLength);
    harmonyMemory(i, featureLength + 1) = calFitness(trainFeatures, trainLabels, testFeatures, testLabels, agents, classifierName, paramValue);
end

initialPerformance = modifiedClassify(trainFeatures, trainLabels, testFeatures, testLabels, harmonyMemory(1, 1:featureLength), classifierName, paramValue);
finalPerformance = initialPerformance;

% mode == 1 : hybrid
% mode == 2 : HS
% mode == 3 : AEFA
iteration = [];
fitness = [];

for it = 1:maxIteration + mod(maxIteration + 1, 2)
    if mode == 1
        % Hybrid
        if mod(it,2)
            fprintf("HS: Iteration: %d, Initial Performance: %f, ", it, initialPerformance);
            [harmonyMemory, finalPerformance] = FS_HS(harmonyMemory);
        else
            fprintf("AEFA: Iteration: %d, Initial Performance: %f, ", it, initialPerformance);
            [harmonyMemory, finalPerformance] = AEFA(harmonyMemory(:, 1:featureLength), harmonyMemory(:, 1 + featureLength), it, maxIteration);
        end
        iteration = [iteration, it];
        fitness = [fitness, finalPerformance];
    elseif mode == 2
        % HS
        fprintf("HS: Iteration: %d, Initial Performance: %f, ", it, initialPerformance);
        [harmonyMemory, finalPerformance] = FS_HS(harmonyMemory);
        iteration = [iteration, it];
        fitness = [fitness, finalPerformance];
    else
        % AEFA
        fprintf("AEFA: Iteration: %d, Initial Performance: %f, ", it, initialPerformance);
        [harmonyMemory, finalPerformance] = AEFA(harmonyMemory(:, 1:featureLength), harmonyMemory(:, 1 + featureLength), it, maxIteration);
        iteration = [iteration, it];
        fitness = [fitness, finalPerformance];
    end
end

harmonyMemory = specialSort(harmonyMemory);

initFeatures = featureLength;
finalFeatures = sum(harmonyMemory(1, 1:featureLength));
finalFeaturesSelected = harmonyMemory(1, 1:featureLength);

clc;
fprintf("NfeatureB: %d, NfeatureA: %d\n", initFeatures, finalFeatures);
fprintf("Initial Performance: %f, Final Accuracy: %f\n", initialPerformance, finalPerformance);
end



