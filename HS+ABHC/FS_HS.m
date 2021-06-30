function [finalPerformance, selectedFeatures, initialPerformance, harmonyMemory] = FS_HS(trainFeatures, trainLabels, testFeatures, testLabels, countHarmony, maxIteration)
% Feature selection using Harmony Search
% Paper : "A New Heuristic Optimization Algorithm : Harmony Search"\ 
% Author : Zong Woo Geem and Joong Hoon K1im, G. V. Loganathan
% 
% <STEPS OF HARMOMY SEARCH ALGORITH>
% Step 1. Initialize a Harmony Memory (HM).
% Step 2. Improvise a new harmony from HM.
% Step 3. If the new harmony is better than minimum harmony in HM,
%         include the new harmony in HM, and exclude the minimum harmony from HM.
% Step 4. If stopping criteria are not satisfied, go to Step 2.
% <=================================>
% 
% 
% Here each feature is considered a musical instrument
% and each musical instrument aka. feature has two tones 
% 1 and 0 : 1 represents the selection of the the particular feature
% (instrument), 0 represent the oposite operation
% 
% AIM
% Our aim is to select a most optimum feature set using harmony search
% algorithm
% 
% 
% Code Author: Khalid Hassan Sheikh
% BCSE, Jadavpur University (2018-2022)
% 
% parameter: train and test features and class labels
% output : Final selected featues with optimal accuracy 


% <Frequently used variables> %
featureLength = size(trainFeatures, 2);
% countHarmony = 100;
accuracyThresold = 1.00;
harmonyMemorySize = [countHarmony, featureLength + 1];
initialPerformance = 0.00;
finalPerformance  = 0.00;
% <\Frequently used variables> %

% <Intialisation> %


% Score calculation
mi = mutualInformation(trainFeatures, trainLabels); % Mutual Informantion
pcc = sum(corrcoef(trainFeatures), 1)./featureLength; % Pearson correlation coefficient
score = mi.*pcc;

% Initialise harmony memory
harmonyMemory = zeros(harmonyMemorySize);

% All features selected
% harmonyMemory(1, :) = ones(1, featureLength + 1);

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
%    for j = 1:featureLength
%       randomIndex = randi(i-1);
%       harmonyMemory(i, j) = harmonyMemory(randomIndex, j);
%    end
   
   for j = 1:ceil(0.05*featureLength)
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

%calculate score
for i = 1:countHarmony
%     clc;
%     fprintf("Knn run No: %d\n", i);
    chromosome = harmonyMemory(i, 1:featureLength);
%     performanceScore = svmClassifier(trainFeatures, trainLabels, testFeatures, testLabels, chromosome);
    performanceScore = SVC(trainFeatures, trainLabels, testFeatures, testLabels, chromosome, 'polynomial');
%     performanceScore = sum(score(chromosome == 1))/sum(chromosome)*SVC(trainFeatures, trainLabels, testFeatures, testLabels, chromosome, 'polynomial');
    harmonyMemory(i, featureLength + 1) = performanceScore;
%     fprintf("Accuracy %f\n", 100*performanceScore);
%     pause();
end

% initialPerformance = svmClassifier(trainFeatures, trainLabels, testFeatures, testLabels, harmonyMemory(i, 1:featureLength));
initialPerformance = SVC(trainFeatures, trainLabels, testFeatures, testLabels, harmonyMemory(1, 1:featureLength), 'polynomial');

harmonyMemory = specialSort(harmonyMemory);
% <\Intialisation> %


% <Improvise> %
terminateCondition = false;
iteration = 1;

Interations = [0];
Accuracies = [initialPerformance];

while iteration < maxIteration && ~terminateCondition
    clc;
    fprintf("Iteration NO: %d\n", iteration);
    chromosome = zeros(1, featureLength);
    
    % Best performance
%     finalPerformance = harmonyMemory(1, featureLength + 1);
%     
%     fprintf("Initial Accuracy : %f\n", 100*initialPerformance);
%     fprintf("Final Accuracy : %f\n", 100*finalPerformance);
    
    for i = 1:featureLength
        randomIndex = randi(countHarmony);
        chromosome(i) = harmonyMemory(randomIndex, i);
    end
    
%     for i = 1:ceil(0.1*featureLength)
%        randomIndex = randi(featureLength);
%        chromosome(randomIndex) = 1 - chromosome(randomIndex);
%     end
    
    % <\Improvise> %
    
    % Calculate the performance score for newly improvised chromosome
%     performanceScore = svmClassifier(trainFeatures, trainLabels, testFeatures, testLabels, chromosome);
    performanceScore = SVC(trainFeatures, trainLabels, testFeatures, testLabels, chromosome, 'polynomial');
%     performanceScore = sum(score(chromosome == 1))/sum(chromosome)*SVC(trainFeatures, trainLabels, testFeatures, testLabels, chromosome, 'polynomial');
    
    % If better hramony is generated by improvisation replace the minimum
    % quality harmony from harmony memory
    % <Better Harmony Generation> %
    if performanceScore > harmonyMemory(countHarmony, featureLength + 1)
        harmonyMemory(countHarmony, :) = [chromosome, performanceScore];
        
        % Sort according to performanceScore
        harmonyMemory = specialSort(harmonyMemory);
    end
    % <\Better Harmony Generation> %
    
    iteration = iteration + 1;
    
    % If desired performance is met, then break out of the loop
    % Or termination condition satified
    if performanceScore > accuracyThresold
        terminateCondition = true;
%         break;
    end
    
    adaptiveBeta(harmonyMemory)
    finalPerformance = harmonyMemory(1, featureLength + 1);
    
    if mod(iteration, 50) == 0
        Interations = [Interations, iteration];
        Accuracies = [Accuracies, finalPerformance];
    end
    % <\Stopping Criterion> %
end


% Best performance
% finalPerformance = svmClassifier(trainFeatures, trainLabels, testFeatures, testLabels, harmonyMemory(1, 1:featureLength));
finalPerformance = SVC(trainFeatures, trainLabels, testFeatures, testLabels, harmonyMemory(1, 1:featureLength), 'polynomial');

% Best possible harmony
selectedFeatures = harmonyMemory(1, 1:featureLength);

clc;
fprintf("Initial Accuracy : %f\n", initialPerformance);
fprintf("Final Accuracy : %f\n", finalPerformance);

ln = plot(Interations, Accuracies);
xlabel('Iterations')
ylabel('Performance')
title('Performance vs Iterations')

ln.LineWidth = 2;
ln.Color = [0 0.5 0.5];
ln.Marker = 'o';
ln.MarkerEdgeColor = 'b';

% for i = 1:countHarmony
%    fprintf("%d %f\n", sum(harmonyMemory(i, 1:featureLength)), harmonyMemory(i, featureLength + 1));
% end

end

