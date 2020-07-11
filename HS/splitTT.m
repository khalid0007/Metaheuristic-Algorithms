function [trainSet, testSet] = splitTT(featureSet, ratio)

maxClassLabel = -1;

[row, column] = size(featureSet);

for i = 1:row
    maxClassLabel = max(maxClassLabel, featureSet(i, column));
end

countClass = zeros(1, maxClassLabel);
countTrain = zeros(1, maxClassLabel);

for i = 1:row
    countClass(featureSet(i, column)) = countClass(featureSet(i, column)) + 1;
end

trainSetSize = 0;

for i = 1:maxClassLabel
    countTrain(i) = floor(ratio*countClass(i));
    trainSetSize = trainSetSize + countTrain(i);
end

trainSet = zeros(trainSetSize, column);
testSet = zeros(row - trainSetSize, column);

countTemp = zeros(1, maxClassLabel);
trainIndex = 1; % Count
testIndex = 1;


for i = 1:row
   countTemp(featureSet(i, column)) = countTemp(featureSet(i, column)) + 1;
   
   if countTemp(featureSet(i, column)) <= countTrain(featureSet(i, column))
       trainSet(trainIndex, :) = featureSet(i, :);
       trainIndex = trainIndex + 1;
   else
       testSet(testIndex, :) = featureSet(i, :);
       testIndex = testIndex + 1;
   end
end

% countClass

end
