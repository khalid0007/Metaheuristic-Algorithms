function [fitness, accuracy] = calFitness(trainFeatures, trainLabels, testFeatures, testLabels, agents, classifierName, paramValue)
numAgents = size(agents, 1);
featureLength = size(agents, 2);
fitness = zeros([numAgents, 1]);
accuracy = zeros([numAgents, 1]);
contributionAcc = 0.99;

for i = 1:numAgents
   % Accuracy normalised in [0, 1]
   accuracy(i) = modifiedClassify(trainFeatures, trainLabels, testFeatures, testLabels, agents(i, :), classifierName, paramValue);
   fitness(i) = contributionAcc*accuracy(i) + (1 - contributionAcc)*(1 - sum(agents(i, :))/featureLength);
end

end