function [consistency] = RandomForest(Tr, cl1, Ts, cl2, agent, paramValue)
consistency = 0;

nTrees = paramValue; % nuber of trees
Tr = Tr(:, agent == 1);
Ts = Ts(:, agent == 1);

if size(Tr, 2) ~= 0
    B = TreeBagger(nTrees,Tr,cl1, 'Method', 'classification'); 
    predChar1 = B.predict(Ts);  % Predictions is a char though. We want it to be a number.
    c = str2double(predChar1);
    consistency=sum(c==cl2)/length(cl2);
end

end

