function [consistency] = RandomForest(Tr, cl1, Ts, cl2)
nTrees=500;
B = TreeBagger(nTrees,Tr,cl1, 'Method', 'classification'); 
predChar1 = B.predict(Ts);  % Predictions is a char though. We want it to be a number.
c = str2double(predChar1);
consistency=sum(c==cl2)/length(cl2);
end

