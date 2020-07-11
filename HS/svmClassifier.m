function [performance]=svmClassifier(x,t,x2,t2,chromosome)
    if(sum(chromosome(1,:) == 1) ~= 0)     
        x2=x2(:,chromosome(1, :) == 1);
        x=x(:,chromosome(1, :) == 1);

        s=size(t,1);
        label=zeros(1,s);
        
        for i=1:s
            label(1,i)= t(i); %find(t(i,:),1);
        end
%         disp(max(t));
%         disp(max(label));
        
%         if max(label)==2
%             svmModel=fitcsvm(x,label,'KernelFunction','polynomial','Standardize',true,'ClassNames',[1 2]);
%         else
            class=zeros(1,max(label));
            for i=1:max(label)
                class(i)=i;
            end
%             disp(class);
            temp = templateSVM('Standardize', 1, 'KernelFunction', 'linear', 'Solver', 'SMO');
            svmModel = fitcecoc(x,label,'Learners',temp,'FitPosterior',1,'ClassNames',class,'Coding','onevsall');
%         end
        
        %svmModel=fitcsvm(x,label,'KernelFunction','linear','Standardize',true,'ClassNames',[1 2]);
        [label, score] = predict(svmModel,x2);
        
        performance = sum(double(label == t2))./size(t2, 1);
        
% %         disp(size(score));
% %         disp(size(label));
%         
%         % save score score;
%         
%         s=size(t2,1);
%         lab=zeros(s,1);
%         for i=1:s
%             lab(i,1)=find(t2(i,:),1);           
%         end
%         c = sum(lab ~= label)/s; % mis-classification rate
%         performance=1-c;
% %         fprintf('Number of features - %d\n',sum(chromosome(1,:)==1));
% %         fprintf('The correct classification is %f\n',(100*performance));
    else
        performance=0.00;
    end
end