function [arr] = specialSort(arr)

% last element is the score for each harmony

[row, column] = size(arr);

for i = 0:row - 1
    for j = 1:row - i - 1
        if arr(j, column) < arr(j + 1, column)
            % Store complete array in temp
            temp = arr(j, :);
            %swap
            arr(j, :) = arr(j + 1, :);
            arr(j + 1, :) = temp;
        end
    end
end

end

