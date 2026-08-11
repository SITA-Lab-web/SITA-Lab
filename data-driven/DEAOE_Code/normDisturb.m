function [selX1] = normDisturb(selX1,n,minVar,maxVar)
    for i = 1:n
        selX1(1,i) = selX1(1,i) + normrnd(0,0.05*(maxVar(i)-minVar(i)));
        if selX1(1,i)<minVar(i)||selX1(1,i)>maxVar(i)
            selX1(1,i) = rand*(maxVar(i)-minVar(i))+minVar(i);
        end
    end
end

