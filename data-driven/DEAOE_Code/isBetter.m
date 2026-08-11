function [flag] = isBetter(popY,popC_ori,offY,offC_ori)
    offC = sum(max(offC_ori,0),2);
    popC = sum(max(popC_ori,0),2);
    flag = 0;
    if popC > 0 && offC == 0
        flag = 1;
    elseif popC>0&&offC>0
        if offC < popC
            flag = 1;
        end
    elseif popC==0&&offC==0
        if offY < popY
            flag = 1;
        end
    end 
end

