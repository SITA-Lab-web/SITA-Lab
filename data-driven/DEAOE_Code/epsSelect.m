function  [p,fit,voi]=epsSelect(p,fit,voi_ori,trial,fittrial,voitrial_ori)
%%
[popsize,~]=size(p);
global VAR 
voi = sum(max(voi_ori,0),2);
voitrial = sum(max(voitrial_ori,0),2);

for i=1:popsize
    if voitrial(i) < VAR && voi(i) < VAR
        if fittrial(i) < fit(i)
            
            p(i,:)=trial(i,:);
            fit(i)=fittrial(i);
            voi(i)=voitrial(i);

        end
    elseif voitrial(i) == voi(i) 
        if fittrial(i) < fit(i)

            p(i,:)=trial(i,:);
            fit(i)=fittrial(i);
            voi(i)=voitrial(i);
            
        end
    elseif voitrial(i) < voi(i)

        p(i,:)=trial(i,:);
        fit(i)=fittrial(i);
        voi(i)=voitrial(i);
        
    end
end


