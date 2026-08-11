function  [p,objF,conV,FES]=diversity(p,objF,conV_ori,minVar,maxVar,FES,problemSetNum,problem,aaa)

[popsize,n]=size(p);
conV = sum(max(conV_ori,0),2);
if std(conV)<1.e-8 && isempty(find(conV==0))
    p(1:popsize,:)=repmat(minVar,popsize,1)+rand(popsize,n).*repmat((maxVar-minVar),popsize,1);
    [objF, conV]=fitness(problemSetNum, p, problem,aaa);
    FES=FES+popsize;
    'diversity'
else
    conV = conV_ori;
end
