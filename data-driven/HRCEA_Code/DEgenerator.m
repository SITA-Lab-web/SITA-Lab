function temp=DEgenerator(p,objF,conV,minVar,maxVar)

global gen maxGen

lu=[minVar;maxVar];
[popsize,n]=size(p);

trial=zeros(3*popsize,n);

for i=1:popsize
    
    
    %% DE current to best
    l=rand;
    if l <= 1/3
        F  = .6;
    elseif l <= 2/3
        F= 0.8;
    else
        F = 1.0;
    end
    
    l=rand;
    if l <= 1/3
        CR  = .1;
    elseif l <= 2/3
        CR = 0.2;
    else
        CR = 1.0;
    end
    
    indexset=1:popsize;
    indexset(i)=[];
    r1=floor(rand*(popsize-1))+1;
    xr1=indexset(r1);
    indexset(r1)=[];
    r2=floor(rand*(popsize-2))+1;
    xr2=indexset(r2);
    indexset(r2)=[];
    r3=floor(rand*(popsize-3))+1;
    xr3=indexset(r3);
    
   
   
    [~,best]=min(objF);
    
    v=p(i,:)+F*(p(best,:)-p(i,:))+F*(p(xr1,:)-p(xr2,:));
    
    
    % Handle the elements of the mutant vector which violate the boundary
    w = find(v < lu(1, :));
    if ~isempty(w)
        l=rand;
        if l < (1-length(find(conV==0))/popsize)^3
            v(1, w) = 2 * lu(1, w) -  v(1, w);
            w1 = find( v(1, w) > lu(2, w));
            if ~isempty(w1)
                v(1, w(w1)) = lu(1, w(w1));
            end
        else
            v(1, w) =  lu(2, w);
        end
    end
    
    y = find(v > lu(2, :));
    if ~isempty(y)
        l=rand;
        if l <  (1-length(find(conV==0))/popsize)^3
            v(1, y) =  2 * lu(2, y) - v(1, y);
            y1 = find(v(1, y) < lu(1, y));
            if ~isempty(y1)
                v(1, y(y1)) = lu(1, y(y1));
            end
        else
            v(1, y) =  lu(2, y);
        end
    end
    
    % Binomial crossover

    t = rand(1, n) < CR;
    j_rand = floor(rand * n) + 1;
    t(1, j_rand) = 1;
    t_ = 1 - t;
    trial(3*(i-1)+1, :) = t .* v + t_ .* p(i,:);
    
%%    
     l=rand;
    if l <= 1/3
        F  = .6;
    elseif l <= 2/3
        F= 0.8;
    else
        F = 1.0;
    end
    
    l=rand;
    if l <= 1/3
        CR  = .1;
    elseif l <= 2/3
        CR = 0.2;
    else
        CR = 1.0;
    end
    
    indexset=1:popsize;
    indexset(i)=[];
    r1=floor(rand*(popsize-1))+1;
    xr1=indexset(r1);
    indexset(r1)=[];
    r2=floor(rand*(popsize-2))+1;
    xr2=indexset(r2);
    indexset(r2)=[];
    r3=floor(rand*(popsize-3))+1;
    xr3=indexset(r3);
     indexset(r3)=[];
    r4=floor(rand*(popsize-4))+1;
    xr4=indexset(r4);
   
   if isempty(find(conV==0))
       [~,best]=min(conV);
   else
       index=find(conV==0);
       r=floor(rand*length(index))+1;
       best=index(r);
   end
   
    v=p(xr1,:)+F*(p(best,:)-p(xr2,:))+F*(p(xr3,:)-p(xr4,:));
    
    
    % Handle the elements of the mutant vector which violate the boundary
    w = find(v < lu(1, :));
    if ~isempty(w)
        l=rand;
        if l < (gen/maxGen)
            v(1, w) = 2 * lu(1, w) -  v(1, w);
            w1 = find( v(1, w) > lu(2, w));
            if ~isempty(w1)
                v(1, w(w1)) = lu(1, w(w1));
            end
        else
            v(1, w) =  lu(2, w);
        end
    end
    
    y = find(v > lu(2, :));
    if ~isempty(y)
        l=rand;
        if l <  (gen/maxGen)
            v(1, y) =  2 * lu(2, y) - v(1, y);
            y1 = find(v(1, y) < lu(1, y));
            if ~isempty(y1)
                v(1, y(y1)) = lu(1, y(y1));
            end
        else
            v(1, y) =  lu(2, y);
        end
    end
    
    % Binomial crossover
    t = rand(1, n) < CR;
    j_rand = floor(rand * n) + 1;
    t(1, j_rand) = 1;
    t_ = 1 - t;
    trial(3*(i-1)+2, :) = t .* v + t_ .* p(i,:);
    
    %% DE current to rand
     l=rand;
    if l <= 1/3
        F  = .6;
    elseif l <= 2/3
        F= 0.8;
    else
        F = 1.0;
    end
    
    
    indexset=1:popsize;
    indexset(i)=[];
    r1=floor(rand*(popsize-1))+1;
    xr1=indexset(r1);
    indexset(r1)=[];
    r2=floor(rand*(popsize-2))+1;
    xr2=indexset(r2);
    indexset(r2)=[];
    r3=floor(rand*(popsize-3))+1;
    xr3=indexset(r3);

    
    v=p(i,:)+rand*(p(xr1,:)-p(i,:))+F*(p(xr3,:)-p(xr2,:));
  
    
    % Handle the elements of the mutant vector which violate the boundary
    w = find(v < lu(1, :));
    if ~isempty(w)
        l=rand;
        if l < 1-(gen/maxGen) || l < (1-length(find(conV==0))/popsize)
            v(1, w) = 2 * lu(1, w) -  v(1, w);
            w1 = find( v(1, w) > lu(2, w));
            if ~isempty(w1)
                v(1, w(w1)) = lu(1, w(w1));
            end
            
        else
 
            v(1, w) =  lu(2, w);
        end
    end
    
    y = find(v > lu(2, :));
    if ~isempty(y)
        l=rand;
        if l <  (1-length(find(conV==0))/popsize)
            v(1, y) =  2 * lu(2, y) - v(1, y);
            y1 = find(v(1, y) < lu(1, y));
            if ~isempty(y1)
                v(1, y(y1)) = lu(1, y(y1));
            end
        else
    
            v(1, y) =  lu(2, y);
        end
    end

    trial(3*(i-1)+3, :) = v ;
  
end
temp=trial;