function [] = main_RBF(problem)
% clc;
% close all;
% clear all;
% problem = 1;
format long;
format compact;
addpath(genpath('../')); 
addpath(genpath('./dace/'))
set(0,'RecursionLimit',2000000);
popsize = 50;
totalTime = 25;
numTrain = 300;
numEvalu = 1;
omega = 2;
maxGen = 1000;%totalFES/numEvalu;
maxSubGen1 = 10;
genBasedG = 5;
problemSetNum = 2010;
if problemSetNum==2006
    problemSet = [1:24];
%     problemIndex = [6]; 
    best2006 = [-15.0000000000,-0.8036191042,-1.0005001000,-30665.5386717834,5126.4967140071,-6961.8138755802...
        24.3062090681,-0.0958250415,680.6300573745,7049.2480205286,0.7499000000,-1.0000000000...
        0.0539415140,-47.7648884595,961.7150222899,-1.9051552586,8853.5396748064,-0.8660254038...
        32.6555929502,0.2049794002,193.7245100700,236.4309755040,-400.0551000000,-5.5080132716];
elseif problemSetNum == 2010
    problemSet = [1:18];
%     problemIndex = [1:18]; 
    n = 30;   
    if n == 30
        numTrain = 500;
    end
else
    fprintf('Error Test Set\n');
end
fprintf('CEC%d_%d\n',problemSetNum,problem);
if problemSetNum == 2006
    [minVar, maxVar, n, aaa] = problemSelection2006(problem);
elseif problemSetNum == 2010
    [minVar,maxVar] = problemSelection2010(problem,n);aaa=0;
end
sizeBioPar = 1;

if problemSetNum == 2006
    if problem == 1
        NC = 9;
    elseif problem == 2 || problem == 6 || problem == 8 || problem == 15 || problem == 24
        NC = 2;
    elseif problem == 7
        NC = 8;
    elseif problem == 9 || problem == 17
        NC = 4;
    elseif problem == 4 || problem == 10 || problem == 21 || problem == 23
        NC = 6;
    elseif problem == 5 || problem == 19
        NC = 5;
    elseif problem == 3 || problem == 11 || problem == 12 || problem == 25
        NC = 1;
    elseif problem == 13 || problem == 14
        NC = 3;
    elseif problem == 16
        NC = 38;
    elseif problem == 18
        NC = 13;
    elseif problem == 20 || problem == 22
        NC = 20;
    end
elseif problemSetNum == 2010 %% 1 7 8 13 14 15
    if problem == 1 || problem == 5 || problem == 6 || problem == 12 || problem == 18
        NC = 2; % 5 6%% 12 18 1+1
    elseif problem == 2 || problem == 13 || problem == 14 || problem == 15 || problem == 17
        NC = 3;%2 17 2+1
    elseif problem == 3 || problem == 7 || problem == 8 || problem == 9 || problem == 10 || problem == 11
        NC = 1; % 3 9 10 11
    elseif problem == 4 || problem == 16
        NC = 4; % 4 4;16 2+2
    end
end
totalFES=1000; %!!!!!!
delete(gcp('nocreate'))
p = parpool(NC+2);
p.IdleTimeout = 1000;
p; 
% q = parallel.pool.DataQueue;
% afterEach(q, @disp)

for time = 1:1%totalTime
    FES = 1;
    gen=1;
    spmd
        labBarrier;
        if labindex == 1
            start_time = cputime;
            run_time = zeros(totalTime,1);
            feasiRatio = zeros(totalFES, totalTime);
            evolveSolution = ones(totalFES,totalTime)*1e16;
            evolveConstrain = ones(totalFES,totalTime)*1e16;
            allConstraints = [];
            allBestIndLoc = [];
            processFilename = sprintf('dis_ToSel_%d_%d_n%d_FEs%d_runs%d_%d_proc.csv', problemSetNum, problem,n, totalFES, totalTime,time);
            processFile = fopen(processFilename,'w');
            start_time = cputime;
            archiveX = [];
            archiveY = [];
            archiveC = [];
            selPoolX = [];
            selPoolY = [];
            selPoolC = [];
            selPoolFlag = zeros(1,NC,genBasedG-1);
            X=0;
%             fprintf('this is %d\n',labindex);
            pop = lhsamp(popsize,n);
            pop = pop.*repmat(maxVar-minVar,popsize,1)+repmat(minVar,popsize,1);
            srgtSRGT_Y = labReceive(NC+2,0);
            for i = 2:NC+1
                srgtSRGT_C(i-1) = labReceive(i,0);
            end
            objF = srgtsRBFEvaluate(pop,srgtSRGT_Y);
            for i = 1:NC
                conV(:,i) = srgtsRBFEvaluate(pop,srgtSRGT_C(i));
            end
            VAR0=max(sum(max(conV(popsize,:),0),2));         
            cp=(-log(VAR0)-6)/log(1-0.5);
%             fprintf('%f\n',VAR0);
        else
%             fprintf('this is %d\n',labindex);
            workerX = lhsamp(numTrain,n);
            workerX = workerX.*repmat(maxVar-minVar,numTrain,1)+repmat(minVar,numTrain,1);
            [workerY,workerC] = fitness(problemSetNum,workerX,problem,aaa);
            workerObj = [workerC,workerY];
            srgtOPT =srgtsRBFSetOptions(workerX(1:numTrain,:),workerObj(1:numTrain,labindex-1));
            srgtSRGT=srgtsRBFFit(srgtOPT);
            labSend(srgtSRGT,1,0);
        end
    end
    while FES <= totalFES
        spmd
            labBarrier;
            if labindex == 1
                if X < 0.5
                    VAR=VAR0*(1-X)^cp;
                else
                    VAR=0;
                end
                % diversity
                conVsum = sum(max(conV,0),2);
                if std(conVsum)<1.e-8 && isempty(find(conVsum==0))
                    pop = lhsamp(popsize,n);
                    pop = pop.*repmat(maxVar-minVar,popsize,1)+repmat(minVar,popsize,1);
                    objF = srgtsRBFEvaluate(pop,srgtSRGT_Y);
                    for i = 1:NC
                        conV(:,i) = srgtsRBFEvaluate(pop,srgtSRGT_C(i));
                    end
                    fprintf('diversity');
                end
                % evolution
                temp=DEgenerator(pop,objF,conVsum,minVar,maxVar,gen,maxGen);
                objFtemp = srgtsRBFEvaluate(temp,srgtSRGT_Y);
                for i = 1:NC
                    conVtemp(:,i) = srgtsRBFEvaluate(temp,srgtSRGT_C(i));
                end
                [trial,objFtrial,conVtrial] = preSelect(temp,objFtemp,conVtemp);
                [trial,objFtrial,conVtrial] = sortAll(trial,objFtrial,conVtrial);
                % firstSelection individual-based selection: constraints
                trialOri = trial;
                objFtrialOri = objFtrial;
                conVtrialOri = conVtrial;
                firstGen = 0;
                infeaC = [];
                for i = 1:NC
                    if min(conVtrial(:,i)) > 0
                        infeaC = [infeaC;i];
                    end
                end
%                 fprintf('infeaC %d\n',length(infeaC));
                while min(sum(max(conVtrial,0),2))>0 && firstGen<maxSubGen1
                    for i = 1:NC
                        % if all individuals are illegal in a constraint
                        firstGenSub = 0;
                        trialNew = trialOri;
                        objFtrialNew = objFtrialOri;
                        conVtrialNew = conVtrialOri;
                        while min(conVtrialNew(:,i)) > 0 && firstGenSub<maxSubGen1 %%% TRY IF
                            trialNewOff = DEBestgenerator(trialNew,conVtrialNew(:,i),minVar,maxVar);
                            objFtrialNewOff = srgtsRBFEvaluate(trialNewOff,srgtSRGT_Y);
                            conVtrialNewOff = zeros(size(trialNewOff,1),NC)*1e16;
                            for j = 1:NC
                                conVtrialNewOff(:,j) = srgtsRBFEvaluate(trialNewOff,srgtSRGT_C(j));
                            end
                            for j = 1:size(trialNew,1)
                                if conVtrialNewOff(j,i) < conVtrialNew(j,i)
                                    trialNew(j,:) = trialNewOff(j,:);
                                    objFtrialNew(j,:) = objFtrialNewOff(j,:);
                                    conVtrialNew(j,:) = conVtrialNewOff(j,:); 
                                end
                            end
                            firstGenSub = firstGenSub + 1;
                        end
                        if firstGenSub ~= 0
                            trial = [trial;trialNewOff];
                            objFtrial = [objFtrial;objFtrialNew];
                            conVtrial = [conVtrial;conVtrialNew];
                        end
                    end
                    firstGen = firstGen + 1;
                end
                [trial,objFtrial,conVtrial] = sortAll(trial,objFtrial,conVtrial);
                selX1 = trial(1,:);
                selY1 = objFtrial(1,:);
                selC1 = conVtrial(1,:);
                if size(archiveX,1)~=0
                    while ismember(selX1,archiveX,'rows')
                        selX1 = normDisturb(selX1,n,minVar,maxVar);
                    end 
                end
                if mod(gen,genBasedG)~=0
                    increasedFES1 = 0;
                    for ii = 1:NC
                        sendFlag = 0;
%                         fprintf('infeaC %d\n',length(infeaC));
                        if ~isempty(infeaC)
                            for jj = 1:length(infeaC)
                                if ii == infeaC(jj)
                                    sendFlag = 1;
                                    break;
                                end
                            end
                        end
%                         fprintf('senFlag %d\n',sendFlag);
%                         fprintf('before send %d %d\n',ii,sendFlag);
                        if sendFlag == 1
                            labSend(selX1,ii+1,gen);
                            selC1(1,ii) = labReceive(ii+1,gen);
                            srgtSRGT_C(ii) = labReceive(ii+1,gen+totalFES);
                        else
                            labSend([],ii+1,gen);
                            ignoreC = labReceive(ii+1,gen);
                            ignoreCmodel = labReceive(ii+1,gen+totalFES);
                        end
                        increasedFES1 = increasedFES1 + sendFlag;
%                         fprintf('increasedFES1 %d\n',length(increasedFES1));
                    end
                    
                    selPoolX = [selPoolX;selX1];
                    selPoolY = [selPoolY;selY1];
                    selPoolC = [selPoolC;selC1];
                    indPool = size(selPoolX,1);
                    selPoolFlag(1,infeaC,indPool) = 1;
%                     if gen > 1
%                         evolveSolution(FES,time) = evolveSolution(gen-1,time);
%                         evolveConstrain(FES,time) = evolveConstrain(gen-1,time);
%                         feasiRatio(FES,time) = feasiRatio(gen-1,time);
%                     end
                elseif mod(gen,genBasedG)==0
                    increasedFES2 = 0;
%                     fprintf('1 %d %d %d %d %d %d\n',size(selPoolX,1),size(selPoolY,1),size(selPoolC,1),size(selPoolFlag,1),size(selPoolFlag,2),size(selPoolFlag,3));
                    [selPoolX,selPoolY,selPoolC,selPoolFlag] = sortAllFlag(selPoolX,selPoolY,selPoolC,selPoolFlag);
%                     fprintf('2 %d %d %d %d %d %d\n',size(selPoolX,1),size(selPoolY,1),size(selPoolC,1),size(selPoolFlag,1),size(selPoolFlag,2),size(selPoolFlag,3));
                    selX2 = selPoolX(1,:);
                    selY2 = selPoolY(1,:);
                    selC2 = selPoolC(1,:);
                    selFlag2 = selPoolFlag(:,:,1);
                    if size(archiveX,1)~=0
                        while ismember(selX2,archiveX,'rows')
                            selX2 = normDisturb(selX2,n,minVar,maxVar);
                        end 
                    end
                    labSend(selX2,NC+2,gen+totalFES*2);
                    selY2 = labReceive(NC+2,gen+totalFES*2);
                    srgtSRGT_Y = labReceive(NC+2,gen+totalFES*3);
                    % evaluate the unknown constraints
                    for ii = 1:NC
                        if selFlag2(ii) == 0
                            labSend(selX2,ii+1,gen+totalFES*3);
                            selC2(1,ii) = labReceive(ii+1,gen+totalFES*4);
                            srgtSRGT_C(ii) = labReceive(ii+1,gen+totalFES*5);
                        else
                            labSend([],ii+1,gen+totalFES*3);
                            ignoreC = labReceive(ii+1,gen+totalFES*4);
                            ignoreCmodel= labReceive(ii+1,gen+totalFES*5);
                        end
%                         labSend(NC-sum(selFlag2)+1,ii+1,gen+totalFES*3);
                    end
                    if size(archiveX,1)~=0
                        archiveX = [selX2;archiveX];
                        archiveY = [selY2;archiveY];
                        archiveC = [selC2;archiveC];
                    else
                        archiveX = selX2;
                        archiveY = selY2;
                        archiveC = selC2;
                    end
                    increasedFES2 = increasedFES2 + NC - sum(selFlag2)+1;
                    selPoolX = [];
                    selPoolY = [];
                    selPoolC = [];
                    selPoolFlag = zeros(1,NC,genBasedG);
                    if size(archiveX,1)>=popsize
                        pop = archiveX(1:popsize,:);
                        objF = archiveY(1:popsize,:);
                        conV = archiveC(1:popsize,:);
                    else
                        addedNum = popsize-size(archiveX,1);
                        addedX = lhsamp(addedNum,n);
                        addedX = addedX.*repmat(maxVar-minVar,addedNum,1)+repmat(minVar,addedNum,1);
                        addedY = srgtsRBFEvaluate(addedX,srgtSRGT_Y);
                        addedC = ones(size(addedX,1),NC)*1e16;
                        for i = 1:NC
                            addedC(:,i) = srgtsRBFEvaluate(addedX,srgtSRGT_C(i));
                        end
                        pop = [archiveX;addedX];
                        objF = [archiveY;addedY];
                        conV = [archiveC;addedC];
                    end
                    [archiveX,archiveY,archiveC] = sortAll(archiveX,archiveY,archiveC);
                    if size(archiveX,1)>=popsize
                        [feaIndP,~] = judgeFeasible(conV);
                    else
                        [feaIndP,~] = judgeFeasible(archiveC);
                    end
                    if ~isempty(feaIndP)
                        feaRatio = length(feaIndP)/popsize;
                    else
                        feaRatio = 0;
                    end
                    [feaIndA,infeaIndA] = judgeFeasible(archiveC);
                    if isempty(feaIndA)
                        bestSolution = NaN;
                        bestConstrain = sum(max(archiveC(1,:),0),2);
                    else
                        bestSolution = archiveY(1,:);
                        bestConstrain = sum(max(archiveC(1,:),0),2);
                    end
%                     fprintf('%d %d %d; %d %d %f; %f %f\n', time, gen, FES, length(feaIndA), length(infeaIndA),feaRatio,bestConstrain, bestSolution);
                    fprintf(processFile,'%d,%d,%d,%d,%d,%f,%f,%f\n', time, gen, FES, length(feaIndA), length(infeaIndA),feaRatio,bestConstrain, bestSolution);
                    evolveSolution(FES,time) = bestSolution;
                    evolveConstrain(FES,time) = bestConstrain;
                    feasiRatio(FES,time) = feaRatio;
                    allConstraints = [allConstraints;archiveC(1,:)];
                    allBestIndLoc = [allBestIndLoc;archiveX(1,:)];
                    X=X+1/maxGen;
                end                
            elseif labindex ~= 1
                if mod(gen,genBasedG)~=0
                    if labindex ~= NC+2
                        selX1 = labReceive(1,gen);
                        if ~isempty(selX1)
                            [selY1,selC1] = fitness(problemSetNum,selX1,problem,aaa);
                            selObj1 = [selC1,selY1];
                            labSend(selObj1(labindex-1),1,gen);
                            if ~ismember(selX1,workerX,'rows')
                                workerX = [workerX;selX1];
                                workerY = [workerY;selY1];
                                workerC = [workerC;selC1];
                                [workerX,workerY,workerC] = sortAll(workerX,workerY,workerC);
                                workerObj = [workerC,workerY];
                                srgtOPT =srgtsRBFSetOptions(workerX(1:numTrain,:),workerObj(1:numTrain,labindex-1));
                                srgtSRGT=srgtsRBFFit(srgtOPT);
                                labSend(srgtSRGT,1,gen+totalFES);
                            end
                        elseif isempty(selX1)
                            labSend([],1,gen);
                            labSend([],1,gen+totalFES);
                        end
                    end
                elseif mod(gen,genBasedG)==0
                    if labindex == NC+2
                        selX2 = labReceive(1,gen+totalFES*2);
                        [selY2,selC2] = fitness(problemSetNum,selX2,problem,aaa);
                        selObj2 = [selC2,selY2];
                        labSend(selObj2(labindex-1),1,gen+totalFES*2);
                        if ~ismember(selX2,workerX,'rows')
                            workerX = [workerX;selX2];
                            workerY = [workerY;selY2];
                            workerC = [workerC;selC2];
                            [workerX,workerY,workerC] = sortAll(workerX,workerY,workerC);
                            workerObj = [workerC,workerY];
                            srgtOPT =srgtsRBFSetOptions(workerX(1:numTrain,:),workerObj(1:numTrain,labindex-1));
                            srgtSRGT=srgtsRBFFit(srgtOPT);
                        end
                        labSend(srgtSRGT,1,gen+totalFES*3);
                    elseif labindex ~= NC+2
                        selX2 = labReceive(1,gen+totalFES*3);
                        if ~isempty(selX2)
                            [selY2,selC2] = fitness(problemSetNum,selX2,problem,aaa);
                            selObj2 = [selC2,selY2];
                            labSend(selObj2(labindex-1),1,gen+totalFES*4);
                            if ~ismember(selX2,workerX,'rows')
                                workerX = [workerX;selX2];
                                workerY = [workerY;selY2];
                                workerC = [workerC;selC2];
                                [workerX,workerY,workerC] = sortAll(workerX,workerY,workerC);
                                workerObj = [workerC,workerY];
                                srgtOPT =srgtsRBFSetOptions(workerX(1:numTrain,:),workerObj(1:numTrain,labindex-1));
                                srgtSRGT=srgtsRBFFit(srgtOPT);
                                labSend(srgtSRGT,1,gen+totalFES*5);
                            end
                        elseif isempty(selX2)
                            labSend([],1,gen+totalFES*4);
                            labSend([],1,gen+totalFES*5);
                        end
                    end
                end
            end 
        end
        if mod(gen,genBasedG)~=0
            FES = FES + increasedFES1{1};
%             fprintf('1 %d\n',increasedFES1{1});
        elseif mod(gen,genBasedG)==0
            FES = FES + increasedFES2{1};
%             fprintf('2 %d\n',increasedFES2{1});
        end
        gen = gen+1;
    end
    spmd
        if labindex == 1
            fclose(processFile);
            run_time(time,1) = cputime-start_time;
            filename = [sprintf('results_allConstraints_%d',problemSetNum) '/' sprintf('dis_SAEA_%d_%d_n%d_FEs%d_runs%d_%d_allCons.csv', problemSetNum, problem,n, totalFES, totalTime,time)];
            dlmwrite(filename, allConstraints, 'precision', '%.6f');
            filename2 = [sprintf('results_allConstraints_%d',problemSetNum) '/' sprintf('dis_SAEA_%d_%d_n%d_FEs%d_runs%d_%d_allIndLoc.csv', problemSetNum, problem,n, totalFES, totalTime,time)];
            dlmwrite(filename2, allBestIndLoc, 'precision', '%.6f');
            
%             filename = sprintf('SA-C2oDE_%d_%d_n%d_FEs%d_runs%d_%d_fit.csv', problemSetNum, problem,n, totalFES, totalTime,time);
%             dlmwrite(filename, evolveSolution(:,time), 'precision', '%.6f');
%             filename = sprintf('SA-C2oDE_%d_%d_n%d_FEs%d_runs%d_%d_con.csv', problemSetNum, problem,n, totalFES, totalTime,time);
%             dlmwrite(filename, evolveConstrain(:,time), 'precision', '%.6f');
%             filename = sprintf('SA-C2oDE_%d_%d_n%d_FEs%d_runs%d_%d_time.csv', problemSetNum, problem,n, totalFES, totalTime,time);
%             dlmwrite(filename, run_time(time,1), 'precision', '%.6f');
%             filename = sprintf('SA-C2oDE_%d_%d_n%d_FEs%d_runs%d_%d_rf.csv', problemSetNum, problem,n, totalFES, totalTime,time);
%             dlmwrite(filename, feasiRatio(:,time), 'precision', '%.6f');
            end_time = cputime;
            (end_time-start_time)/60
        end 
    end
end

delete(gcp('nocreate'))
end



