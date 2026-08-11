warning off;
clc;
close all;
clear all;
format long;
format compact;
addpath(genpath('../')); 
addpath(genpath('./dace/'))
set(0,'RecursionLimit',2000000);
'SA-C2oDE_CAMM'

thresholdFH = 0;
popsize = 40;
totalTime = 25;
totalFES=1000;%240000;
numTrain = 300;
numEvalu = 1;
omega = 2;
global gen maxGen VAR 
maxGen = totalFES/numEvalu;
problemSetNum = 2006;
if problemSetNum==2006
    problemSet = [1:24];
    problemIndex = [2,4,19,24];
    best2006 = [-15.0000000000,-0.8036191042,-1.0005001000,-30665.5386717834,5126.4967140071,-6961.8138755802...
        24.3062090681,-0.0958250415,680.6300573745,7049.2480205286,0.7499000000,-1.0000000000...
        0.0539415140,-47.7648884595,961.7150222899,-1.9051552586,8853.5396748064,-0.8660254038...
        32.6555929502,0.2049794002,193.7245100700,236.4309755040,-400.0551000000,-5.5080132716];
elseif problemSetNum == 2010
    problemSet = [1:18];
    problemIndex = [8:18]; %%%10D 8:18
    n = 10;   
else
    fprintf('Error Test Set\n');
end
for problem = problemSet(problemIndex)
    fprintf('CEC%d_%d\n',problemSetNum,problem);
    if problemSetNum == 2006
        [minVar, maxVar, n, aaa] = problemSelection2006(problem);
    elseif problemSetNum == 2010
        [minVar,maxVar] = problemSelection2010(problem,n);aaa=0;
    end
    sizeBioPar = 1;
    NC = 1;
    run_time = zeros(totalTime,1);
    feasiRatio = zeros(totalFES, totalTime);
    evolveSolution = ones(totalFES,totalTime)*1e16;
    evolveConstrain = ones(totalFES,totalTime)*1e16;
    for time = 1:25
%         processFilename = ['process/' sprintf('LR-C2oDE_%d_%d_n%d_FEs%d_runs%d_%d_fit.csv', problemSetNum, problem,n, totalFES, totalTime,time)];
%         processFile = fopen(processFilename,'w');
%         processFilename2 = ['process/' sprintf('LR-C2oDE_%d_%d_n%d_FEs%d_runs%d_%d_probs.csv', problemSetNum, problem,n, totalFES, totalTime,time)];
%         processFile2 = fopen(processFilename2,'w');
        start_time = cputime;
        archiveX=lhsamp(numTrain,n);
        archiveX = archiveX.*repmat(maxVar-minVar,numTrain,1)+repmat(minVar,numTrain,1);
        [archiveY,archiveC]=fitness(problemSetNum,archiveX,problem,aaa);
        srgtOPT_Y =srgtsRBFSetOptions(archiveX,archiveY);
        srgtSRGT_Y=srgtsRBFFit(srgtOPT_Y);
        for i = 1:NC
            srgtOPT_C(i) =srgtsRBFSetOptions(archiveX,archiveC);
            srgtSRGT_C(i)=srgtsRBFFit(srgtOPT_C(i));
        end
        [archiveX,archiveY,archiveC] = sortAll(archiveX,archiveY,archiveC);
        FES = 0;
        gen=1;    X=0;
        VAR0=max(sum(max(archiveC(popsize,1),0),2));         
        cp=(-log(VAR0)-6)/log(1-0.5);
        trueFea = ones(totalFES,popsize*3)*(-1);
        predFea = ones(totalFES,popsize*3)*(-1);
        for FES = 1:totalFES
            count1 = 0;
            count2 = 0;
            count3 = 0;
            % adjusting the threshold
            if X < 0.5
                VAR=VAR0*(1-X)^cp;
            else
                VAR=0;
            end
            p = archiveX(1:popsize,:);
            objF = archiveY(1:popsize,:);
            conV = archiveC(1:popsize,:);
            [p,objF,conV,FES]=diversity(p,objF,conV,minVar,maxVar,FES,problemSetNum,problem,aaa);
            % generating next generation
            temp=DEgenerator(p,objF,sum(max(conV,0),2),minVar,maxVar);
            objFtemp = zeros(size(temp,1),1);
            conVtemp = zeros(size(temp,1),NC);
%             fprintf(processFile2,'%d,', FES);
%             conTrainX = archiveX;
%             conTrainC = archiveC;
            [conTrainX,conTrainC] = trainingDataSelection(archiveX,archiveC,numTrain);
            feaInd = find(conTrainC<=0);
            infeaInd = find(conTrainC>0);
            weights = ones(size(conTrainX,1),1);
            for i = 1:size(conTrainC,1)
                if conTrainC(i,1)>0
                    conTrainC(i,1) = 0;
                    weights(i,1) = (length(feaInd))/size(conTrainX,1);
                else
                    conTrainC(i,1)=1;
                    weights(i,1) = (length(infeaInd))/size(conTrainX,1);
                end
            end
            mdl = fitglm(conTrainX,conTrainC,'linear','Distribution','Binomial','link','logit');
            [ypred,yci] = predict(mdl,temp);
%             [ypred,yci] = predict(mdl,temp,'Alpha',0.1,'Simultaneous',true);
            mdl.LogLikelihood;
            for i=1:size(temp,1)
                objFtemp(i,1)=srgtsRBFEvaluate(temp(i,:),srgtSRGT_Y);
                for j = 1:NC
                    conVtemp(i,j)=srgtsRBFEvaluate(temp(i,:),srgtSRGT_C(j));
                end
                if length(feaInd)>=thresholdFH
                    if conVtemp(i,1)<=0&&ypred(i,1)<=0.05%min(length(infeaInd),length(feaInd))/min(500,size(archiveX,1))
                        count1 = count1+1;
                        conVtemp(i,1) = -conVtemp(i,1);
                    elseif conVtemp(i,1)>0&&ypred(i,1)>=0.95%max(length(infeaInd),length(feaInd))/min(500,size(archiveX,1)) 
                        count2=count2+1;
                        conVtemp(i,1) = -conVtemp(i,1);
                    else%if ypred(i,1)<max(length(infeaInd),length(feaInd))/min(500,size(archiveX,1)) && ypred(i,1)>min(length(infeaInd),length(feaInd))/min(500,size(archiveX,1))
                        count3 = count3+1;
                    end
                end
%                 fprintf(processFile2,'%.10f,', forProb(i,1));
            end
%             fprintf(processFile2,'\n');
            conVtemp = max(conVtemp,0);
            [trial, objFtrial, conVtrial]=preSelect(temp,objFtemp,conVtemp);
            [trial,objFtrial,conVtrial] = sortAll(trial,objFtrial,conVtrial);
            selX = trial(1,:);
            selY = objFtrial(1,:);
            selC = conVtrial(1,:);
            if ismember(selX,archiveX,'rows')
                for i = 1:n
                    selX(1,i) = selX(1,i) + normrnd(0,0.05*(maxVar(i)-minVar(i)));
                    if selX(1,i)<minVar(i)||selX(1,i)>maxVar(i)
                        selX(1,i) = rand*(maxVar(i)-minVar(i))+minVar(i);
                    end
                end
            end
            [selY,selC] = fitness(problemSetNum,selX,problem,aaa); 
            archiveX = [selX;archiveX];
            archiveY = [selY;archiveY];
            archiveC = [selC;archiveC];
            % [realY,realC] = fitness(problemSetNum,temp,problem,aaa);
            % realCsum = sum(max(realC,0),2);
            % feaOnes = find(realCsum<=0);
            % infeaOnes = find(realCsum>0);
            % trueFea(FES,:) = realCsum';
            % trueFea(FES,feaOnes) = 1;
            % trueFea(FES,infeaOnes) = 0;
            % conVtempsum = sum(max(conVtemp,0),2);
            % feaOnes = find(conVtempsum<=0);
            % infeaOnes = find(conVtempsum>0);
            % predFea(FES,:) = conVtempsum';
            % predFea(FES,feaOnes) = 1;
            % predFea(FES,infeaOnes) = 0;
            [archiveX,archiveY,archiveC] = sortAll(archiveX,archiveY,archiveC);
            srgtOPT_Y =srgtsRBFSetOptions(archiveX,archiveY);
            srgtSRGT_Y=srgtsRBFFit(srgtOPT_Y);
            for i = 1:NC
                srgtOPT_C(i) =srgtsRBFSetOptions(archiveX,archiveC(:,i));
                srgtSRGT_C(i)=srgtsRBFFit(srgtOPT_C(i));
            end
            [feaIndP,infeaIndP] = judgeFeasible(conV);
            if ~isempty(feaIndP)
                feaRatio = length(feaIndP)/(length(feaIndP)+length(infeaIndP));
            else
                feaRatio = 0;
            end
            [feaIndA,infeaIndA] = judgeFeasible(archiveC);
            if isempty(feaIndA)
                bestSolution = NaN;
                bestConstrain = sum(max(archiveC(1,:),0),2);
            else
                bestSolution = archiveY(1,:);
                bestConstrain = 0;
            end
            fprintf('%d %d; %d %d; %f %f; %d %d %d\n', time, FES, length(feaIndA), length(infeaIndA),bestConstrain, bestSolution,count1,count2,count3);
%             fprintf(processFile,'%d,%d,%d,%d,%f,%f,%d,%d %d\n', time, FES, length(feaIndA), length(infeaIndA),bestConstrain, bestSolution,count1,count2,count3);
            evolveSolution(FES,time) = bestSolution;
            evolveConstrain(FES,time) = bestConstrain;
            feasiRatio(FES,time) = feaRatio;
            gen=gen+1;
            X=X+1/maxGen;            
        end
        run_time(time,1) = cputime-start_time;
%         filename = [sprintf('true_pred') '\' sprintf('SA-C2oDE_R+_trueFea_%d_%d_n%d_FEs%d_runs%d_%d_fit.csv', problemSetNum, problem,n, totalFES, totalTime,time)];
%         dlmwrite(filename, trueFea, 'precision', '%.6f');
%         filename = [sprintf('true_pred') '\' sprintf('SA-C2oDE_R+_predFea_%d_%d_n%d_FEs%d_runs%d_%d_fit.csv', problemSetNum, problem,n, totalFES, totalTime,time)];
%         dlmwrite(filename, predFea, 'precision', '%.6f');
        filename = sprintf('SA-C2oDE_CAMM_%d_%d_n%d_FEs%d_runs%d_%d_fit.csv', problemSetNum, problem,n, totalFES, totalTime,time);
        dlmwrite(filename, evolveSolution(:,time), 'precision', '%.6f');
        filename = sprintf('SA-C2oDE_CAMM_%d_%d_n%d_FEs%d_runs%d_%d_con.csv', problemSetNum, problem,n, totalFES, totalTime,time);
        dlmwrite(filename, evolveConstrain(:,time), 'precision', '%.6f');
        filename = sprintf('SA-C2oDE_CAMM_%d_%d_n%d_FEs%d_runs%d_%d_time.csv', problemSetNum, problem,n, totalFES, totalTime,time);
        dlmwrite(filename, run_time(time,1), 'precision', '%.6f');
        filename = sprintf('SA-C2oDE_CAMM_%d_%d_n%d_FEs%d_runs%d_%d_rf.csv', problemSetNum, problem,n, totalFES, totalTime,time);
        dlmwrite(filename, feasiRatio(:,time), 'precision', '%.6f');
%         fclose(processFile);
%         fclose(processFile2);
    end
%     filename = sprintf('LR-C2oDE_%d_%d_n%d_FEs%d_runs%d_fit.csv', problemSetNum, problem,n, totalFES, totalTime);
%     dlmwrite(filename, evolveSolution, 'precision', '%.6f');
%     filename = sprintf('LR-C2oDE_%d_%d_n%d_FEs%d_runs%d_con.csv', problemSetNum, problem,n, totalFES, totalTime);
%     dlmwrite(filename, evolveConstrain, 'precision', '%.6f');
%     filename = sprintf('LR-C2oDE_%d_%d_n%d_FEs%d_runs%d_time.csv', problemSetNum, problem,n, totalFES, totalTime);
%     dlmwrite(filename, run_time, 'precision', '%.6f');
%     filename = sprintf('LR-C2oDE_%d_%d_n%d_FEs%d_runs%d_rf.csv', problemSetNum, problem,n, totalFES, totalTime);
%     dlmwrite(filename, feasiRatio, 'precision', '%.6f');
end
