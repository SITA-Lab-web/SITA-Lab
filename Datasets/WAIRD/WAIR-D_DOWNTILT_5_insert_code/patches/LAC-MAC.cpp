#include <fstream>
#include <string>
#include <cstdio>
#include <curl/curl.h>
#include <sstream> 
#include <nlohmann/json.hpp>
#include <regex> 
#include <vector>
#include <iostream>
#include <Python.h>
#include <mpi.h>
#include <thread>
#include <Eigen/Dense>
#include"../LLM/mycode.h"
#include"../LLM/myconnect.h"
#include"../LLM/mylog.h"
#include"../LLM/mypy.h"
#include"../LLM/myclass.h"
#include "../framework/framework.h"
#include "./internal_optimizer.h"
using namespace std;
using namespace Eigen;

void print_matrix_dim(const MatrixXd& mat, const std::string& name) {
    std::cout << name << " 维度: " 
              << mat.rows() << "行 x " << mat.cols() << "列" << std::endl;
}
// parameter setting
int swarmSize = 100; // 5-D application: 50-100 is usually sufficient
int interval = 4;
int no_improve_tolerant=10;
double termination_threshold = 1E-5;
double w0[2]={0.7,1.3}; // Initial agent-internal velocity coefficient of variation
double w[2]={1,1}; // Current agent-internal learned velocity coefficient of variation
const int T0 = 2000; // Pre-experiment iterations
const int interactionInterval_v = (int)ceil(T0/3);  // Agent-internal adaptive learning interval
const int interactionInterval_w = (int)ceil(T0/exp(1));  // Agent-external adaptive learning interval
const int maxOutputLength = 200;
std::string filePath = "./iteration__data.txt"; 
std::string currentCodeFilePath = __FILE__; 
std::string logFilePath = "./code_update_log.txt";
vector<InteractionData> history;

void agent_function(Framework* handler){
    // get optimization problem information
    int dim = handler->get_problem_dim();
    double lb = handler->get_lower_bound();
    double ub = handler->get_upper_bound();
    vector<int> nei_list = handler->get_neighbor_id();
    vector<double> nei_weight = handler->get_adjacent_weights();
    vector<double> last_nei_weight;//n
    int nei_num = nei_list.size();
    int myrank = handler->get_self_id();
    //std::string filename = "agent_rank_" + std::to_string(myrank) + "_fitness.csv";
    if (nei_num > 0) {
        double avg_weight = 1.0 / (double)nei_num;
        nei_weight.assign(nei_num, avg_weight);
        last_nei_weight = nei_weight;
    } else {
        nei_weight.clear();
        last_nei_weight.clear();
    }
    vector<int> activate_nei_idx(nei_num,0);
    iota(activate_nei_idx.begin(), activate_nei_idx.end(), 0);
    vector<NeighborPerformance> neighbor_perfs(nei_num);
    for (int i = 0; i < nei_num; i++) {
        neighbor_perfs[i].neighbor_id = nei_list[i];
    }

    // population initialization
    MatrixXd swarm_x = (MatrixXd::Random(swarmSize,dim).array()/2 + 0.5).array() * (ub-lb) + lb;
    MatrixXd external_swarm_v = MatrixXd::Zero(swarmSize,dim);
    VectorXd swarm_fit = VectorXd::Zero(swarmSize);
    for(int j = 0;j<swarmSize;j++){
        VectorXd inv = swarm_x.row(j);
        swarm_fit(j) = handler->local_evaluation(inv.data());
    }
    LLSO* internal_opt = new LLSO();
    internal_opt->bestFit = swarm_fit.minCoeff();

    // neighboring communication initialization
    vector<MatrixXd> nei_buffers;
    MPI_Request* terminate_recv_req = new MPI_Request[nei_num]; 
    MPI_Request* message_recv_req = new MPI_Request[nei_num];
    int if_nei_terminate[nei_num];
    for (int i =0 ;i <nei_num;i++) {
        // communication for optimization information, tag is 1
        MatrixXd buffer(swarmSize,dim);
        nei_buffers.push_back(buffer);
        MPI_Request recv_req_2;
        handler->Message_Irecv(nei_buffers[i].data(),swarmSize*dim,MPI_DOUBLE, nei_list[i],1,&recv_req_2);               
        message_recv_req[i] = recv_req_2;

        // communication for termination notification, tag is 5
        if_nei_terminate[i] =0;
        MPI_Request recv_req;
        handler->Message_Irecv(&if_nei_terminate[i], 1, MPI_INT, nei_list[i], 5, &recv_req);  
        terminate_recv_req[i] = recv_req;
    }

    double best_inv_fit = DBL_MAX;
    int no_improve_count=0;
    int final_iter = 0;
    int final_reason = 0;
    double final_velocity_norm = 0.0;
    
    for(int iter=1;;iter++){

        // internal learning
        MatrixXd swarm_v = external_swarm_v.array();
        for(int i=0;i<interval;i++){
            internal_opt->step(&swarm_x,&swarm_v,swarm_fit);
            swarm_x = swarm_x.cwiseMin(ub);
            swarm_x = swarm_x.cwiseMax(lb);
            for(int j = 0;j<swarmSize;j++){
                VectorXd inv = swarm_x.row(j);
                swarm_fit(j) = handler->local_evaluation(inv.data());
            }
            internal_opt->update_performance(swarm_fit);            
        }

        // send message to neighbors
        double* trans_data = swarm_x.data();
        MPI_Request req[nei_num];
        for(int k:activate_nei_idx){
            handler->Message_Isend(trans_data,swarmSize*dim,MPI_DOUBLE,nei_list[k],1,&req[k]);
        }
        
        // wait for message from neighbors
        vector<int> new_activate_nei_idx;
        for(int k:activate_nei_idx){
            MPI_Request* nei_req = new MPI_Request[2];
            nei_req[0] = terminate_recv_req[k];
            nei_req[1] = message_recv_req[k];
            int index;
            MPI_Status status;
            MPI_Waitany(2, nei_req, &index, &status);
            if(status.MPI_TAG == 5){
                MPI_Cancel(&message_recv_req[k]);
            }else{
                new_activate_nei_idx.push_back(k);
                MPI_Request recv_req;
                MatrixXd neighbor_swarm = nei_buffers[k];
                MatrixXd neighbor_mean = neighbor_swarm.colwise().mean();
                MatrixXd neighbor_diff = neighbor_swarm - neighbor_mean.replicate(swarmSize, 1);
                double neighbor_disagreement = neighbor_diff.array().square().sum() / swarmSize;
                VectorXd neighbor_fit(swarmSize);
                for (int j = 0; j < swarmSize; j++) {
                    neighbor_fit(j) = handler->local_evaluation(neighbor_swarm.row(j).data());
                }
                double neighbor_avg_fitness = neighbor_fit.mean();
                if (neighbor_perfs[k].fitness_history.size() >= 10) {
                    neighbor_perfs[k].fitness_history.erase(neighbor_perfs[k].fitness_history.begin());
                    neighbor_perfs[k].disagreement_history.erase(neighbor_perfs[k].disagreement_history.begin());
                }
                neighbor_perfs[k].fitness_history.push_back(neighbor_avg_fitness);
                neighbor_perfs[k].disagreement_history.push_back(neighbor_disagreement);
                handler->Message_Irecv(nei_buffers[k].data(),swarmSize*dim,MPI_DOUBLE, nei_list[k],1,&recv_req);               
                message_recv_req[k] = recv_req;
            }
        }
        activate_nei_idx = new_activate_nei_idx;

        if (iter % interactionInterval_v == 0 && iter<T0) {
            try{
                std::stringstream prompt_ss;
                prompt_ss << "任务:" << dim << "维问题,优化前是"<<T0<<"次迭代左右。当前参数:d=" << w[0] << ", c=" << w[1] 
                        << "状态|目前迭代" << iter << "次|前19次数据|";
                        for (size_t j = 0; j < history.size(); j++) {
                            int recordIndex = iter - 19 + j;
                            prompt_ss << "第" << recordIndex << "次 - "
                                 << "fitness: " << history[j].fitness << ", "
                                 << "disagreement: " << history[j].disagreement <<"|";
                        };
                        prompt_ss <<"需求:若fitness停滞且disagreement低,增大c;若fitness下降慢且disagreement高.增大d。仅返回新参数,使用小括号包裹,用逗号分隔(d在0.5-1,c在1-1.8),例:(0.7,1.3)";
                std::string prompt = prompt_ss.str();
                std::string finalResponse = "";
                bool isDone = false;
                while (!isDone) {
                    std::string currentResponse = callOllama(prompt, maxOutputLength);
                    if (currentResponse.empty()) {
                        std::cerr << "CallOllama failed: No response or request error" << std::endl;
                    }
                    // std::cout << "模型完整响应:\n" << currentResponse << std::endl;
                    size_t start = 0;
                    while (start < currentResponse.size()) {
                        std::string jsonChunk = extractFullJson(currentResponse, start);
                        if (!jsonChunk.empty()) {
                            try {
                                nlohmann::json jsonPart = nlohmann::json::parse(jsonChunk);
                                std::string responsePart = jsonPart["response"].get<std::string>();
                                finalResponse += responsePart;
                                isDone = jsonPart["done"].get<bool>();
                            } catch (const nlohmann::json::parse_error& e) {
                                std::cerr << "JSON decoding error: " << e.what() << std::endl;
                            }
                        }
                    }
                }
                double* w0 = extractParameters(finalResponse);
                logCodeUpdate(iter,w,w0,logFilePath);
                for (int i = 0; i < 2; i++) {
                    w[i] = w0[i];
                }
                delete[] w0;   
            }catch (const std::exception& e) {
                std::cerr << "iter:" << iter << "error: " << e.what() << std::endl;
                continue; 
            }
        }
        if(iter>T0){
            for (int i = 0; i < 2; i++) {
                w[i] = 1;
            }
        }                
        
        if (iter % interactionInterval_w == 0) {
            try{
                std::stringstream prompt_ss;
                prompt_ss << "任务:更新多智能体优化的邻居权重矩阵。|"
                << "当前智能体的邻居数量:" << nei_num 
                << "|权重更新规则:" 
                << "1. 若邻居的fitness(适应度)低且disagreement(不一致性)低,说明该邻居信息优质且稳定,应提高其权重(0.3~0.5);" 
                << "2. 若邻居的fitness高且disagreement高,说明该邻居信息质量差,应降低其权重(0.1~0.2);"
                << "3. 明确任务目标是需要尽可能低的fitness值,所以优先参考fitness值,disagreement值只是作为一个平衡,高disagreement值可以说明粒子充分探索,低disagreement值也能说明粒子有很好的共识;"
                << "4. 所有邻居权重之和需为1,确保共识收敛。|" 
                << "邻居性能历史(最近10次迭代):" ;
                for (int k = 0; k < nei_num; k++) {
                    prompt_ss << "|邻居ID:" << neighbor_perfs[k].neighbor_id 
                                << "  平均fitness:" << accumulate(neighbor_perfs[k].fitness_history.begin(), 
                                                                neighbor_perfs[k].fitness_history.end(), 0.0) 
                                                                / neighbor_perfs[k].fitness_history.size() 
                                << ",  平均disagreement:" << accumulate(neighbor_perfs[k].disagreement_history.begin(), 
                                                                    neighbor_perfs[k].disagreement_history.end(), 0.0) 
                                                                    / neighbor_perfs[k].disagreement_history.size() ;
                }
                prompt_ss << "|请返回更新后的权重,输出格式严格用[0,1]的小数表示,按邻居ID顺序,用逗号分隔,并用中括号包裹,输出示例:[0.2,0.5,0.3]";
                std::string prompt = prompt_ss.str();
                std::string finalResponse = "";
                bool isDone = false;
                while (!isDone) {
                    std::string currentResponse = callOllama(prompt, maxOutputLength);
                    if (currentResponse.empty()) {
                        std::cerr << "CallOllama failed: No response or request error" << std::endl;
                    }
                    size_t start = 0;
                    while (start < currentResponse.size()) {
                        std::string jsonChunk = extractFullJson(currentResponse, start);
                        if (!jsonChunk.empty()) {
                            try {
                                nlohmann::json jsonPart = nlohmann::json::parse(jsonChunk);
                                std::string responsePart = jsonPart["response"].get<std::string>();
                                finalResponse += responsePart;
                                isDone = jsonPart["done"].get<bool>();
                            } catch (const nlohmann::json::parse_error& e) {
                                std::cerr << "JSON decoding error: " << e.what() << std::endl;
                            }
                        }
                    }
                }
                vector<double> new_weights = extract_weights_from_llm_response(finalResponse, nei_num);
                double sum = accumulate(new_weights.begin(), new_weights.end(), 0.0);
                    for (auto& value : new_weights) value = value / sum;
                    nei_weight = new_weights;
            }catch (const std::exception& e) {
                if (!nei_weight.empty()) {
                    cerr << "iter:" << iter << "error: " << e.what() << ", revert to the previous weights" << endl;
                } else {
                    nei_weight.assign(nei_num, DEFAULT_WEIGHT);
                    cerr << "iter:" << iter << "error: " << e.what() << ", adopt default weights" << endl;
                }
                continue; 
            }
        }

        MatrixXd swarm_mean = swarm_x.colwise().mean(); 
        MatrixXd diff = swarm_x - swarm_mean.replicate(swarm_x.rows(), 1); 
        double current_disagreement = diff.array().square().sum() / swarmSize;
        double perturb_scale = 1.0;
        if (current_disagreement > 1e-1) { 
            perturb_scale = w[0]; 
        } else if (current_disagreement < 1e-2) {
            perturb_scale = w[1];
        }
        MatrixXd randmtx = (MatrixXd::Random(swarmSize, 1).array() / 2 + 0.5)
                        .replicate(1, dim) * perturb_scale;
        external_swarm_v = external_swarm_v.cwiseProduct(randmtx);
        for(int k : activate_nei_idx) {
            external_swarm_v += (nei_buffers[k] - swarm_x) * nei_weight[k];
        }
        swarm_x += external_swarm_v;
        swarm_x = swarm_x.cwiseMin(ub);
        swarm_x = swarm_x.cwiseMax(lb);
        for(int j = 0; j < swarmSize; j++) {
            VectorXd inv = swarm_x.row(j);
            swarm_fit(j) = handler->local_evaluation(inv.data());
        }
        if (history.size() >= 19) {
            history.erase(history.begin());
        }
        history.push_back({swarm_fit.mean(), current_disagreement});
    
        // adaptive communication interval
        if(swarm_fit.minCoeff() < best_inv_fit){
            best_inv_fit = swarm_fit.minCoeff();
            no_improve_count = 0;
        }else{
            no_improve_count ++;
        }
        if(no_improve_count > no_improve_tolerant && interval > 2){
            interval = interval-1;
            no_improve_count = 0;
            best_inv_fit = swarm_fit.minCoeff();
        }

        // termination decision
        const double velocity_l1 = external_swarm_v.array().abs().sum();
        const bool stop_by_velocity = (velocity_l1 < termination_threshold);
        const bool stop_by_budget = handler->reachMaxEva();
        if(stop_by_velocity || stop_by_budget){
            final_iter = iter;
            final_velocity_norm = velocity_l1;
            final_reason = stop_by_velocity ? 1 : 2;
            int terminate_flag = 1;
            MPI_Request terminate_send_req[nei_num]; 
            for(int k = 0;k<nei_num;k++)    
                handler->Message_Isend(&terminate_flag,1,MPI_INT,nei_list[k],5,&terminate_send_req[k]);
            for(int k:activate_nei_idx){
                MPI_Cancel(&message_recv_req[k]);
            }
            std::cout << "\n iter:" << iter << ", current agent-internal learned velocity coefficient of variation:\[" << w[0] << ","<< w[1] << "\]。" <<std::endl;
            for (int i = 0; i < 2; i++) {
                w[i] = w0[i];
            }
            break;
        }
    }

    // record optimization results
    VectorXd final_x = swarm_x.colwise().mean();
    double final_local_fitness = handler->local_evaluation(final_x.data());
    handler->submit_final_solution(final_x.data());
    handler->submit_run_stats(final_iter, final_local_fitness, final_velocity_norm, final_reason);
    return;
}