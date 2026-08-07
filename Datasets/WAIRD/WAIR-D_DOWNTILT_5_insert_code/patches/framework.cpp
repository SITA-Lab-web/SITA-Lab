#include "../framework/framework.h"
#include "../Benchmarks/Benchmarks.h"
#include <mpi.h>
#include <iostream>
using namespace std;

Framework::Framework(int agent_id, string func_id){
    this->agent_id = agent_id;
    commu_cost = 0;
    final_iteration = 0;
    final_local_fitness = 0.0;
    final_velocity_norm = 0.0;
    termination_reason = 0;
    this->pFunc = new Benchmarks(func_id);
    this->neighbor_id = this->pFunc->getNeighbors(this->agent_id);
    this->final_solution = new double[this->get_problem_dim()]{0};
    Func=func_id;
}

Framework::~Framework(){

}


int Framework::Message_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag){
    bool valid = false;
    for(int id : this->neighbor_id){
        if(id == dest)
            valid = true;
    }
    int res;
    if(valid){
        res = MPI_Send(buf,count,datatype,dest,tag,MPI_COMM_WORLD);
        int type_size = 0;
        MPI_Type_size(datatype, &type_size);
        this->commu_cost += count * type_size;
    }
    else{
        cout<<"the message send from "<<this->agent_id<<" to "<<dest<<" is invalid. Because they are not neighbors.\n";
        res = -1;
    }
    return res;
}

int Framework::Message_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Status *status){
    bool valid = false;
    for(int id : this->neighbor_id){
        if(id == source)
            valid = true;
    }
    int res;
    if(valid){
        res = MPI_Recv(buf,count,datatype,source,tag,MPI_COMM_WORLD,status);
    }
    else{
        cout<<"It is invalid to wait for message from "<<source<<" to "<<this->agent_id<<". Because they are not neighbors.\n";
        res = -1;
    }
    return res;
}

int Framework::Message_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,MPI_Request  *request){
    bool valid = false;
    for(int id : this->neighbor_id){
        if(id == dest)
            valid = true;
    }
    int res;
    if(valid){
        res = MPI_Isend(buf,count,datatype,dest,tag,MPI_COMM_WORLD,request);
        int type_size = 0;
        MPI_Type_size(datatype, &type_size);
        this->commu_cost += count * type_size;
    }
    else{
        cout<<"the message send from "<<this->agent_id<<" to "<<dest<<" is invalid. Because they are not neighbors.\n";
        res = -1;
    }
    return res;
}

int Framework::Message_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,MPI_Request  *request){
    bool valid = false;
    for(int id : this->neighbor_id){
        if(id == source)
            valid = true;
    }
    int res;
    if(valid){
        res = MPI_Irecv(buf,count,datatype,source,tag,MPI_COMM_WORLD,request);
    }
    else{
        cout<<"It is invalid to wait for message from "<<source<<" to "<<this->agent_id<<". Because they are not neighbors.\n";
        res = -1;
    }
    return res;
}

bool Framework::reachMaxEva(){
    return pFunc->reachMaxEva();
}

double Framework::local_evaluation(double* x){
    return this->pFunc->local_eva(x,this->agent_id);
}

int Framework::get_self_id(){
    return this->agent_id;
}

vector<int> Framework::get_neighbor_id(){
    vector<int> new_vec(this->neighbor_id);
    return new_vec;
}

int Framework::get_agent_num(){
    return this->pFunc->getNodeNum();
}

int Framework::get_problem_dim(){
    return this->pFunc->getDimension();
}

double Framework::get_lower_bound(){
    return this->pFunc->getMinX();
}

double Framework::get_upper_bound(){
    return this->pFunc->getMaxX();
}

void Framework::submit_final_solution(double* x){
    memcpy(this->final_solution,x,this->get_problem_dim()* sizeof(double));
    return;
}

void Framework::submit_run_stats(int iteration, double local_fitness, double velocity_norm, int reason){
    this->final_iteration = iteration;
    this->final_local_fitness = local_fitness;
    this->final_velocity_norm = velocity_norm;
    this->termination_reason = reason;
}

int Framework::get_final_iteration(){ return this->final_iteration; }
double Framework::get_final_local_fitness(){ return this->final_local_fitness; }
double Framework::get_final_velocity_norm(){ return this->final_velocity_norm; }
int Framework::get_termination_reason(){ return this->termination_reason; }

int Framework::get_commu_cost(){
    return this->commu_cost;
}

vector<double> Framework::get_adjacent_weights(){
    double** W = this->pFunc->getNetworkGraph();
    vector<double> weights;
    for(int nei_id:this->neighbor_id){
        weights.push_back(W[this->agent_id][nei_id]);
    }
    return weights;
}