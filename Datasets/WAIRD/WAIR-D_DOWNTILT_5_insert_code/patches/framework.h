#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <string>
#include <vector>
#include <mpi.h>
#include "../Benchmarks/Benchmarks.h"
using std::string;
using std::vector;

class Framework{
private:
    int agent_id;
    vector<int> neighbor_id;
    int commu_cost;
    Benchmarks* pFunc;
    int final_iteration;
    double final_local_fitness;
    double final_velocity_norm;
    int termination_reason; // 1: velocity threshold, 2: max evaluations

public:
    Framework(int, string);
    ~Framework();
    double* final_solution;
    int get_self_id();
    vector<int> get_neighbor_id();
    int get_agent_num();
    int get_problem_dim();
    double get_lower_bound();
    double get_upper_bound();
    double local_evaluation(double*);
    vector<double> get_adjacent_weights();
    int Message_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag);
    int Message_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Status *status);
    int Message_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,MPI_Request *request);
    int Message_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,MPI_Request *request);
    void submit_final_solution(double*);
    void submit_run_stats(int iteration, double local_fitness, double velocity_norm, int reason);
    int get_final_iteration();
    double get_final_local_fitness();
    double get_final_velocity_norm();
    int get_termination_reason();
    int get_commu_cost();
    bool reachMaxEva();

    string Func;
};

#endif
