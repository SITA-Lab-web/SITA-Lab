#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <sys/time.h>
#include "../framework/framework.h"
#include "../util/utils.h"
#include <curl/curl.h>
using namespace std;

void agent_function(Framework*);
long getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static double euclidean_distance(const double* a, const double* b, int dim){
    double s = 0.0;
    for(int d = 0; d < dim; ++d){
        const double z = a[d] - b[d];
        s += z * z;
    }
    return std::sqrt(s);
}

static const char* termination_name(int reason){
    if(reason == 1) return "velocity_threshold";
    if(reason == 2) return "evaluation_budget";
    return "unknown";
}

int main(int argc, char* argv[]){
    int myrank, nprocs, name;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Get_processor_name(proc_name, &name);
    curl_global_init(CURL_GLOBAL_ALL);

    if (argc < 2) {
        if (myrank == 0) cerr << "Usage: mpirun -np <agents> ./LAC-MAC <benchmark_id>\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    string func_id = argv[argc-1];
    Framework* handler = new Framework(myrank,func_id);
    if (nprocs != handler->get_agent_num()) {
        if (myrank == 0) cerr << "MPI process count must equal benchmark node_num="
                              << handler->get_agent_num() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    srand(getCurrentTime()+myrank);
    agent_function(handler);
    MPI_Barrier(MPI_COMM_WORLD);

    const int dimension = handler->get_problem_dim();
    double *gather_final_solution = NULL;
    int *gather_commu_cost = NULL;
    int *gather_iterations = NULL;
    int *gather_reasons = NULL;
    double *gather_local_fitness = NULL;
    double *gather_velocity_norm = NULL;

    if (myrank == 0) {
        gather_final_solution = new double[dimension * nprocs]();
        gather_commu_cost = new int[nprocs]();
        gather_iterations = new int[nprocs]();
        gather_reasons = new int[nprocs]();
        gather_local_fitness = new double[nprocs]();
        gather_velocity_norm = new double[nprocs]();
    }

    MPI_Gather(handler->final_solution, dimension, MPI_DOUBLE,
               gather_final_solution, dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int commu_cost = handler->get_commu_cost();
    int final_iteration = handler->get_final_iteration();
    int termination_reason = handler->get_termination_reason();
    double final_local_fitness = handler->get_final_local_fitness();
    double final_velocity_norm = handler->get_final_velocity_norm();

    MPI_Gather(&commu_cost, 1, MPI_INT, gather_commu_cost, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&final_iteration, 1, MPI_INT, gather_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&termination_reason, 1, MPI_INT, gather_reasons, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&final_local_fitness, 1, MPI_DOUBLE, gather_local_fitness, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&final_velocity_norm, 1, MPI_DOUBLE, gather_velocity_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myrank == 0){
        double **total_solutions = new double*[nprocs];
        for(int i = 0; i < nprocs; i++){
            total_solutions[i] = new double[dimension]();
            memcpy(total_solutions[i], gather_final_solution + i * dimension,
                   dimension * sizeof(double));
        }

        double *average_solution = get_mtx_mean(total_solutions,nprocs,dimension);
        Benchmarks* global_func = new Benchmarks(func_id);
        double fitness = global_func->global_eva(average_solution);
        double disagreement = get_mtx_std(total_solutions,nprocs,dimension);
        double average_commu_cost = get_array_mean(gather_commu_cost,nprocs);

        cout << fixed << setprecision(6);
        cout << "\n========== FINAL AGENT CONFIGURATIONS ==========\n";
        for(int i = 0; i < nprocs; ++i){
            cout << "Agent " << i << " theta=[";
            for(int d = 0; d < dimension; ++d){
                if(d) cout << ", ";
                cout << total_solutions[i][d];
            }
            cout << "]"
                 << " local_fitness=" << gather_local_fitness[i]
                 << " iterations=" << gather_iterations[i]
                 << " stop=" << termination_name(gather_reasons[i])
                 << " final_velocity_L1=" << gather_velocity_norm[i]
                 << " communication_bytes=" << gather_commu_cost[i]
                 << "\n";
        }

        cout << "Mean theta=[";
        for(int d = 0; d < dimension; ++d){
            if(d) cout << ", ";
            cout << average_solution[d];
        }
        cout << "]\n";

        double max_pair_distance = 0.0;
        double mean_pair_distance = 0.0;
        int pair_count = 0;
        for(int i = 0; i < nprocs; ++i){
            for(int j = i + 1; j < nprocs; ++j){
                const double dist = euclidean_distance(total_solutions[i], total_solutions[j], dimension);
                mean_pair_distance += dist;
                if(dist > max_pair_distance) max_pair_distance = dist;
                ++pair_count;
            }
        }
        if(pair_count > 0) mean_pair_distance /= pair_count;

        cout << "Per-dimension range:";
        for(int d = 0; d < dimension; ++d){
            double min_v = total_solutions[0][d];
            double max_v = total_solutions[0][d];
            for(int i = 1; i < nprocs; ++i){
                if(total_solutions[i][d] < min_v) min_v = total_solutions[i][d];
                if(total_solutions[i][d] > max_v) max_v = total_solutions[i][d];
            }
            cout << " theta" << d << "=" << (max_v - min_v);
        }
        cout << "\n";
        cout << "Consensus diagnostics: mean_pair_L2=" << mean_pair_distance
             << ", max_pair_L2=" << max_pair_distance
             << ", framework_disagreement=" << disagreement << "\n";
        cout << "algorithm performance: [fitness:" << scientific << setprecision(4) << fitness
             << ", disagreement:" << disagreement
             << ", communication cost:" << average_commu_cost << "]\n";

        for(int i = 0; i < nprocs; ++i) delete[] total_solutions[i];
        delete[] total_solutions;
        delete[] average_solution;
        delete global_func;
        delete[] gather_final_solution;
        delete[] gather_commu_cost;
        delete[] gather_iterations;
        delete[] gather_reasons;
        delete[] gather_local_fitness;
        delete[] gather_velocity_norm;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    delete handler;
    curl_global_cleanup();
    MPI_Finalize();
    return 0;
}
