#ifndef WAIRD_DOWNTILT_5_H
#define WAIRD_DOWNTILT_5_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct WairdTiltRow {
    int ue;
    int subcarrier;
    int serving_bs;
    double phi[5];
    double gain[5];
};

struct WairdTiltConfig {
    int num_bs;
    double noise_power;
    double vertical_hpbw_deg;
    double max_atten_db;
    double outage_weight;
    double rate_floor;
    std::string environment_id;
    std::vector< std::vector<WairdTiltRow> > local_rows;
    bool loaded;
    WairdTiltConfig() : num_bs(5), noise_power(0.1), vertical_hpbw_deg(10.0),
        max_atten_db(20.0), outage_weight(0.10), rate_floor(1.0), loaded(false) {}
};

static WairdTiltConfig g_waird_tilt;

static std::string waird_env_from_runtime(const std::string& fallback) {
    const char* value = std::getenv("WAIRD_ENV_ID");
    if (value != NULL && value[0] != '\0') return std::string(value);
    return fallback;
}

static double waird_vertical_gain(double phi_deg, double downtilt_deg) {
    double d = (phi_deg - downtilt_deg) / g_waird_tilt.vertical_hpbw_deg;
    double atten_db = std::min(12.0 * d * d, g_waird_tilt.max_atten_db);
    return std::pow(10.0, -atten_db / 10.0);
}

static bool waird_parse_row(const std::string& line, WairdTiltRow* row) {
    if (line.empty() || line[0] == '#') return false;
    std::stringstream ss(line);
    ss >> row->ue >> row->subcarrier >> row->serving_bs;
    for (int b = 0; b < 5; ++b) ss >> row->phi[b];
    for (int b = 0; b < 5; ++b) ss >> row->gain[b];
    if (!ss) throw std::runtime_error("Malformed WAIR-D row: " + line);
    return true;
}

//
static void waird_downtilt5_default_metrics();

static void waird_downtilt5_load(const std::string& data_root,
                                  int node_num,
                                  int num_bs,
                                  double snr_db,
                                  double vertical_hpbw_deg,
                                  double max_atten_db,
                                  double outage_weight,
                                  double rate_floor,
                                  const std::string& fallback_env) {
    if (node_num != 5 || num_bs != 5) {
        throw std::runtime_error("WAIRD_DOWNTILT_5 requires exactly 5 MPI agents/BSs");
    }
    g_waird_tilt.num_bs = num_bs;
    g_waird_tilt.noise_power = std::pow(10.0, -snr_db / 10.0);
    g_waird_tilt.vertical_hpbw_deg = vertical_hpbw_deg;
    g_waird_tilt.max_atten_db = max_atten_db;
    g_waird_tilt.outage_weight = outage_weight;
    g_waird_tilt.rate_floor = rate_floor;
    g_waird_tilt.environment_id = waird_env_from_runtime(fallback_env);
    g_waird_tilt.local_rows.assign(node_num, std::vector<WairdTiltRow>());

    for (int i = 0; i < node_num; ++i) {
        std::ostringstream path;
        path << data_root << "/" << g_waird_tilt.environment_id << "/bs_";
        if (i < 10) path << "0";
        path << i << ".txt";
        std::ifstream in(path.str().c_str());
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open WAIR-D local file: " + path.str());
        }
        std::string line;
        while (std::getline(in, line)) {
            WairdTiltRow row;
            if (waird_parse_row(line, &row)) g_waird_tilt.local_rows[i].push_back(row);
        }
    }
    g_waird_tilt.loaded = true;
    // ============================================================
    // Baseline: conventional fixed downtilt configuration
    // Added for real-world experiment comparison
    // ============================================================
    waird_downtilt5_default_metrics();

}

static double waird_user_rate(const std::vector<WairdTiltRow>& rows,
                              size_t begin, size_t end,
                              const double* theta) {
    double sum = 0.0;
    int count = 0;
    for (size_t p = begin; p < end; ++p) {
        const WairdTiltRow& r = rows[p];
        int s = r.serving_bs;
        double signal = waird_vertical_gain(r.phi[s], theta[s]) * r.gain[s];
        double interference = 0.0;
        for (int b = 0; b < 5; ++b) {
            if (b == s) continue;
            interference += waird_vertical_gain(r.phi[b], theta[b]) * r.gain[b];
        }
        double sinr = signal / (interference + g_waird_tilt.noise_power);
        sum += std::log(1.0 + std::max(0.0, sinr)) / std::log(2.0);
        ++count;
    }
    return count > 0 ? sum / (double)count : 0.0;
}

static double waird_downtilt5_local_eva(const double* theta, int dim, int agent_id) {
    if (!g_waird_tilt.loaded) throw std::runtime_error("WAIR-D benchmark is not loaded");
    if (dim != 5 || agent_id < 0 || agent_id >= 5) {
        throw std::runtime_error("Invalid WAIR-D dimension or agent id");
    }
    const std::vector<WairdTiltRow>& rows = g_waird_tilt.local_rows[agent_id];
    double sum_rate = 0.0;
    double sum_shortfall = 0.0;
    int users = 0;
    size_t begin = 0;
    while (begin < rows.size()) {
        size_t end = begin + 1;
        while (end < rows.size() && rows[end].ue == rows[begin].ue) ++end;
        double rate = waird_user_rate(rows, begin, end, theta);
        sum_rate += rate;
        double shortfall = std::max(0.0, g_waird_tilt.rate_floor - rate);
        sum_shortfall += shortfall * shortfall;
        ++users;
        begin = end;
    }
    if (users == 0) return 0.0;
    double mean_rate = sum_rate / (double)users;
    double mean_shortfall = sum_shortfall / (double)users;
    return -mean_rate + g_waird_tilt.outage_weight * mean_shortfall;
}

//
static double waird_downtilt5_global_eva(
    const double* theta,
    int dim)
{
    std::vector<double> user_rates;


    // Collect all UE spectral efficiencies
    for (int i = 0; i < 5; ++i)
    {
        const std::vector<WairdTiltRow>& rows =
            g_waird_tilt.local_rows[i];


        size_t begin = 0;


        while (begin < rows.size())
        {
            size_t end = begin + 1;


            while (
                end < rows.size()
                &&
                rows[end].ue == rows[begin].ue
            )
            {
                ++end;
            }


            double rate =
                waird_user_rate(
                    rows,
                    begin,
                    end,
                    theta
                );


            user_rates.push_back(rate);


            begin = end;
        }
    }


    if (user_rates.empty())
    {
        return 0.0;
    }


    // Calculate UE-level AvgSE
    double avg_se = 0.0;


    for (size_t i = 0; i < user_rates.size(); ++i)
    {
        avg_se += user_rates[i];
    }


    avg_se /= (double)user_rates.size();


    // Minimize fitness -> maximize AvgSE
    return -avg_se;
}

static void waird_downtilt5_metrics(const double* theta, int dim,
                                    double* avg_se, double* p10_se,
                                    double* p5_se, double* min_se,
                                    double* bs_balance) {
    std::vector<double> user_rates;
    double bs_mean[5] = {0,0,0,0,0};
    for (int i = 0; i < 5; ++i) {
        const std::vector<WairdTiltRow>& rows = g_waird_tilt.local_rows[i];
        size_t begin = 0;
        int users = 0;
        while (begin < rows.size()) {
            size_t end = begin + 1;
            while (end < rows.size() && rows[end].ue == rows[begin].ue) ++end;
            double rate = waird_user_rate(rows, begin, end, theta);
            user_rates.push_back(rate);
            bs_mean[i] += rate;
            ++users;
            begin = end;
        }
        if (users > 0) bs_mean[i] /= (double)users;
    }
    if (user_rates.empty()) {
        *avg_se = *p10_se = *p5_se = *min_se = *bs_balance = 0.0;
        return;
    }
    double sum = 0.0;
    for (size_t i = 0; i < user_rates.size(); ++i) sum += user_rates[i];
    *avg_se = sum / (double)user_rates.size();
    std::sort(user_rates.begin(), user_rates.end());
    *min_se = user_rates.front();
    *p10_se = user_rates[(size_t)(0.10 * (user_rates.size() - 1))];
    *p5_se = user_rates[(size_t)(0.05 * (user_rates.size() - 1))];
    double mean_bs = 0.0;
    for (int i = 0; i < 5; ++i) mean_bs += bs_mean[i];
    mean_bs /= 5.0;
    double var = 0.0;
    for (int i = 0; i < 5; ++i) {
        double d = bs_mean[i] - mean_bs;
        var += d * d;
    }
    *bs_balance = var / 5.0;
}

// ============================================================
// Baseline: conventional fixed downtilt configuration
// Added for real-world experiment comparison
// ============================================================
static void waird_downtilt5_default_metrics()
{
    // Conventional BS downtilt setting
    // Each BS uses the same default downtilt angle
    double theta_default[5] =
    {
        7.0,
        7.0,
        7.0,
        7.0,
        7.0
    };


    double avg_se = 0.0;
    double p10_se = 0.0;
    double p5_se = 0.0;
    double min_se = 0.0;
    double bs_balance = 0.0;


    waird_downtilt5_metrics(
        theta_default,
        5,
        &avg_se,
        &p10_se,
        &p5_se,
        &min_se,
        &bs_balance
    );


    std::cout
        << "\n========== DEFAULT DOWNTILT BASELINE =========="
        << std::endl;

    std::cout
        << "theta=[7,7,7,7,7]"
        << std::endl;

    std::cout
        << "AvgSE="
        << avg_se
        << ", P10-SE="
        << p10_se
        << ", P5-SE="
        << p5_se
        << ", Min-SE="
        << min_se
        << ", BS-balance="
        << bs_balance
        << std::endl;

    std::cout
        << "==============================================\n"
        << std::endl;
}

static void waird_downtilt5_free() {
    g_waird_tilt.local_rows.clear();
    g_waird_tilt.loaded = false;
}

#endif
