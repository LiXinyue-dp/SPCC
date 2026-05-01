#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <tuple>
#include <stdio.h>
#include <string.h>
#include <array>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include "mt19937ar.h"
#include <stdlib.h>
#include "MemoryOperation.h"
#include "include/stats.hpp"
#include <cstdlib>
#include <ctime>
#include <utility>
#include <algorithm>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <vector>
#include <random>
using namespace std;

int q;
long long OriginalCliqueCount = 0;

string EdgeFile;
string DatasetPath;
int NodeNum;
double Eps;
string Eps_s;
int k_star;
double eta = 0.22;
double alpha_val = 0.22;
double lambda_val = 0.65;
double beta_val = 1.1;
stats::rand_engine_t engine(1776);

struct BucketInfo {
    vector<vector<int>> buckets;
    unordered_map<int, int> node_to_bucket;
    unordered_map<int, double> node_true_degree;
    unordered_map<int, int> noisy_degrees;
    int d_hat_max;
    int n;
};

double BinomCoeff(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    double result = 1;
    for (int i = 0; i < k; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

long long BinomCoeffLL(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    long long result = 1;
    for (int i = 0; i < k; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

double LaplaceNoise(double scale, mt19937 &gen) {
    uniform_real_distribution<double> uniform(-0.5, 0.5);
    double u = uniform(gen);
    return (u > 0) ? -scale * log(1 - 2 * fabs(u)) : scale * log(1 + 2 * fabs(u));
}

FILE *FileOpen(string filename, const char *mode)
{
    FILE *fp;
    if ((fp = fopen(filename.c_str(), mode)) == NULL)
    {
        cout << "cannot open " << filename << endl;
        exit(-1);
    }
    return fp;
}

bool checkFileExistence(const std::string &str)
{
    std::ifstream ifs(str);
    return ifs.is_open();
}

void ReadEdgeFile(const std::string &edgeFile,
                  std::set<int> &actualNodeIDs,
                  std::unordered_map<int, std::set<int>> &data)
{
    std::ifstream file(edgeFile);
    if (!file.is_open())
    {
        cout << "Error: Cannot open edge file: " << edgeFile << endl;
        return;
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string first, second;
        std::getline(ss, first, ',');
        std::getline(ss, second, ',');
        if (std::isdigit(first[0]) && std::isdigit(second[0]))
        {
            int firstNum = std::stoi(first);
            int secondNum = std::stoi(second);
            data[firstNum].insert(secondNum);
            data[secondNum].insert(firstNum);
            actualNodeIDs.insert(firstNum);
            actualNodeIDs.insert(secondNum);
        }
    }
    file.close();
}

long long CountCliques(const std::set<int> &actualNodeIDs,
                       const std::unordered_map<int, std::set<int>> &data,
                       int qVal)
{
    long long count = 0;
    if (qVal == 3)
    {
        for (const auto &node_i : actualNodeIDs)
        {
            const auto &neighbors_i = data.at(node_i);
            for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
            {
                if (*it_j <= node_i) continue;
                for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
                {
                    if (*it_k <= node_i) continue;
                    if (data.at(*it_j).count(*it_k))
                        count++;
                }
            }
        }
    }
    else if (qVal == 4)
    {
        for (const auto &node_i : actualNodeIDs)
        {
            const auto &neighbors_i = data.at(node_i);
            for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
            {
                if (*it_j <= node_i) continue;
                for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
                {
                    if (*it_k <= node_i) continue;
                    if (!data.at(*it_j).count(*it_k)) continue;
                    for (auto it_l = next(it_k); it_l != neighbors_i.end(); ++it_l)
                    {
                        if (*it_l <= node_i) continue;
                        if (data.at(*it_j).count(*it_l) && data.at(*it_k).count(*it_l))
                            count++;
                    }
                }
            }
        }
    }
    return count;
}

vector<int> GraphProjection(const set<int> &neighbors, int d_hat_max, mt19937 &gen) {
    vector<int> proj(neighbors.begin(), neighbors.end());
    if ((int)proj.size() > d_hat_max) {
        shuffle(proj.begin(), proj.end(), gen);
        proj.resize(d_hat_max);
    }
    return proj;
}

vector<vector<int>> BuildKStars(const vector<int> &neighbors, int k) {
    vector<vector<int>> result;
    int n = (int)neighbors.size();
    if (k == 2) {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                result.push_back({neighbors[i], neighbors[j]});
    } else if (k == 3) {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                for (int l = j + 1; l < n; l++)
                    result.push_back({neighbors[i], neighbors[j], neighbors[l]});
    } else if (k == 1) {
        for (int i = 0; i < n; i++)
            result.push_back({neighbors[i]});
    }
    return result;
}

vector<vector<int>> PaddingAndDropping(vector<vector<int>> &kstars, int L, int dum_id, int k, mt19937 &gen) {
    vector<vector<int>> result;
    if ((int)kstars.size() >= L) {
        shuffle(kstars.begin(), kstars.end(), gen);
        result.assign(kstars.begin(), kstars.begin() + L);
    } else {
        result = kstars;
        vector<int> dummy_kstar(k, dum_id);
        while ((int)result.size() < L)
            result.push_back(dummy_kstar);
    }
    return result;
}

int PerturbAGP(int val, const vector<int> &domain, double eps_prime, mt19937 &gen) {
    int domain_size = (int)domain.size();
    if (domain_size <= 1) return val;
    double e_eps = exp(eps_prime);
    double p_retain = e_eps / (e_eps * domain_size + domain_size - 1);

    uniform_real_distribution<double> dist(0.0, 1.0);
    if (dist(gen) < p_retain) {
        return val;
    } else {
        vector<int> candidates;
        candidates.reserve(domain_size - 1);
        for (int x : domain) {
            if (x != val) candidates.push_back(x);
        }
        if (candidates.empty()) return val;
        uniform_int_distribution<int> pick(0, (int)candidates.size() - 1);
        return candidates[pick(gen)];
    }
}

double AGPRetainProb(int domain_size, double eps_prime) {
    if (domain_size <= 1) return 1.0;
    double e_eps = exp(eps_prime);
    return e_eps / (e_eps * domain_size + domain_size - 1);
}

BucketInfo BuildBuckets(const std::set<int> &actualNodeIDs,
                        const std::unordered_map<int, std::set<int>> &data,
                        double eps1, int L, int n, mt19937 &gen) {
    BucketInfo info;
    info.n = n;
    info.d_hat_max = 0;

    double laplace_scale = 2.0 / eps1;
    int max_real_degree = 0;
    for (const auto &node : actualNodeIDs) {
        int degree = (int)data.at(node).size();
        info.node_true_degree[node] = degree;
        if (degree > max_real_degree) max_real_degree = degree;
        double noisy = degree + LaplaceNoise(laplace_scale, gen);
        noisy = round(noisy);
        noisy = max(1.0, noisy);
        noisy = min((double)max_real_degree, noisy);
        info.noisy_degrees[node] = (int)noisy;
    }

    for (const auto &nd : info.noisy_degrees)
        if (nd.second > info.d_hat_max) info.d_hat_max = nd.second;

    vector<int> sorted_noisy;
    for (const auto &nd : info.noisy_degrees) sorted_noisy.push_back(nd.second);
    sort(sorted_noisy.begin(), sorted_noisy.end());
    int q50 = sorted_noisy[sorted_noisy.size() / 2];
    int q90 = sorted_noisy[sorted_noisy.size() * 9 / 10];

    auto sizeConstraint = [&](double mu) -> int {
        if (mu > q90) return (int)ceil(lambda_val * L);
        if (mu > q50) return (int)ceil(alpha_val * L);
        return (int)ceil(beta_val * L);
    };

    map<int, vector<int>> deg_groups;
    for (const auto &node : actualNodeIDs)
        deg_groups[info.noisy_degrees[node]].push_back(node);

    for (auto &grp : deg_groups) {
        int bucket_id = (int)info.buckets.size();
        info.buckets.push_back(grp.second);
        for (int node : grp.second)
            info.node_to_bucket[node] = bucket_id;
    }

    vector<int> worklist;
    for (int i = 0; i < (int)info.buckets.size(); i++)
        worklist.push_back(i);

    while (!worklist.empty()) {
        int bucket_idx = worklist.back();
        worklist.pop_back();
        auto &bk = info.buckets[bucket_idx];
        if ((int)bk.size() <= 1) continue;

        double mu = 0;
        for (int node : bk) mu += info.node_true_degree[node];
        mu /= bk.size();

        double sigma = 0;
        for (int node : bk) {
            double diff = info.node_true_degree[node] - mu;
            sigma += diff * diff;
        }
        sigma = sqrt(sigma / bk.size());

        if (mu > 0 && sigma > eta * mu) {
            sort(bk.begin(), bk.end(), [&](int a, int b) {
                return info.node_true_degree[a] < info.node_true_degree[b];
            });
            int mid = (int)bk.size() / 2;
            vector<int> upper(bk.begin() + mid, bk.end());
            bk.resize(mid);

            int new_bucket_id = (int)info.buckets.size();
            info.buckets.push_back(upper);
            for (int node : upper)
                info.node_to_bucket[node] = new_bucket_id;

            worklist.push_back(bucket_idx);
            worklist.push_back(new_bucket_id);
        }
    }

    for (int i = 0; i < (int)info.buckets.size(); i++) {
        auto &bk = info.buckets[i];
        if (bk.empty()) continue;
        double mu = 0;
        for (int node : bk) mu += info.node_true_degree[node];
        mu /= bk.size();
        int max_size = sizeConstraint(mu);
        if ((int)bk.size() > max_size && bk.size() > 2) {
            sort(bk.begin(), bk.end(), [&](int a, int b) {
                return info.node_true_degree[a] < info.node_true_degree[b];
            });
            int mid = max_size;
            vector<int> upper(bk.begin() + mid, bk.end());
            bk.resize(mid);

            int new_bucket_id = (int)info.buckets.size();
            info.buckets.push_back(upper);
            for (int node : upper)
                info.node_to_bucket[node] = new_bucket_id;
        }
    }

    info.node_to_bucket.clear();
    for (int i = 0; i < (int)info.buckets.size(); i++)
        for (int node : info.buckets[i])
            info.node_to_bucket[node] = i;

    cout << "Built " << info.buckets.size() << " buckets" << endl;
    cout << "d_hat_max: " << info.d_hat_max << endl;

    return info;
}

int CalculateOptimalL(const std::set<int> &actualNodeIDs,
                      const std::unordered_map<int, std::set<int>> &data,
                      double eps2, int k_val, int n) {
    unordered_map<int, double> ks_counts;
    double total_ks = 0;
    long long total_tri = OriginalCliqueCount;

    for (const auto &node : actualNodeIDs) {
        int deg = (int)data.at(node).size();
        ks_counts[node] = BinomCoeff(deg, k_val);
        total_ks += ks_counts[node];
    }

    double avg_ks = total_ks / n;
    double q_bar = (total_ks > 0) ? (double)total_tri / total_ks : 0.01;
    if (q_bar < 1e-6) q_bar = (q == 3) ? 0.1 : 0.01;

    double possible_edges = (double)n * (n - 1) / 2.0;
    double total_edges = 0;
    for (const auto &node : actualNodeIDs)
        total_edges += data.at(node).size();
    total_edges /= 2.0;
    double density = (possible_edges > 0) ? total_edges / possible_edges : 0.01;
    if (density < 1e-6) density = 0.01;

    double avg_bucket_size = beta_val * 10.0;
    double gamma_base = (double)(k_val * k_val * k_val) / (eps2 * eps2 * avg_bucket_size * avg_bucket_size);

    cout << "CalculateOptimalL: avg_ks=" << avg_ks << " q_bar=" << q_bar
         << " density=" << density << " gamma_base=" << gamma_base << endl;

    int min_L = max(1, (int)(avg_ks * 0.3));
    int max_L = min(n, max(100, (int)(avg_ks * 3.0)));
    int step = max(1, (max_L - min_L) / 100);

    double min_loss = 1e30;
    int optimal_L = min_L;

    for (int L = min_L; L <= max_L; L += step) {
        double total_drop_bias = 0.0;
        for (const auto &node : actualNodeIDs) {
            double ks = ks_counts[node];
            if (ks > L) {
                total_drop_bias += (ks - L);
            }
        }
        double bias_sq = pow(q_bar * total_drop_bias, 2);
        double gamma_L = gamma_base * L * L * L;
        double variance = gamma_L * n * L;
        double total_loss = bias_sq + variance;

        if (total_loss < min_loss) {
            min_loss = total_loss;
            optimal_L = L;
        }
    }

    int fine_min = max(1, optimal_L - step * 2);
    int fine_max = min(max_L, optimal_L + step * 2);
    for (int L = fine_min; L <= fine_max; L++) {
        double total_drop_bias = 0.0;
        for (const auto &node : actualNodeIDs) {
            double ks = ks_counts[node];
            if (ks > L) total_drop_bias += (ks - L);
        }
        double bias_sq = pow(q_bar * total_drop_bias, 2);
        double gamma_L = gamma_base * L * L * L;
        double variance = gamma_L * n * L;
        double total_loss = bias_sq + variance;
        if (total_loss < min_loss) {
            min_loss = total_loss;
            optimal_L = L;
        }
    }

    cout << "Optimal L: " << optimal_L << " (loss=" << min_loss << ")" << endl;
    return optimal_L;
}

void EPCC_3Clique(double &relative_error) {
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int n = (int)actualNodeIDs.size();
    long long tri_count = CountCliques(actualNodeIDs, data, 3);
    cout << "Original triangle count: " << tri_count << endl;
    OriginalCliqueCount = tri_count;

    double p = exp(Eps) / (exp(Eps) + 1.0);

    std::unordered_map<int, std::unordered_set<int>> perturbed_edges;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (const auto &node_i : actualNodeIDs) {
        for (const auto &node_j : actualNodeIDs) {
            if (node_i == node_j) continue;
            bool original = data.at(node_i).count(node_j) > 0;
            bool perturbed = original;
            if (dist(gen) > p)
                perturbed = !original;
            if (perturbed)
                perturbed_edges[node_i].insert(node_j);
        }
    }

    double e_eps = exp(Eps);
    double y_when_edge = e_eps / (e_eps - 1.0);
    double y_when_no_edge = -1.0 / (e_eps - 1.0);

    vector<int> nodes(actualNodeIDs.begin(), actualNodeIDs.end());
    int num_samples = min(1000000, (int)(n * (n - 1LL) * (n - 2LL) / 6));

    double sample_sum = 0.0;
    for (int s = 0; s < num_samples; s++) {
        int idx_i = rand() % n;
        int idx_j = rand() % (n - 1);
        int idx_k = rand() % (n - 2);
        if (idx_j >= idx_i) idx_j++;
        if (idx_k >= idx_i) idx_k++;
        if (idx_k >= idx_j) idx_k++;

        int i = nodes[idx_i];
        int j = nodes[idx_j];
        int k = nodes[idx_k];

        double y_ij = perturbed_edges[i].count(j) ? y_when_edge : y_when_no_edge;
        double y_ik = perturbed_edges[i].count(k) ? y_when_edge : y_when_no_edge;
        double y_jk = perturbed_edges[j].count(k) ? y_when_edge : y_when_no_edge;
        sample_sum += y_ij * y_ik * y_jk;
    }

    double total_combos = BinomCoeff(n, 3);
    double estimate = total_combos * sample_sum / num_samples;

    relative_error = std::abs((estimate - tri_count) / (double)tri_count);
    cout << "EPCC estimated triangles: " << estimate << endl;
    cout << "Relative error (EPCC, 3-clique): " << relative_error << endl;
}

void EPCC_4Clique(double &relative_error) {
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int n = (int)actualNodeIDs.size();
    long long clique_count = CountCliques(actualNodeIDs, data, 4);
    cout << "Original 4-clique count: " << clique_count << endl;
    OriginalCliqueCount = clique_count;

    double p = exp(Eps) / (exp(Eps) + 1.0);

    std::unordered_map<int, std::unordered_set<int>> perturbed_edges;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (const auto &node_i : actualNodeIDs) {
        for (const auto &node_j : actualNodeIDs) {
            if (node_i == node_j) continue;
            bool original = data.at(node_i).count(node_j) > 0;
            bool perturbed = original;
            if (dist(gen) > p) perturbed = !original;
            if (perturbed) perturbed_edges[node_i].insert(node_j);
        }
    }

    double e_eps = exp(Eps);
    double y_when_edge = e_eps / (e_eps - 1.0);
    double y_when_no_edge = -1.0 / (e_eps - 1.0);

    vector<int> nodes(actualNodeIDs.begin(), actualNodeIDs.end());
    int num_samples = 1000000;

    double sample_sum = 0.0;
    for (int s = 0; s < num_samples; s++) {
        int idx_a = rand() % n;
        int idx_b = rand() % (n - 1);
        int idx_c = rand() % (n - 2);
        int idx_d = rand() % (n - 3);
        if (idx_b >= idx_a) idx_b++;
        if (idx_c >= idx_a) idx_c++;
        if (idx_d >= idx_a) idx_d++;
        if (idx_c >= idx_b) idx_c++;
        if (idx_d >= idx_b) idx_d++;
        if (idx_d >= idx_c) idx_d++;

        int a = nodes[idx_a], b = nodes[idx_b];
        int c = nodes[idx_c], d = nodes[idx_d];

        double prod = 1.0;
        prod *= perturbed_edges[a].count(b) ? y_when_edge : y_when_no_edge;
        prod *= perturbed_edges[a].count(c) ? y_when_edge : y_when_no_edge;
        prod *= perturbed_edges[a].count(d) ? y_when_edge : y_when_no_edge;
        prod *= perturbed_edges[b].count(c) ? y_when_edge : y_when_no_edge;
        prod *= perturbed_edges[b].count(d) ? y_when_edge : y_when_no_edge;
        prod *= perturbed_edges[c].count(d) ? y_when_edge : y_when_no_edge;
        sample_sum += prod;
    }

    double total_combos = BinomCoeff(n, 4);
    double estimate = total_combos * sample_sum / num_samples;

    relative_error = (clique_count > 0) ? std::abs((estimate - clique_count) / (double)clique_count) : 0.0;
    cout << "EPCC estimated 4-cliques: " << estimate << endl;
    cout << "Relative error (EPCC, 4-clique): " << relative_error << endl;
}

void SPCC_3Clique(double &relative_error) {
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int n = (int)actualNodeIDs.size();
    long long tri_count = CountCliques(actualNodeIDs, data, 3);
    cout << "Original triangle count: " << tri_count << endl;
    OriginalCliqueCount = tri_count;

    double eps1 = Eps / 2.0;
    double eps2 = Eps / 2.0;
    int k = q - 1;

    cout << "eps1=" << eps1 << " eps2=" << eps2 << " k=" << k << endl;

    std::random_device rd;
    std::mt19937 gen(rd());

    int L = CalculateOptimalL(actualNodeIDs, data, eps2, k, n);
    BucketInfo buckets = BuildBuckets(actualNodeIDs, data, eps1, L, n, gen);

    int d_hat_max = buckets.d_hat_max;
    int dum_id = n + 1;

    double eps_prime = eps2 / (double)(k * L);
    cout << "eps_prime=" << eps_prime << " L=" << L << endl;

    vector<vector<vector<int>>> all_reports;
    vector<int> node_list(actualNodeIDs.begin(), actualNodeIDs.end());
    unordered_map<int, int> node_to_idx;
    for (int i = 0; i < n; i++) node_to_idx[node_list[i]] = i;

    vector<vector<int>> bucket_plus_dum = buckets.buckets;
    for (auto &bk : bucket_plus_dum)
        bk.push_back(dum_id);

    vector<int> all_nodes_vec(actualNodeIDs.begin(), actualNodeIDs.end());

    for (int idx = 0; idx < n; idx++) {
        int node_u = node_list[idx];
        const auto &neighbors = data.at(node_u);

        vector<int> proj_neighbors = GraphProjection(neighbors, d_hat_max, gen);
        vector<vector<int>> kstars = BuildKStars(proj_neighbors, k);
        vector<vector<int>> padded = PaddingAndDropping(kstars, L, dum_id, k, gen);

        vector<vector<int>> perturbed_kstars;
        for (auto &kstar : padded) {
            vector<int> perturbed_kstar;
            for (int v_i : kstar) {
                vector<int> domain;
                if (v_i == dum_id) {
                    uniform_int_distribution<int> pick_node(0, n - 1);
                    int x = all_nodes_vec[pick_node(gen)];
                    domain = bucket_plus_dum[buckets.node_to_bucket[x]];
                } else {
                    domain = bucket_plus_dum[buckets.node_to_bucket[v_i]];
                }
                perturbed_kstar.push_back(PerturbAGP(v_i, domain, eps_prime, gen));
            }
            perturbed_kstars.push_back(perturbed_kstar);
        }
        all_reports.push_back(perturbed_kstars);
    }

    unordered_map<int, unordered_map<int, double>> A_hat;

    for (int idx = 0; idx < n; idx++) {
        int node_u = node_list[idx];
        int d_tilde_u = buckets.noisy_degrees[node_u];
        long long R_hat = BinomCoeffLL(d_tilde_u - 1, k - 1);
        if (R_hat <= 0) continue;

        unordered_map<int, double> y_sums;

        for (auto &kstar : all_reports[idx]) {
            for (int v_tilde : kstar) {
                if (v_tilde == dum_id) continue;

                if (buckets.node_to_bucket.find(v_tilde) == buckets.node_to_bucket.end()) continue;
                int bk_id = buckets.node_to_bucket[v_tilde];
                int domain_size = (int)bucket_plus_dum[bk_id].size();

                double p_v = AGPRetainProb(domain_size, eps_prime);

                double denom_plus = domain_size - 1;
                double one_minus_pv = max(1e-15, 1.0 - p_v);
                double flip_prob = one_minus_pv / denom_plus;
                double denominator = p_v - flip_prob;
                if (fabs(denominator) < 1e-15) denominator = 1e-15;

                double y_self = (1.0 - flip_prob) / denominator;
                double y_other = -flip_prob / denominator;

                y_sums[v_tilde] += y_self;
                for (int v_in_bk : buckets.buckets[bk_id]) {
                    if (v_in_bk != v_tilde)
                        y_sums[v_in_bk] += y_other;
                }
            }
        }

        for (auto &ys : y_sums) {
            double a_val = ys.second / (double)R_hat;
            if (fabs(a_val) > 1e-10)
                A_hat[node_u][ys.first] = a_val;
        }
    }

    double estimate = 0.0;
    set<tuple<int, int, int>> counted_cliques;

    for (int idx = 0; idx < n; idx++) {
        int u = node_list[idx];

        for (auto &kstar : all_reports[idx]) {
            if (k == 2 && kstar.size() == 2) {
                int v1 = kstar[0], v2 = kstar[1];

                if (v1 == dum_id || v2 == dum_id) continue;
                if (actualNodeIDs.find(v1) == actualNodeIDs.end()) continue;
                if (actualNodeIDs.find(v2) == actualNodeIDs.end()) continue;

                vector<int> clique = {u, v1, v2};
                sort(clique.begin(), clique.end());

                tuple<int, int, int> clique_key(clique[0], clique[1], clique[2]);
                if (counted_cliques.count(clique_key)) continue;
                counted_cliques.insert(clique_key);

                int a = clique[0], b = clique[1], c = clique[2];
                double a_ab = A_hat[a].count(b) ? A_hat[a][b] : 0.0;
                double a_ac = A_hat[a].count(c) ? A_hat[a][c] : 0.0;
                double a_bc = A_hat[b].count(c) ? A_hat[b][c] : 0.0;

                double product = a_ab * a_ac * a_bc;
                estimate += product;
            }
        }
    }

    relative_error = (tri_count > 0) ? std::abs((estimate - tri_count) / (double)tri_count) : 0.0;
    cout << "SPCC estimated triangles: " << estimate << endl;
    cout << "Relative error (SPCC, 3-clique): " << relative_error << endl;
}

void SPCC_4Clique(double &relative_error) {
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int n = (int)actualNodeIDs.size();
    long long clique_count = CountCliques(actualNodeIDs, data, 4);
    cout << "Original 4-clique count: " << clique_count << endl;
    OriginalCliqueCount = clique_count;

    double eps1 = Eps / 2.0;
    double eps2 = Eps / 2.0;
    int k = q - 1;

    cout << "eps1=" << eps1 << " eps2=" << eps2 << " k=" << k << endl;

    std::random_device rd;
    std::mt19937 gen(rd());

    int L = CalculateOptimalL(actualNodeIDs, data, eps2, k, n);
    BucketInfo buckets = BuildBuckets(actualNodeIDs, data, eps1, L, n, gen);
    int d_hat_max = buckets.d_hat_max;
    int dum_id = n + 1;

    double eps_prime = eps2 / (double)(k * L);
    cout << "eps_prime=" << eps_prime << " L=" << L << endl;

    vector<vector<vector<int>>> all_reports;
    vector<int> node_list(actualNodeIDs.begin(), actualNodeIDs.end());
    unordered_map<int, int> node_to_idx;
    for (int i = 0; i < n; i++) node_to_idx[node_list[i]] = i;

    vector<vector<int>> bucket_plus_dum = buckets.buckets;
    for (auto &bk : bucket_plus_dum)
        bk.push_back(dum_id);

    vector<int> all_nodes_vec(actualNodeIDs.begin(), actualNodeIDs.end());

    for (int idx = 0; idx < n; idx++) {
        int node_u = node_list[idx];
        const auto &neighbors = data.at(node_u);

        vector<int> proj_neighbors = GraphProjection(neighbors, d_hat_max, gen);
        vector<vector<int>> kstars = BuildKStars(proj_neighbors, k);
        vector<vector<int>> padded = PaddingAndDropping(kstars, L, dum_id, k, gen);

        vector<vector<int>> perturbed_kstars;
        for (auto &kstar : padded) {
            vector<int> perturbed_kstar;
            for (int v_i : kstar) {
                vector<int> domain;
                if (v_i == dum_id) {
                    uniform_int_distribution<int> pick_node(0, n - 1);
                    int x = all_nodes_vec[pick_node(gen)];
                    domain = bucket_plus_dum[buckets.node_to_bucket[x]];
                } else {
                    domain = bucket_plus_dum[buckets.node_to_bucket[v_i]];
                }
                perturbed_kstar.push_back(PerturbAGP(v_i, domain, eps_prime, gen));
            }
            perturbed_kstars.push_back(perturbed_kstar);
        }
        all_reports.push_back(perturbed_kstars);
    }

    unordered_map<int, unordered_map<int, double>> A_hat;

    for (int idx = 0; idx < n; idx++) {
        int node_u = node_list[idx];
        int d_tilde_u = buckets.noisy_degrees[node_u];
        long long R_hat = BinomCoeffLL(d_tilde_u - 1, k - 1);
        if (R_hat <= 0) continue;

        unordered_map<int, double> y_sums;

        for (auto &kstar : all_reports[idx]) {
            for (int v_tilde : kstar) {
                if (v_tilde == dum_id) continue;
                if (buckets.node_to_bucket.find(v_tilde) == buckets.node_to_bucket.end()) continue;
                int bk_id = buckets.node_to_bucket[v_tilde];
                int domain_size = (int)bucket_plus_dum[bk_id].size();

                double p_v = AGPRetainProb(domain_size, eps_prime);

                double denom_plus = domain_size - 1;
                double one_minus_pv = max(1e-15, 1.0 - p_v);
                double flip_prob = one_minus_pv / denom_plus;
                double denominator = p_v - flip_prob;
                if (fabs(denominator) < 1e-15) denominator = 1e-15;

                double y_self = (1.0 - flip_prob) / denominator;
                double y_other = -flip_prob / denominator;

                y_sums[v_tilde] += y_self;
                for (int v_in_bk : buckets.buckets[bk_id]) {
                    if (v_in_bk != v_tilde)
                        y_sums[v_in_bk] += y_other;
                }
            }
        }

        for (auto &ys : y_sums) {
            double a_val = ys.second / (double)R_hat;
            if (fabs(a_val) > 1e-10)
                A_hat[node_u][ys.first] = a_val;
        }
    }

    double estimate = 0.0;
    set<tuple<int, int, int, int>> counted_cliques;

    for (int idx = 0; idx < n; idx++) {
        int u = node_list[idx];

        for (auto &kstar : all_reports[idx]) {
            if (k == 3 && kstar.size() == 3) {
                int v1 = kstar[0], v2 = kstar[1], v3 = kstar[2];

                if (v1 == dum_id || v2 == dum_id || v3 == dum_id) continue;
                if (actualNodeIDs.find(v1) == actualNodeIDs.end()) continue;
                if (actualNodeIDs.find(v2) == actualNodeIDs.end()) continue;
                if (actualNodeIDs.find(v3) == actualNodeIDs.end()) continue;

                vector<int> clique = {u, v1, v2, v3};
                sort(clique.begin(), clique.end());

                tuple<int, int, int, int> clique_key(clique[0], clique[1], clique[2], clique[3]);
                if (counted_cliques.count(clique_key)) continue;
                counted_cliques.insert(clique_key);

                int a = clique[0], b = clique[1], c = clique[2], d = clique[3];
                double a_ab = A_hat[a].count(b) ? A_hat[a][b] : 0.0;
                double a_ac = A_hat[a].count(c) ? A_hat[a][c] : 0.0;
                double a_ad = A_hat[a].count(d) ? A_hat[a][d] : 0.0;
                double a_bc = A_hat[b].count(c) ? A_hat[b][c] : 0.0;
                double a_bd = A_hat[b].count(d) ? A_hat[b][d] : 0.0;
                double a_cd = A_hat[c].count(d) ? A_hat[c][d] : 0.0;

                double product = a_ab * a_ac * a_ad * a_bc * a_bd * a_cd;
                estimate += product;
            }
        }
    }

    relative_error = (clique_count > 0) ? std::abs((estimate - clique_count) / (double)clique_count) : 0.0;
    cout << "SPCC estimated 4-cliques: " << estimate << endl;
    cout << "Relative error (SPCC, 4-clique): " << relative_error << endl;
}

int main(int argc, char *argv[])
{
    unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 4;
    init_by_array(init, length);

    if (argc < 5)
    {
        printf("Usage: %s [DatasetPath] [q] [Method] [Eps] ([EdgeFile (default: edges.csv)])\n\n", argv[0]);
        printf("[DatasetPath]: Path to the dataset directory\n");
        printf("[q]: Target clique size (3 or 4)\n");
        printf("[Method]: 1 for EPCC, 2 for SPCC\n");
        printf("[Eps]: Total privacy budget epsilon\n");
        printf("[EdgeFile]: Edge file name (default: edges.csv)\n");
        return -1;
    }

    string dataset_path = argv[1];
    DatasetPath = dataset_path;
    q = atoi(argv[2]);
    k_star = q - 1;
    int method = atoi(argv[3]);

    Eps = atof(argv[4]);
    Eps_s = argv[4];

    string edge_file_name = "edges.csv";
    if (argc >= 6)
        edge_file_name = argv[5];

    EdgeFile = dataset_path + "/" + edge_file_name;

    printf("Total Epsilon: %s (eps1 = eps2 = %g)\n", Eps_s.c_str(), Eps / 2.0);
    printf("Method: %d (%s)\n", method, (method == 1) ? "EPCC" : "SPCC");

    {
        std::unordered_set<int> nodes;
        std::ifstream ef(EdgeFile);
        std::string line;
        if (ef.is_open())
        {
            std::getline(ef, line);
            while (std::getline(ef, line))
            {
                std::stringstream ss(line);
                std::string a, b;
                std::getline(ss, a, ',');
                std::getline(ss, b, ',');
                if (!a.empty() && !b.empty() && std::isdigit(a[0]) && std::isdigit(b[0]))
                {
                    nodes.insert(std::stoi(a));
                    nodes.insert(std::stoi(b));
                }
            }
            ef.close();
        }
        if (!nodes.empty())
            NodeNum = (int)nodes.size();
        cout << "Detected NodeNum: " << NodeNum << endl;
    }

    {
        std::set<int> tmpNodes;
        std::unordered_map<int, std::set<int>> tmpData;
        ReadEdgeFile(EdgeFile, tmpNodes, tmpData);
        OriginalCliqueCount = CountCliques(tmpNodes, tmpData, q);
        cout << "Ground truth " << q << "-clique count: " << OriginalCliqueCount << endl;
    }

    const int NUM_EXPERIMENTS = 10;
    std::vector<double> relative_errors;

    cout << "\n=== Running " << NUM_EXPERIMENTS << " experiments ===" << endl;
    cout << "Dataset: " << dataset_path << endl;
    cout << "Target clique: " << q << "-clique" << endl;
    cout << "Method: " << (method == 1 ? "EPCC" : "SPCC") << endl;

    for (int exp = 1; exp <= NUM_EXPERIMENTS; exp++)
    {
        cout << "\n--- Experiment " << exp << " ---" << endl;

        double err = 0.0;

        if (q == 3)
        {
            if (method == 1) EPCC_3Clique(err);
            else SPCC_3Clique(err);
        }
        else if (q == 4)
        {
            if (method == 1) EPCC_4Clique(err);
            else SPCC_4Clique(err);
        }

        cout << "Experiment " << exp << " completed. Relative error: " << err << endl;
        relative_errors.push_back(err);
    }

    double average = 0.0;
    for (double val : relative_errors) average += val;
    average /= NUM_EXPERIMENTS;

    cout << "\n=== Final Results ===" << endl;
    cout << "Number of experiments: " << NUM_EXPERIMENTS << endl;
    cout << "Target clique: " << q << "-clique" << endl;
    cout << "Original count: " << OriginalCliqueCount << endl;
    cout << "Method: " << (method == 1 ? "EPCC" : "SPCC") << endl;
    cout << "Total epsilon: " << Eps << " (eps1=eps2=" << Eps / 2.0 << ")" << endl;
    cout << "k-star size (k): " << k_star << endl;
    cout << "Average relative error: " << average << endl;

    return 0;
}
