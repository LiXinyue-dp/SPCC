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
#include <cstdlib>   // for rand() and srand()
#include <ctime>     // for time()
#include <utility>   // for std::pair
#include <algorithm> // for std::find
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
using namespace std;

double optimal_pad;
int q;
long long OriginalCliqueCount = 0;

string EdgeFile;
string DatasetPath;
int NodeNum;
double Eps;
string Eps_s;
double EpsNsMaxDeg;
int NSType;
int ItrNum;
int Alg;
double Balloc[3];
char *Balloc_s[3];
stats::rand_engine_t engine(1776);

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

// ============================
// Read edge file into graph structure
// ============================
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
    std::getline(file, line); // Skip header
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

// ============================
// Calculate clique count in the graph
// ============================
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

// ============================
// Lopt: Calculate optimal padding (Section 4.2)
// ============================
double CalculateOptimalPadding(double Eps, int qVal, double clique_ratio = 1.0)
{
    std::string dataset_path = DatasetPath + "/edges.csv";
    std::unordered_map<int, std::set<int>> graph;
    std::set<int> actualNodeIDs;
    ReadEdgeFile(dataset_path, actualNodeIDs, graph);

    int n = actualNodeIDs.size();
    cout << "Total nodes: " << n << endl;

    // Count k-stars and cliques
    double total_kstars = 0.0;
    std::unordered_map<int, int> kstar_counts;

    if (qVal == 3)
    {
        OriginalCliqueCount = CountCliques(actualNodeIDs, graph, 3);
        for (const auto &node : actualNodeIDs)
        {
            int degree = graph[node].size();
            int kstar_count = (degree * (degree - 1)) / 2;
            kstar_counts[node] = kstar_count;
            total_kstars += kstar_count;
        }
    }
    else if (qVal == 4)
    {
        OriginalCliqueCount = CountCliques(actualNodeIDs, graph, 4);
        for (const auto &node : actualNodeIDs)
        {
            int degree = graph[node].size();
            int kstar_count = (degree * (degree - 1) * (degree - 2)) / 6;
            kstar_counts[node] = kstar_count;
            total_kstars += kstar_count;
        }
    }

    double total_degree = 0.0;
    for (const auto &node : actualNodeIDs)
        total_degree += graph[node].size();
    double avg_degree = total_degree / n;
    double avg_kstars = total_kstars / n;

    cout << "Average degree: " << avg_degree << endl;
    cout << "Average k-stars per node: " << avg_kstars << endl;

    double q_bar = (total_kstars > 0) ? (double)OriginalCliqueCount / total_kstars : 0.0;
    if (q_bar < 0.001)
    {
        q_bar = (qVal == 3) ? 0.1 : 0.01;
        cout << "Using empirical q_bar: " << q_bar << endl;
    }
    cout << "Average clique contribution per k-star (q_bar): " << q_bar << endl;

    // Search for optimal L
    int min_L = 1;
    int max_L = std::max(50, static_cast<int>(2 * avg_kstars));
    int step = std::max(1, (max_L - min_L) / 100);
    cout << "Searching for optimal L in range [" << min_L << ", " << max_L << "] with step " << step << endl;

    std::vector<std::pair<int, double>> loss_values;
    double min_loss = std::numeric_limits<double>::max();
    int optimal_L = min_L;

    for (int L = min_L; L <= max_L; L += step)
    {
        double total_loss = 0.0;
        for (const auto &node : actualNodeIDs)
        {
            int kstar_count = kstar_counts[node];
            double bias_drop = 0.0;
            if (kstar_count > L)
                bias_drop = q_bar * (kstar_count - L);

            double bias_pad = 0.0;
            if (kstar_count < L)
            {
                double p_false_clique = avg_degree / n;
                bias_pad = q_bar * p_false_clique * (L - kstar_count);
            }
            total_loss += std::pow(bias_drop + bias_pad, 2);
        }

        loss_values.emplace_back(L, total_loss);
        if (total_loss < min_loss)
        {
            min_loss = total_loss;
            optimal_L = L;
        }
        if (L % (5 * step) == 0)
            cout << "L = " << L << ", Loss = " << total_loss << endl;
    }

    cout << "=== Optimal Padding Length Results ===" << endl;
    cout << "Optimal L: " << optimal_L << endl;
    cout << "Minimum loss: " << min_loss << endl;
    cout << "Average k-stars: " << avg_kstars << endl;
    cout << "Ratio L_opt / avg_kstars: " << (double)optimal_L / avg_kstars << endl;

    std::sort(loss_values.begin(), loss_values.end(),
              [](const std::pair<int,double> &a, const std::pair<int,double> &b) { return a.second < b.second; });

    cout << "Top 5 candidate L values:" << endl;
    for (int i = 0; i < std::min(5, (int)loss_values.size()); i++)
        cout << "L = " << loss_values[i].first << ", Loss = " << loss_values[i].second << endl;

    return optimal_L;
}

double FindOptimalPaddingForDataset()
{
    double Eps = 1.0;
    double clique_ratio = 1.0;
    double optimal_pad = CalculateOptimalPadding(Eps, q, clique_ratio);
    cout << "Recommended optimal padding length: " << optimal_pad << endl;
    return optimal_pad;
}

// ============================
// Build degree groups using Laplace noise + quantile-based splitting (Section 4.3)
// ============================
struct DegreeGroupInfo
{
    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree;
    double avg_group_size;
};

DegreeGroupInfo BuildDegreeGroups(const std::set<int> &actualNodeIDs,
                                   const std::unordered_map<int, std::set<int>> &data,
                                   double localEps)
{
    DegreeGroupInfo info;
    info.maxDegree = 0;

    for (const auto &node : actualNodeIDs)
    {
        double degree = data.at(node).size();
        info.node_degrees[node] = degree;
        if ((int)degree > info.maxDegree)
            info.maxDegree = (int)degree;
    }
    cout << "maxDegree: " << info.maxDegree << endl;

    // Laplace noise for degree obfuscation (Section 4.3)
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / localEps;

    auto laplace_random = [&](double scale)
    {
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);
        double u = uniform(gen);
        return (u > 0) ? -scale * log(1 - 2 * abs(u)) : scale * log(1 + 2 * abs(u));
    };

    std::unordered_map<int, int> noisy_degrees;
    for (const auto &node : actualNodeIDs)
    {
        double noise = laplace_random(laplace_scale);
        double noisy_degree = info.node_degrees[node] + noise;
        noisy_degree = round(noisy_degree);
        noisy_degree = max(1, static_cast<int>(noisy_degree));
        noisy_degree = min(info.maxDegree, static_cast<int>(noisy_degree));
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degree
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        info.degreeGroups[degree].push_back(node);
    }

    // Quantile-based grouping (Section 4.3)
    vector<int> noisy_degree_values;
    for (const auto &node : actualNodeIDs)
        noisy_degree_values.push_back(noisy_degrees[node]);
    sort(noisy_degree_values.begin(), noisy_degree_values.end());

    int median_degree = noisy_degree_values[noisy_degree_values.size() / 2];
    int q90_degree = noisy_degree_values[noisy_degree_values.size() * 9 / 10];

    auto get_target_size = [&](int degree) -> int
    {
        if (degree >= q90_degree) return 3;
        else if (degree >= median_degree) return 10;
        else return 20;
    };

    map<int, vector<int>> optimizedGroups;
    for (auto &group : info.degreeGroups)
    {
        int degree = group.first;
        auto &nodes = group.second;
        int target_size = get_target_size(degree);

        if (nodes.size() > (size_t)(target_size * 1.5))
        {
            sort(nodes.begin(), nodes.end(),
                 [&](int a, int b) { return data.at(a).size() < data.at(b).size(); });
            int split_pos = nodes.size() / 2;
            vector<int> lower(nodes.begin(), nodes.begin() + split_pos);
            vector<int> upper(nodes.begin() + split_pos, nodes.end());
            int rep1 = data.at(lower[lower.size() / 2]).size();
            int rep2 = data.at(upper[upper.size() / 2]).size();
            optimizedGroups[rep1] = lower;
            optimizedGroups[rep2] = upper;
        }
        else if (nodes.size() < (size_t)(target_size * 0.7))
        {
            auto closest_it = optimizedGroups.begin();
            int min_diff = abs(closest_it->first - degree);
            for (auto it = optimizedGroups.begin(); it != optimizedGroups.end(); ++it)
            {
                int diff = abs(it->first - degree);
                if (diff < min_diff) min_diff = diff;
            }
            if (min_diff < median_degree / 2)
                closest_it->second.insert(closest_it->second.end(), nodes.begin(), nodes.end());
            else
                optimizedGroups[degree] = nodes;
        }
        else
        {
            optimizedGroups[degree] = nodes;
        }
    }
    info.degreeGroups = optimizedGroups;

    info.avg_group_size = 0.0;
    for (const auto &group : info.degreeGroups)
        info.avg_group_size += group.second.size();
    info.avg_group_size /= info.degreeGroups.size();
    cout << "avg_group_size: " << info.avg_group_size << endl;

    return info;
}

// ============================
// RR perturbation for a single element (common helper)
// ============================
int PerturbElement(int element, int node_i, int actualNodeCount,
                   const DegreeGroupInfo &dgInfo,
                   const std::unordered_map<int, std::set<int>> &data)
{
    if (element == NodeNum)
    {
        // Generate random replacement from valid node range
        int x;
        do
        {
            std::random_device rd2;
            unsigned seed2 = rd2() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::mt19937 gen2(seed2);
            std::uniform_int_distribution<> dis2(0, NodeNum);
            x = dis2(gen2);
        } while (x == node_i);
        return x;
    }

    auto it = dgInfo.node_degrees.find(element);
    if (it != dgInfo.node_degrees.end())
    {
        double degree = it->second;
        auto degreeGroup = dgInfo.degreeGroups.find((int)degree);
        if (degreeGroup != dgInfo.degreeGroups.end())
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double prob = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
            if (dist(gen) > prob)
            {
                std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                int r;
                do
                {
                    r = degreeGroup->second[dis_idx(gen)];
                } while (r == node_i);
                return r;
            }
        }
    }
    return element;
}

// ============================
// EPCC: Enhanced Privacy-Preserving Clique Counting (Section 5.1)
// Method 1 - Enhanced baseline with Lopt + degree-based grouping
// ============================
void EPCC_3Clique(double &relative_error, double custom_pad = -1)
{
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());

    // Calculate original triangle count
    long long tri_count = CountCliques(actualNodeIDs, data, 3);
    cout << "Original triangle count: " << tri_count << endl;
    OriginalCliqueCount = tri_count;

    // Set pad (Lopt)
    double pad = (custom_pad > 0) ? custom_pad : 20;
    double localEps = Eps / pad; // Privacy budget per element
    cout << "Local epsilon per element: " << localEps << endl;

    // Build subDict: for each node, list of k-star tuples (pairs for 3-clique)
    map<int, vector<pair<int, int>>> subDict;
    long long star = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (size_t l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
                subDict[node_i].push_back(make_pair(neighbors[0], NodeNum));
            for (size_t m = l + 1; m < neighbors.size(); m++)
            {
                subDict[node_i].push_back(make_pair(neighbors[l], neighbors[m]));
                star++;
            }
        }
    }

    // Build degree groups (Section 4.3)
    DegreeGroupInfo dgInfo = BuildDegreeGroups(actualNodeIDs, data, localEps);

    // Padding and dropping (Section 5.1)
    for (const auto &node_i : actualNodeIDs)
    {
        if ((int)subDict[node_i].size() > (int)pad)
        {
            auto it = subDict[node_i].begin();
            std::advance(it, (int)pad);
            subDict[node_i].erase(it, subDict[node_i].end());
        }
        if ((int)subDict[node_i].size() < (int)pad)
        {
            int ttque = (int)pad - subDict[node_i].size();
            for (int que = 1; que <= ttque; que++)
                subDict[node_i].push_back(make_pair(NodeNum, NodeNum));
        }
    }

    // Perturbation (Section 3.2): each node perturbs its k-star tuples using RR
    map<int, vector<pair<int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_pair : subDictPerturbed[node_i])
        {
            edge_pair.first = PerturbElement(edge_pair.first, node_i, actualNodeCount, dgInfo, data);
            if (edge_pair.second != -1 && edge_pair.second != NodeNum)
                edge_pair.second = PerturbElement(edge_pair.second, node_i, actualNodeCount, dgInfo, data);
            else if (edge_pair.second == NodeNum)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double prob = exp(Eps) / (exp(Eps) + dgInfo.degreeGroups.size());
                if (dist(gen) > prob)
                {
                    // Generate random replacement
                    int x;
                    do
                    {
                        std::random_device rd2;
                        unsigned seed2 = rd2() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen2(seed2);
                        std::uniform_int_distribution<> dis2(0, NodeNum);
                        x = dis2(gen2);
                    } while (x == node_i);
                    edge_pair.second = x;
                }
                else
                {
                    edge_pair.second = NodeNum; // keep as "no second node"
                }
            }

            myDeg[node_i].insert(edge_pair.first);
            if (edge_pair.second != -1 && edge_pair.second != NodeNum)
                myDeg[node_i].insert(edge_pair.second);
        }
    }

    // Count perturbed triangles
    double tri_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 2) continue;
        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (size_t j = 0; j < neighborList.size(); j++)
            for (size_t k = j + 1; k < neighborList.size(); k++)
                if (myDeg[neighborList[j]].count(neighborList[k]))
                    tri_num += 1;
    }

    relative_error = std::abs((tri_num - tri_count) / (double)tri_count);
    cout << "Perturbed triangle count: " << tri_num << endl;
    cout << "Relative error (EPCC, 3-clique): " << relative_error << endl;
}

void EPCC_4Clique(double &relative_error, double custom_pad = -1)
{
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());

    long long clique_count = CountCliques(actualNodeIDs, data, 4);
    cout << "Original 4-clique count: " << clique_count << endl;
    OriginalCliqueCount = clique_count;

    double pad = (custom_pad > 0) ? custom_pad : 20;
    double localEps = Eps / pad;
    cout << "Local epsilon per element: " << localEps << endl;

    // Build subDict: for each node, list of 3-star tuples
    map<int, vector<tuple<int, int, int>>> subDict;
    long long star = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (size_t l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
                subDict[node_i].push_back(make_tuple(neighbors[0], NodeNum, NodeNum));
            for (size_t m = l + 1; m < neighbors.size(); m++)
            {
                if (neighbors.size() == 2)
                    subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], NodeNum));
                for (size_t n = m + 1; n < neighbors.size(); n++)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], neighbors[n]));
                    star++;
                }
            }
        }
    }

    DegreeGroupInfo dgInfo = BuildDegreeGroups(actualNodeIDs, data, localEps);

    // Padding and dropping
    for (const auto &node_i : actualNodeIDs)
    {
        if ((int)subDict[node_i].size() > (int)pad)
        {
            auto it = subDict[node_i].begin();
            std::advance(it, (int)pad);
            subDict[node_i].erase(it, subDict[node_i].end());
        }
        if ((int)subDict[node_i].size() < (int)pad)
        {
            int ttque = (int)pad - subDict[node_i].size();
            for (int que = 1; que <= ttque; que++)
                subDict[node_i].push_back(make_tuple(NodeNum, NodeNum, NodeNum));
        }
    }

    // Perturbation
    map<int, vector<tuple<int, int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_tuple : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 3; t++)
            {
                int *target = nullptr;
                switch (t)
                {
                case 0: target = &get<0>(edge_tuple); break;
                case 1: target = &get<1>(edge_tuple); break;
                case 2: target = &get<2>(edge_tuple); break;
                }

                if (*target == -1) continue;

                if (*target == NodeNum)
                {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<double> dist(0.0, 1.0);
                    double prob = exp(Eps) / (exp(Eps) + dgInfo.degreeGroups.size());
                    if (dist(gen) > prob)
                    {
                        int x;
                        do
                        {
                            std::random_device rd2;
                            unsigned seed2 = rd2() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                            std::mt19937 gen2(seed2);
                            std::uniform_int_distribution<> dis2(0, NodeNum);
                            x = dis2(gen2);
                        } while (x == node_i);
                        *target = x;
                    }
                    else
                    {
                        // Keep as NodeNum (no replacement)
                    }
                }
                else
                {
                    *target = PerturbElement(*target, node_i, actualNodeCount, dgInfo, data);
                }
            }

            myDeg[node_i].insert(get<0>(edge_tuple));
            if (get<1>(edge_tuple) != -1 && get<1>(edge_tuple) != NodeNum)
                myDeg[node_i].insert(get<1>(edge_tuple));
            if (get<2>(edge_tuple) != -1 && get<2>(edge_tuple) != NodeNum)
                myDeg[node_i].insert(get<2>(edge_tuple));
        }
    }

    // Count perturbed 4-cliques
    double clique_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 3) continue;
        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (size_t j = 0; j < neighborList.size(); j++)
            for (size_t k = j + 1; k < neighborList.size(); k++)
                for (size_t l = k + 1; l < neighborList.size(); l++)
                {
                    int a = neighborList[j], b = neighborList[k], c = neighborList[l];
                    if (myDeg[a].count(b) && myDeg[a].count(c) && myDeg[b].count(c))
                        clique_num += 1;
                }
    }

    relative_error = std::abs((clique_num - clique_count) / (double)clique_count);
    cout << "Perturbed 4-clique count: " << clique_num << endl;
    cout << "Relative error (EPCC, 4-clique): " << relative_error << endl;
}

// ============================
// SPCC: Secure Privacy-Preserving Clique Counting (Section 5.2)
// Method 2 - K-Star with Lopt + grouping + cross-checking + candidate validation
// ============================
void SPCC_3Clique(double &relative_error, double custom_pad = -1)
{
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());

    long long tri_count = CountCliques(actualNodeIDs, data, 3);
    cout << "Original triangle count: " << tri_count << endl;
    OriginalCliqueCount = tri_count;

    double pad = (custom_pad > 0) ? custom_pad : 20;
    double localEps = Eps / pad;
    cout << "Local epsilon per element: " << localEps << endl;

    // Build subDict (2-star tuples for 3-clique)
    map<int, vector<pair<int, int>>> subDict;
    long long star = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (size_t l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
                subDict[node_i].push_back(make_pair(neighbors[0], NodeNum));
            for (size_t m = l + 1; m < neighbors.size(); m++)
            {
                subDict[node_i].push_back(make_pair(neighbors[l], neighbors[m]));
                star++;
            }
        }
    }

    DegreeGroupInfo dgInfo = BuildDegreeGroups(actualNodeIDs, data, localEps);

    // Padding and dropping
    for (const auto &node_i : actualNodeIDs)
    {
        if ((int)subDict[node_i].size() > (int)pad)
        {
            auto it = subDict[node_i].begin();
            std::advance(it, (int)pad);
            subDict[node_i].erase(it, subDict[node_i].end());
        }
        if ((int)subDict[node_i].size() < (int)pad)
        {
            int ttque = (int)pad - subDict[node_i].size();
            for (int que = 1; que <= ttque; que++)
                subDict[node_i].push_back(make_pair(NodeNum, NodeNum));
        }
    }

    // Perturbation
    map<int, vector<pair<int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_pair : subDictPerturbed[node_i])
        {
            edge_pair.first = PerturbElement(edge_pair.first, node_i, actualNodeCount, dgInfo, data);
            if (edge_pair.second != -1 && edge_pair.second != NodeNum)
                edge_pair.second = PerturbElement(edge_pair.second, node_i, actualNodeCount, dgInfo, data);
            else if (edge_pair.second == NodeNum)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double prob = exp(Eps) / (exp(Eps) + dgInfo.degreeGroups.size());
                if (dist(gen) > prob)
                {
                    int x;
                    do
                    {
                        std::random_device rd2;
                        unsigned seed2 = rd2() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen2(seed2);
                        std::uniform_int_distribution<> dis2(0, NodeNum);
                        x = dis2(gen2);
                    } while (x == node_i);
                    edge_pair.second = x;
                }
                else
                {
                    edge_pair.second = NodeNum;
                }
            }

            myDeg[node_i].insert(edge_pair.first);
            if (edge_pair.second != -1 && edge_pair.second != NodeNum)
                myDeg[node_i].insert(edge_pair.second);
        }
    }

    // Cross-checking (Section 5.2): for each reported edge (i,j) in G_i,
    // keep only if i is also in G_j's report
    std::unordered_map<int, std::set<int>> myDegChecked;
    for (const auto &node_i : actualNodeIDs)
    {
        for (int neighbor : myDeg[node_i])
        {
            if (neighbor == node_i) continue;
            // Check if node_i is in G_j's reported set
            if (myDeg.find(neighbor) != myDeg.end() && myDeg[neighbor].count(node_i))
            {
                myDegChecked[node_i].insert(neighbor);
            }
        }
    }

    // Count perturbed triangles using cross-checked graph
    double tri_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDegChecked[node_i];
        if (neighbors.size() < 2) continue;
        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (size_t j = 0; j < neighborList.size(); j++)
            for (size_t k = j + 1; k < neighborList.size(); k++)
                if (myDegChecked[neighborList[j]].count(neighborList[k]))
                    tri_num += 1;
    }

    relative_error = std::abs((tri_num - tri_count) / (double)tri_count);
    cout << "Perturbed triangle count: " << tri_num << endl;
    cout << "Relative error (SPCC, 3-clique): " << relative_error << endl;
}

void SPCC_4Clique(double &relative_error, double custom_pad = -1)
{
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());

    long long clique_count = CountCliques(actualNodeIDs, data, 4);
    cout << "Original 4-clique count: " << clique_count << endl;
    OriginalCliqueCount = clique_count;

    double pad = (custom_pad > 0) ? custom_pad : 20;
    double localEps = Eps / pad;
    cout << "Local epsilon per element: " << localEps << endl;

    // Build subDict (3-star tuples for 4-clique)
    map<int, vector<tuple<int, int, int>>> subDict;
    long long star = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (size_t l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
                subDict[node_i].push_back(make_tuple(neighbors[0], NodeNum, NodeNum));
            for (size_t m = l + 1; m < neighbors.size(); m++)
            {
                if (neighbors.size() == 2)
                    subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], NodeNum));
                for (size_t n = m + 1; n < neighbors.size(); n++)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], neighbors[n]));
                    star++;
                }
            }
        }
    }

    DegreeGroupInfo dgInfo = BuildDegreeGroups(actualNodeIDs, data, localEps);

    // Padding and dropping
    for (const auto &node_i : actualNodeIDs)
    {
        if ((int)subDict[node_i].size() > (int)pad)
        {
            auto it = subDict[node_i].begin();
            std::advance(it, (int)pad);
            subDict[node_i].erase(it, subDict[node_i].end());
        }
        if ((int)subDict[node_i].size() < (int)pad)
        {
            int ttque = (int)pad - subDict[node_i].size();
            for (int que = 1; que <= ttque; que++)
                subDict[node_i].push_back(make_tuple(NodeNum, NodeNum, NodeNum));
        }
    }

    // Perturbation
    map<int, vector<tuple<int, int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_tuple : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 3; t++)
            {
                int *target = nullptr;
                switch (t)
                {
                case 0: target = &get<0>(edge_tuple); break;
                case 1: target = &get<1>(edge_tuple); break;
                case 2: target = &get<2>(edge_tuple); break;
                }

                if (*target == -1) continue;

                if (*target == NodeNum)
                {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<double> dist(0.0, 1.0);
                    double prob = exp(Eps) / (exp(Eps) + dgInfo.degreeGroups.size());
                    if (dist(gen) > prob)
                    {
                        int x;
                        do
                        {
                            std::random_device rd2;
                            unsigned seed2 = rd2() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                            std::mt19937 gen2(seed2);
                            std::uniform_int_distribution<> dis2(0, NodeNum);
                            x = dis2(gen2);
                        } while (x == node_i);
                        *target = x;
                    }
                }
                else
                {
                    *target = PerturbElement(*target, node_i, actualNodeCount, dgInfo, data);
                }
            }

            myDeg[node_i].insert(get<0>(edge_tuple));
            if (get<1>(edge_tuple) != -1 && get<1>(edge_tuple) != NodeNum)
                myDeg[node_i].insert(get<1>(edge_tuple));
            if (get<2>(edge_tuple) != -1 && get<2>(edge_tuple) != NodeNum)
                myDeg[node_i].insert(get<2>(edge_tuple));
        }
    }

    // Cross-checking (Section 5.2)
    std::unordered_map<int, std::set<int>> myDegChecked;
    for (const auto &node_i : actualNodeIDs)
    {
        for (int neighbor : myDeg[node_i])
        {
            if (neighbor == node_i) continue;
            if (myDeg.find(neighbor) != myDeg.end() && myDeg[neighbor].count(node_i))
                myDegChecked[node_i].insert(neighbor);
        }
    }

    // Candidate validation (Section 5.2): for each potential 3-clique (i,j,k),
    // verify that all cross-edges exist among the candidates
    double clique_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDegChecked[node_i];
        if (neighbors.size() < 3) continue;
        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (size_t j = 0; j < neighborList.size(); j++)
            for (size_t k = j + 1; k < neighborList.size(); k++)
                for (size_t l = k + 1; l < neighborList.size(); l++)
                {
                    int a = neighborList[j], b = neighborList[k], c = neighborList[l];
                    // Verify all mutual edges exist in cross-checked graph
                    if (myDegChecked[a].count(b) && myDegChecked[a].count(c) && myDegChecked[b].count(c))
                        clique_num += 1;
                }
    }

    relative_error = std::abs((clique_num - clique_count) / (double)clique_count);
    cout << "Perturbed 4-clique count: " << clique_num << endl;
    cout << "Relative error (SPCC, 4-clique): " << relative_error << endl;
}

// ============================
// Baseline RR (for comparison, without Lopt)
// ============================
void BaselineRR3Clique(double &relative_error)
{
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    long long tri_count = CountCliques(actualNodeIDs, data, 3);
    cout << "Original triangle count: " << tri_count << endl;

    double localEps = Eps;
    double p = exp(localEps) / (exp(localEps) + 1);

    std::unordered_map<int, std::set<int>> perturbed_data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : actualNodeIDs)
        {
            if (node_i == node_j) continue;
            bool original_edge = data[node_i].count(node_j) > 0;
            bool perturbed_edge = original_edge;
            if (dist(gen) > p)
                perturbed_edge = !original_edge;
            if (perturbed_edge)
                perturbed_data[node_i].insert(node_j);
        }
    }

    long long observed_edges = 0;
    for (const auto &pair : perturbed_data)
        observed_edges += pair.second.size();
    observed_edges /= 2;

    long long n = actualNodeCount;
    long long possible_edges = n * (n - 1) / 2;
    double hat_m = (observed_edges - possible_edges * (1 - p)) / (2 * p - 1);
    double theta = (observed_edges > 0) ? std::min(1.0, std::max(0.0, hat_m / observed_edges)) : 0.0;
    cout << "Observed edges: " << observed_edges << ", Estimated true edges: " << hat_m << ", Theta: " << theta << endl;

    std::unordered_map<int, std::set<int>> corrected_data;
    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : perturbed_data[node_i])
        {
            if (node_i < node_j && dist(gen) <= theta)
            {
                corrected_data[node_i].insert(node_j);
                corrected_data[node_j].insert(node_i);
            }
        }
    }

    double perturbed_tri_count = 0;
    std::vector<int> nodes(actualNodeIDs.begin(), actualNodeIDs.end());
    for (size_t i = 0; i < nodes.size(); i++)
        for (size_t j = i + 1; j < nodes.size(); j++)
            for (size_t k = j + 1; k < nodes.size(); k++)
            {
                int n1 = nodes[i], n2 = nodes[j], n3 = nodes[k];
                if (corrected_data[n1].count(n2) && corrected_data[n1].count(n3) && corrected_data[n2].count(n3))
                    perturbed_tri_count++;
            }

    relative_error = std::abs((perturbed_tri_count - tri_count) / (double)tri_count);
    cout << "Perturbed triangle count: " << perturbed_tri_count << endl;
    cout << "Relative error for 3-clique (Baseline RR): " << relative_error << endl;
}

void BaselineRR4Clique(double &relative_error)
{
    std::set<int> actualNodeIDs;
    std::unordered_map<int, std::set<int>> data;
    ReadEdgeFile(EdgeFile, actualNodeIDs, data);

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    long long clique4_count = CountCliques(actualNodeIDs, data, 4);
    cout << "Original 4-clique count: " << clique4_count << endl;

    double localEps = Eps;
    double p = exp(localEps) / (exp(localEps) + 1);

    std::unordered_map<int, std::set<int>> perturbed_data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : actualNodeIDs)
        {
            if (node_i == node_j) continue;
            bool original_edge = data[node_i].count(node_j) > 0;
            bool perturbed_edge = original_edge;
            if (dist(gen) > p)
                perturbed_edge = !original_edge;
            if (perturbed_edge)
                perturbed_data[node_i].insert(node_j);
        }
    }

    long long observed_edges = 0;
    for (const auto &pair : perturbed_data)
        observed_edges += pair.second.size();
    observed_edges /= 2;

    long long n = actualNodeCount;
    long long possible_edges = n * (n - 1) / 2;
    double hat_m = (observed_edges - possible_edges * (1 - p)) / (2 * p - 1);
    double theta = (observed_edges > 0) ? std::min(1.0, std::max(0.0, hat_m / observed_edges)) : 0.0;
    cout << "Observed edges: " << observed_edges << ", Estimated true edges: " << hat_m << ", Theta: " << theta << endl;

    std::unordered_map<int, std::set<int>> corrected_data;
    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : perturbed_data[node_i])
        {
            if (node_i < node_j && dist(gen) <= theta)
            {
                corrected_data[node_i].insert(node_j);
                corrected_data[node_j].insert(node_i);
            }
        }
    }

    std::vector<int> nodes(actualNodeIDs.begin(), actualNodeIDs.end());
    double perturbed_clique4_count = 0;
    for (size_t i = 0; i < nodes.size(); i++)
        for (size_t j = i + 1; j < nodes.size(); j++)
            for (size_t k = j + 1; k < nodes.size(); k++)
                for (size_t l = k + 1; l < nodes.size(); l++)
                {
                    int n1 = nodes[i], n2 = nodes[j], n3 = nodes[k], n4 = nodes[l];
                    if (corrected_data[n1].count(n2) && corrected_data[n1].count(n3) && corrected_data[n1].count(n4) &&
                        corrected_data[n2].count(n3) && corrected_data[n2].count(n4) &&
                        corrected_data[n3].count(n4))
                        perturbed_clique4_count++;
                }

    relative_error = (clique4_count > 0) ? std::abs((perturbed_clique4_count - clique4_count) / (double)clique4_count) : 0.0;
    cout << "Perturbed 4-clique count: " << perturbed_clique4_count << endl;
    cout << "Relative error for 4-clique (Baseline RR): " << relative_error << endl;
}

// ============================
// Main
// ============================
int main(int argc, char *argv[])
{
    unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 4;
    init_by_array(init, length);

    if (argc < 5)
    {
        printf("Usage: %s [DatasetPath] [q] [Method] [Eps] ([EdgeFile (default: edges.csv)])\n\n", argv[0]);
        printf("[DatasetPath]: Path to the dataset directory\n");
        printf("[q]: Target clique size (3 or 4)\n");
        printf("[Method]: 1 for EPCC (Enhanced PCC), 2 for SPCC (Secure PCC)\n");
        printf("[Eps]: Privacy parameter epsilon\n");
        printf("[EdgeFile]: Edge file name (default: edges.csv)\n");
        return -1;
    }

    string dataset_path = argv[1];
    DatasetPath = dataset_path;
    q = atoi(argv[2]);
    int method = atoi(argv[3]);

    double inputEps = atof(argv[4]);
    Eps_s = argv[4];

    // Set epsilon based on clique size q
    if (q == 3)
        Eps = inputEps / 2.0;
    else if (q == 4)
        Eps = inputEps / 3.0;
    else
        Eps = inputEps;

    string edge_file_name = "edges.csv";
    if (argc >= 6)
        edge_file_name = argv[5];

    EdgeFile = dataset_path + "/" + edge_file_name;

    printf("Input Eps: %s, Adjusted Eps (used): %g\n", Eps_s.c_str(), Eps);
    printf("Method: %d (%s)\n", method, (method == 1) ? "EPCC" : "SPCC");

    NodeNum = -1;
    EpsNsMaxDeg = Eps / 10;
    NSType = 0;
    ItrNum = 1;
    Alg = 2;

    for (int i = 0; i < 3; i++)
    {
        Balloc[i] = 1.0;
        Balloc_s[i] = (char *)"1";
    }

    // Calculate optimal padding (Lopt, Section 4.2)
    optimal_pad = FindOptimalPaddingForDataset();
    cout << "Computed optimal padding: " << optimal_pad << endl;

    // Detect number of nodes
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

    // Run experiments
    const int NUM_EXPERIMENTS = 10;
    std::vector<double> relative_errors;

    cout << "\n=== Running " << NUM_EXPERIMENTS << " experiments ===" << endl;
    cout << "Dataset: " << dataset_path << endl;
    cout << "Target clique: " << q << "-clique" << endl;
    cout << "Method: " << (method == 1 ? "EPCC" : "SPCC") << endl;

    for (int exp = 1; exp <= NUM_EXPERIMENTS; exp++)
    {
        cout << "--- Experiment " << exp << " ---" << endl;

        double clique_num_ns = 0.0;

        if (q == 3)
        {
            if (method == 1)
            {
                cout << "Calling EPCC_3Clique..." << endl;
                EPCC_3Clique(clique_num_ns, optimal_pad);
            }
            else
            {
                cout << "Calling SPCC_3Clique..." << endl;
                SPCC_3Clique(clique_num_ns, optimal_pad);
            }
        }
        else if (q == 4)
        {
            if (method == 1)
            {
                cout << "Calling EPCC_4Clique..." << endl;
                EPCC_4Clique(clique_num_ns, optimal_pad);
            }
            else
            {
                cout << "Calling SPCC_4Clique..." << endl;
                SPCC_4Clique(clique_num_ns, optimal_pad);
            }
        }

        cout << "Experiment " << exp << " completed." << endl << endl;
        relative_errors.push_back(clique_num_ns);
    }

    double average = 0.0;
    for (double val : relative_errors)
        average += val;
    average /= NUM_EXPERIMENTS;

    cout << "=== Final Results ===" << endl;
    cout << "Number of experiments: " << NUM_EXPERIMENTS << endl;
    cout << "Target clique: " << q << "-clique" << endl;
    cout << "Original " << (q == 3 ? "triangle" : "4-clique") << " count: " << OriginalCliqueCount << endl;
    cout << "Method: " << (method == 1 ? "EPCC" : "SPCC") << endl;
    cout << "Optimal padding (Lopt): " << optimal_pad << endl;
    cout << "Average relative error: " << average << endl;

    return 0;
}
