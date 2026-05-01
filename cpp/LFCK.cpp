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
#include <fstream>
#include <iomanip>
using namespace std;
double optimal_pad;
int q;
long long OriginalCliqueCount = 0;

string EdgeFile;
string DatasetPath; // Global dataset path
int NodeNum;
double Eps;
string Eps_s;
double EpsNsMaxDeg;
int NSType;
int ItrNum;
int Alg;
double Balloc[3];
char *Balloc_s[3];
// Initialization of statslib
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

// Lopt
double CalculateOptimalPadding(double Eps, int q, double clique_ratio = 1.0)
{
    // Use global DatasetPath to read the dataset
    std::string dataset_path = DatasetPath + "/edges.csv";
    std::unordered_map<int, std::set<int>> graph;
    std::set<int> actualNodeIDs;

    std::ifstream file(dataset_path);
    std::string line;

    // Skip header line (if present)
    std::getline(file, line);

    // Read edge data
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

            graph[firstNum].insert(secondNum);
            graph[secondNum].insert(firstNum);

            actualNodeIDs.insert(firstNum);
            actualNodeIDs.insert(secondNum);
        }
    }
    file.close();

    int n = actualNodeIDs.size();
    cout << "Total nodes: " << n << endl;
    
    // Calculate k-star count and average degree for each node
    std::unordered_map<int, int> kstar_counts;
    double total_degree = 0.0;
    double total_kstars = 0.0;
    int kstar_count;
    if (q == 3) {
        // Calculate 3-clique (triangle) count
        for (const auto &node_i : actualNodeIDs) {
            const auto &neighbors_i = graph[node_i];
            for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j) {
                if (*it_j <= node_i) continue;
                for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k) {
                    if (*it_k <= node_i) continue;
                    if (graph[*it_j].count(*it_k)) {
                        OriginalCliqueCount++;
                    }
                }
            }
        }
        
        // Calculate 2-star count
        for (const auto& node : actualNodeIDs) {
            int degree = graph[node].size();
            // Number of 2-stars = C(degree, 2)
            int kstar_count = (degree * (degree - 1)) / 2;
            kstar_counts[node] = kstar_count;
            total_kstars += kstar_count;
        }
    } 
    else if (q == 4) {
        // Calculate 4-clique count
        std::vector<int> nodes(actualNodeIDs.begin(), actualNodeIDs.end());
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                for (int k = j + 1; k < nodes.size(); k++) {
                    for (int l = k + 1; l < nodes.size(); l++) {
                        int n1 = nodes[i], n2 = nodes[j], n3 = nodes[k], n4 = nodes[l];
                        if (graph[n1].count(n2) && graph[n1].count(n3) && graph[n1].count(n4) &&
                            graph[n2].count(n3) && graph[n2].count(n4) &&
                            graph[n3].count(n4)) {
                            OriginalCliqueCount++;
                        }
                    }
                }
            }
        }
        
        // Calculate 3-star count
        for (const auto& node : actualNodeIDs) {
            int degree = graph[node].size();
            // Number of 3-stars = C(degree, 3)
            int kstar_count = (degree * (degree - 1) * (degree - 2)) / 6;
            kstar_counts[node] = kstar_count;
            total_kstars += kstar_count;
        }
    }
    for (const auto &node : actualNodeIDs)
    {
        int degree = graph[node].size();
        total_degree += degree;
    }
    double avg_degree = total_degree / n;
    double avg_kstars = total_kstars / n;

    cout << "Average degree: " << avg_degree << endl;
    cout << "Average k-stars per node: " << avg_kstars << endl;

   double q_bar = (total_kstars > 0) ? OriginalCliqueCount / total_kstars : 0.0;

    if (q_bar < 0.001) {
        if (q == 3) {
            q_bar = 0.1; // Empirical conversion rate from 2-star to triangle
        } else {
            q_bar = 0.01; // Empirical conversion rate from 3-star to 4-clique
        }
        cout << "Using empirical q_bar: " << q_bar << endl;
    }

    cout << "Average clique contribution per k-star (q_bar): " << q_bar << endl;

    // Define search range
    int min_L = 1;
    int max_L = std::max(50, static_cast<int>(2 * avg_kstars)); // Search at least up to 50
    int step = std::max(1, (max_L - min_L) / 100); // Finer search step size

    cout << "Searching for optimal L in range [" << min_L << ", " << max_L << "] with step " << step << endl;

    // Calculate loss function value for each candidate L
    std::vector<std::pair<int, double>> loss_values;
    double min_loss = std::numeric_limits<double>::max();
    int optimal_L = min_L;

    for (int L = min_L; L <= max_L; L += step)
    {
        double total_loss = 0.0;

        for (const auto &node : actualNodeIDs)
        {
            int kstar_count = kstar_counts[node];

            // Calculate dropping bias
            double bias_drop = 0.0;
            if (kstar_count > L)
            {
                bias_drop = q_bar * (kstar_count - L);
            }

            // Calculate padding bias
            double bias_pad = 0.0;
            if (kstar_count < L)
            {
                // Probability of fake edge forming clique ≈ average degree / total nodes
                double p_false_clique = avg_degree / n;
                bias_pad = q_bar * p_false_clique * (L - kstar_count);
            }

            // Accumulate squared loss
            total_loss += std::pow(bias_drop + bias_pad, 2);
        }

        loss_values.emplace_back(L, total_loss);

        if (total_loss < min_loss)
        {
            min_loss = total_loss;
            optimal_L = L;
        }

        // Output debug information (optional)
        if (L % (5 * step) == 0)
        {
            cout << "L = " << L << ", Loss = " << total_loss << endl;
        }
    }

    // Output optimal results
    cout << "=== Optimal Padding Length Results ===" << endl;
    cout << "Optimal L: " << optimal_L << endl;
    cout << "Minimum loss: " << min_loss << endl;
    cout << "Average k-stars: " << avg_kstars << endl;
    cout << "Ratio L_opt / avg_kstars: " << (double)optimal_L / avg_kstars << endl;

    // Optional: Output the first few minimum values of the loss function curve
    std::sort(loss_values.begin(), loss_values.end(),
              [](const auto &a, const auto &b)
              { return a.second < b.second; });

    cout << "Top 5 candidate L values:" << endl;
    for (int i = 0; i < std::min(5, (int)loss_values.size()); i++)
    {
        cout << "L = " << loss_values[i].first << ", Loss = " << loss_values[i].second << endl;
    }

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

void NewCalNLocTriedge2k3cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{

    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double tri_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;

    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;
    cout << "Eps" << Eps << endl;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
    // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    // Calculate original triangle count
    int tri_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (data[*it_j].count(*it_k))
                {
                    tri_count++;
                }
            }
        }
    }
    cout << "Original triangle count: " << tri_count << endl;
    OriginalCliqueCount = tri_count;
    double star;
    cout << "NodeNum" << NodeNum << endl;

    // Build subDict (using actual node IDs)

    map<int, vector<pair<int, int>>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
            {
                subDict[node_i].push_back(make_pair(neighbors[0], NodeNum)); // Use -1 to indicate no second node
            }
            for (int m = l + 1; m < neighbors.size(); m++)
            {
                subDict[node_i].push_back(make_pair(neighbors[l], neighbors[m]));
                star++;
            }
        }
    }

    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;
    vector<int> degrees;
    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        degrees.push_back(data[node].size());
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);//
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;
    // 2. Sort degrees
    sort(degrees.begin(), degrees.end());
    int n1 = degrees.size();
    if (n1 == 0)
    {
        cerr << "Error: No degree data for Gini calculation." << endl;
        return;
    }

    // 3. Calculate Gini coefficient
    double sum = 0.0;
    for (int i = 0; i < n1; ++i)
    {
        sum += (2 * (i + 1) - n1 - 1) * degrees[i];
    }
    double total_degree = accumulate(degrees.begin(), degrees.end(), 0.0);
    double gini = (total_degree > 0) ? sum / (n1 * total_degree) : 0.0;
    cout << "gini" << gini << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }
    // Statistics of noisy degree distribution
    vector<int> noisy_degree_values;
    for (const auto &node : actualNodeIDs)
    {
        noisy_degree_values.push_back(noisy_degrees[node]);
    }
    sort(noisy_degree_values.begin(), noisy_degree_values.end());

    // Calculate key quantiles
    int median_degree = noisy_degree_values[noisy_degree_values.size() / 2];
    int q90_degree = noisy_degree_values[noisy_degree_values.size() * 9 / 10];
    // Define dynamic grouping rule
    auto get_target_size = [&](int degree) -> int
    {
        if (degree >= q90_degree)
            return 3; // High-degree nodes: small groups
        else if (degree >= median_degree)
            return 10; // Medium-degree nodes
        else
            return 20; // Low-degree nodes: large groups
    };

    // First round: Initial grouping (by noisy degree)
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // Second round: Dynamic group adjustment
    map<int, vector<int>> optimizedGroups;
    for (auto &group : degreeGroups)
    {
        int degree = group.first;
        auto &nodes = group.second;
        int target_size = get_target_size(degree);

        if (nodes.size() > target_size * 1.5)
        {
            // Split logic: sort by true degree and split evenly
            sort(nodes.begin(), nodes.end(),
                 [&](int a, int b)
                 { return data[a].size() < data[b].size(); });

            int split_pos = nodes.size() / 2;
            vector<int> lower(nodes.begin(), nodes.begin() + split_pos);
            vector<int> upper(nodes.begin() + split_pos, nodes.end());

            int rep1 = data[lower[lower.size() / 2]].size(); // Use median of true degrees
            int rep2 = data[upper[upper.size() / 2]].size();
            optimizedGroups[rep1] = lower;
            optimizedGroups[rep2] = upper;
        }
        else if (nodes.size() < target_size * 0.7)
        {
            // Merge logic: merge into the group with closest degree
            auto closest_it = optimizedGroups.begin();
            int min_diff = abs(closest_it->first - degree);
            for (auto it = optimizedGroups.begin(); it != optimizedGroups.end(); ++it)
            {
                int diff = abs(it->first - degree);
                if (diff < min_diff)
                    min_diff = diff;
            }
            if (min_diff < median_degree / 2)
            { // Only merge when degrees are close
                closest_it->second.insert(closest_it->second.end(), nodes.begin(), nodes.end());
            }
            else
            {
                optimizedGroups[degree] = nodes;
            }
        }
        else
        {
            optimizedGroups[degree] = nodes;
        }
    }
    degreeGroups = optimizedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;

    // padding and dropping

    double triangle_star = tri_count / star; // Number of triangles affected by each 2-star
    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {

            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(make_pair(NodeNum, NodeNum));
            }
        }
    }

    // Perturbation process
    map<int, vector<pair<int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_pair : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 2; t++)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                int *target = (t == 0) ? &edge_pair.first : &edge_pair.second;
                if (*target == -1)
                    continue; // Skip invalid edge
                if (*target == NodeNum)
                {
                    int x;
                    do
                    {

                        std::random_device rd;
                        unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen(seed);
                        std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                        x = dis_x(gen);

                    } while (x == node_i);
                }

                auto it = node_degrees.find(*target);
                if (it != node_degrees.end())
                {
                    double degree = it->second;
                    auto degreeGroup = degreeGroups.find(degree);
                    if (degreeGroup != degreeGroups.end())
                    {
                        double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                        if (dist(gen) > q)
                        {
                            std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                            int r;
                            do
                            {
                                r = degreeGroup->second[dis_idx(gen)];
                            } while (r == node_i);
                            *target = r;
                        }
                    }
                }
            }
            myDeg[node_i].insert(edge_pair.first);
            if (edge_pair.second != -1)
            {
                myDeg[node_i].insert(edge_pair.second);
            }
        }
    }

    // Calculate perturbed triangle count
    tri_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        // cout<<node_i<<endl;
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 2)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                int a = neighborList[j];
                int b = neighborList[k];
                if (myDeg[a].count(b))
                {
                    tri_num += 1;
                }
            }
        }
    }

    // double theoretical_var = tri_count * pow((exp(Eps) + avg_group_size - 1), 3) / pow((exp(Eps) - 1), 6);
    //  theoretical_var = theoretical_var/9;

    // Calculate error
    double MSE = (tri_count - tri_num) * (tri_count - tri_num);

    cout << "MSE: " << MSE << endl;
    cout << "Perturbed triangle count: " << tri_num << endl;

    // Calculate relative error
    double relative_error = std::abs((tri_num - tri_count) / tri_count);

    // Pass relative error back to caller and output
    clique_num_ns = relative_error;
    std::cout << "Relative error: " << relative_error << std::endl;
}
void NewCalNLocTriedge2k4cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{
    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double clique_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;
   
    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
    // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }
    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    // Calculate original 4-clique count
    int clique_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];

        // First level neighbor iteration
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting

            // Second level neighbor iteration
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (!data[*it_j].count(*it_k))
                    continue; // Ensure j-k edge exists

                // Third level neighbor iteration
                for (auto it_l = next(it_k); it_l != neighbors_i.end(); ++it_l)
                {
                    if (*it_l <= node_i)
                        continue;

                    // Check if the other three edges exist
                    if (data[*it_j].count(*it_l) &&
                        data[*it_k].count(*it_l))
                    {
                        clique_count++;
                    }
                }
            }
        }
    }
    cout << "Original 4clique count: " << clique_count << endl;
    OriginalCliqueCount = clique_count;
    double star;

    // Build subDict (using actual node IDs)

    map<int, vector<pair<int, int>>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
            {
                subDict[node_i].push_back(make_pair(neighbors[0], NodeNum)); // Use -1 to indicate no second node
            }
            for (int m = l + 1; m < neighbors.size(); m++)
            {
                subDict[node_i].push_back(make_pair(neighbors[l], neighbors[m]));
                star++;
            }
        }
    }
    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;
    vector<int> degrees;
    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        degrees.push_back(data[node].size());
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);//
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;

    // 2. Sort degrees
    sort(degrees.begin(), degrees.end());
    int n1 = degrees.size();
    if (n1 == 0)
    {
        cerr << "Error: No degree data for Gini calculation." << endl;
        return;
    }

    // 3. Calculate Gini coefficient
    double sum = 0.0;
    for (int i = 0; i < n1; ++i)
    {
        sum += (2 * (i + 1) - n1 - 1) * degrees[i];
    }
    double total_degree = accumulate(degrees.begin(), degrees.end(), 0.0);
    double gini = (total_degree > 0) ? sum / (n1 * total_degree) : 0.0;
    cout << "gini" << gini << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }
    // Statistics of noisy degree distribution
    vector<int> noisy_degree_values;
    for (const auto &node : actualNodeIDs)
    {
        noisy_degree_values.push_back(noisy_degrees[node]);
    }
    sort(noisy_degree_values.begin(), noisy_degree_values.end());

    // Calculate key quantiles
    int median_degree = noisy_degree_values[noisy_degree_values.size() / 2];
    int q90_degree = noisy_degree_values[noisy_degree_values.size() * 9 / 10];
    // Define dynamic grouping rule
    auto get_target_size = [&](int degree) -> int
    {
        if (degree >= q90_degree)
            return 3; // High-degree nodes: small groups
        else if (degree >= median_degree)
            return 10; // Medium-degree nodes
        else
            return 20; // Low-degree nodes: large groups
    };

    // First round: Initial grouping (by noisy degree)
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // Second round: Dynamic group adjustment
    map<int, vector<int>> optimizedGroups;
    for (auto &group : degreeGroups)
    {
        int degree = group.first;
        auto &nodes = group.second;
        int target_size = get_target_size(degree);

        if (nodes.size() > target_size * 1.5)
        {
            // Split logic: sort by true degree and split evenly
            sort(nodes.begin(), nodes.end(),
                 [&](int a, int b)
                 { return data[a].size() < data[b].size(); });

            int split_pos = nodes.size() / 2;
            vector<int> lower(nodes.begin(), nodes.begin() + split_pos);
            vector<int> upper(nodes.begin() + split_pos, nodes.end());

            int rep1 = data[lower[lower.size() / 2]].size(); // Use median of true degrees
            int rep2 = data[upper[upper.size() / 2]].size();
            optimizedGroups[rep1] = lower;
            optimizedGroups[rep2] = upper;
        }
        else if (nodes.size() < target_size * 0.7)
        {
            // Merge logic: merge into the group with closest degree
            auto closest_it = optimizedGroups.begin();
            int min_diff = abs(closest_it->first - degree);
            for (auto it = optimizedGroups.begin(); it != optimizedGroups.end(); ++it)
            {
                int diff = abs(it->first - degree);
                if (diff < min_diff)
                    min_diff = diff;
            }
            if (min_diff < median_degree / 2)
            { // Only merge when degrees are close
                closest_it->second.insert(closest_it->second.end(), nodes.begin(), nodes.end());
            }
            else
            {
                optimizedGroups[degree] = nodes;
            }
        }
        else
        {
            optimizedGroups[degree] = nodes;
        }
    }
    degreeGroups = optimizedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;
    // padding and sampling

    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {

            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(make_pair(NodeNum, NodeNum));
            }
        }
    }

    // Perturbation process
    map<int, vector<pair<int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_pair : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 2; t++)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                int *target = (t == 0) ? &edge_pair.first : &edge_pair.second;
                if (*target == -1)
                    continue; // Skip invalid edge
                if (*target == NodeNum)
                {
                    int x;
                    do
                    {

                        std::random_device rd;
                        unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen(seed);
                        std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                        x = dis_x(gen);

                    } while (x == node_i);
                }

                auto it = node_degrees.find(*target);
                if (it != node_degrees.end())
                {
                    double degree = it->second;
                    auto degreeGroup = degreeGroups.find(degree);
                    if (degreeGroup != degreeGroups.end())
                    {
                        double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                        if (dist(gen) > q)
                        {
                            std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                            int r;
                            do
                            {
                                r = degreeGroup->second[dis_idx(gen)];
                            } while (r == node_i);
                            *target = r;
                        }
                    }
                }
            }
            myDeg[node_i].insert(edge_pair.first);
            if (edge_pair.second != -1)
            {
                myDeg[node_i].insert(edge_pair.second);
            }
        }
    }

    // Calculate perturbed clique count
    clique_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 3)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                for (int l = k + 1; l < neighborList.size(); l++)
                {
                    int a = neighborList[j];
                    int b = neighborList[k];
                    int c = neighborList[l];

                    if (myDeg[a].count(b) && myDeg[a].count(c) && myDeg[b].count(c))
                    {
                        clique_num += 1;
                    }
                }
            }
        }
    }

    cout << "Perturbed triangle count: " << clique_num << endl;

    // Calculate relative error
    double relative_error = std::abs((clique_num - clique_count) / clique_count);

    // Pass relative error back to caller and output
    clique_num_ns = relative_error;
    std::cout << "Relative error: " << relative_error << std::endl;
}
void NewCalNLocTriedge1k3cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{
    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double tri_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;
    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;

    Eps = Eps * 2;
    cout << "Eps" << Eps << endl;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
    // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    // Calculate original triangle count
    int tri_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (data[*it_j].count(*it_k))
                {
                    tri_count++;
                }
            }
        }
    }
    cout << "Original triangle count: " << tri_count << endl;
    double star;

    // Build subDict (using actual node IDs)
    map<int, vector<int>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            subDict[node_i].push_back(neighbors[l]);
        }
    }

    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;

    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees

    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // const int MIN_GROUP_SIZE = 10;
    // map<int, vector<int>> mergedGroups;
    // // Merge small degree groups
    // for (auto it = degreeGroups.begin(); it != degreeGroups.end(); ++it)
    // {
    //     int degree = it->first;
    //     const auto &nodes = it->second;
    //     // Merge to nearest multiple of 10
    //     int target_degree = (nodes.size() < MIN_GROUP_SIZE) ? (degree / 10) * 10 : degree;
    //     mergedGroups[target_degree].insert(mergedGroups[target_degree].end(), nodes.begin(), nodes.end());
    // }
    // degreeGroups = mergedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;

    // padding and sampling
    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        // cout<<"i"<<node_i<<endl;
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {
            // cout << "111" << endl;
            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            // cout << "222" << endl;
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(NodeNum);
            }
        }
    }

    // Perturbation process
    map<int, vector<int>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;
    cout << "raodong" << endl;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge : subDictPerturbed[node_i])
        {
            if (edge == NodeNum)
            {
                int x;
                do
                {

                    std::random_device rd;
                    unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                    std::mt19937 gen(seed);
                    std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                    x = dis_x(gen);

                } while (x == node_i);
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            auto it = node_degrees.find(edge);

            if (it != node_degrees.end())
            {
                double degree = it->second;
                auto degreeGroup = degreeGroups.find(degree);
                if (degreeGroup != degreeGroups.end())
                {
                    double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                    if (dist(gen) > q)
                    {
                        std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                        int r;
                        do
                        {
                            r = degreeGroup->second[dis_idx(gen)];

                        } while (r == node_i);
                        edge = r;
                    }
                }
            }

            myDeg[node_i].insert(edge);
        }
    }

    // Calculate perturbed triangle count
    tri_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 2)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                int a = neighborList[j];
                int b = neighborList[k];
                if (myDeg[a].count(b))
                {
                    tri_num += 1;
                }
            }
        }
    }

    double theoretical_var = tri_count * pow((exp(Eps) + avg_group_size - 1), 3) / pow((exp(Eps) - 1), 6);
    // theoretical_var = theoretical_var/9;

    // Calculate error
    double MSE = (tri_count - tri_num) * (tri_count - tri_num);

    cout << "MSE: " << MSE << endl;
    cout << "theoretical_var: " << theoretical_var << endl;

    cout << "Perturbed triangle count: " << tri_num << endl;

    // Calculate relative error
    double relative_error = std::abs((tri_num - tri_count) / tri_count);

    // Output relative error
    std::cout << "Relative error: " << relative_error << std::endl;
}
void NewCalNLocTriedge1k4cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{
    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double clique_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;
    
    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;
    Eps = Eps * 2;
    cout << "Eps" << Eps << endl;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
    // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }
    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    // Calculate original 4-clique count
    int clique_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];

        // First level neighbor iteration
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting

            // Second level neighbor iteration
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (!data[*it_j].count(*it_k))
                    continue; // Ensure j-k edge exists

                // Third level neighbor iteration
                for (auto it_l = next(it_k); it_l != neighbors_i.end(); ++it_l)
                {
                    if (*it_l <= node_i)
                        continue;

                    // Check if the other three edges exist
                    if (data[*it_j].count(*it_l) &&
                        data[*it_k].count(*it_l))
                    {
                        clique_count++;
                    }
                }
            }
        }
    }
    cout << "Original 4clique count: " << clique_count << endl;
    double star;

    // Build subDict (using actual node IDs)
    map<int, vector<int>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            subDict[node_i].push_back(neighbors[l]);
        }
    }

    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;
    vector<int> degrees;
    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        degrees.push_back(data[node].size());
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);//
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;

    // 2. Sort degrees
    sort(degrees.begin(), degrees.end());
    int n1 = degrees.size();
    if (n1 == 0)
    {
        cerr << "Error: No degree data for Gini calculation." << endl;
        return;
    }

    // 3. Calculate Gini coefficient
    double sum = 0.0;
    for (int i = 0; i < n1; ++i)
    {
        sum += (2 * (i + 1) - n1 - 1) * degrees[i];
    }
    double total_degree = accumulate(degrees.begin(), degrees.end(), 0.0);
    double gini = (total_degree > 0) ? sum / (n1 * total_degree) : 0.0;
    cout << "gini" << gini << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }
    // Statistics of noisy degree distribution
    vector<int> noisy_degree_values;
    for (const auto &node : actualNodeIDs)
    {
        noisy_degree_values.push_back(noisy_degrees[node]);
    }
    sort(noisy_degree_values.begin(), noisy_degree_values.end());

    // Calculate key quantiles
    int median_degree = noisy_degree_values[noisy_degree_values.size() / 2];
    int q90_degree = noisy_degree_values[noisy_degree_values.size() * 9 / 10];
    // Define dynamic grouping rule
    auto get_target_size = [&](int degree) -> int
    {
        if (degree >= q90_degree)
            return 3; // High-degree nodes: small groups
        else if (degree >= median_degree)
            return 10; // Medium-degree nodes
        else
            return 20; // Low-degree nodes: large groups
    };

    // First round: Initial grouping (by noisy degree)
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // Second round: Dynamic group adjustment
    map<int, vector<int>> optimizedGroups;
    for (auto &group : degreeGroups)
    {
        int degree = group.first;
        auto &nodes = group.second;
        int target_size = get_target_size(degree);

        if (nodes.size() > target_size * 1.5)
        {
            // Split logic: sort by true degree and split evenly
            sort(nodes.begin(), nodes.end(),
                 [&](int a, int b)
                 { return data[a].size() < data[b].size(); });

            int split_pos = nodes.size() / 2;
            vector<int> lower(nodes.begin(), nodes.begin() + split_pos);
            vector<int> upper(nodes.begin() + split_pos, nodes.end());

            int rep1 = data[lower[lower.size() / 2]].size(); // Use median of true degrees
            int rep2 = data[upper[upper.size() / 2]].size();
            optimizedGroups[rep1] = lower;
            optimizedGroups[rep2] = upper;
        }
        else if (nodes.size() < target_size * 0.7)
        {
            // Merge logic: merge into the group with closest degree
            auto closest_it = optimizedGroups.begin();
            int min_diff = abs(closest_it->first - degree);
            for (auto it = optimizedGroups.begin(); it != optimizedGroups.end(); ++it)
            {
                int diff = abs(it->first - degree);
                if (diff < min_diff)
                    min_diff = diff;
            }
            if (min_diff < median_degree / 2)
            { // Only merge when degrees are close
                closest_it->second.insert(closest_it->second.end(), nodes.begin(), nodes.end());
            }
            else
            {
                optimizedGroups[degree] = nodes;
            }
        }
        else
        {
            optimizedGroups[degree] = nodes;
        }
    }
    degreeGroups = optimizedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;

    // padding and sampling
    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        // cout<<"i"<<node_i<<endl;
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {
            // cout << "111" << endl;
            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            // cout << "222" << endl;
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(NodeNum);
            }
        }
    }

    // Perturbation process
    map<int, vector<int>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;
    cout << "raodong" << endl;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge : subDictPerturbed[node_i])
        {
            if (edge == NodeNum)
            {
                int x;
                do
                {

                    std::random_device rd;
                    unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                    std::mt19937 gen(seed);
                    std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                    x = dis_x(gen);

                } while (x == node_i);
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            auto it = node_degrees.find(edge);
            if (it != node_degrees.end())
            {
                double degree = it->second;
                auto degreeGroup = degreeGroups.find(degree);
                if (degreeGroup != degreeGroups.end())
                {
                    double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                    if (dist(gen) > q)
                    {
                        std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                        int r;
                        do
                        {
                            r = degreeGroup->second[dis_idx(gen)];
                        } while (r == node_i);
                        edge = r;
                    }
                }
            }

            myDeg[node_i].insert(edge);
        }
    }

    // Calculate perturbed clique count
    clique_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 3)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                for (int l = k + 1; l < neighborList.size(); l++)
                {
                    int a = neighborList[j];
                    int b = neighborList[k];
                    int c = neighborList[l];

                    if (myDeg[a].count(b) && myDeg[a].count(c) && myDeg[b].count(c))
                    {
                        clique_num += 1;
                    }
                }
            }
        }
    }

    cout << "Perturbed triangle count: " << clique_num << endl;

    // Calculate relative error
    double relative_error = std::abs((clique_num - clique_count) / clique_count);

    // Pass relative error back to caller and output
    clique_num_ns = relative_error;
    std::cout << "Relative error: " << relative_error << std::endl;
}
void NewCalNLocTriedge3k3cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{
    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double tri_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;
    
    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;
    Eps = Eps * 2 / 3;
    cout << "Eps" << Eps << endl;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
       
     // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    // Calculate original triangle count
    int tri_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (data[*it_j].count(*it_k))
                {
                    tri_count++;
                }
            }
        }
    }
    cout << "Original triangle count: " << tri_count << endl;
    OriginalCliqueCount = tri_count;
    double star;

    // Build subDict (using actual node IDs)

    map<int, vector<tuple<int, int, int>>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
            {
                subDict[node_i].push_back(make_tuple(neighbors[0], NodeNum, NodeNum)); // Use -1 to indicate no second node
            }
            for (int m = l + 1; m < neighbors.size(); m++)
            {
                if (neighbors.size() == 2)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[0], neighbors[1], NodeNum)); // Use -1 to indicate no second node
                }
                for (int n = m + 1; n < neighbors.size(); n++)
                {
                    // Create tuple and store in map
                    subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], neighbors[n]));
                    star++;
                }
            }
        }
    }

    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;

    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees

    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // const int MIN_GROUP_SIZE = 10;
    // map<int, vector<int>> mergedGroups;
    // // Merge small degree groups
    // for (auto it = degreeGroups.begin(); it != degreeGroups.end(); ++it)
    // {
    //     int degree = it->first;
    //     const auto &nodes = it->second;
    //     // Merge to nearest multiple of 10
    //     int target_degree = (nodes.size() < MIN_GROUP_SIZE) ? (degree / 10) * 10 : degree;
    //     mergedGroups[target_degree].insert(mergedGroups[target_degree].end(), nodes.begin(), nodes.end());
    // }
    // degreeGroups = mergedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;
    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {

            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(make_tuple(NodeNum, NodeNum, NodeNum));
            }
        }
    }

    // cout << "dropping_error" << dropping_error << endl;
    // cout << "padding_error" << padding_error << endl;

    // Perturbation process
    map<int, vector<tuple<int, int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_tuple : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 3; t++)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                // Modified line - use switch statement to handle triples
                int *target = nullptr;
                switch (t)
                {
                case 0:
                    target = &get<0>(edge_tuple);
                    break;
                case 1:
                    target = &get<1>(edge_tuple);
                    break;
                case 2:
                    target = &get<2>(edge_tuple);
                    break;
                }

                if (*target == -1)
                    continue; // Skip invalid edge

                if (*target == NodeNum)
                {
                    int x;
                    do
                    {

                        std::random_device rd;
                        unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen(seed);
                        std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                        x = dis_x(gen);

                    } while (x == node_i);
                }

                auto it = node_degrees.find(*target);
                if (it != node_degrees.end())
                {
                    double degree = it->second;
                    auto degreeGroup = degreeGroups.find(degree);
                    if (degreeGroup != degreeGroups.end())
                    {
                        double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                        if (dist(gen) > q)
                        {
                            std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                            int r;
                            do
                            {
                                r = degreeGroup->second[dis_idx(gen)];
                            } while (r == node_i);
                            *target = r;
                        }
                    }
                }
            }

            // Insert three elements into myDeg
            myDeg[node_i].insert(get<0>(edge_tuple));
            if (get<1>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<1>(edge_tuple));
            }
            if (get<2>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<2>(edge_tuple));
            }
        }
    }
    // Calculate perturbed triangle count
    tri_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 2)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                int a = neighborList[j];
                int b = neighborList[k];
                if (myDeg[a].count(b))
                {
                    tri_num += 1;
                }
            }
        }
    }

    double theoretical_var = tri_count * pow((exp(Eps) + avg_group_size - 1), 3) / pow((exp(Eps) - 1), 6);
    // theoretical_var = theoretical_var/9;

    // Calculate error
    double MSE = (tri_count - tri_num) * (tri_count - tri_num);

    cout << "MSE: " << MSE << endl;
    cout << "theoretical_var: " << theoretical_var << endl;

    cout << "Perturbed triangle count: " << tri_num << endl;

    // Calculate relative error
    double relative_error = std::abs((tri_num - tri_count) / tri_count);

    // Pass relative error back to caller and output
    clique_num_ns = relative_error;
    std::cout << "Relative error: " << relative_error << std::endl;
}
void NewCalNLocTriedge4k3cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{
    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double tri_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;
  
    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;
    Eps = Eps * 2 / 4;
    cout << "Eps" << Eps << endl;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
     // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    // Calculate original triangle count
    int tri_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (data[*it_j].count(*it_k))
                {
                    tri_count++;
                }
            }
        }
    }
    cout << "Original triangle count: " << tri_count << endl;
    double star;

    // Build subDict (using actual node IDs)

    map<int, vector<tuple<int, int, int, int>>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
            {
                subDict[node_i].push_back(make_tuple(neighbors[0], NodeNum, NodeNum, NodeNum)); // Use -1 to indicate no second node
            }
            for (int m = l + 1; m < neighbors.size(); m++)
            {
                if (neighbors.size() == 2)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[0], neighbors[1], NodeNum, NodeNum)); // Use -1 to indicate no second node
                }
                for (int n = m + 1; n < neighbors.size(); n++)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[0], neighbors[1], neighbors[2], NodeNum));
                    for (int o = n + 1; o < neighbors.size(); o++)
                    {
                        subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], neighbors[n], neighbors[o]));
                        star++;
                    }
                    // Create tuple and store in map
                }
            }
        }
    }

    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;
    vector<int> degrees;
    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        degrees.push_back(data[node].size());
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);//
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;

    // 2. Sort degrees
    sort(degrees.begin(), degrees.end());
    int n1 = degrees.size();
    if (n1 == 0)
    {
        cerr << "Error: No degree data for Gini calculation." << endl;
        return;
    }

    // 3. Calculate Gini coefficient
    double sum = 0.0;
    for (int i = 0; i < n1; ++i)
    {
        sum += (2 * (i + 1) - n1 - 1) * degrees[i];
    }
    double total_degree = accumulate(degrees.begin(), degrees.end(), 0.0);
    double gini = (total_degree > 0) ? sum / (n1 * total_degree) : 0.0;
    cout << "gini" << gini << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }
    // Statistics of noisy degree distribution
    vector<int> noisy_degree_values;
    for (const auto &node : actualNodeIDs)
    {
        noisy_degree_values.push_back(noisy_degrees[node]);
    }
    sort(noisy_degree_values.begin(), noisy_degree_values.end());

    // Calculate key quantiles
    int median_degree = noisy_degree_values[noisy_degree_values.size() / 2];
    int q90_degree = noisy_degree_values[noisy_degree_values.size() * 9 / 10];
    // Define dynamic grouping rule
    auto get_target_size = [&](int degree) -> int
    {
        if (degree >= q90_degree)
            return 3; // High-degree nodes: small groups
        else if (degree >= median_degree)
            return 10; // Medium-degree nodes
        else
            return 20; // Low-degree nodes: large groups
    };

    // First round: Initial grouping (by noisy degree)
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // Second round: Dynamic group adjustment
    map<int, vector<int>> optimizedGroups;
    for (auto &group : degreeGroups)
    {
        int degree = group.first;
        auto &nodes = group.second;
        int target_size = get_target_size(degree);

        if (nodes.size() > target_size * 1.5)
        {
            // Split logic: sort by true degree and split evenly
            sort(nodes.begin(), nodes.end(),
                 [&](int a, int b)
                 { return data[a].size() < data[b].size(); });

            int split_pos = nodes.size() / 2;
            vector<int> lower(nodes.begin(), nodes.begin() + split_pos);
            vector<int> upper(nodes.begin() + split_pos, nodes.end());

            int rep1 = data[lower[lower.size() / 2]].size(); // Use median of true degrees
            int rep2 = data[upper[upper.size() / 2]].size();
            optimizedGroups[rep1] = lower;
            optimizedGroups[rep2] = upper;
        }
        else if (nodes.size() < target_size * 0.7)
        {
            // Merge logic: merge into the group with closest degree
            auto closest_it = optimizedGroups.begin();
            int min_diff = abs(closest_it->first - degree);
            for (auto it = optimizedGroups.begin(); it != optimizedGroups.end(); ++it)
            {
                int diff = abs(it->first - degree);
                if (diff < min_diff)
                    min_diff = diff;
            }
            if (min_diff < median_degree / 2)
            { // Only merge when degrees are close
                closest_it->second.insert(closest_it->second.end(), nodes.begin(), nodes.end());
            }
            else
            {
                optimizedGroups[degree] = nodes;
            }
        }
        else
        {
            optimizedGroups[degree] = nodes;
        }
    }
    degreeGroups = optimizedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;

    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {

            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(make_tuple(NodeNum, NodeNum, NodeNum, NodeNum));
            }
        }
    }

    // cout << "dropping_error" << dropping_error << endl;
    // cout << "padding_error" << padding_error << endl;

    // Perturbation process
    map<int, vector<tuple<int, int, int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_tuple : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 4; t++)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                // Modified line - use switch statement to handle triples
                int *target = nullptr;
                switch (t)
                {
                case 0:
                    target = &get<0>(edge_tuple);
                    break;
                case 1:
                    target = &get<1>(edge_tuple);
                    break;
                case 2:
                    target = &get<2>(edge_tuple);
                    break;
                case 3:
                    target = &get<3>(edge_tuple);
                    break;
                }

                if (*target == -1)
                    continue; // Skip invalid edge

                if (*target == NodeNum)
                {
                    int x;
                    do
                    {

                        std::random_device rd;
                        unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen(seed);
                        std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                        x = dis_x(gen);

                    } while (x == node_i);
                }

                auto it = node_degrees.find(*target);
                if (it != node_degrees.end())
                {
                    double degree = it->second;
                    auto degreeGroup = degreeGroups.find(degree);
                    if (degreeGroup != degreeGroups.end())
                    {
                        double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                        if (dist(gen) > q)
                        {
                            std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                            int r;
                            do
                            {
                                r = degreeGroup->second[dis_idx(gen)];
                            } while (r == node_i);
                            *target = r;
                        }
                    }
                }
            }

            // Insert four elements into myDeg
            myDeg[node_i].insert(get<0>(edge_tuple));
            if (get<1>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<1>(edge_tuple));
            }
            if (get<2>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<2>(edge_tuple));
            }
            if (get<3>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<3>(edge_tuple));
            }
        }
    }
    // Calculate perturbed triangle count
    tri_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 2)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                int a = neighborList[j];
                int b = neighborList[k];
                if (myDeg[a].count(b))
                {
                    tri_num += 1;
                }
            }
        }
    }

    double theoretical_var = tri_count * pow((exp(Eps) + avg_group_size - 1), 3) / pow((exp(Eps) - 1), 6);
    // theoretical_var = theoretical_var/9;

    // Calculate error
    double MSE = (tri_count - tri_num) * (tri_count - tri_num);

    cout << "MSE: " << MSE << endl;
    cout << "theoretical_var: " << theoretical_var << endl;

    cout << "Perturbed triangle count: " << tri_num << endl;

    // Calculate relative error
    double relative_error = std::abs((tri_num - tri_count) / tri_count);

    // Output relative error
    std::cout << "Relative error: " << relative_error << std::endl;
}
void NewCalNLocTriedge3k4cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{
    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double clique_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;
    
    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;
    Eps = Eps * 2 / 3;
    cout << "Eps" << Eps << endl;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
    // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    /// Calculate original 4-clique count
    int clique_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];

        // First level neighbor iteration
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting

            // Second level neighbor iteration
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (!data[*it_j].count(*it_k))
                    continue; // Ensure j-k edge exists

                // Third level neighbor iteration
                for (auto it_l = next(it_k); it_l != neighbors_i.end(); ++it_l)
                {
                    if (*it_l <= node_i)
                        continue;

                    // Check if the other three edges exist
                    if (data[*it_j].count(*it_l) &&
                        data[*it_k].count(*it_l))
                    {
                        clique_count++;
                    }
                }
            }
        }
    }
    cout << "Original 4clique count: " << clique_count << endl;
    double star;

    // Build subDict (using actual node IDs)

    map<int, vector<tuple<int, int, int>>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
            {
                subDict[node_i].push_back(make_tuple(neighbors[0], NodeNum, NodeNum)); // Use -1 to indicate no second node
            }
            for (int m = l + 1; m < neighbors.size(); m++)
            {
                if (neighbors.size() == 2)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[0], neighbors[1], NodeNum)); // Use -1 to indicate no second node
                }
                for (int n = m + 1; n < neighbors.size(); n++)
                {
                    // Create tuple and store in map
                    subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], neighbors[n]));
                    star++;
                }
            }
        }
    }

    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;
    vector<int> degrees;
    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        degrees.push_back(data[node].size());
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);//
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;

    // 2. Sort degrees
    sort(degrees.begin(), degrees.end());
    int n1 = degrees.size();
    if (n1 == 0)
    {
        cerr << "Error: No degree data for Gini calculation." << endl;
        return;
    }

    // 3. Calculate Gini coefficient
    double sum = 0.0;
    for (int i = 0; i < n1; ++i)
    {
        sum += (2 * (i + 1) - n1 - 1) * degrees[i];
    }
    double total_degree = accumulate(degrees.begin(), degrees.end(), 0.0);
    double gini = (total_degree > 0) ? sum / (n1 * total_degree) : 0.0;
    cout << "gini" << gini << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }
    // Statistics of noisy degree distribution
    vector<int> noisy_degree_values;
    for (const auto &node : actualNodeIDs)
    {
        noisy_degree_values.push_back(noisy_degrees[node]);
    }
    sort(noisy_degree_values.begin(), noisy_degree_values.end());

    // Calculate key quantiles
    int median_degree = noisy_degree_values[noisy_degree_values.size() / 2];
    int q90_degree = noisy_degree_values[noisy_degree_values.size() * 9 / 10];
    // Define dynamic grouping rule
    auto get_target_size = [&](int degree) -> int
    {
        if (degree >= q90_degree)
            return 3; // High-degree nodes: small groups
        else if (degree >= median_degree)
            return 10; // Medium-degree nodes
        else
            return 20; // Low-degree nodes: large groups
    };

    // First round: Initial grouping (by noisy degree)
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // Second round: Dynamic group adjustment
    map<int, vector<int>> optimizedGroups;
    for (auto &group : degreeGroups)
    {
        int degree = group.first;
        auto &nodes = group.second;
        int target_size = get_target_size(degree);

        if (nodes.size() > target_size * 1.5)
        {
            // Split logic: sort by true degree and split evenly
            sort(nodes.begin(), nodes.end(),
                 [&](int a, int b)
                 { return data[a].size() < data[b].size(); });

            int split_pos = nodes.size() / 2;
            vector<int> lower(nodes.begin(), nodes.begin() + split_pos);
            vector<int> upper(nodes.begin() + split_pos, nodes.end());

            int rep1 = data[lower[lower.size() / 2]].size(); // Use median of true degrees
            int rep2 = data[upper[upper.size() / 2]].size();
            optimizedGroups[rep1] = lower;
            optimizedGroups[rep2] = upper;
        }
        else if (nodes.size() < target_size * 0.7)
        {
            // Merge logic: merge into the group with closest degree
            auto closest_it = optimizedGroups.begin();
            int min_diff = abs(closest_it->first - degree);
            for (auto it = optimizedGroups.begin(); it != optimizedGroups.end(); ++it)
            {
                int diff = abs(it->first - degree);
                if (diff < min_diff)
                    min_diff = diff;
            }
            if (min_diff < median_degree / 2)
            { // Only merge when degrees are close
                closest_it->second.insert(closest_it->second.end(), nodes.begin(), nodes.end());
            }
            else
            {
                optimizedGroups[degree] = nodes;
            }
        }
        else
        {
            optimizedGroups[degree] = nodes;
        }
    }
    degreeGroups = optimizedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;

    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {

            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(make_tuple(NodeNum, NodeNum, NodeNum));
            }
        }
    }

    // Perturbation process
    map<int, vector<tuple<int, int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_tuple : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 3; t++)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                // Modified line - use switch statement to handle triples
                int *target = nullptr;
                switch (t)
                {
                case 0:
                    target = &get<0>(edge_tuple);
                    break;
                case 1:
                    target = &get<1>(edge_tuple);
                    break;
                case 2:
                    target = &get<2>(edge_tuple);
                    break;
                }

                if (*target == -1)
                    continue; // Skip invalid edge

                if (*target == NodeNum)
                {
                    int x;
                    do
                    {

                        std::random_device rd;
                        unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen(seed);
                        std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                        x = dis_x(gen);

                    } while (x == node_i);
                }

                auto it = node_degrees.find(*target);
                if (it != node_degrees.end())
                {
                    double degree = it->second;
                    auto degreeGroup = degreeGroups.find(degree);
                    if (degreeGroup != degreeGroups.end())
                    {
                        double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                        if (dist(gen) > q)
                        {
                            std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                            int r;
                            do
                            {
                                r = degreeGroup->second[dis_idx(gen)];
                            } while (r == node_i);
                            *target = r;
                        }
                    }
                }
            }

            // Insert three elements into myDeg
            myDeg[node_i].insert(get<0>(edge_tuple));
            if (get<1>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<1>(edge_tuple));
            }
            if (get<2>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<2>(edge_tuple));
            }
        }
    }
    // Calculate perturbed clique count
    clique_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 3)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                for (int l = k + 1; l < neighborList.size(); l++)
                {
                    int a = neighborList[j];
                    int b = neighborList[k];
                    int c = neighborList[l];

                    if (myDeg[a].count(b) && myDeg[a].count(c) && myDeg[b].count(c))
                    {
                        clique_num += 1;
                    }
                }
            }
        }
    }

    cout << "Perturbed 4clique count: " << clique_num << endl;
    // Calculate relative error
    double relative_error = std::abs((clique_num - clique_count) / clique_count);

    // Pass relative error back to caller and output
    clique_num_ns = relative_error;
    std::cout << "Relative error: " << relative_error << std::endl;
}
void NewCalNLocTriedge4k4cliquenoise(map<int, int> *a_mat, int *deg, string outfile, double &clique_num_ns, int emp, double custom_pad = -1)
{
    map<int, int> *a_mat_ns;
    map<int, int>::iterator aitr;
    map<int, int>::iterator aitr2;
    int *deg_ns;
    long long tot_edge_num_ns;
    long long st2_num, ed2_num, ed1_num, non_num;
    double clique_num;
    double alp, alp_1_3, q_inv_11, q_inv_21, q_inv_31, q_inv_41;
    int i, j, k;
    int l, m, n;
    int left, right;
   
    // Modify pad selection logic
    double pad;
    if (custom_pad > 0)
    {
        pad = custom_pad; // Use the optimal pad value passed from outside
    }
    else
    {
        pad = 20; // Default value
    }
    Eps = Eps / pad;
    Eps = Eps * 2 / 4;
    cout << "Eps" << Eps << endl;

    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
     // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    // No longer use NodeNum, switch to actual node count
    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Build adjacency list (using actual node IDs)
    std::unordered_map<int, std::vector<int>> adj_list;
    for (const auto &pair : data)
    {
        adj_list[pair.first] = std::vector<int>(pair.second.begin(), pair.second.end());
    }

    /// Calculate original 4-clique count
    int clique_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];

        // First level neighbor iteration
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue; // Avoid duplicate counting

            // Second level neighbor iteration
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (!data[*it_j].count(*it_k))
                    continue; // Ensure j-k edge exists

                // Third level neighbor iteration
                for (auto it_l = next(it_k); it_l != neighbors_i.end(); ++it_l)
                {
                    if (*it_l <= node_i)
                        continue;

                    // Check if the other three edges exist
                    if (data[*it_j].count(*it_l) &&
                        data[*it_k].count(*it_l))
                    {
                        clique_count++;
                    }
                }
            }
        }
    }
    cout << "Original 4clique count: " << clique_count << endl;
    double star;

    // Build subDict (using actual node IDs)

    map<int, vector<tuple<int, int, int, int>>> subDict;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        for (int l = 0; l < neighbors.size(); l++)
        {
            if (neighbors.size() == 1)
            {
                subDict[node_i].push_back(make_tuple(neighbors[0], NodeNum, NodeNum, NodeNum)); // Use -1 to indicate no second node
            }
            for (int m = l + 1; m < neighbors.size(); m++)
            {
                if (neighbors.size() == 2)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[0], neighbors[1], NodeNum, NodeNum)); // Use -1 to indicate no second node
                }
                for (int n = m + 1; n < neighbors.size(); n++)
                {
                    subDict[node_i].push_back(make_tuple(neighbors[0], neighbors[1], neighbors[2], NodeNum));
                    for (int o = n + 1; o < neighbors.size(); o++)
                    {
                        subDict[node_i].push_back(make_tuple(neighbors[l], neighbors[m], neighbors[n], neighbors[o]));
                        star++;
                    }
                    // Create tuple and store in map
                }
            }
        }
    }

    std::unordered_map<int, double> node_degrees;
    map<int, vector<int>> degreeGroups;
    int maxDegree = 0;
    vector<int> degrees;
    // Group by degree
    for (const auto &node : actualNodeIDs)
    {
        double degree = data[node].size();
        degrees.push_back(data[node].size());
        node_degrees[node] = degree;
        // degreeGroups[degree].push_back(node);//
        if (degree > maxDegree)
        {
            maxDegree = degree;
        }
    }
    cout << "maxDegree" << maxDegree << endl;

    // 2. Sort degrees
    sort(degrees.begin(), degrees.end());
    int n1 = degrees.size();
    if (n1 == 0)
    {
        cerr << "Error: No degree data for Gini calculation." << endl;
        return;
    }

    // 3. Calculate Gini coefficient
    double sum = 0.0;
    for (int i = 0; i < n1; ++i)
    {
        sum += (2 * (i + 1) - n1 - 1) * degrees[i];
    }
    double total_degree = accumulate(degrees.begin(), degrees.end(), 0.0);
    double gini = (total_degree > 0) ? sum / (n1 * total_degree) : 0.0;
    cout << "gini" << gini << endl;

    /// Add Laplace noise to degrees
    std::random_device rd;
    std::mt19937 gen(rd());
    double laplace_scale = 1.0 / Eps; // Scale parameter for Laplace noise

    // Custom Laplace distribution random number generation
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
        double noisy_degree = node_degrees[node] + noise;
        // Ensure degree is within reasonable range
        noisy_degree = round(noisy_degree);                            // Round to nearest integer
        noisy_degree = max(1, static_cast<int>(noisy_degree));         // Not less than 1
        noisy_degree = min(maxDegree, static_cast<int>(noisy_degree)); // Not greater than max degree
        noisy_degrees[node] = static_cast<int>(noisy_degree);
    }

    // Group by noisy degrees
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }
    // Statistics of noisy degree distribution
    vector<int> noisy_degree_values;
    for (const auto &node : actualNodeIDs)
    {
        noisy_degree_values.push_back(noisy_degrees[node]);
    }
    sort(noisy_degree_values.begin(), noisy_degree_values.end());

    // Calculate key quantiles
    int median_degree = noisy_degree_values[noisy_degree_values.size() / 2];
    int q90_degree = noisy_degree_values[noisy_degree_values.size() * 9 / 10];
    // Define dynamic grouping rule
    auto get_target_size = [&](int degree) -> int
    {
        if (degree >= q90_degree)
            return 3; // High-degree nodes: small groups
        else if (degree >= median_degree)
            return 10; // Medium-degree nodes
        else
            return 20; // Low-degree nodes: large groups
    };

    // First round: Initial grouping (by noisy degree)
    for (const auto &node : actualNodeIDs)
    {
        int degree = noisy_degrees[node];
        degreeGroups[degree].push_back(node);
    }

    // Second round: Dynamic group adjustment
    map<int, vector<int>> optimizedGroups;
    for (auto &group : degreeGroups)
    {
        int degree = group.first;
        auto &nodes = group.second;
        int target_size = get_target_size(degree);

        if (nodes.size() > target_size * 1.5)
        {
            // Split logic: sort by true degree and split evenly
            sort(nodes.begin(), nodes.end(),
                 [&](int a, int b)
                 { return data[a].size() < data[b].size(); });

            int split_pos = nodes.size() / 2;
            vector<int> lower(nodes.begin(), nodes.begin() + split_pos);
            vector<int> upper(nodes.begin() + split_pos, nodes.end());

            int rep1 = data[lower[lower.size() / 2]].size(); // Use median of true degrees
            int rep2 = data[upper[upper.size() / 2]].size();
            optimizedGroups[rep1] = lower;
            optimizedGroups[rep2] = upper;
        }
        else if (nodes.size() < target_size * 0.7)
        {
            // Merge logic: merge into the group with closest degree
            auto closest_it = optimizedGroups.begin();
            int min_diff = abs(closest_it->first - degree);
            for (auto it = optimizedGroups.begin(); it != optimizedGroups.end(); ++it)
            {
                int diff = abs(it->first - degree);
                if (diff < min_diff)
                    min_diff = diff;
            }
            if (min_diff < median_degree / 2)
            { // Only merge when degrees are close
                closest_it->second.insert(closest_it->second.end(), nodes.begin(), nodes.end());
            }
            else
            {
                optimizedGroups[degree] = nodes;
            }
        }
        else
        {
            optimizedGroups[degree] = nodes;
        }
    }
    degreeGroups = optimizedGroups;

    double avg_group_size = 0.0;
    double tot_degree = 0.0;
    for (const auto &group : degreeGroups)
    {
        tot_degree += group.second.size();
        avg_group_size += group.second.size();
    }
    avg_group_size /= degreeGroups.size();
    cout << "avg_group_size: " << avg_group_size << endl;

    cout << "origin" << endl;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = adj_list[node_i];
        if (neighbors.size() > pad)
        {

            auto it = subDict[node_i].begin();
            std::advance(it, pad);                            // Move to the xth element
            subDict[node_i].erase(it, subDict[node_i].end()); // Delete all elements from xth to end
        }
        if (neighbors.size() < pad)
        {
            int ttque = pad - neighbors.size();
            for (int que = 1; que <= ttque; que++)
            {

                subDict[node_i].push_back(make_tuple(NodeNum, NodeNum, NodeNum, NodeNum));
            }
        }
    }

    // Perturbation process
    map<int, vector<tuple<int, int, int, int>>> subDictPerturbed = subDict;
    std::unordered_map<int, std::set<int>> myDeg;

    for (const auto &node_i : actualNodeIDs)
    {
        for (auto &edge_tuple : subDictPerturbed[node_i])
        {
            for (int t = 0; t < 4; t++)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                // Modified line - use switch statement to handle triples
                int *target = nullptr;
                switch (t)
                {
                case 0:
                    target = &get<0>(edge_tuple);
                    break;
                case 1:
                    target = &get<1>(edge_tuple);
                    break;
                case 2:
                    target = &get<2>(edge_tuple);
                    break;
                case 3:
                    target = &get<3>(edge_tuple);
                    break;
                }

                if (*target == -1)
                    continue; // Skip invalid edge

                if (*target == NodeNum)
                {
                    int x;
                    do
                    {

                        std::random_device rd;
                        unsigned seed = rd() ^ (std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        std::mt19937 gen(seed);
                        std::uniform_int_distribution<> dis_x(0, NodeNum); // Define distribution range
                        x = dis_x(gen);

                    } while (x == node_i);
                }

                auto it = node_degrees.find(*target);
                if (it != node_degrees.end())
                {
                    double degree = it->second;
                    auto degreeGroup = degreeGroups.find(degree);
                    if (degreeGroup != degreeGroups.end())
                    {
                        double q = exp(Eps) / (exp(Eps) + degreeGroup->second.size());
                        if (dist(gen) > q)
                        {
                            std::uniform_int_distribution<> dis_idx(0, degreeGroup->second.size() - 1);
                            int r;
                            do
                            {
                                r = degreeGroup->second[dis_idx(gen)];
                            } while (r == node_i);
                            *target = r;
                        }
                    }
                }
            }

            // Insert three elements into myDeg
            myDeg[node_i].insert(get<0>(edge_tuple));
            if (get<1>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<1>(edge_tuple));
            }
            if (get<2>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<2>(edge_tuple));
            }
            if (get<3>(edge_tuple) != -1)
            {
                myDeg[node_i].insert(get<3>(edge_tuple));
            }
        }
    }
    // Calculate perturbed clique count
    clique_num = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors = myDeg[node_i];
        if (neighbors.size() < 3)
            continue;

        std::vector<int> neighborList(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborList.size(); j++)
        {
            for (int k = j + 1; k < neighborList.size(); k++)
            {
                for (int l = k + 1; l < neighborList.size(); l++)
                {
                    int a = neighborList[j];
                    int b = neighborList[k];
                    int c = neighborList[l];

                    if (myDeg[a].count(b) && myDeg[a].count(c) && myDeg[b].count(c))
                    {
                        clique_num += 1;
                    }
                }
            }
        }
    }

    cout << "Perturbed triangle count: " << clique_num << endl;
    // Calculate relative error
    double relative_error = std::abs((clique_num - clique_count) / clique_count);

    // Output relative error
    std::cout << "Relative error: " << relative_error << std::endl;
}

void BaselineRR3Clique(map<int, int> *a_mat, int *deg, string outfile, double &tri_num_ns, int emp, double custom_pad = -1)
{
    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
    // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // Calculate original triangle count
    int tri_count = 0;
    for (const auto &node_i : actualNodeIDs)
    {
        const auto &neighbors_i = data[node_i];
        for (auto it_j = neighbors_i.begin(); it_j != neighbors_i.end(); ++it_j)
        {
            if (*it_j <= node_i)
                continue;
            for (auto it_k = next(it_j); it_k != neighbors_i.end(); ++it_k)
            {
                if (*it_k <= node_i)
                    continue;
                if (data[*it_j].count(*it_k))
                {
                    tri_count++;
                }
            }
        }
    }
    cout << "Original triangle count: " << tri_count << endl;

    // Parameter settings
    double Eps = 2.0;                     // Total privacy budget
    double p = exp(Eps) / (exp(Eps) + 1); // RR retention probability

    // Step 1: Local Perturbation - Apply RR to adjacency vector of each node
    std::unordered_map<int, std::set<int>> perturbed_data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : actualNodeIDs)
        {
            if (node_i == node_j)
                continue;

            bool original_edge = data[node_i].count(node_j) > 0;
            bool perturbed_edge = original_edge;

            if (dist(gen) > p)
            {
                perturbed_edge = !original_edge; // Flip with probability 1-p
            }

            if (perturbed_edge)
            {
                perturbed_data[node_i].insert(node_j);
            }
        }
    }

    // Step 2: Data Aggregation with Linear Correction
    // Calculate the number of observed edges
    long long observed_edges = 0;
    for (const auto &pair : perturbed_data)
    {
        observed_edges += pair.second.size();
    }
    observed_edges /= 2; // Each edge is counted twice

    // Calculate unbiased estimate of edge count
    long long n = actualNodeCount;
    long long possible_edges = n * (n - 1) / 2;
    double hat_m = (observed_edges - possible_edges * (1 - p)) / (2 * p - 1);

    // Calculate retention probability theta
    double theta = (observed_edges > 0) ? std::min(1.0, std::max(0.0, hat_m / observed_edges)) : 0.0;

    cout << "Observed edges: " << observed_edges << ", Estimated true edges: " << hat_m << ", Theta: " << theta << endl;

    // Apply linear correction
    std::unordered_map<int, std::set<int>> corrected_data;
    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : perturbed_data[node_i])
        {
            if (node_i < node_j) // Avoid duplicate processing
            {
                if (dist(gen) <= theta) // Retain edge with probability theta
                {
                    corrected_data[node_i].insert(node_j);
                    corrected_data[node_j].insert(node_i);
                }
            }
        }
    }

    // Step 3: 3-Clique Counting on corrected graph
    double perturbed_tri_count = 0;
    std::vector<int> nodes(actualNodeIDs.begin(), actualNodeIDs.end());

    // 枚举所有可能的三元组
    for (int i = 0; i < nodes.size(); i++)
    {
        for (int j = i + 1; j < nodes.size(); j++)
        {
            for (int k = j + 1; k < nodes.size(); k++)
            {
                int node1 = nodes[i];
                int node2 = nodes[j];
                int node3 = nodes[k];

                // 检查是否形成三角形
                if (corrected_data[node1].count(node2) &&
                    corrected_data[node1].count(node3) &&
                    corrected_data[node2].count(node3))
                {
                    perturbed_tri_count++;
                }
            }
        }
    }

    cout << "Perturbed triangle count: " << perturbed_tri_count << endl;

    // Calculate relative error
    double relative_error = std::abs((perturbed_tri_count - tri_count) / tri_count);
    tri_num_ns = relative_error;

    cout << "Relative error for 3-clique: " << relative_error << endl;
}

void BaselineRR4Clique(map<int, int> *a_mat, int *deg, string outfile, double &clique4_num_ns, int emp, double custom_pad = -1)
{
    // Use actual node ID set
    std::set<int> actualNodeIDs;

    // Change data structure to use actual node IDs
    std::unordered_map<int, std::set<int>> data;
    // Use global EdgeFile, no longer hardcoded path
    std::ifstream file(EdgeFile);
    if (!file.is_open()) {
        cout << "Error: Cannot open edge file: " << EdgeFile << endl;
        return;
    }

    std::string line;

    // Read edge file and collect all actual node IDs
    std::getline(file, line); // Skip header line
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

    int actualNodeCount = actualNodeIDs.size();
    cout << "Actual number of nodes: " << actualNodeCount << endl;

    // 计算原始4-clique数量
    int clique4_count = 0;
    std::vector<int> nodes(actualNodeIDs.begin(), actualNodeIDs.end());

    // 枚举所有可能的4节点组合
    for (int i = 0; i < nodes.size(); i++)
    {
        for (int j = i + 1; j < nodes.size(); j++)
        {
            for (int k = j + 1; k < nodes.size(); k++)
            {
                for (int l = k + 1; l < nodes.size(); l++)
                {
                    int n1 = nodes[i], n2 = nodes[j], n3 = nodes[k], n4 = nodes[l];

                    // 检查是否形成4-clique (需要6条边)
                    if (data[n1].count(n2) && data[n1].count(n3) && data[n1].count(n4) &&
                        data[n2].count(n3) && data[n2].count(n4) &&
                        data[n3].count(n4))
                    {
                        clique4_count++;
                    }
                }
            }
        }
    }
    cout << "Original 4-clique count: " << clique4_count << endl;

    // Parameter settings
    double Eps = 2.0;                     // Total privacy budget
    double p = exp(Eps) / (exp(Eps) + 1); // RR retention probability

    // Step 1: Local Perturbation - Apply RR to adjacency vector of each node
    std::unordered_map<int, std::set<int>> perturbed_data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : actualNodeIDs)
        {
            if (node_i == node_j)
                continue;

            bool original_edge = data[node_i].count(node_j) > 0;
            bool perturbed_edge = original_edge;

            if (dist(gen) > p)
            {
                perturbed_edge = !original_edge; // Flip with probability 1-p
            }

            if (perturbed_edge)
            {
                perturbed_data[node_i].insert(node_j);
            }
        }
    }

    // Step 2: Data Aggregation with Linear Correction
    // Calculate the number of observed edges
    long long observed_edges = 0;
    for (const auto &pair : perturbed_data)
    {
        observed_edges += pair.second.size();
    }
    observed_edges /= 2; // Each edge is counted twice

    // Calculate unbiased estimate of edge count
    long long n = actualNodeCount;
    long long possible_edges = n * (n - 1) / 2;
    double hat_m = (observed_edges - possible_edges * (1 - p)) / (2 * p - 1);

    // Calculate retention probability theta
    double theta = (observed_edges > 0) ? std::min(1.0, std::max(0.0, hat_m / observed_edges)) : 0.0;

    cout << "Observed edges: " << observed_edges << ", Estimated true edges: " << hat_m << ", Theta: " << theta << endl;

    // Apply linear correction
    std::unordered_map<int, std::set<int>> corrected_data;
    for (const auto &node_i : actualNodeIDs)
    {
        for (const auto &node_j : perturbed_data[node_i])
        {
            if (node_i < node_j) // Avoid duplicate processing
            {
                if (dist(gen) <= theta) // Retain edge with probability theta
                {
                    corrected_data[node_i].insert(node_j);
                    corrected_data[node_j].insert(node_i);
                }
            }
        }
    }

    // Step 3: 4-Clique Counting on corrected graph
    double perturbed_clique4_count = 0;

    // 枚举所有可能的4节点组合
    for (int i = 0; i < nodes.size(); i++)
    {
        for (int j = i + 1; j < nodes.size(); j++)
        {
            for (int k = j + 1; k < nodes.size(); k++)
            {
                for (int l = k + 1; l < nodes.size(); l++)
                {
                    int n1 = nodes[i], n2 = nodes[j], n3 = nodes[k], n4 = nodes[l];

                    // 检查是否形成4-clique (需要6条边)
                    if (corrected_data[n1].count(n2) && corrected_data[n1].count(n3) && corrected_data[n1].count(n4) &&
                        corrected_data[n2].count(n3) && corrected_data[n2].count(n4) &&
                        corrected_data[n3].count(n4))
                    {
                        perturbed_clique4_count++;
                    }
                }
            }
        }
    }

    cout << "Perturbed 4-clique count: " << perturbed_clique4_count << endl;

    // Calculate relative error
    double relative_error = (clique4_count > 0) ? std::abs((perturbed_clique4_count - clique4_count) / clique4_count) : 0.0;
    clique4_num_ns = relative_error;

    cout << "Relative error for 4-clique: " << relative_error << endl;
}

int main(int argc, char *argv[])
{
    // Initialization of Mersennne Twister
    unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 4;
    init_by_array(init, length);

    if (argc < 5)
    {
        printf("Usage: %s [DatasetPath] [q] [Method] [Eps] ([EdgeFile (default: edges.csv)])\n\n", argv[0]);
        printf("[DatasetPath]: Path to the dataset directory\n");
        printf("[q]: Target clique size (3 or 4)\n");
        printf("[Method]: 1 for Baseline RR, 2 for K-Star method\n");
        printf("[Eps]: Privacy parameter epsilon (will be adjusted by q)\n");
        printf("[EdgeFile]: Edge file name (default: edges.csv)\n");
        return -1;
    }

    // Parse command-line arguments
    string dataset_path = argv[1];
    DatasetPath = dataset_path; // Set global variable
    q = atoi(argv[2]);
    int method = atoi(argv[3]);

    // Parse epsilon (input by user). Will be adjusted depending on q.
    double inputEps = atof(argv[4]);
    Eps_s = argv[4];
    if (q == 3)
    {
        Eps = inputEps / 2.0;
    }
    else if (q == 4)
    {
        Eps = inputEps / 3.0;
    }
    else
    {
        Eps = inputEps;
    }

    string edge_file_name = "edges.csv";
    if (argc >= 6)
        edge_file_name = argv[5];

    EdgeFile = dataset_path + "/" + edge_file_name;

    printf("Input Eps: %s, Adjusted Eps (used): %g\n", Eps_s.c_str(), Eps);
    printf("Method: %d (%s)\n", method, (method == 1) ? "Baseline RR" : "K-Star");

    // Use fixed parameters for this simplified main function
    NodeNum = -1; // will attempt to detect below
    EpsNsMaxDeg = Eps / 10;
    NSType = 0;
    ItrNum = 1;
    Alg = 2;

    for (int i = 0; i < 3; i++)
    {
        Balloc[i] = 1.0;
        Balloc_s[i] = (char *)"1";
    }

    // Calculate optimal padding for the dataset
    optimal_pad = FindOptimalPaddingForDataset();
    cout << "Computed optimal padding: " << optimal_pad << endl;

    // Try to detect number of nodes from the edge file to avoid allocating huge arrays
    {
        std::unordered_set<int> nodes;
        std::ifstream ef(EdgeFile);
        std::string line;
        if (ef.is_open())
        {
            std::getline(ef, line); // skip header if present
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
        {
            NodeNum = (int)nodes.size();
        }
        cout << "Detected NodeNum: " << NodeNum << endl;
    }

    // Run 10 experiments
    const int NUM_EXPERIMENTS = 10;
    std::vector<double> relative_errors;

    cout << "\n=== Running " << NUM_EXPERIMENTS << " experiments ===" << endl;
    cout << "Dataset: " << dataset_path << endl;
    cout << "Target clique: " << q << "-clique" << endl
         << endl;
    cout << "Method: " << (method == 1 ? "Baseline RR" : "K-Star") << endl
         << endl;

    for (int exp = 1; exp <= NUM_EXPERIMENTS; exp++)
    {
        cout << "--- Experiment " << exp << " ---" << endl;

        // Create temporary adjacency matrix and degree array
        map<int, int> *a_mat = new map<int, int>[NodeNum];
        int *deg = new int[NodeNum];
        double clique_num_ns = 0.0;

        // Call appropriate function based on q value
        if (q == 3)
        {
            if (method == 1)
            {
                cout << "Calling BaselineRR3Clique..." << endl;
                BaselineRR3Clique(a_mat, deg, "", clique_num_ns, 0, optimal_pad);
            }
            else
            {
                cout << "Calling NewCalNLocTriedge2k3cliquenoise..." << endl;
                NewCalNLocTriedge2k3cliquenoise(a_mat, deg, "", clique_num_ns, 0, optimal_pad);
            }
        }
        else if (q == 4)
        {
            if (method == 1)
            {
                cout << "Calling BaselineRR4Clique..." << endl;
                BaselineRR4Clique(a_mat, deg, "", clique_num_ns, 0, optimal_pad);
            }
            else
            {
                cout << "Calling NewCalNLocTriedge3k4cliquenoise..." << endl;
                NewCalNLocTriedge3k4cliquenoise(a_mat, deg, "", clique_num_ns, 0, optimal_pad);
            }
        }
        cout << "Experiment " << exp << " completed." << endl
             << endl;
        relative_errors.push_back(clique_num_ns);

        delete[] a_mat;
        delete[] deg;
    }

    // Calculate average
    double average = 0.0;
    for (double val : relative_errors)
        average += val;
    average /= NUM_EXPERIMENTS;

    cout << "=== Final Results ===" << endl;
    cout << "Number of experiments: " << NUM_EXPERIMENTS << endl;
    cout << "Target clique: " << q << "-clique" << endl;
    cout << "Original " << (q == 3 ? "triangle" : "4-clique") << " count: " << OriginalCliqueCount << endl;
    cout << "Method: " << (method == 1 ? "Baseline RR" : "K-Star") << endl;
    cout << "Optimal padding: " << optimal_pad << endl;
    cout << "Average relative error: " << average << endl;

    return 0;
}
