#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
using namespace std;

double epsilon = 1e-5;

struct song
{
    double danceability;
    double energy;
    double tempo;
    double popularity;
};

vector<string> parse_csv_line(const string &line)
{
    vector<string> result;
    stringstream ss(line);
    string word;
    char delimiter = ',';
    bool inside_quotes = false;
    string field;

    for (char ch : line)
    {
        if (ch == '"')
        {
            inside_quotes = !inside_quotes;
        }
        else if (ch == delimiter && !inside_quotes)  //when you reach delimiter meaning one field is complete then you push it in result
        {
            result.push_back(field);   
            field.clear(); 
        }
        else
        {
            field += ch;
        }
    }
    result.push_back(field); // add the last field since it will not have comma at end
    return result;
}

vector<song> read_csv(const string &file_name)
{
    vector<song> data_set;
    ifstream file(file_name);
    string line;
    if (!file.is_open())
    {
        cout << "COULDN'T OPEN THE FILE!!" << endl;
        return data_set;
    }
    getline(file, line); // skip title line
    while (getline(file, line))
    {
        vector<string> fields = parse_csv_line(line);
        song obj;
        try
        {
            obj.danceability = stod(fields[8]);
            obj.energy = stod(fields[9]);
            obj.popularity = stod(fields[5]);
            obj.tempo = stod(fields[18]);
            if(obj.popularity==0) continue;
            data_set.push_back(obj);
        }
        catch (const invalid_argument &e)
        {
            cerr << "Invalid argument encountered: " << e.what() << endl;
        }
    }
    file.close();
    return data_set;
}
class LinearRegression
{
    vector<double> thetha;
    double alpha;
    int iterations;

public:
    LinearRegression(vector<double> thethas, double learning_rate, int m)
    {
        thetha = thethas;
        alpha = learning_rate;
        iterations = m;
    }

    double hypothesis_fxn(song songs)
    {
        double result = thetha[0];
        result = result + thetha[1] * songs.danceability + thetha[2] * songs.energy+thetha[3]*songs.tempo;
        return result;
    }
    double cost_function(vector<song> &songs)
    {
        double j_thetha = 0.0;
        int m=songs.size();
        for (int i = 0; i < m; i++)
        {
            double h = hypothesis_fxn(songs[i]);
            j_thetha += pow((h - songs[i].popularity), 2);
        }
        return j_thetha / (2 * m);
    }
    void cost_derivative(vector<song> &songs, vector<double> &gradients)
    {
        fill(gradients.begin(), gradients.end(), 0.0);
        int m=songs.size();
        for (int i = 0; i < m; i++)
        {
            double h = hypothesis_fxn(songs[i]);
            gradients[0] += (h - songs[i].popularity);
            gradients[1] += (h - songs[i].popularity) * songs[i].danceability;
            gradients[2] += (h - songs[i].popularity) * songs[i].energy;
            gradients[3] += (h - songs[i].popularity) * songs[i].tempo;
        }
        for (int i = 0; i < gradients.size(); i++)
        {
            gradients[i] /= m;
        }
    }
    void gradient_descent(vector<song> &songs)
    {
        int n = thetha.size();
        vector<double> new_thetha(n, 0.0);
        double previous_cost = cost_function(songs);
        for (int i = 0; i < iterations; i++)
        {
            vector<double> gradients(n, 0.0);
            cost_derivative(songs, gradients);
            for (int j = 0; j < n; j++)
            {
                new_thetha[j] = thetha[j] - gradients[j] * alpha;
            }
            thetha = new_thetha;
            double new_cost = cost_function(songs);
            if(i%100==0)
            {
             cout<<"Iteration : "<<i<<"  Cost = " << new_cost << endl;
                cout<<"Theta values: ";
                for (double theta_val : thetha)
                {
                    cout << theta_val << " ";
                }
                cout << endl;
                cout << "Gradients: ";
                for (double grad : gradients)
                {
                    cout << grad << " ";
                }
                cout << endl;
            }
            if (fabs(new_cost - previous_cost) < epsilon)
            {
                cout << "converged at iteration no : " << i << endl;
                break;
            }
            previous_cost = new_cost;
        }
    }
    double predict(song s)
    {
        return hypothesis_fxn(s);
    }
};

void normalize_features(vector<song> &songs)
{
    double mean_danceability = 0, mean_energy = 0, mean_tempo = 0;
    double std_danceability = 0, std_energy = 0,  std_tempo = 0;
    int n = songs.size();

    for (const auto &s : songs)
    {
        mean_danceability += s.danceability;
        mean_energy += s.energy;
        mean_tempo += s.tempo;
    }

    mean_danceability /= n;
    mean_energy /= n;
    mean_tempo /= n;

    for (const auto &s : songs)
    {
        std_danceability += pow(s.danceability - mean_danceability, 2);
        std_energy += pow(s.energy - mean_energy, 2);
        std_tempo += pow(s.tempo - mean_tempo, 2);
    }

    std_danceability = sqrt(std_danceability / n);
    std_energy = sqrt(std_energy / n);
    std_tempo = sqrt(std_tempo / n);

    for (auto &s : songs)
    {
        s.danceability = (s.danceability - mean_danceability) / std_danceability;
        s.energy = (s.energy - mean_energy) / std_energy;
        s.tempo = (s.tempo - mean_tempo) / std_tempo;
    }
}

void train_test_split(vector<song>&songs,vector<song>&train_set,vector<song>&test_set,double test_size)
{
    unsigned seed=chrono::system_clock::now().time_since_epoch().count();
    shuffle(songs.begin(),songs.end(),default_random_engine(seed));
    int test_count=static_cast<int>(songs.size()*test_size);
    test_set.assign(songs.begin(), songs.begin() + test_count);
    train_set.assign(songs.begin() + test_count, songs.end());
}

int main()
{
    vector<song> data=read_csv("SpotifyTrackset.csv");
    vector<song> test_data,train_data;
    train_test_split(data,train_data,test_data,0.2);

    vector<double> initial_thetha = {0.0, 0.0, 0.0, 0.0};
    double alpha = 0.001;
    int iterations = 3000;

    normalize_features(train_data);
    normalize_features(test_data);

    LinearRegression LR(initial_thetha, alpha, iterations);
    LR.gradient_descent(train_data);

    double total_error = 0.0;
    int count=0;
    for (const auto &test_song : test_data)
    {
        double prediction = LR.predict(test_song);
        if(count%1000==0)
        cout << "Actual popularity: " << test_song.popularity << ", Predicted popularity: " << prediction << endl;
        total_error += pow(prediction - test_song.popularity, 2);
        count++;
    }
    double mse = total_error / test_data.size();
    cout << "Mean Squared Error (MSE) on test set: " << mse << endl;

    return 0;
}
