#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
using namespace std;

#define EPSILON 1e-7

void read_csv(vector<vector<double>>&data,const string& file_name)
{
    ifstream file(file_name);
    string line;
    getline(file,line);
    while(getline(file,line))
    {
        stringstream ss(line);
        string entry;
        vector<double>row;
        bool skip=false;
        while(getline(ss,entry,','))
        {
            if(entry.empty())
            {
                skip=true;
                break;
            }
            row.push_back(stod(entry));
        }
        if(!skip && !row.empty())
        {
            row.insert(row.begin(), 1.0);
            data.push_back(row);
        }
    }
}

double hypothesis_function(vector<double>&training_eg,vector<double>&thethas)
{
    int n=thethas.size();
    double h=0;
    for(int i=0;i<n;i++)
    {
        h+=thethas[i]*training_eg[i];
    }
    return h;
}

double cost_function(vector<vector<double>>&training_data,vector<double>&thethas)
{
    double cost=0;
    int m=training_data.size();
    int n=training_data[0].size();
    for(int i=0;i<m;i++)
    {
        double h=hypothesis_function(training_data[i],thethas);
        double j=(pow((h-training_data[i][n-1]),2))/2;
        cost+=j;
    }
    return cost/m;
}

double find_gradient(vector<vector<double>>&training_data,vector<double>&thethas,int j)
{
    double gradient=0;
    int m=training_data.size();
    int n=training_data[0].size();
    for(int i=0;i<m;i++)
    {
        double h=hypothesis_function(training_data[i],thethas);
        gradient+=((h-training_data[i][n-1])*training_data[i][j]);
    }
    return gradient/m;
}
void batch_gradient_descent(vector<double>&thethas,vector<vector<double>>&training_data)
{
    int n=thethas.size();
    int m=training_data.size();
    double alpha=0.001;
    double old_cost=cost_function(training_data,thethas);
    int i=0;
    while(i<20000)
    {
        for(int j=0;j<n;j++)
        {
            thethas[j]=thethas[j]-alpha*find_gradient(training_data,thethas,j);
        }
        double new_cost=cost_function(training_data,thethas);
        if(abs(new_cost-old_cost)<EPSILON)
        {
            cout<<"completed gradient desent";
            break;
        }
        old_cost=new_cost;
        if(i%1000==0)
        {
            cout<<"Cost : "<<old_cost<<endl;
            for(int j=0;j<n;j++)
            {
                cout<<thethas[j]<<" ";
            }
            cout<<endl;
        }
        i++;
    }
}

double predict(vector<double>&data,vector<double>&thethas)
{
    return hypothesis_function(data,thethas);
}

double mean_squared_error(vector<vector<double>>&test_data,vector<double>&thethas)
{
    int m=test_data.size();
    int n=test_data[0].size();
    double sum=0;
    double std_target=1.15619125e+05;
    double mean_target=2.07194694e+05;
    for(int i=0;i<m;i++)
    {
        double y_cap=predict(test_data[i],thethas)*std_target+mean_target;
        double y=test_data[i][n-1]*std_target+mean_target;
        cout<<"y predicted"<<y_cap<<" "<<"y_original"<<y<<" "<<endl;
        sum+=(pow(y_cap-y,2));
    }
    double mean=sum/m;
    double rmse=sqrt(mean);
    double percentage = (rmse / mean_target) * 100;
    cout << "RMSE is " << percentage << "% of the mean house price." << endl;
    return sqrt(mean);
}

int main()
{
    vector<vector<double>>training_data;
    read_csv(training_data,"train_normalised2.csv");
    vector<vector<double>>test_data;
    read_csv(test_data,"test_normalised2.csv");
    int n=training_data[0].size();
    vector<double>thethas(n-1,0);
    batch_gradient_descent(thethas,training_data);
    double rmse=mean_squared_error(test_data,thethas);
    cout<<"Mean Squared error is : "<<rmse<<endl;
    return 0;
}