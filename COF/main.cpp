#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <random>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;
//using namespace cv;


////定义数据点类型
//struct Point2f
//{
//    float x, y;
//    float lrd; //局部可达密度
//    float lof;//局部离群因子
//};
//
////计算欧几里得距离
//float euclidean_distance(Point2f p1, Point2f p2)
//{
//    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
//}
//
////计算K近邻
//vector<Point2f> k_nearest_neighbors(Point2f p, vector<Point2f> &points, int k)
//{
//    vector<Point2f> neighbors;
//    for (auto it = points.begin(); it != points.end(); it++)
//    {
//        if (p.x == it->x && p.y == it->y)
//        { //排除自身点
//            continue;
//        }
//        float dist = euclidean_distance(p, *it);
//        if (neighbors.size() < k)
//        {
//            //添加到近邻集合中
//            neighbors.push_back(*it);
//            it->lrd = dist; //初始化局部可达密度为距离
//        }
//        else
//        {
//            auto max_it = max_element(neighbors.begin(), neighbors.end(), [](Point2f &p1, Point2f &p2)
//            { return p1.lrd < p2.lrd; });
//            if (dist < max_it->lrd)
//            { //替换掉当前最远近邻
//                *max_it = *it;
//                max_it->lrd = dist;
//            }
//        }
//    }
//    return neighbors;
//}
//
////计算可达距离
//float reachable_distance(Point2f p, Point2f o, vector<Point2f> &points)
//{
//    float max_dist = euclidean_distance(p, o);
//    //两点之间的距离
//    for (auto it = points.begin(); it != points.end(); it++)
//    {
//        if (it->x == p.x && it->y == p.y || it->x == o.x && it->y == o.y)
//        {
//            //排除自身点和当前比较的点
//            continue;
//        }
//        float dist = euclidean_distance(p, *it);
//        if (dist < max_dist)
//        { max_dist = dist; }
//    }
//    return max_dist;
//}
//
////计算局部可达密度
//float local_reachability_density(Point2f p, vector<Point2f> &points, int k)
//{
//    auto neighbors = k_nearest_neighbors(p, points, k);
//    float sum = 0;
//    for (auto it = neighbors.begin(); it != neighbors.end(); it++)
//    {
//        it->lrd = reachable_distance(p, *it, neighbors);
//        sum += it->lrd;
//    }
//    if (sum > 0)
//    { return neighbors.size() / sum; }
//    else
//    { return 0; }
//}
//
////计算COF值
//float count_outliers_factor(Point2f p, vector<Point2f> &points, int k)
//{
//    auto neighbors = k_nearest_neighbors(p, points, k);
//    float sum = 0;
//    for (auto it = neighbors.begin(); it != neighbors.end(); it++)
//    {
//        auto it_neighbors = k_nearest_neighbors(*it, points, k);
//        float lrd_o = local_reachability_density(*it, points, k);
//        sum += lrd_o / p.lrd;
//    }
//    if (neighbors.size() > 0)
//    { return sum / neighbors.size(); }
//    else
//    { return 0; }
//}

////生成随机坐标
//void generate_random_points(vector<Point2f> &points, int n, float x_min, float y_min, float x_max, float y_max)
//{
//    srand(time(0))
//
//    for (int i = 0; i < n; i++)
//    {
//        float x = rand() % 1000;
//        float y = rand() % 1000;
//        points.push_back({x, y, 0, 0});
//    }
//}

// 生成一些随机点
constexpr int num_points = 60;
constexpr int Width = 1000;
constexpr int K = 3;
constexpr float Threshold = 1.5;

float getRand(int max_val)
{
//    return rng.uniform(0.0f, max_val);
    return rand() % max_val;
}

// 计算给定点与其它所有点的欧几里得距离
vector<float> euclidean_distance(vector<cv::Point> points, cv::Point p)
{
    vector<float> dists(points.size(), 0.0);
    for (int i = 0; i < points.size(); ++i)
    {
        float dx = points[i].x - p.x;
        float dy = points[i].y - p.y;
        dists[i] = sqrt(dx * dx + dy * dy);
    }
    return dists;
}

// 计算给定点的 COF（离群度量）
float calculate_cof(vector<float> dists)
{
    int k = dists.size() / K; // 选取距离最小的K个点
    nth_element(dists.begin(), dists.begin() + k, dists.end());
    float nearest = dists[k];
    float sum = 0;
    for (int i = 0; i < k; ++i)
    {
        sum += dists[i];
    }
    return nearest / (sum / static_cast<float>(k));
}


int main()
{
    srand(time(0) + 1);
    vector<cv::Point> points(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        points[i] = cv::Point(getRand(Width), getRand(Width));
    }

    // 计算每个点的 COF
    vector<float> cof_values(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        vector<float> dists = euclidean_distance(points, points[i]);
        cof_values[i] = calculate_cof(dists);
    }

    // 显示 COF 值
    cv::Mat img(Width, Width, CV_8UC3, cv::Scalar(255, 255, 255));
    double min_cof, max_cof;
    cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::minMaxLoc(cof_values, &min_cof, &max_cof);
    for (int i = 0; i < num_points; ++i)
    {
        float value = cof_values[i] / static_cast<float>(max_cof) * 255;
        circle(img, points[i], 3, cv::Scalar(value, value, value), -1);
        std::cout << cof_values[i] << std::endl;
        if (cof_values[i] < Threshold)
        {
            circle(img, points[i], 10, cv::Scalar(value, value, value), 1);
        }
    }
    imshow("COF values", img);
    cv::waitKey(0);

    return 0;
}