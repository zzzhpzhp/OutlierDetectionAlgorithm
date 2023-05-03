#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <random>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;
////using namespace cv;
//

//定义数据点类型
struct Point2f
{
    Point2f() = default;

    Point2f(float tx, float ty)
    {
        x = tx;
        y = ty;
    }

    float x{0}, y{0};
    float lrd{0}; //局部可达密度
    float lof{0}; //局部离群因子
};

//计算欧几里得距离
float euclidean_distance(Point2f p1, Point2f p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 计算K近邻
vector<Point2f> k_nearest_neighbors(Point2f p, vector<Point2f> &points, int k)
{
    vector<Point2f> neighbors;
    for (auto it = points.begin(); it != points.end(); it++)
    {
        if (p.x == it->x && p.y == it->y)
        { //排除自身点
            continue;
        }
        float dist = euclidean_distance(p, *it);
        if (neighbors.size() < k)
        {
            //添加到近邻集合中
            neighbors.push_back(*it);
            it->lrd = dist; //初始化局部可达密度为距离
        }
        else
        {
            auto max_it = max_element(neighbors.begin(), neighbors.end(), [](Point2f &p1, Point2f &p2)
            {
                return p1.lrd < p2.lrd;
            });
            if (dist > max_it->lrd)
            {
                // 替换掉当前最远近邻
                *max_it = *it;
                max_it->lrd = dist;
            }
        }
    }
    return neighbors;
}

//计算可达距离
float reachable_distance(Point2f p, Point2f o, vector<Point2f> &points)
{
    float max_dist = euclidean_distance(p, o);
    //两点之间的距离
    for (auto it = points.begin(); it != points.end(); it++)
    {
        if (it->x == p.x && it->y == p.y || it->x == o.x && it->y == o.y)
        {
            //排除自身点和当前比较的点
            continue;
        }
        float dist = euclidean_distance(p, *it);
        if (dist < max_dist)
        { max_dist = dist; }
    }
    return max_dist;
}

//计算局部可达密度
float local_reachability_density(Point2f p, vector<Point2f> &points, int k)
{
    auto neighbors = k_nearest_neighbors(p, points, k);
    float sum = 0;
    for (auto it = neighbors.begin(); it != neighbors.end(); it++)
    {
        it->lrd = reachable_distance(p, *it, neighbors);
        sum += it->lrd;
    }
    if (sum > 0)
    { return neighbors.size() / sum; }
    else
    { return 0; }
}

//计算COF值
float count_outliers_factor(Point2f p, vector<Point2f> &points, int k)
{
    auto neighbors = k_nearest_neighbors(p, points, k);
    float sum = 0;
    for (auto it = neighbors.begin(); it != neighbors.end(); it++)
    {
        if (it->x == p.x && it->y == p.y)
        {
            continue;
        }
        auto it_neighbors = k_nearest_neighbors(*it, points, k);
        float lrd_o = local_reachability_density(*it, it_neighbors, k);
        sum += lrd_o / p.lrd;
    }
    if (neighbors.size() > 0)
    {
        return sum / neighbors.size();
    }
    else
    {
        return 0;
    }
}

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

int main()
{
    srand(time(0) + 1);
    vector<Point2f> points(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        points[i] = Point2f(getRand(Width), getRand(Width));
    }

    // 计算每个点的 COF
    vector<float> cof_values(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        cof_values[i] = count_outliers_factor(points[i], points, K);
        std::cout << cof_values[i] << std::endl;
    }

    // 显示 COF 值
    cv::Mat img(Width, Width, CV_8UC3, cv::Scalar(255, 255, 255));
    double min_cof, max_cof;
    cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::minMaxLoc(cof_values, &min_cof, &max_cof);
    for (int i = 0; i < num_points; ++i)
    {
        float value = cof_values[i] / static_cast<float>(max_cof) * 255;
        cv::Point pt(points[i].x, points[i].y);
        circle(img, pt, 3, cv::Scalar(value, value, value), -1);
        if (cof_values[i] < Threshold)
        {
            circle(img, pt, 10, cv::Scalar(value, value, value), 1);
        }
    }
    imshow("COF values", img);
    cv::waitKey(0);

    return 0;
}



//
////定义点类
//class Point
//{
//public:
//    double x, y;
//
//    Point(double x = 0, double y = 0) : x(x), y(y)
//    {}
//};
//
////计算点p与点q之间的欧几里得距离
//double EuclideanDistance(const Point &p, const Point &q)
//{ return sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2)); }
//
////计算点p的k距离（即与p最近的k个点之间的最远距离）
//double kDistance(const Point &p, const std::vector<Point> &points, int k)
//{
//    std::vector<double> distances;
//    for (auto &q : points)
//    {
//        if (q.x == p.x && q.y == p.y)
//        {
//            continue;
//        }
//        distances.push_back(EuclideanDistance(p, q));
//    }
//    std::nth_element(distances.begin(), distances.begin() + k, distances.end());
//    return distances[k];
//}
//
////计算点p的局部可达密度（即与p距离小于等于k距离的点的个数）
//double reachabilityDensity(const Point &p, const std::vector<Point> &points, int k)
//{
//    int count = 0;
//    double kdistance = kDistance(p, points, k);
//    for (auto &q : points)
//    {
//        if (q.x == p.x && q.y == p.y)
//        {
//            continue;
//        }
//        if (EuclideanDistance(p, q) <= kdistance)
//        { ++count; }
//    }
//    return count / M_PI / pow(kdistance, 2);
//}
//
////计算点p的局部离群因子
//double LOF(const Point &p, const std::vector<Point> &points, int k)
//{
//    double rd = reachabilityDensity(p, points, k);
//    double sum = 0;
//    for (auto &q : points)
//    {
//        if (q.x == p.x && q.y == p.y)
//        {
//            continue;
//        }
//        { sum += reachabilityDensity(q, points, k) / rd; }
//    }
//    return sum / (points.size() - 1);
//}
//
////计算点p的COF离群度量
//double COF(const Point &p, const std::vector<Point> &points, int k)
//{
//    double sum = 0;
//    for (auto &q : points)
//    {
//        if (&p != &q)
//        {
//            double pd = EuclideanDistance(p, q);
//            double rd = reachabilityDensity(p, points, k);
//            sum += (rd - reachabilityDensity(q, points, k)) / rd / pd;
//        }
//    }
//    return sum / (points.size() - 1);
//}
//
//int main()
//{
//    //生成随机点
//    std::vector<Point> points;
//    for (int i = 0; i < 20; ++i)
//    { points.emplace_back(rand() % 1000, rand() % 1000); }
//
//    //计算每个点的COF离群度量
//    std::vector<double> cofs;
//    for (auto &p : points)
//    {
//        cofs.push_back(COF(p, points, 3));
//    }
//
//    //输出结果
//    for (int i = 0; i < points.size(); ++i)
//    {
//        std::cout << "Point (" << points[i].x << "," << points[i].y << ") COF: " << cofs[i] << std::endl;
//    }
//
//    //通过OpenCV可视化显示结果
//    cv::Mat image(1000, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
//    for (auto &p : points)
//    {
//        cv::circle(image, cv::Point(p.x, p.y), 5, cv::Scalar(0, 0, 255), -1);
//    }
//    for (int i = 0; i < points.size(); ++i)
//    {
//        if (cofs[i] > 0)
//        {
//            cv::circle(image, cv::Point(points[i].x, points[i].y), 10, cv::Scalar(0, 0, 255), 1);
//        }
//    }
//    imshow("result", image);
//    cv::waitKey(0);
//
//}