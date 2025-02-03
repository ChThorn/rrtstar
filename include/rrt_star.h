#ifndef RRT_STAR_H
#define RRT_STAR_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>
#include "nanoflann.hpp"

namespace rrt {

struct Point3D {
    double x, y, z;
    
    Point3D(double x = 0, double y = 0, double z = 0);
    double distance(const Point3D& other) const;
    Point3D operator+(const Point3D& other) const;
    Point3D operator-(const Point3D& other) const;
    Point3D operator*(double scalar) const;
    bool operator==(const Point3D& other) const;

    // Add the subscript operators
    double& operator[](size_t index) {
        switch(index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: throw std::out_of_range("Invalid dimension");
        }
    }

    const double& operator[](size_t index) const {
        switch(index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: throw std::out_of_range("Invalid dimension");
        }
    }
};

// class KDTree {
//     struct PointCloud {
//         std::vector<Point3D> pts;
        
//         inline size_t kdtree_get_point_count() const { return pts.size(); }
        
//         inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
//             if (dim == 0) return pts[idx].x;
//             if (dim == 1) return pts[idx].y;
//             return pts[idx].z;
//         }
        
//         template <class BBOX>
//         bool kdtree_get_bbox(BBOX&) const { return false; }
//     };

//     PointCloud cloud_;
//     using KDTreeT = nanoflann::KDTreeSingleIndexAdaptor
//         nanoflann::L2_Simple_Adaptor<double, PointCloud>,
//         PointCloud,
//         3>;
//     std::unique_ptr<KDTreeT> index_;

// public:
//     explicit KDTree(const std::vector<Point3D>& points);
//     std::vector<size_t> radius_search(const Point3D& point, double radius) const;
// };

struct Obstacle {
    Point3D min;
    Point3D max;
    Obstacle(const Point3D& min, const Point3D& max);
};

class KDTree {
    struct PointCloud {
        std::vector<Point3D> pts;
        
        inline size_t kdtree_get_point_count() const { return pts.size(); }
        
        inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
            if (dim == 0) return pts[idx].x;
            if (dim == 1) return pts[idx].y;
            return pts[idx].z;
        }
        
        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
    };

    PointCloud cloud_;
    using KDTreeT = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud>,
        PointCloud,
        3>;
    std::unique_ptr<KDTreeT> index_;

public:
    explicit KDTree(const std::vector<Point3D>& points);
    std::vector<size_t> radius_search(const Point3D& point, double radius) const;
};

class RRTStar {
public:
    // struct Config {
    //     Point3D bounds {10, 10, 10};
    //     double step_size = 0.5;
    //     int max_iterations = 2000;
    //     double goal_bias = 0.1;
    //     double goal_threshold = 0.5;
    //     double safety_margin = 0.3;
    //     double rewire_radius = 2.5;
    // };

    struct Config {
        Point3D bounds;
        double step_size;
        int max_iterations;
        double goal_bias;
        double goal_threshold;
        double safety_margin;
        double rewire_radius;

        // Constructor with default values
        Config()
            : bounds{10, 10, 10},
            step_size(0.5),
            max_iterations(2000),
            goal_bias(0.1),
            goal_threshold(0.5),
            safety_margin(0.3),
            rewire_radius(2.5) {}
    };

    RRTStar(const Point3D& start, 
           const Point3D& goal,
           const std::vector<Obstacle>& obstacles,
           const Config& config = Config());
    
    std::vector<Point3D> plan();

private:
    struct Node {
        Point3D point;
        size_t parent;
        double cost;
    };

    // Core algorithm components
    Point3D random_sample() const;
    size_t nearest_node(const Point3D& target) const;
    Point3D steer(const Point3D& from, const Point3D& to) const;
    bool is_collision_free(const Point3D& a, const Point3D& b) const;
    bool line_aabb_intersection(const Point3D& p1, const Point3D& p2,
                               const Point3D& box_min, const Point3D& box_max) const;
    std::vector<size_t> find_near_nodes(const Point3D& point) const;
    std::vector<Point3D> trace_path(size_t end_index) const;
    std::vector<Point3D> smooth_path(const std::vector<Point3D>& path) const;

    // Spatial indexing
    void rebuild_kd_tree();
    std::vector<size_t> radius_search(const Point3D& point, double radius) const;

    // Member variables
    Point3D start_;
    Point3D goal_;
    std::vector<Obstacle> obstacles_;
    Config config_;
    std::vector<Node> nodes_;
    std::unique_ptr<class KDTree> kd_tree_;
};

} // namespace rrt

#endif // RRT_STAR_H