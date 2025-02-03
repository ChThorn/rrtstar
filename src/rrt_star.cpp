#include "rrt_star.h"
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include "nanoflann.hpp"

using namespace rrt;
using namespace nanoflann;

namespace rrt {
    KDTree::KDTree(const std::vector<Point3D>& points) {
        cloud_.pts = points;
        index_.reset(new KDTreeT(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
        index_->buildIndex();
    }
}

// Point3D Implementation
Point3D::Point3D(double x, double y, double z) : x(x), y(y), z(z) {}

double Point3D::distance(const Point3D& other) const {
    return std::sqrt(
        std::pow(x - other.x, 2) + 
        std::pow(y - other.y, 2) + 
        std::pow(z - other.z, 2)
    );
}

Point3D Point3D::operator+(const Point3D& other) const {
    return Point3D(x + other.x, y + other.y, z + other.z);
}

Point3D Point3D::operator-(const Point3D& other) const {
    return Point3D(x - other.x, y - other.y, z - other.z);
}

Point3D Point3D::operator*(double scalar) const {
    return Point3D(x * scalar, y * scalar, z * scalar);
}

// Obstacle Implementation
Obstacle::Obstacle(const Point3D& min, const Point3D& max) 
    : min(min), max(max) {}

// KD-Tree Adapter
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

// class KDTree {
//     PointCloud cloud_;
//     using KDTreeT = KDTreeSingleIndexAdaptor<
//         L2_Simple_Adaptor<double, PointCloud>,
//         PointCloud,
//         3>;
//     std::unique_ptr<KDTreeT> index_;

// public:
//     KDTree::KDTree(const std::vector<Point3D>& points) {
//         cloud_.pts = points;
//         index_.reset(new KDTreeT(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
//         index_->buildIndex();
//     }

//     std::vector<size_t> KDTree::radius_search(const Point3D& point, double radius) const {
//         std::vector<nanoflann::ResultItem<size_t, double>> ret_matches;
        
//         const double query_pt[3] = {point.x, point.y, point.z};
//         nanoflann::RadiusResultSet<double, size_t> resultSet(radius * radius, ret_matches);
//         index_->findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
        
//         std::vector<size_t> indices;
//         indices.reserve(ret_matches.size());
//         for (const auto& match : ret_matches) {
//             indices.push_back(match.first);
//         }
//         return indices;
//     }
// };

// RRTStar Implementation
RRTStar::RRTStar(const Point3D& start, const Point3D& goal,
               const std::vector<Obstacle>& obstacles,
               const Config& config)
    : start_(start), goal_(goal), obstacles_(obstacles), config_(config) {
    nodes_.push_back({start_, 0, 0.0});
    rebuild_kd_tree();
}

void RRTStar::rebuild_kd_tree() {
    std::vector<Point3D> points;
    points.reserve(nodes_.size());
    for (const auto& node : nodes_)
        points.push_back(node.point);
    kd_tree_.reset(new KDTree(points));
}

std::vector<size_t> RRTStar::radius_search(const Point3D& point, double radius) const {
    return kd_tree_->radius_search(point, radius);
}

std::vector<size_t> KDTree::radius_search(const Point3D& point, double radius) const {
    std::vector<ResultItem<size_t, double>> ret_matches;
    
    const double query_pt[3] = {point.x, point.y, point.z};
    nanoflann::RadiusResultSet<double, size_t> resultSet(radius * radius, ret_matches);
    index_->findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
    
    std::vector<size_t> indices;
    indices.reserve(ret_matches.size());
    for (const auto& match : ret_matches) {
        indices.push_back(match.first);
    }
    return indices;
}

Point3D RRTStar::random_sample() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    if (dis(gen) < config_.goal_bias) {
        return goal_;
    }
    return Point3D(
        dis(gen) * config_.bounds.x,
        dis(gen) * config_.bounds.y,
        dis(gen) * config_.bounds.z
    );
}

size_t RRTStar::nearest_node(const Point3D& target) const {
    const auto indices = radius_search(target, config_.step_size);
    if (!indices.empty()) return indices[0];
    
    // Fallback linear search (should never happen with proper KD-Tree)
    size_t best_idx = 0;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const double d = nodes_[i].point.distance(target);
        if (d < best_dist) {
            best_dist = d;
            best_idx = i;
        }
    }
    return best_idx;
}

Point3D RRTStar::steer(const Point3D& from, const Point3D& to) const {
    const Point3D direction = to - from;
    const double dist = from.distance(to);
    if (dist <= config_.step_size) return to;
    return from + direction * (config_.step_size / dist);
}

bool RRTStar::is_collision_free(const Point3D& a, const Point3D& b) const {
    for (const auto& obs : obstacles_) {
        if (line_aabb_intersection(a, b, 
            obs.min - Point3D(config_.safety_margin, config_.safety_margin, config_.safety_margin),
            obs.max + Point3D(config_.safety_margin, config_.safety_margin, config_.safety_margin))) {
            return false;
        }
    }
    return true;
}

bool RRTStar::line_aabb_intersection(const Point3D& p1, const Point3D& p2,
                                   const Point3D& box_min, 
                                   const Point3D& box_max) const {
    const Point3D dir = (p2 - p1) * 0.5;
    const Point3D midpoint = (p1 + p2) * 0.5;
    const Point3D abs_dir(std::abs(dir.x), std::abs(dir.y), std::abs(dir.z));
    
    const Point3D box_center = (box_min + box_max) * 0.5;
    const Point3D box_half = (box_max - box_min) * 0.5;
    const Point3D t = midpoint - box_center;

    // Check axis overlap
    for (int i = 0; i < 3; ++i) {
        const double e = box_half[i] + abs_dir[i];
        if (std::abs(t[i]) > e) return false;
    }

    // Check cross axes
    for (int i = 0; i < 3; ++i) {
        const int a = (i + 1) % 3;
        const int b = (i + 2) % 3;
        const double radius = box_half[a] * std::abs(dir[b]) + 
                            box_half[b] * std::abs(dir[a]);
        const double distance = std::abs(t[a] * dir[b] - t[b] * dir[a]);
        if (distance > radius) return false;
    }

    return true;
}

std::vector<size_t> RRTStar::find_near_nodes(const Point3D& point) const {
    return radius_search(point, config_.rewire_radius);
}

std::vector<Point3D> RRTStar::trace_path(size_t end_index) const {
    std::vector<Point3D> path;
    size_t current_idx = end_index;
    
    while (true) {
        path.push_back(nodes_[current_idx].point);
        if (current_idx == 0) break; // Reached start node
        current_idx = nodes_[current_idx].parent;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Point3D> RRTStar::smooth_path(const std::vector<Point3D>& path) const {
    // Basic simplification - remove colinear points
    std::vector<Point3D> smoothed;
    if (path.size() < 3) return path;

    smoothed.push_back(path[0]);
    for (size_t i = 1; i < path.size() - 1; ++i) {
        const Point3D& prev = smoothed.back();
        const Point3D& next = path[i+1];
        if (!is_collision_free(prev, next)) {
            smoothed.push_back(path[i]);
        }
    }
    smoothed.push_back(path.back());

    // Simple linear interpolation for demonstration
    std::vector<Point3D> final_path;
    const double step = config_.step_size * 0.5;
    for (size_t i = 0; i < smoothed.size() - 1; ++i) {
        const Point3D& current = smoothed[i];
        const Point3D& next = smoothed[i+1];
        const double dist = current.distance(next);
        const int steps = std::ceil(dist / step);
        
        for (int j = 0; j <= steps; ++j) {
            const double t = static_cast<double>(j) / steps;
            final_path.push_back(current + (next - current) * t);
        }
    }
    
    return final_path;
}

std::vector<Point3D> RRTStar::plan() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        const auto q_rand = random_sample();
        const auto nearest_idx = nearest_node(q_rand);
        const auto& q_near = nodes_[nearest_idx].point;
        const auto q_new = steer(q_near, q_rand);

        if (is_collision_free(q_near, q_new)) {
            const auto near_indices = find_near_nodes(q_new);
            
            // Find best parent
            double min_cost = std::numeric_limits<double>::max();
            size_t best_parent = nearest_idx;
            
            for (const auto idx : near_indices) {
                const auto& node = nodes_[idx];
                if (is_collision_free(node.point, q_new)) {
                    const double cost = node.cost + node.point.distance(q_new);
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_parent = idx;
                    }
                }
            }

            // Add new node
            nodes_.push_back({q_new, best_parent, min_cost});
            rebuild_kd_tree();

            // Rewiring
            for (const auto idx : near_indices) {
                auto& node = nodes_[idx];
                const double cost_via_new = min_cost + q_new.distance(node.point);
                if (cost_via_new < node.cost && is_collision_free(q_new, node.point)) {
                    node.parent = nodes_.size() - 1;
                    node.cost = cost_via_new;
                }
            }

            // Check goal proximity
            if (q_new.distance(goal_) <= config_.goal_threshold) {
                auto path = trace_path(nodes_.size() - 1);
                if (is_collision_free(path.back(), goal_)) {
                    path.push_back(goal_);
                }
                return smooth_path(path);
            }
        }
    }
    return {};
}