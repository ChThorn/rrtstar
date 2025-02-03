#include "rrt_star.h"
#include <iostream>

int main() {
    using namespace rrt;

    std::vector<Obstacle> obstacles = {
        {{3, 3, 2}, {7, 7, 4}},
        {{2, 5, 5}, {4, 7, 8}},
        {{6, 2, 3}, {8, 4, 6}}
    };

    RRTStar::Config config;
    config.step_size = 0.7;
    config.max_iterations = 5000;
    config.goal_threshold = 0.7;

    RRTStar planner(
        {1, 1, 1},   // Start
        {9, 9, 9},   // Goal
        obstacles,
        config
    );

    const auto path = planner.plan();
    std::cout << "Path contains " << path.size() << " points\n";
    if (!path.empty()) {
        std::cout << "Final point: (" << path.back().x << ", "
                  << path.back().y << ", " << path.back().z << ")\n";
    }
    return 0;
}