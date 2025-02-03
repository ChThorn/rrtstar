import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree

class RRT3DEnhanced:
    def __init__(self, start, goal, obstacles, 
                 bounds=(10, 10, 10),
                 step_size=0.5, 
                 max_iter=2000,
                 goal_bias=0.1,
                 goal_threshold=0.5,
                 smooth_iter=100,
                 visualize=True,
                 safety_margin=0.3):
        
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = np.array(bounds)
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.goal_threshold = goal_threshold
        self.smooth_iter = smooth_iter
        self.visualize = visualize
        self.safety_margin = safety_margin
        self.obstacles = [(np.array(o[0]), np.array(o[1])) for o in obstacles]
        self.nodes = [self.start]
        self.parents = {tuple(self.start): None}
        
        if self.visualize:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._init_visuals()

    def _init_visuals(self):
        self.ax.set_xlim(0, self.bounds[0])
        self.ax.set_ylim(0, self.bounds[1])
        self.ax.set_zlim(0, self.bounds[2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("3D RRT* Path Planning")
        
        for obs in self.obstacles:
            self._plot_cuboid(obs[0], obs[1], color='gray', alpha=0.3)
            
        self.ax.scatter(*self.start, c='green', s=100, marker='o', label='Start')
        self.ax.scatter(*self.goal, c='red', s=100, marker='^', label='Goal')
        self.ax.legend()

    def _plot_cuboid(self, min_point, max_point, color='gray', alpha=0.5):
        x_min, y_min, z_min = min_point
        x_max, y_max, z_max = max_point

        x = [x_min, x_max]
        y = [y_min, y_max]
        z = [z_min, z_max]

        X, Y = np.meshgrid(x, y)
        self.ax.plot_surface(X, Y, np.full_like(X, z_min), color=color, alpha=alpha)
        self.ax.plot_surface(X, Y, np.full_like(X, z_max), color=color, alpha=alpha)

        X, Z = np.meshgrid(x, z)
        self.ax.plot_surface(X, np.full_like(X, y_min), Z, color=color, alpha=alpha)
        self.ax.plot_surface(X, np.full_like(X, y_max), Z, color=color, alpha=alpha)

        Y, Z = np.meshgrid(y, z)
        self.ax.plot_surface(np.full_like(Y, x_min), Y, Z, color=color, alpha=alpha)
        self.ax.plot_surface(np.full_like(Y, x_max), Y, Z, color=color, alpha=alpha)

    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _random_sample(self):
        if np.random.rand() < self.goal_bias:
            return self.goal
        return np.random.uniform(0, self.bounds, size=3)

    def _nearest_node(self, target):
        return min(self.nodes, key=lambda node: self.distance(node, target))

    def _steer(self, q_nearest, q_target):
        direction = q_target - q_nearest
        distance = self.distance(q_nearest, q_target)
        if distance <= self.step_size:
            return q_target
        return q_nearest + (direction / distance) * self.step_size

    def _is_point_in_obstacle(self, point):
        for (min_p, max_p) in self.obstacles:
            if np.all(point >= min_p) and np.all(point <= max_p):
                return True
        return False

    def _is_collision_free(self, q1, q2):
        for (min_p, max_p) in self.obstacles:
            expanded_min = min_p - self.safety_margin
            expanded_max = max_p + self.safety_margin
            if self._line_aabb_intersection(q1, q2, expanded_min, expanded_max):
                return False
        return True
    
    def _line_aabb_intersection(self, p1, p2, box_min, box_max):
        dir_vec = (p2 - p1) * 0.5
        midpoint = (p1 + p2) * 0.5
        extent = np.abs(dir_vec)
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)

        box_center = (box_min + box_max) * 0.5
        box_half = (box_max - box_min) * 0.5
        t = midpoint - box_center

        for i in range(3):
            e = box_half[i] + extent[i]
            if abs(t[i]) > e:
                return False

        for i in range(3):
            a = (i+1)%3
            b = (i+2)%3
            radius = box_half[a] * abs(dir_vec[b]) + box_half[b] * abs(dir_vec[a])
            distance = abs(t[a] * dir_vec[b] - t[b] * dir_vec[a])
            if distance > radius:
                return False

        return True

    def plan(self):
        for _ in range(self.max_iter):
            q_rand = self._random_sample()
            q_near = self._nearest_node(q_rand)
            q_new = self._steer(q_near, q_rand)
            
            if self._is_collision_free(q_near, q_new):
                self.nodes.append(q_new)
                self.parents[tuple(q_new)] = tuple(q_near)
                
                if self.visualize:
                    self.ax.plot([q_near[0], q_new[0]],
                                [q_near[1], q_new[1]],
                                [q_near[2], q_new[2]], 
                                'b-', linewidth=0.5)
                    plt.pause(0.001)
                
                if self.distance(q_new, self.goal) <= self.goal_threshold:
                    path = self._trace_path(q_new)
                    if self._is_collision_free(q_new, self.goal):
                        path[-1] = self.goal
                    else:
                        path.append(self.goal)
                    
                    smoothed_path = self._smooth_path(path)
                    
                    if self.visualize:
                        self._plot_final_path(smoothed_path)
                        plt.ioff()
                        plt.show()
                    
                    return smoothed_path

        print("Maximum iterations reached")
        if self.visualize:
            plt.ioff()
            plt.show()
        return []

    def _trace_path(self, q_end):
        path = [q_end]
        current = tuple(q_end)
        while current is not None:
            parent = self.parents.get(current)
            if parent is not None:
                path.append(np.array(parent))
            current = parent
        return path[::-1]

    def _plot_final_path(self, path):
        if len(path) > 1:
            path_array = np.array(path)
            self.ax.plot(path_array[:,0], path_array[:,1], path_array[:,2],
                        'm-', linewidth=3, label='Optimized Path')
            self.ax.scatter(path_array[:,0], path_array[:,1], path_array[:,2],
                          c='orange', s=10, marker='*', alpha=0.5)
            self.ax.legend()
            self.fig.canvas.draw_idle()

class RRT3DStarEnhanced(RRT3DEnhanced):
    def __init__(self, *args, rewire_radius=2.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewire_radius = rewire_radius
        self.costs = {tuple(self.start): 0.0}
        self.kd_tree = KDTree(self.nodes)

    def _find_near_nodes(self, new_node):
        indices = self.kd_tree.query_ball_point(new_node, self.rewire_radius)
        return [self.nodes[i] for i in indices]

    def plan(self):
        for _ in range(self.max_iter):
            q_rand = self._random_sample()
            q_near = self._nearest_node(q_rand)
            q_new = self._steer(q_near, q_rand)
            
            if self._is_collision_free(q_near, q_new):
                near_nodes = self._find_near_nodes(q_new)
                
                min_cost = float('inf')
                best_parent = q_near
                
                for node in near_nodes:
                    if self._is_collision_free(node, q_new):
                        cost = self.costs[tuple(node)] + self.distance(node, q_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_parent = node
                
                self.nodes.append(q_new)
                self.parents[tuple(q_new)] = tuple(best_parent)
                self.costs[tuple(q_new)] = min_cost
                self.kd_tree = KDTree(self.nodes)
                
                for node in near_nodes:
                    if np.array_equal(node, best_parent):
                        continue
                    cost_via_new = self.costs[tuple(q_new)] + self.distance(q_new, node)
                    if (self.costs[tuple(node)] > cost_via_new and 
                        self._is_collision_free(q_new, node)):
                        self.parents[tuple(node)] = tuple(q_new)
                        self.costs[tuple(node)] = cost_via_new
                
                if self.visualize:
                    self.ax.plot([best_parent[0], q_new[0]],
                                [best_parent[1], q_new[1]],
                                [best_parent[2], q_new[2]], 
                                'b-', linewidth=0.5)
                    plt.pause(0.001)
                
                if self.distance(q_new, self.goal) <= self.goal_threshold:
                    path = self._trace_path(q_new)
                    if self._is_collision_free(path[-1], self.goal):
                        path[-1] = self.goal
                    else:
                        path.append(self.goal)
                    
                    smoothed_path = self._smooth_path(path)
                    
                    if not np.allclose(smoothed_path[-1], self.goal, atol=self.goal_threshold):
                        smoothed_path.append(self.goal)
                    
                    if self.visualize:
                        self._plot_final_path(smoothed_path)
                        plt.ioff()
                        plt.show()
                    
                    return smoothed_path

        print("Maximum iterations reached")
        if self.visualize:
            plt.ioff()
            plt.show()
        return []
    
    def _smooth_path(self, path):
        if not path:
            return path

        if not np.allclose(path[-1], self.goal, atol=self.goal_threshold):
            path.append(self.goal.copy())

        simplified = []
        for point in path:
            if len(simplified) < 2:
                simplified.append(point)
                continue
            if not self._is_colinear(simplified[-2], simplified[-1], point):
                simplified.append(point)
        simplified.append(self.goal)

        path_length = sum(self.distance(p1, p2) for p1, p2 in zip(simplified[:-1], simplified[1:]))
        smooth_points = max(10, min(int(path_length / self.step_size) * 2, 100))

        try:
            tck, u = splprep(np.array(simplified).T, s=0, k=min(3, len(simplified)-1), nest=-1)
            new_u = np.linspace(0, 1, smooth_points)
            smooth_path = splev(new_u, tck)
            smooth_path = np.column_stack(smooth_path)
        except Exception as e:
            print(f"Smoothing failed: {e}, using simplified path")
            smooth_path = simplified

        final_path = [smooth_path[0]]
        for point in smooth_path[1:]:
            if self.distance(point, self.goal) < self.goal_threshold:
                final_path.append(self.goal)
                break
            
            if self._is_collision_free(final_path[-1], point):
                final_path.append(point)
            else:
                dir_vec = (point - final_path[-1])
                dist = self.distance(final_path[-1], point)
                steps = max(2, int(dist / self.step_size))
                for t in np.linspace(0, 1, steps)[1:]:
                    intermediate = final_path[-1] + t * dir_vec
                    if self._is_collision_free(final_path[-1], intermediate):
                        final_path.append(intermediate)
                    else:
                        break

        if not np.allclose(final_path[-1], self.goal, atol=self.goal_threshold):
            if self._is_collision_free(final_path[-1], self.goal):
                final_path.append(self.goal)
            else:
                return [p for p in path if self._is_collision_free(final_path[-1], p)] + [self.goal]

        return np.unique(final_path, axis=0).tolist()

    def _is_colinear(self, p1, p2, p3, tol=1e-3):
        if p1 is None or p2 is None or p3 is None:
            return False
        v1 = p2 - p1
        v2 = p3 - p2
        cross = np.linalg.norm(np.cross(v1, v2))
        return cross < tol

if __name__ == "__main__":
    obstacles = [
        ((3, 3, 2), (7, 7, 4)),
        ((2, 5, 5), (4, 7, 8)),
        ((6, 2, 3), (8, 4, 6))
    ]
    
    rrt_star = RRT3DStarEnhanced(
        start=(1, 1, 1),
        goal=(9, 9, 9),
        obstacles=obstacles,
        bounds=(10, 10, 10),
        step_size=0.7,
        max_iter=5000,
        smooth_iter=100,
        visualize=True,
        safety_margin=0.3,
        rewire_radius=3.0,
        goal_threshold=0.7
    )

    path = rrt_star.plan()
    print(f"Optimized path points: {len(path)}")
    print(f"Final position: {path[-1] if path else 'No path found'}")