import numpy as np
from metadrive.engine.core.draw import ColorSphereNodePath
from metadrive.utils.interpolating_line import InterpolatingLine


class WaypointTrajectory(object):
    def __init__(self):
        self.local_waypoints = np.zeros([1 + 16, 2])  # 16 predict points + 1 current point
        self.vehicle_position = np.zeros([2])  # forward,left
        self.vehicle_heading = 0.0

    @staticmethod
    def create_vehicle_line(vehicle_position, vehicle_heading):
        vehicle_head_pos = vehicle_position + np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)]) * 1.0
        vehicle_line = InterpolatingLine([vehicle_position, vehicle_head_pos])
        return vehicle_line

    def set_local_waypoint(self, waypoints, vehicle_position, vehicle_heading):
        extra_waypoints = np.vstack((np.zeros([1, 2]), np.asarray(waypoints).reshape(-1, 2)))
        self.local_waypoints = extra_waypoints
        self.vehicle_position = np.asarray(vehicle_position)
        self.vehicle_heading = vehicle_heading

    def local_waypoint_to_global(self, waypoints, vehicle_position, vehicle_heading):
        self.set_local_waypoint(waypoints, vehicle_position, vehicle_heading)

        vehicle_line = self.create_vehicle_line(vehicle_position, vehicle_heading)
        global_waypoints = np.zeros_like(self.local_waypoints)
        for ind in range(self.local_waypoints.shape[0]):
            global_waypoints[ind] = vehicle_line.position(self.local_waypoints[ind][0], -self.local_waypoints[ind][1])
        return global_waypoints

    def set_global_waypoint(self, waypoints, vehicle_position, vehicle_heading):
        extra_waypoints = np.vstack((np.asarray(vehicle_position), np.asarray(waypoints).reshape(-1, 2)))
        vehicle_line = self.create_vehicle_line(vehicle_position, vehicle_heading)
        self.local_waypoints = np.zeros_like(extra_waypoints)
        for ind in range(extra_waypoints.shape[0]):
            long, lat = vehicle_line.local_coordinates(extra_waypoints[ind])
            self.local_waypoints[ind] = np.array([long, -lat])  # Frenet coordinates and world axis are different
        self.vehicle_position = np.asarray(vehicle_position)
        self.vehicle_heading = vehicle_heading

    def global_waypoint_to_local(self, waypoints, vehicle_position, vehicle_heading):
        self.set_global_waypoint(waypoints, vehicle_position, vehicle_heading)
        return self.local_waypoints

    def draw_in_sim_local(self, drawer: ColorSphereNodePath, rgba=np.array([0, 105 / 255, 180 / 255, 1])):
        """
        call after set waypoint, before step
        """
        points = []
        colors = []
        for local_pos in self.local_waypoints:
            points.append((local_pos[0], local_pos[1], 0.5))
            colors.append(np.clip(rgba, 0., 1.0))
        # drawer.reset()
        drawer.draw_points(points, colors)

    def draw_in_sim_global(self, drawer: ColorSphereNodePath):
        """
        call after set waypoint, before step, using global drawer
        """
        global_waypoints = self.local_waypoint_to_global(self.local_waypoints, self.vehicle_position,
                                                         self.vehicle_heading)
        points = []
        colors = []
        for global_pos in global_waypoints:
            points.append((global_pos[0], global_pos[1], 0.5))
            colors.append(np.clip(np.array([0, 105 / 255, 180 / 255, 1]), 0., 1.0))
        drawer.reset()
        drawer.draw_points(points, colors)

    def get_ahead_position(self, vehicle_back_position, vehicle_look_ahead_dis):
        """
        call after set waypoint
        """
        global_waypoints = self.local_waypoint_to_global(self.local_waypoints, self.vehicle_position,
                                                         self.vehicle_heading)
        poly_line = InterpolatingLine(global_waypoints)
        final_s = poly_line.local_coordinates(poly_line._end_points[-1])[0]
        now_s = 0.1
        point = poly_line.position(now_s, 0)
        while vehicle_look_ahead_dis > poly_line.points_distance(vehicle_back_position, poly_line.position(now_s, 0)):
            now_s += 0.1
            point = poly_line.position(now_s, 0)
            if now_s > final_s:
                break
        return point


if __name__ == "__main__":
    traj = WaypointTrajectory()
    global_waypoint = traj.local_waypoint_to_global(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
                                                    np.array([5.0, 0.0]), np.pi / 6)
    ahead_point = traj.get_ahead_position(np.array([4.0, 0.0]), 1.0)
    print(global_waypoint)
    print(ahead_point)
    print("=====")

    local_waypoint = traj.global_waypoint_to_local(np.array([[6.0, 0.0], [7.0, 0.0], [8.0, 0.0]]),
                                                    np.array([5.0, 0.0]), np.pi / 6)
    ahead_point = traj.get_ahead_position(np.array([4.0, 0.0]), 1.0)
    print(local_waypoint)
    print(ahead_point)