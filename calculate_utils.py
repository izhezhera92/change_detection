from geopy.distance import geodesic
import geocoder
from math import cos, sin, pi, pow, sqrt, radians, atan2, degrees, tan
import numpy as np
import matplotlib.pyplot as plt
import requests
#import pymavlink
from shapely.geometry.polygon import LinearRing

class Geodesy(object):

    def __init__(self):
        self.earth_radius = 6371000

    def point_of_intersection_L2L(k1: float, k2: float, b1: float, b2: float):
        """
        Method of 2 lines intersection coordinate calculation
        input  -> k1, k2, b1, b2
        output -> intersection coordinate
        """
        M = np.array([[k1, -1], [k2, -1]])
        v = np.array([-b1, -b2])
        return np.linalg.solve(M, v)

    def lineEquationFromPoints(p1, p2):
        """
        Get equation from 2 points
        input  -> p1, p2: points (x, y)
        output -> k, b: slope and intercept
        """
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - k * p1[0]
        return round(k, 4), round(b, 4)

    def get_xy_by_course_and_distance(
        self, x:int = 0, y:int = 0,
        l: float = 0.0,
        azimuth: float = 0.0)-> tuple:
        xx = 0
        yy = 0
        if x > 0 and y > 0:
            xx = x + (l * cos(azimuth))
            yy = y + (l * sin(azimuth))
        return xx, yy

    def get_lat_lon_by_course_and_distance(
        self, lat: float = 0.0,
        lon: float = 0.0, 
        l: float = 0.0, 
        azimuth: float = 0.0)-> tuple:
        new_lat = lat + l * cos(azimuth * pi / 180) / (self.earth_radius * pi / 180)
        new_lon = lon + l * sin(azimuth * pi / 180) / cos(lat * pi / 180) / (self.earth_radius * pi / 180)
        print("> ", new_lat, new_lon)
        return (new_lat, new_lon)

    def get_2d_course_angle_image_coord_sys(self, 
        x1:int = 0, y1:int = 0, 
        x2:int = 0, y2:int = 0
        ) -> float:
        """
        Method for 2d course angle calculation.
        Zero direction angle in the top of image, over on 1st point.
        Using image coordinate system.

        input  -> int: x1, y1, x2, y2
        output -> float: course_2d
        """
        x1 = x1 - x2
        y1 = y1 - y2
        x2, y2 = 0, 0

        angle = degrees(atan2(x2-x1, y2-y1))

        if angle <= 0 and angle >= -180:
            return -angle

        elif angle >= 0 and angle <= 180:
            return 360 - angle

        
    def get_real_course_angle(self, 
        course_2d : float = 0.0, 
        mavlink_course_angle : float = 0.0)-> float:
        """
        Method for the real cource angle calculation.
        Consist from magnetic course from mavlink and wear angle from image.
        input -> float:  course_2d, mavlink_course_angle
        output -> float: real_course_angle
        """
        real_course_angle = mavlink_course_angle + course_2d
        if real_course_angle >= 360:
            return real_course_angle - 360

        return real_course_angle

    def course_angle(self, current_angle, previous_angle):
        if previous_angle is None:
            return current_angle
        else:
            course_change = current_angle - previous_angle

            if course_change > 180:
                return course_change - 360
            elif course_change < -180:
                return course_change + 360
            else:
                return course_change

    def calculate_distance_xy(self, x1, y1, x2, y2):
        """
        Method for dist calculation between 2 points in 2D
        input    -> p1, p2: (x1,y1), (x2,y2)
        output   -> dist 
        """
        if x1 and y1 and x2 and y2:
            return sqrt(pow(abs(x1 - x2), 2) + pow(abs(y1 - y2),2))

        else:
            return None

    def calculate_distance_wgs84(self, lat1, lon1, lat2, lon2):
        if lat1 and lon1 and lat2 and lon2:
            coords_1 = (lat1, lon1)
            coords_2 = (lat2, lon2)
            distance = geodesic(coords_1, coords_2).meters
            return distance
        else:
            return None

    def calculate_wear_angle(self, lat1, lon1, lat2, lon2):
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        dlon = lon2 - lon1
        x = cos(lat2) * sin(dlon)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

        azimuth = atan2(x, y)
        azimuth = degrees(azimuth)

        if azimuth < 0:
            azimuth += 360

        return azimuth

    def get_height(self, latitude, longitude):
        """
        Method for altitude getting
        """
        url = f"https://api.opentopodata.org/v1/aster30m?locations={latitude},{longitude}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                elevation = data['results'][0]['elevation']
                return elevation

            else:
                print("Error:", data['status'])
        else:
            print("Error:", response.status_code)
        return None



    def get_meter_per_pixel(self, alt_relative = 150.0, 
            camera_angle_vertical = 70.0, 
            camera_angle_horizontal = 86.0, 
            angle = 0.0,
            distance = 0.0,
            img = None):

        if img is not None:
            absolut_image_height_m = 2 * tan(camera_angle_vertical / 2 * pi / 180) * alt_relative
            absolut_image_wight_m = 2 * tan(camera_angle_horizontal / 2 * pi / 180) * alt_relative
            image_height, image_width, _ = img.shape

            k_x_meters = absolut_image_wight_m / image_width
            k_y_meters = absolut_image_height_m / image_height

            #print("K: ", k_x_meters, k_y_meters)

            #delta_x_pos_meters = abs(old_pos[0] - curr_pos[0]) * k_x_meters
            #delta_y_pos_meters = abs(old_pos[1] - curr_pos[1]) * k_y_meters

            #movement = sqrt(pow(delta_x_pos_meters, 2) + pow(delta_y_pos_meters, 2))        
        return k_x_meters, k_y_meters








