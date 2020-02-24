import numpy as np


class Seashell(object):
    def __init__(self, r0: float, z0: float, growth: float, d_theta: float):
        self.__base_radius = r0
        self.__base_z = z0
        self.__growth_rate = growth
        self.__increment = d_theta
        self.__base = pow(growth, 1/d_theta)

    def __frenet_frame(self, t: float) -> np.ndarray:
        vec1 = self.__first_partial(t)
        vec1 /= np.linalg.norm(vec1)

        vec3 = np.cross(vec1, self.__second_partial(t))
        vec3 /= np.linalg.norm(vec3)

        vec2 = np.cross(vec3, vec1)

        return np.array([vec1, vec2, vec3])

    def __first_partial(self, t: float) -> np.ndarray:
        x = self.__base_radius * (t * pow(self.__base, t - 1) * np.cos(t) - pow(self.__base, t) * np.sin(t))
        y = self.__base_radius * (t * pow(self.__base, t - 1) * np.sin(t) + pow(self.__base, t) * np.cos(t))
        z = t * self.__base_z * pow(self.__base, t - 1)
        return np.array([x, y, z])

    def __second_partial(self, t: float) -> np.ndarray:
        x = self.__base_radius * pow(self.__base, t) * ((pow(np.log(self.__base), 2) - 1) * np.cos(t)
                                                        - 2 * np.log(self.__base) * np.sin(t))
        y = self.__base_radius * pow(self.__base, t) * ((pow(np.log(self.__base), 2) - 1) * np.sin(t)
                                                        + 2 * np.log(self.__base) * np.cos(t))
        z = self.__base_z * pow(self.__base, t) * pow(np.log(self.__base), 2)
        return np.arraY([x, y, z])