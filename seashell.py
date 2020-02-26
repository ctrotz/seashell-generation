import math

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from mpl_toolkits.mplot3d import Axes3D


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

    def helicospiral(self, t: float) -> np.ndarray:
        x = self.__base_radius * pow(self.__base, t) * np.cos(t)
        y = self.__base_radius * pow(self.__base, t) * np.sin(t)
        z = self.__base_z * pow(self.__base, t)
        return np.array([x, y, z])

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
        return np.array([x, y, z])

    def generate_curve_sample(self, f: Callable[[float], np.ndarray], t: float, s: float):
        pt = f(s)
        scale = pow(self.__base, t)
        pt *= scale
        rot_mat = self.__frenet_frame(t).transpose()
        pt = np.matmul(rot_mat, pt)
        # print(pt)
        pt += self.helicospiral(t)

        return pt

    def generate_curve_samples(self, f: Callable[[float], np.ndarray], t: float, n: int) -> list:
        ss = np.linspace(0, 2 * np.pi, n)
        samples = []
        for s in ss:
            samples.append(a.generate_curve_sample(f, t, s))
        return samples

    def generate_vertices_faces(self, f: Callable[[float], np.ndarray], t_max: float, n: int) -> tuple:
        faces = []
        cur = self.generate_curve_samples(f, 0, n)  # first samples
        vertices = cur
        ts = np.linspace(0, t_max, math.floor(t_max/self.__increment))[1:]
        for i in range(1, len(ts)):
            cur = self.generate_curve_samples(f, ts[i], n)
            faces.extend(self.generate_faces(n, i))
            vertices.extend(cur)

        return vertices, faces

    def generate_faces(self, n, i):
        assert i > 0
        prev_offset = (i - 1) * n
        cur_offset = i * n
        faces = []
        for j in range(n - 1):
            v1_cur = j + 1 + cur_offset
            v2_cur = j + cur_offset
            v1_prev = j + 1 + prev_offset
            v2_prev = j + prev_offset
            faces.append([v1_cur, v1_prev, v2_cur])
            faces.append([v2_cur, v1_prev, v2_prev])
        v1_cur = cur_offset
        v2_cur = (n-1) + cur_offset
        v1_prev = prev_offset
        v2_prev = (n-1) + prev_offset
        faces.append([v1_cur, v1_prev, v2_cur])
        faces.append([v2_cur, v1_prev, v2_prev])

        return faces
if __name__ == "__main__":
    a = Seashell(0.04, 1.9, 1.007, 0.174533)

    def func(x): return np.array([np.cos(x), np.sin(x), 0]) * (1 + (1/10.0) * np.sin(10 * x))
    vertices, faces = a.generate_vertices_faces(func, 30 * np.pi, 30)

    file = open("test4.obj", "w")
    for vertex in vertices:
        file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
    for face in faces:
        file.write(f'f {face[0]} {face[1]} {face[2]}\n')
    file.close()


    # pts = []
    # samples = []
    # counter = 0
    # for t in ts:
    #     pts.append(a.helicospiral(t))
    #     counter += 1
    #     if counter == 20:
    #         for s in ss:
    #             samples.append(a.generate_sample(func, t, s))
    #         counter = 0
    #
    # pts = np.array(pts)
    #
    # x = pts[:, 0]
    # y = pts[:, 1]
    # z = pts[:, 2]
    #
    #
    # plt.rcParams['legend.fontsize'] = 10
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # ax.plot(x, y, z, label='parametric curve')
    # samples = np.array(samples)
    # print(samples.shape)
    # x = samples[:, 0]
    # y = samples[:, 1]
    # z = samples[:, 2]
    #
    # ax.plot(x, y, z, label='samples')
    #
    # ax.legend()
    #
    # plt.show()
    # # for angle in range(0, 360):
    # #     ax.view_init(30, angle)
    # #     plt.draw()
    # #     plt.pause(.001)