import sys
import numpy as np
import scipy
import nibabel as nib
import pandas as pd
import argparse
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import simnibs
import math

class TmsAtlasPoints:
    def __init__(self, file_path, points=None):
        # self.points = o3d.geometry.PointCloud()
        self.points = np.array([])
        self.normals = np.array([])
        self.queryTree = None
        if file_path.endswith('nii') or file_path.endswith('nii.tar.gz') or file_path.endswith('nii.gz'):
            nii_ = nib.load(file_path)
            affine = nii_.affine
            grid_ = nii_.get_fdata()
            coord = np.nonzero(grid_)
            coord = np.stack(coord).T
            grid_list = np.array(coord)
            length = len(grid_list)
            affine = np.array(affine)
            grid_coord = np.concatenate([coord, np.ones([length, 1])], axis=1)
            SUB_coord = grid_coord.dot(affine.T)      
            SUB_coord = SUB_coord.round(2)
            self.queryTree = scipy.spatial.cKDTree(self.points)
            # self.points.points = o3d.utility.Vector3dVector(SUB_coord[:, :3])
            # self.points.normals = o3d.utility.Vector3dVector(np.zeros(
            #     (1, 3)))  # invalidate existing normals
            self.points = SUB_coord[:, :3]
            # self.points.estimate_normals()
        elif file_path.endswith('.msh') and points is None:
            msh = simnibs.read_msh(file_path)
            msh_surf = msh.crop_mesh(elm_type=2)
            skin = msh_surf.crop_mesh([5, 1005])
            # skin = simnibs_interface.ta_get_skin_surface(msh)
            SUB_coord = skin.nodes.node_coord
            SUB_normal = skin.nodes_normals().value
            self.points = SUB_coord[:, :3]
            self.normals = SUB_normal
            self.queryTree = scipy.spatial.cKDTree(self.points)
            # self.points.normalize_normals = o3d.utility.Vector3dVector(skin.nodes_normals())
            # self.points = SUB_coord[:, :3]
            # self.normals = SUB_normal
            # o3d.visualization.draw_geometries([self.points], point_show_normal=True)
        elif file_path.endswith('.msh') and points is not None:
            msh = simnibs.read_msh(file_path)
            msh_surf = msh.crop_mesh(elm_type=2)
            skin = msh_surf.crop_mesh([5, 1005])
            SUB_coord = skin.nodes.node_coord
            SUB_normal = skin.nodes_normals().value
            m_tree = scipy.spatial.cKDTree(SUB_coord)
            dists, indexes = m_tree.query(points)
            self.points = SUB_coord[indexes, :3]
            self.normals = SUB_normal[indexes, :]
            self.queryTree = scipy.spatial.cKDTree(self.points)
        
            # self.points.normalize_normals = o3d.utility.Vector3dVector(self.normals)
            # o3d.visualization.draw_geometries([self.points], point_show_normal=True)



    def define_area(self, point1, point2, point3):
        """
        法向量    ：n={A,B,C}
        空间上某点：p={x0,y0,z0}
        点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)

        (Ax, By, Cz, D)代表：Ax + By + Cz + D = 0
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        point3 = np.asarray(point3)
        AB = np.asmatrix(point2 - point1)
        AC = np.asmatrix(point3 - point1)
        N = np.cross(AB, AC)  # 向量叉乘，求法向量
        # Ax+By+Cz
        Ax = N[0, 0]
        By = N[0, 1]
        Cz = N[0, 2]
        D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
        return Ax, By, Cz, D

    def cal_normal(self, Ax, By, Cz, D, p1, p2):
        """
        计算过点p:(p1+ p2)/2的且在平面Ax + By + Cz + D = 0上的，与p1-p2垂直的向量
        """
        p = (p1 + p2) / 2.0
        # 将p p1 p2 投影到平面Ax + By + Cz + D = 0上
        p_plane_t = (Ax * p[0] + By * p[1] + Cz * p[2] + D) / (Ax * Ax + By * By + Cz * Cz)
        p1_plane_t = (Ax * p1[0] + By * p1[1] + Cz * p1[2] + D) / (Ax * Ax + By * By + Cz * Cz)
        p2_plane_t = (Ax * p2[0] + By * p2[1] + Cz * p2[2] + D) / (Ax * Ax + By * By + Cz * Cz)
        p_plane = np.array([p[0] - Ax * p_plane_t, p[1] - By * p_plane_t, p[2] - Cz * p_plane_t])
        p1_plane = np.array([p1[0] - Ax * p1_plane_t, p1[1] - By * p1_plane_t, p1[2] - Cz * p1_plane_t])
        p2_plane = np.array([p2[0] - Ax * p2_plane_t, p2[1] - By * p2_plane_t, p2[2] - Cz * p2_plane_t])
        # 向量为p_n = p + (m, n, k)
        # 其中(m,n,k)为方向向量
        # p_n满足条件Ax + By + Cz + D = 0
        # (m,n,k)与p1-p2垂直
        # p1-p2.x p1-p2.y p1-p2.z    m     0
        # Ax      By      Cz      *  n  =  0
        #                            k  
        # 令 k = 1
        # p1-p2.x p1-p2.y    m    -p1-p2.z
        # Ax      By      *  n  = -Cz
        p1_p2 = p2_plane - p1_plane
        A = np.array([
            [p1_p2[0], p1_p2[1]],
            [Ax,       By]
        ])
        b = np.array([-p1_p2[2], -Cz])
        k = scipy.linalg.solve(A, b)
        k = np.array([k[0], k[1], 1.0])
        p_n = p + k
        return p_n

    def get_surface_line(self, _p1, _p2, _p3, dis=0.5):
        p1 = np.array(_p1)
        p2 = np.array(_p2)
        p3 = np.array(_p3)
        _, p1_index = self.queryTree.query(p1)
        _, p2_index = self.queryTree.query(p2)
        _, p3_index = self.queryTree.query(p3)
        p1 = self.points[p1_index, :]
        p2 = self.points[p2_index, :]
        p3 = self.points[p3_index, :]
        points_hull = np.asarray(self.points)#.points)#[np.where(np.asarray(self.points.points)[:, 2] > min_z)[0]]
        normal_hull = np.asarray(self.normals)#[np.where(np.asarray(self.points.points)[:, 2] > min_z)[0]]

        Ax, By, Cz, D = self.define_area(p1, p2, p3)
        N = np.array([Ax, By, Cz])
        mod_d = np.dot(points_hull, N) + D
        mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
        d = np.abs(mod_d) / mod_area 
        # surface = o3d.geometry.PointCloud()

        points = points_hull[np.where(d < dis)[0]]
        # 将points 投影到平面Ax + By + Cz + D = 0上
        p_plane_t = (Ax * points[:, 0] + By * points[:, 1] + Cz * points[:, 2] + D) / (Ax * Ax + By * By + Cz * Cz)
        points[:, 0] = points[:, 0] - Ax * p_plane_t
        points[:, 1] = points[:, 1] - By * p_plane_t
        points[:, 2] = points[:, 2] - Cz * p_plane_t
        # points = np.array([points[:, 0] - Ax * p_plane_t, points[:, 1] - By * p_plane_t, points[:, 2] - Cz * p_plane_t])
        normal_hull = normal_hull[np.where(d < dis)[0]]

        center = (np.array(p1) + np.array(p2))/2.0
        normals = points - center
        # 定义center-p1为起点
        start_n = p1 - center
        normal_dot = np.dot(normals, start_n)
        degree_cos = normal_dot / (np.linalg.norm(normals, axis=1) * np.linalg.norm(start_n))
        degree_cos[np.where(degree_cos > 1.0)[0]] = 1.0
        degree_cos[np.where(degree_cos < -1.0)[0]] = -1.0 
        degrees = np.arccos(degree_cos) * 180.0 / math.pi
        # 定义center-p3为正方向，去掉反方向的
        p_n = self.cal_normal(Ax, By, Cz, D, p1, p2)
        # surface = o3d.geometry.PointCloud()
        # positions_t = np.zeros(shape=(points.shape[0] + 2,3))
        # positions_t[:-2, :] = points
        # positions_t[-2, :] = center
        # positions_t[-1, :] = p_n
        # surface.points = o3d.utility.Vector3dVector(positions_t)
        # o3d.visualization.draw_geometries([surface])
        positive_norm = p_n - center
        positive_dot = np.dot(normals, positive_norm)
    
        points = points[np.where(positive_dot>=0)[0]]
        degrees = degrees[np.where(positive_dot>=0)[0]]
        normal_hull = normal_hull[np.where(positive_dot>=0)[0]]
        index = np.argsort(degrees)

        temp_points = points[index]
        temp_degrees = degrees[index]
        temp_normal_hull = normal_hull[index]

        new_points = None  # 坐标值
        new_degrees = None # 角度值
        new_normals = None # 法向量
        all_inters = np.array([], dtype=np.int64)
        # 去除距离异常值
        for i in range(0, 181, 1):
            start_i = 0
            end_i = len(temp_points)
            if len(np.where(temp_degrees >= i)[0]) > 0:
                start_i = np.where(temp_degrees >= i)[0][0]
            if len(np.where(temp_degrees >= i + 5)[0]) > 0:
                end_i = np.where(temp_degrees >= i + 5)[0][0]
            p_c = np.linalg.norm(temp_points[start_i:end_i, :] - center, axis=1)
            if len(p_c) == 0:
                continue
            inliers = np.where(p_c >= np.max(p_c) * 0.9)[0]
            # if len(inliers) != len(p_c):
            #     print(len(p_c)- len(inliers))
            if len(inliers) > 0:
                all_inters = np.union1d(all_inters, inliers + start_i)

        new_points = temp_points[all_inters]
        new_degrees = temp_degrees[all_inters]
        new_normals = temp_normal_hull[all_inters]
        arc = np.zeros(new_degrees.shape)
        

        # 计算p3所处位置
        p3_n = p3 - center
        normal_dot = np.dot(p3_n, start_n)
        degree_cos = normal_dot / (np.linalg.norm(p3_n) * np.linalg.norm(start_n))
        degrees = np.arccos(degree_cos) * 180.0 / math.pi
        # normal_cross = np.cross(p3_n, start_n)
        # if normal_cross[2] < 0:
        #     degrees = 360.0 - degrees
        p3_index = 0
        if len(np.where(new_degrees <= degrees)[0]) > 0:
            p3_index = np.where(new_degrees <= degrees)[0][-1]

        # 求弧线总长度
        arc_length = 0
        p3_arc = 0
        for i in range(len(new_points)):
            if i == 0:
                continue
            if i == p3_index:
                p3_arc = arc_length + np.linalg.norm(p3 - new_points[i-1, :])
            arc_length = arc_length + np.linalg.norm(new_points[i, :] - new_points[i-1, :])
            arc[i] = arc_length

        return [new_points, new_degrees, arc, p3_arc, degrees, new_normals]

    def cal_equator(self, p1, p2, p3, dis):
        self.dis = dis
        self.equator_points, self.equator_degrees ,self.equator_arc, _, _, _ = self.get_surface_line(p1, p2, p3, dis)
        # eq = o3d.geometry.PointCloud()
        # eq.points = o3d.utility.Vector3dVector(self.equator_points)
        # o3d.visualization.draw_geometries([eq], 
        #                             zoom=1.412,
        #                             front=[1.0, 0.0, 0.0],
        #                             lookat=[0.0, 1.0, 0.0],
        #                             up=[-0.0694, -0.9768, 0.2024])
        return self.equator_arc[-1]

    def cal_longitude(self, al, ar, p):
        self.al = al
        self.ar = ar
        self.points_long, _, arc, p_long, p_degree_l, _ = self.get_surface_line(al, ar, p, self.dis)
        
        # eq = o3d.geometry.PointCloud()
        # eq.points = o3d.utility.Vector3dVector(self.points_long)
        # o3d.visualization.draw_geometries([eq], 
        #                             zoom=1.412,
        #                             front=[0.0, 0.0, 1.0],
        #                             lookat=[0.0, 1.0, 0.0],
        #                             up=[-0.0694, -0.9768, 0.2024])
        # p经度值
        p_long = p_long / arc[-1]
        # 计算经度与赤道的交点
        distance = scipy.spatial.distance.cdist(self.equator_points, self.points_long)
        index_eq, index_lg = np.unravel_index(distance.argmin(), distance.shape)
        # p赤道值
        p_eq = self.equator_arc[index_eq] / self.equator_arc[-1]
        p_degree_e = self.equator_degrees[index_eq]
        # 赤道角度需要插值
        if index_eq > 0 and index_eq < len(self.equator_degrees) - 1:
            p_degree_e_0 = self.equator_degrees[index_eq - 1]
            p_degree_e_1 = self.equator_degrees[index_eq]
            p_degree_e_2 = self.equator_degrees[index_eq + 1]
            d0 = distance[index_eq - 1, index_lg]
            d1 = distance[index_eq, index_lg]
            d2 = distance[index_eq + 1, index_lg]
            yd = np.array([d0, d1, d2])
            xd = np.array([0.0, 1.0, 2.0])
            dd = np.polyfit(xd, yd, 2)
            dmin = -dd[1]/dd[0]/2.0
            if dmin >= 1.0 and dmin <= 2.0:
                p_degree_e = p_degree_e_1 + (p_degree_e_2 - p_degree_e_1) * (dmin - 1)
            elif dmin >= 0.0 and dmin < 1.0:
                p_degree_e = p_degree_e_0 + (p_degree_e_1 - p_degree_e_0) * (dmin)
        return [p_eq, p_long,  p_degree_e, p_degree_l, arc[-1]]

    def cal_mni_zhu(self, p_eq, p_long):
        p_arc_eq = self.equator_arc[-1] * p_eq
        p_index_eq = np.where(self.equator_arc <= p_arc_eq)[0][-1]
        # 插值
        p1 = self.equator_points[p_index_eq]
        p2 = self.equator_points[p_index_eq + 1]
        p_inter = (p_arc_eq - self.equator_arc[p_index_eq]) / (self.equator_arc[p_index_eq + 1] - self.equator_arc[p_index_eq])
        p3 = p1 + (p2 - p1) * p_inter

        points, _, arc, p_arc_l, _, _ = self.get_surface_line(self.al, self.ar, p3)
        p_arc_long = arc[-1] * p_long
        p_index_long = np.where(arc <= p_arc_long)[0][-1]
        # 插值
        p1 = points[p_index_long]
        p2 = points[p_index_long + 1]
        p_inter = (p_arc_long - arc[p_index_long]) / (arc[p_index_long + 1] - arc[p_index_long])
        p_f = p1 + (p2 - p1) * p_inter
        return p_f

    def cal_point(self, p_eq_angle, p_long_angle):
        angle = p_eq_angle
        # 找到赤道点
        indexs = np.where(self.equator_degrees >= angle)[0]
        i1 = 0
        i2 = -1
        if len(indexs) > 0:
            if indexs[0] > 0:
                i1 = indexs[0] - 1
                if i1 + 1< len(self.equator_degrees):
                    i2 = i1 + 1
                else:
                    i2 = i1
            else:
                i1 = indexs[0]
                i2 = i1
        else:
            i1 = -1
            i2 = -1
        # 插值
        p1 = self.equator_points[i1]
        p2 = self.equator_points[i2]
        p_inter = 0.0
        if i2 != i1:
            p_inter = (angle - self.equator_degrees[i1]) / (self.equator_degrees[i2] - self.equator_degrees[i1])
        p_f = p1 + (p2 - p1) * p_inter
        # 确定经度线
        points_lg, angles_lg, _, _, _, points_n = self.get_surface_line(self.al, self.ar, p_f, self.dis)
       
        angle_lg = p_long_angle
        # 找到经度点
        indexs_lg = np.where(angles_lg >= angle_lg)[0]
        i1 = 0
        i2 = -1
        if len(indexs_lg) > 0:
            if indexs_lg[0] > 0:
                i1 = indexs_lg[0] - 1
                if i1 + 1< len(angles_lg):
                    i2 = i1 + 1
                else:
                    i2 = i1
            else:
                i1 = indexs_lg[0]
                i2 = i1
        else:
            i1 = -1
            i2 = -1
        # 插值
        p1 = points_lg[i1]
        p2 = points_lg[i2]
        pn1 = points_n[i1]
        pn2 = points_n[i2]
        p_inter = 0.0
        if i2 != i1:
            p_inter = (angle_lg - angles_lg[i1]) / (angles_lg[i2] - angles_lg[i1])
        p_f = p1 + (p2 - p1) * p_inter
        closest = np.argmin(np.linalg.norm(self.points - p_f, axis=1))
        p_n = self.normals[closest]
        return [p_f, p_n]

    def cal_pointmap_of_mni(self, _coord):
        # sub_coord = simnibs_interface.ta_trans_mni2sub_coords(m2m_file, mni_coord)
        p3 = np.array(_coord)
        _, _,  eq, lg, _ = self.cal_longitude(self.al, self.ar, p3)
        return [eq, lg]

    def get_surface(self):
        # radii = [0.0005, 0.002, 0.002, 0.002]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(self.points, o3d.utility.DoubleVector(radii))
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.points, depth=5, scale=1.0)
        
        # temp = o3d.geometry.PointCloud()
        # temp.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        # o3d.visualization.draw_geometries([temp])
        pass
        # self.points.

    def discrete_angle(self, discrete_num):
        pointmap = np.zeros((discrete_num, discrete_num, 3), dtype=np.float64)
        pointmap_normal = np.zeros((discrete_num, discrete_num, 3), dtype=np.float64)
        pointmap_angle = np.zeros((discrete_num, discrete_num, 2), dtype=np.float64)
        # mesh_number = (discrete_num - 2)
        # mesh = np.zeros((discrete_num * discrete_num, 3), dtype=np.float64)
        for i in range(discrete_num):
            angle = 180.0 / discrete_num * i
            # 找到赤道点
            indexs = np.where(self.equator_degrees >= angle)[0]
            i1 = 0
            i2 = -1
            if len(indexs) > 0:
                if indexs[0] > 0:
                    i1 = indexs[0] - 1
                    if i1 + 1< len(self.equator_degrees):
                        i2 = i1 + 1
                    else:
                        i2 = i1
                else:
                    i1 = indexs[0]
                    i2 = i1
            else:
                i1 = -1
                i2 = -1
            # 插值
            p1 = self.equator_points[i1]
            p2 = self.equator_points[i2]
            p_inter = 0.0
            if i2 != i1:
                p_inter = (angle - self.equator_degrees[i1]) / (self.equator_degrees[i2] - self.equator_degrees[i1])
            p_f = p1 + (p2 - p1) * p_inter
            # 确定经度线
            points_lg, angles_lg, _, _, _, points_n = self.get_surface_line(self.al, self.ar, p_f, self.dis)
            for j in range(discrete_num):
                angle_lg = 180.0 / discrete_num * j
                # 找到经度点
                indexs_lg = np.where(angles_lg >= angle_lg)[0]
                i1 = 0
                i2 = -1
                if len(indexs_lg) > 0:
                    if indexs_lg[0] > 0:
                        i1 = indexs_lg[0] - 1
                        if i1 + 1< len(angles_lg):
                            i2 = i1 + 1
                        else:
                            i2 = i1
                    else:
                        i1 = indexs_lg[0]
                        i2 = i1
                else:
                    i1 = -1
                    i2 = -1
                # 插值
                p1 = points_lg[i1]
                p2 = points_lg[i2]
                pn1 = points_n[i1]
                pn2 = points_n[i2]
                p_inter = 0.0
                if i2 != i1:
                    p_inter = (angle_lg - angles_lg[i1]) / (angles_lg[i2] - angles_lg[i1])
                p_f = p1 + (p2 - p1) * p_inter
                p_n = pn1 + (pn2 - pn1) * p_inter
                pointmap[i,j,:] = p_f
                closest = np.argmin(np.linalg.norm(self.points - p_f, axis=1))
                pointmap_normal[i,j,:] = self.normals[closest]
                pointmap_angle[i,j,:] = [angle, angle_lg]
            # if i >= 99:
            #     # print(i)
            #     temp = o3d.geometry.PointCloud()
            #     temp.points = o3d.utility.Vector3dVector(points_lg)
            #     # temp.points = o3d.utility.Vector3dVector(pointmap[i,:,:])
            #     o3d.visualization.draw_geometries([temp])
        return pointmap, pointmap_normal, pointmap_angle

    def discrete_angle_old(self, pointmap):
        pointmaplist = pointmap.reshape((-1, 3))
        _, indexes = self.queryTree.query(pointmaplist)
        pointmap = self.points[indexes].reshape(pointmap.shape)
        pointmap_normal = self.normals[indexes].reshape(pointmap.shape)
        return pointmap, pointmap_normal

    def make_mesh(self):
        import open3d as o3d
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(self.points)
        points.normals = o3d.utility.Vector3dVector(self.normals)
        # estimate radius for rolling ball
        distances = points.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist 
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                points,
                o3d.utility.DoubleVector([radius, radius * 2]))
        o3d.visualization.draw_geometries([mesh, points], window_name='PointMap', width=800, height=600, left=50,
                                  top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False,)



def getSUBPointmap(sub):
    dataPath = '/DATA/TMS_DATA/'
    import os

    pointmap = np.load(dataPath + "MNI_pointmap_final.npy")
    sub_pointmap = simnibs.mni2subject_coords(
        pointmap.reshape((-1, 3), dataPath + sub + '/m2m_' + sub,
        transformation_type='nonl')
    )

    points = TmsAtlasPoints(dataPath + sub + '/' + sub + '.msh', sub_pointmap)
    arc_length = points.cal_equator(sub_pointmap[50,:], sub_pointmap[10150,:], sub_pointmap[5100,:], 2.0)
    p_eq, p_long, p_eq_degree, p_long_degree, lg_arc_length = points.cal_longitude(sub_pointmap[5050,:],  sub_pointmap[5150,:], sub_pointmap[5100,:])
    points.subpointmap = sub_pointmap.reshape(pointmap.shape)
    return points