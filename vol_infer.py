import os
import glob
import open3d as o3d
import numpy as np
import pandas as pd


def rotate(mesh):
    # メッシュの頂点座標を取得
    points = np.asarray(mesh.vertices)

    # PCAを適用
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 第三主成分を取得（固有値に基づく）
    third_pc = eigenvectors[:, 2]

    # 第三主成分をZ軸に合わせるための回転行列を計算
    z_axis = np.array([0, 0, 1])
    rotation_matrix = np.eye(3)

    # コサイン類似度に基づいて軸を合わせます
    cosine_similarity = np.dot(third_pc, z_axis)
    rotation_axis = np.cross(third_pc, z_axis)
    rotation_angle = np.arccos(cosine_similarity)

    if np.linalg.norm(rotation_axis) > 1e-6:  # 回転軸が定義可能な場合
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle
        )

    # 再構築されたメッシュのバウンディングボックスを計算
    bbox = mesh.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()

    # 軸の順番を大きい順に並べ替える
    sorted_indices = np.argsort(-extents)
    largest_axis = sorted_indices[0]
    second_largest_axis = sorted_indices[1]
    smallest_axis = sorted_indices[2]

    # 並べ替えた軸に基づいて回転行列を作成
    rotation_matrix = np.eye(3)
    rotation_matrix[0, :] = np.array([1 if i == largest_axis else 0 for i in range(3)])
    rotation_matrix[1, :] = np.array(
        [1 if i == second_largest_axis else 0 for i in range(3)]
    )
    rotation_matrix[2, :] = np.array([1 if i == smallest_axis else 0 for i in range(3)])
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    return mesh


def infer_volumes(mesh_path):
    print(f"Processing {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # xyz軸上に沿うように回転
    mesh = rotate(mesh)

    if not mesh.is_watertight():
        # ポアソン再構築でメッシュを再構築し、watertightにする
        while True:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                mesh.sample_points_poisson_disk(number_of_points=5000), depth=8
            )
            if mesh.is_watertight():
                break

    # 再構築されたメッシュのバウンディングボックスを計算
    bbox = mesh.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()
    sorted_extents = np.sort(extents)
    second_largest_extent = sorted_extents[1]

    # 高さを1に正規化
    scale_factor = 1 / second_largest_extent
    mesh.scale(scale_factor, center=mesh.get_center())

    # メッシュの体積を計算
    volume = mesh.get_volume()
    print(f"Volume: {volume}")
    return volume
