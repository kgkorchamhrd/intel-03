import cv2
import numpy as np
import torch
import torch.nn.functional as F
import openvino as ov
import requests
import time
import os
from pathlib import Path
from huggingface_hub import hf_hub_download



def download_file(url, filename):
    """Downloads a file from a URL to the specified path."""
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def clone_repo(repo_url):
    """Clones a repository using git."""
    import os
    import subprocess
    
    # Extract repo name from URL
    repo_name = repo_url.split('/')[-1]
    
    # Check if the repo directory already exists
    if not Path(repo_name).exists():
        subprocess.run(['git', 'clone', repo_url])
        print(f"Repository {repo_name} cloned successfully.")
    else:
        print(f"Repository {repo_name} already exists.")


def setup_environment():
    """Sets up the necessary environment for Depth Anything V2."""
    # Download utility scripts if they don't exist
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)

    if not Path("cmd_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
        )
        open("cmd_helper.py", "w").write(r.text)

    # Import utility after downloading
    from cmd_helper import clone_repo
    
    # Clone repository
    clone_repo("https://huggingface.co/spaces/depth-anything/Depth-Anything-V2")
    
    # Fix attention file to work without xformers
    attention_file_path = Path("./Depth-Anything-V2/depth_anything_v2/dinov2_layers/attention.py")
    orig_attention_path = attention_file_path.parent / ("orig_" + attention_file_path.name)

    if not orig_attention_path.exists():
        attention_file_path.rename(orig_attention_path)

        with orig_attention_path.open("r") as f:
            data = f.read()
            data = data.replace("XFORMERS_AVAILABLE = True", "XFORMERS_AVAILABLE = False")
            with attention_file_path.open("w") as out_f:
                out_f.write(data)


def get_depth_map(output, w, h):
    """Convert depth output to a colored depth map."""
    depth = cv2.resize(output, (w, h))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    return depth


def get_raw_depth(output, w, h):
    """Convert depth output to raw depth values."""
    depth = cv2.resize(output, (w, h))
    # Normalize depth to 0-1 range
    normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
    # Scale depth to realistic values (in meters) - this is an approximation
    # Assuming the depth range is roughly 0.1 to 10 meters
    scaled_depth = normalized_depth * 4.9 + 0.1
    return scaled_depth


def create_slam_grid_map_front_only(rgb_image, depth_map, normals=None, ground_mask=None, grid_size=100, grid_resolution=0.05, max_range=5.0):
    """
    카메라 앞에 있는 실제 장애물만 인식하도록 조정된 SLAM 그리드 맵 생성 함수.
    
    Args:
        rgb_image: 원본 RGB 이미지
        depth_map: 미터 단위의 원시 깊이 맵
        normals: 사전 계산된 표면 법선 벡터 (선택 사항)
        ground_mask: 사전 계산된 지면 평면 마스크 (선택 사항)
        grid_size: 셀 단위의 그리드 크기
        grid_resolution: 미터 단위의 각 셀 크기
        max_range: 고려할 최대 범위(미터)
        
    Returns:
        2D 이진 점유 그리드 맵 시각화 및 그리드 맵
    """
    h, w = depth_map.shape
    
    # 카메라 매개변수
    fx = 525.0
    fy = 525.0
    cx = w / 2
    cy = h / 2
    
    # 성능을 위한 다운샘플링
    stride = 4
    y_coords, x_coords = np.mgrid[0:h:stride, 0:w:stride]
    
    # 깊이 맵을 3D 포인트로 변환
    Z = depth_map[::stride, ::stride]
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    
    # 그리드 맵 초기화
    grid_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
    grid_map.fill(128)  # 128 = 회색 (알 수 없음)
    
    # 카메라는 그리드 좌표의 중앙에 위치
    camera_grid_x = grid_size // 2
    camera_grid_y = grid_size // 2
    
    # ----- 앞쪽 장애물 감지를 위한 개선된 필터링 -----
    
    # 1. 관심 영역(ROI) 설정 - 카메라 앞쪽 중앙 영역에 집중
    # 이미지 중앙에서 수평으로 40%, 수직으로 60%만 고려
    roi_width_ratio = 0.8
    roi_height_ratio = 0.6
    roi_left = int(w * (0.5 - roi_width_ratio/2)) // stride
    roi_right = int(w * (0.5 + roi_width_ratio/2)) // stride
    roi_top = int(h * 0.2) // stride  # 상단 20%는 일반적으로 벽/천장
    roi_bottom = int(h * 0.75) // stride  # 하단 20%는 일반적으로 책상
    
    roi_mask = np.zeros_like(Z, dtype=bool)
    roi_mask[roi_top:roi_bottom, roi_left:roi_right] = True
    
    # 2. 깊이 기반 필터링 - 매우 가까운 점(책상)과 먼 점(배경) 제외
    # 0.5m에서 3.0m 사이의 점만 고려 - 장애물이 주로 존재하는 범위
    near_threshold = 0.25  # 너무 가까운 점 제외 (책상 표면)
    far_threshold = 2.0   # 너무 먼 점 제외 (배경 벽, 천장 등)
    depth_mask = (Z > near_threshold) & (Z < far_threshold)
    
    # 3. 높이 기반 필터링 - 카메라 높이 중심의 일정 범위만 고려
    # 카메라 높이 기준 위아래로 특정 범위만 포함
    lower_height_limit = -0.3  # 카메라 아래 30cm까지 (책상 위 낮은 장애물 포함)
    upper_height_limit = 0.5   # 카메라 위 50cm까지 (선반, 모니터 등 포함)
    height_mask = (Y > lower_height_limit) & (Y < upper_height_limit)
    
    # 4. 표면 법선 기반 필터링 - 수직 표면(벽, 장애물)과 수평 표면(책상) 구분
    if normals is None:
        normals = compute_surface_normals_fast(depth_map)
    
    # 다운샘플링된 법선
    normals_sampled = normals[::stride, ::stride]
    
    # 수직 표면 감지 (Y 방향 법선 성분이 낮음)
    # 값이 -0.3에서 0.3 사이이면 대략 수직 표면
    vertical_surface_mask = np.abs(normals_sampled[:, :, 1]) < 0.3
    
    # 5. 바닥(책상) 제외를 위한 지면 마스크 적용
    if ground_mask is None:
        ground_mask = np.zeros_like(Z, dtype=bool)
    else:
        ground_mask = ground_mask[::stride, ::stride]
    
    # 6. 모든 필터 마스크 결합
    # 관심 영역 내 + 적절한 깊이 + 적절한 높이 + 수직 표면 + 지면 아님
    valid_points_mask = roi_mask & depth_mask & height_mask & vertical_surface_mask & ~ground_mask
    
    # ----- 그리드 맵 생성 -----
    for i in range(valid_points_mask.shape[0]):
        for j in range(valid_points_mask.shape[1]):
            # 유효한 점만 처리
            if not valid_points_mask[i, j]:
                continue
                
            # 3D 좌표 가져오기
            x, y, z = X[i, j], Y[i, j], Z[i, j]
            
            # 유효성 검사 한 번 더
            if z <= 0 or z > max_range or np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            
            # 월드 좌표로 변환(카메라 앞이 +z, 오른쪽이 -x)
            world_x = z
            world_y = -x
            
            # 그리드 좌표로 변환
            grid_x = int(camera_grid_x + world_y / grid_resolution)
            grid_y = int(camera_grid_y - world_x / grid_resolution)
            
            # 그리드 내부인지 확인
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                # 장애물로 표시
                grid_map[grid_y, grid_x] = 0  # 검은색(점유)
                
                # 카메라에서 장애물까지의 공간은 비어있음으로 표시
                free_cells = bresenham_line(camera_grid_x, camera_grid_y, grid_x, grid_y)
                for cell_x, cell_y in free_cells[:-1]:  # 마지막 점(장애물) 제외
                    if 0 <= cell_x < grid_size and 0 <= cell_y < grid_size:
                        if grid_map[cell_y, cell_x] == 128:  # 알 수 없는 영역만 변경
                            grid_map[cell_y, cell_x] = 255  # 흰색(비어있음)
    
    # 맵 노이즈 제거 및 시각화 향상
    kernel = np.ones((3, 3), np.uint8)
    
    # 장애물 영역에 closing 연산 적용 (작은 구멍 메우기)
    obstacle_mask = (grid_map == 0).astype(np.uint8) * 255
    closed_obstacles = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
    
    # 작은 장애물 제거 (노이즈 제거)
    min_obstacle_size = 5  # 최소 장애물 크기 (픽셀)
    filtered_obstacles = cv2.morphologyEx(closed_obstacles, cv2.MORPH_OPEN, 
                                          np.ones((min_obstacle_size, min_obstacle_size), np.uint8))
    
    # 최종 맵 생성
    clean_map = np.full((grid_size, grid_size), 255, dtype=np.uint8)  # 흰색 배경(비어있음)
    clean_map[grid_map == 128] = 192  # 알 수 없는 영역은 연한 회색
    clean_map[filtered_obstacles > 0] = 0  # 필터링된 장애물은 검은색
    
    # 시각화용 컬러 맵 생성
    clean_rgb = np.stack([clean_map, clean_map, clean_map], axis=2)
    
    # 카메라 위치 및 FOV 표시
    center_radius = 2
    cv2.circle(clean_rgb, (camera_grid_x, camera_grid_y), center_radius, (0, 0, 255), -1)
    
    fov_angle = 60  # FOV 각도
    fov_length = 15
    angle1_rad = np.radians(-fov_angle / 2)
    angle2_rad = np.radians(fov_angle / 2)
    
    end1_x = int(camera_grid_x + fov_length * np.sin(angle1_rad))
    end1_y = int(camera_grid_y - fov_length * np.cos(angle1_rad))
    end2_x = int(camera_grid_x + fov_length * np.sin(angle2_rad))
    end2_y = int(camera_grid_y - fov_length * np.cos(angle2_rad))
    
    cv2.line(clean_rgb, (camera_grid_x, camera_grid_y), (end1_x, end1_y), (0, 0, 255), 1)
    cv2.line(clean_rgb, (camera_grid_x, camera_grid_y), (end2_x, end2_y), (0, 0, 255), 1)
    
    # 시각화용 크기 조정
    display_size = 400
    grid_display = cv2.resize(clean_rgb, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    
    # 그레이스케일로 변환
    grid_display_gray = cv2.cvtColor(grid_display, cv2.COLOR_BGR2GRAY)
    
    return grid_display_gray, clean_map


def compute_surface_normals_fast(depth_map, stride=4, window_size=5):
    """
    성능이 최적화된 표면 법선 계산 함수.
    
    Args:
        depth_map: 깊이 맵
        stride: 픽셀 건너뛰기 간격 (높을수록 더 빠름)
        window_size: 법선 계산에 사용할 윈도우 크기
        
    Returns:
        법선 벡터 맵
    """
    h, w = depth_map.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    
    # 더 빠른 법선 계산을 위해 간소화된 방법 사용
    # 소벨 필터를 사용하여 깊이 기울기 계산
    sobelx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    
    # 카메라 매개변수
    fx = 525.0
    fy = 525.0
    
    # 더 간소화된 법선 계산
    # 전체 픽셀이 아닌 일부 샘플만 계산(stride만큼 건너뛰기)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if depth_map[y, x] <= 0:
                continue
                
            # 간소화된 법선 계산
            dzdx = sobelx[y, x] / fx
            dzdy = sobely[y, x] / fy
            
            # 법선 벡터 = (-dzdx, -dzdy, 1) 정규화
            nx, ny, nz = -dzdx, -dzdy, 1.0
            norm = np.sqrt(nx*nx + ny*ny + nz*nz)
            
            if norm > 0:
                normals[y, x] = [nx/norm, ny/norm, nz/norm]
    
    # 누락된 픽셀 채우기 (필요시)
    if stride > 1:
        mask = np.all(normals == 0, axis=2)
        for c in range(3):
            channel = normals[:,:,c]
            channel[mask] = cv2.resize(channel[::stride, ::stride], (w, h), 
                                    interpolation=cv2.INTER_LINEAR)[mask]
    
    return normals


def detect_ground_plane_fast(depth_map, normals):
    """
    단순화된 빠른 지면 평면 감지
    
    Args:
        depth_map: 깊이 맵
        normals: 법선 벡터 맵
        
    Returns:
        지면 마스크
    """
    h, w = depth_map.shape
    
    # 빠른 계산을 위해 RANSAC 대신 간단한 임계값 처리 사용
    # 상향 법선을 가진 지점(Y 성분이 강한)과 특정 깊이 범위 내의 지점을 책상으로 간주
    desk_height_threshold = 0.1  # 위에서 10cm 이내
    height_mask = np.zeros_like(depth_map, dtype=bool)
    
    # Y 좌표 계산 (카메라 좌표계)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    fx = 525.0
    fy = 525.0
    cx = w / 2
    cy = h / 2
    Z = depth_map.copy()
    Y = (y_coords - cy) * Z / fy
    
    # 높이 기반 필터링 (카메라 높이 아래 특정 범위)
    height_mask = (Y > -desk_height_threshold) & (Y < 0) & (Z > 0)
    
    # 법선 기반 필터링 (상향 법선)
    normal_mask = normals[:,:,1] < -0.8
    
    # 두 조건 결합
    ground_mask = height_mask & normal_mask
    
    # 노이즈 제거를 위한 모폴로지 연산 적용
    kernel = np.ones((5, 5), np.uint8)
    ground_mask = ground_mask.astype(np.uint8) * 255
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)
    
    return ground_mask > 0


def estimate_camera_rotation(prev_frame, curr_frame):
    """Estimate camera rotation using feature matching"""
    if prev_frame is None or curr_frame is None:
        return 0, 0, 0
        
    try:
        # Convert images to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=500)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        
        # If no features found, return no movement
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0, 0, 0
        
        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = bf.match(des1, des2)
        
        # Sort them by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use only good matches (top 20%)
        good_matches = matches[:int(len(matches) * 0.2)]
        
        if len(good_matches) < 4:
            return 0, 0, 0
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Calculate homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return 0, 0, 0
        
        # Calculate rotation and translation from homography
        h, w = prev_gray.shape
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Calculate center of original and transformed image
        orig_center = np.mean(corners, axis=0)[0]
        trans_center = np.mean(transformed_corners, axis=0)[0]
        
        # Movement is the difference between centers
        dx = (trans_center[0] - orig_center[0]) * 0.03
        dy = (trans_center[1] - orig_center[1]) * 0.03
        
        # Estimate rotation (simplified)
        # Calculate relative rotation between corners
        orig_top_vector = corners[3][0] - corners[0][0]
        trans_top_vector = transformed_corners[3][0] - transformed_corners[0][0]
        
        # Calculate angle between vectors
        orig_angle = np.arctan2(orig_top_vector[1], orig_top_vector[0])
        trans_angle = np.arctan2(trans_top_vector[1], trans_top_vector[0])
        rotation = trans_angle - orig_angle
        
        # Scale rotation to be more sensitive
        rotation = rotation * 5
        
        return dx, dy, rotation
    
    except Exception as e:
        print(f"Camera rotation estimation error: {e}")
        return 0, 0, 0


def bresenham_line(x0, y0, x1, y1):
    """
    Implementation of Bresenham's line algorithm to get cells along a line.
    Used to mark free space between camera and obstacle.
    
    Returns list of (x, y) coordinates of points on the line.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
    return points


def create_accumulated_map(grid_size=400, persistence=0.7):
    """
    Initialize or get accumulated map for SLAM-style mapping over time.
    
    Args:
        grid_size: Size of the map
        persistence: How much to persist old observations (0-1)
        
    Returns:
        The current accumulated map
    """
    # Initialize static map if it doesn't exist
    if not hasattr(create_accumulated_map, "map"):
        # Initialize with unknown (gray) values
        create_accumulated_map.map = np.full((grid_size, grid_size), 128, dtype=np.uint8)
        # Keep track of confidence in each cell
        create_accumulated_map.confidence = np.zeros((grid_size, grid_size), dtype=np.float32)
        # Initial camera position and orientation
        create_accumulated_map.camera_x = grid_size // 2
        create_accumulated_map.camera_y = grid_size // 2
        create_accumulated_map.camera_angle = 0  # in radians
        
    return (create_accumulated_map.map, 
            create_accumulated_map.confidence,
            create_accumulated_map.camera_x,
            create_accumulated_map.camera_y,
            create_accumulated_map.camera_angle)


def update_accumulated_map_improved(local_map, dx, dy, rotation):
    """
    실제 장애물 감지에 최적화된 누적 맵 업데이트 함수.
    시간적 필터링을 추가하여 일시적인 오류를 줄임.
    
    Args:
        local_map: 새로운 로컬 그리드 맵 관측
        dx, dy: x,y 방향의 카메라 이동
        rotation: 라디안 단위의 카메라 회전
        
    Returns:
        업데이트된 누적 맵
    """
    global_map, confidence, camera_x, camera_y, camera_angle = create_accumulated_map()
    
    # 카메라 위치 및 방향 업데이트
    camera_angle += rotation
    
    # 현재 방향을 기준으로 이동을 전역 좌표로 변환
    global_dx = dx * np.cos(camera_angle) - dy * np.sin(camera_angle)
    global_dy = dx * np.sin(camera_angle) + dy * np.cos(camera_angle)
    
    # 카메라 위치 업데이트
    camera_x += int(global_dx * 3)
    camera_y += int(global_dy * 3)
    
    # 카메라가 맵 경계 내에 있도록 보장
    grid_size = global_map.shape[0]
    camera_x = max(50, min(grid_size - 50, camera_x))
    camera_y = max(50, min(grid_size - 50, camera_y))
    
    # 업데이트된 값 저장
    create_accumulated_map.camera_x = camera_x
    create_accumulated_map.camera_y = camera_y
    create_accumulated_map.camera_angle = camera_angle
    
    # 맵 크기 가져오기
    global_size = global_map.shape[0]
    local_size = local_map.shape[0]
    
    # 회전 행렬 생성
    cos_theta = np.cos(camera_angle)
    sin_theta = np.sin(camera_angle)
    
    # 로컬 맵 중심
    local_center_x = local_size // 2
    local_center_y = local_size // 2
    
    # ----- 향상된 맵 통합 -----
    for y in range(local_size):
        for x in range(local_size):
            # 알 수 없는 셀 건너뛰기
            if local_map[y, x] == 128:
                continue
                
            # 로컬 맵 중심 기준 상대 위치
            rel_x = x - local_center_x
            rel_y = y - local_center_y
            
            # 회전 적용하여 전역 좌표 계산
            rot_x = int(rel_x * cos_theta - rel_y * sin_theta)
            rot_y = int(rel_x * sin_theta + rel_y * cos_theta)
            
            # 전역 맵에서의 위치
            global_x = camera_x + rot_x
            global_y = camera_y + rot_y
            
            # 경계 확인
            if 0 <= global_x < global_size and 0 <= global_y < global_size:
                # 장애물과 자유 공간의 신뢰도 조정
                if local_map[y, x] == 0:  # 장애물
                    # 더 높은 신뢰도 증가 - 실제 장애물은 신뢰성 높게
                    confidence[global_y, global_x] += 0.4
                    # 신뢰도 상한 설정
                    confidence[global_y, global_x] = min(2.0, confidence[global_y, global_x])
                elif local_map[y, x] == 255:  # 자유 공간
                    # 더 낮은 신뢰도 증가 - 자유 공간은 보수적으로
                    confidence[global_y, global_x] += 0.1
                
                # 장애물 인식에 더 높은, 자유 공간 인식에 더 낮은 임계값 사용
                # 오탐지 감소 효과
                if confidence[global_y, global_x] > 1.5 and local_map[y, x] == 0:
                    global_map[global_y, global_x] = 0  # 장애물
                elif confidence[global_y, global_x] > 1.0 and local_map[y, x] == 255:
                    global_map[global_y, global_x] = 255  # 자유 공간
    
    # 시간이 지남에 따라 신뢰도 감소 (오래된 관측의 가중치 감소)
    confidence = confidence * 0.99
    
    # 맵 개선을 위한 후처리
    kernel = np.ones((3, 3), np.uint8)
    obstacle_mask = (global_map == 0).astype(np.uint8) * 255
    
    # 더 큰 커널로 closing 연산 적용 (작은 구멍 메우기)
    cleaned_obstacles = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
    
    # 작은 장애물(노이즈) 제거
    cleaned_obstacles = cv2.morphologyEx(cleaned_obstacles, cv2.MORPH_OPEN, 
                                         np.ones((2, 2), np.uint8))
    
    # 최종 깨끗한 맵 생성
    clean_map = np.full_like(global_map, 128)  # 모두 알 수 없음으로 시작
    clean_map[global_map == 255] = 255  # 자유 공간
    clean_map[cleaned_obstacles > 0] = 0  # 장애물
    
    return clean_map, camera_x, camera_y, camera_angle


def visualize_accumulated_map(clean_map, camera_x, camera_y, camera_angle, display_size=500):
    """Create a visualization of the accumulated map with camera position and orientation"""
    # Resize for display
    map_display = cv2.resize(clean_map, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    
    # Convert to color for visualization
    map_color = cv2.cvtColor(map_display, cv2.COLOR_GRAY2BGR)
    
    # Calculate camera position in display coordinates
    grid_size = clean_map.shape[0]
    display_x = int(camera_x * display_size / grid_size)
    display_y = int(camera_y * display_size / grid_size)
    
    # Draw camera position
    cv2.circle(map_color, (display_x, display_y), 5, (0, 0, 255), -1)
    
    # Draw camera orientation (direction the camera is facing)
    arrow_length = 20
    end_x = int(display_x + arrow_length * np.sin(camera_angle))
    end_y = int(display_y - arrow_length * np.cos(camera_angle))
    cv2.line(map_color, (display_x, display_y), (end_x, end_y), (0, 0, 255), 2)
    
    # Draw field of view
    fov_angle = np.radians(60)  # 60 degree FOV
    left_angle = camera_angle - fov_angle/2
    right_angle = camera_angle + fov_angle/2
    
    left_x = int(display_x + arrow_length * 1.5 * np.sin(left_angle))
    left_y = int(display_y - arrow_length * 1.5 * np.cos(left_angle))
    
    right_x = int(display_x + arrow_length * 1.5 * np.sin(right_angle))
    right_y = int(display_y - arrow_length * 1.5 * np.cos(right_angle))
    
    cv2.line(map_color, (display_x, display_y), (left_x, left_y), (0, 255, 0), 1)
    cv2.line(map_color, (display_x, display_y), (right_x, right_y), (0, 255, 0), 1)
    
    # Add grid lines
    grid_step = display_size // 10
    for i in range(0, display_size, grid_step):
        cv2.line(map_color, (i, 0), (i, display_size-1), (100, 100, 100), 1)
        cv2.line(map_color, (0, i), (display_size-1, i), (100, 100, 100), 1)
    
    return map_color


def handle_accumulated_mapping_mode_front_only(frame, raw_depth, prev_frame, grid_size):
    """
    실제 카메라 앞 장애물만 인식하도록 조정된 누적 매핑 처리 함수.
    
    Args:
        frame: 현재 RGB 프레임
        raw_depth: 미터 단위의 깊이 맵
        prev_frame: 이전 RGB 프레임
        grid_size: 로컬 그리드 맵의 크기
        
    Returns:
        map_display: 맵 시각화
        mapping_time: 매핑에 소요된 시간
        camera_x, camera_y, camera_angle: 카메라 위치 및 방향
    """
    start_time = time.time()
    
    # 표면 법선 계산 (최적화된 버전)
    normals = compute_surface_normals_fast(raw_depth)
    
    # 지면(책상 표면) 감지
    ground_mask = detect_ground_plane_fast(raw_depth, normals)
    
    # 앞쪽 장애물 감지에 최적화된 그리드 맵 생성
    grid_display, local_map = create_slam_grid_map_front_only(
        frame, 
        raw_depth,
        normals,
        ground_mask,
        grid_size=grid_size
    )
    
    # 카메라 이동 추정
    dx, dy, rotation = estimate_camera_rotation(prev_frame, frame)
    print(f"Camera moved: dx={dx:.2f}, dy={dy:.2f}, rotation={np.degrees(rotation):.1f}°")
    
    # 누적된 맵 업데이트 - 더 높은 신뢰도 임계값 사용
    clean_map, camera_x, camera_y, camera_angle = update_accumulated_map_improved(
        local_map, dx, dy, rotation
    )
    
    # 시각화 생성
    map_display = visualize_accumulated_map(clean_map, camera_x, camera_y, camera_angle)
    
    # 처리 시간 계산
    mapping_time = time.time() - start_time
    
    return map_display, mapping_time, camera_x, camera_y, camera_angle




def check_available_devices():
    """Check and print all available OpenVINO devices."""
    core = ov.Core()
    devices = core.available_devices
    
    print("\n=== Available OpenVINO Devices ===")
    for device in devices:
        device_name = device
        try:
            device_info = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"• {device}: {device_info}")
        except:
            print(f"• {device}")
    print("==================================\n")
    
    return devices


def main():
    """Main function to run Depth Anything V2 with webcam input and 2D SLAM-style mapping."""
    # Setup environment and download necessary files
    print("Setting up environment...")
    setup_environment()
    
    # Check available devices
    available_devices = check_available_devices()
    
    # Import model after setup
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    
    # Model configuration
    encoder = "vits"
    model_type = "Small"
    model_id = f"depth_anything_v2_{encoder}"
    
    # We'll use the original input resolution to match the model
    input_height = 518
    input_width = 518
    
    # Set up transformation pipeline
    transform = Compose(
        [
            Resize(
                width=input_width,
                height=input_height,
                resize_target=False,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    
    # Download model from Hugging Face
    print("Downloading model...")
    model_path = hf_hub_download(
        repo_id=f"depth-anything/Depth-Anything-V2-{model_type}", 
        filename=f"{model_id}.pth", 
        repo_type="model"
    )
    
    # Initialize PyTorch model
    print("Initializing model...")
    model = DepthAnythingV2(encoder=encoder, features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Convert to OpenVINO model if not already converted
    OV_DEPTH_ANYTHING_PATH = Path(f"{model_id}.xml")
    
    # Use existing model or convert
    if OV_DEPTH_ANYTHING_PATH.exists():
        print(f"Using existing OpenVINO model: {OV_DEPTH_ANYTHING_PATH}")
    else:
        print(f"Converting to OpenVINO format (input size: {input_width}x{input_height})...")
        ov_model = ov.convert_model(
            model, 
            example_input=torch.rand(1, 3, input_height, input_width), 
            input=[1, 3, input_height, input_width]
        )
        ov.save_model(ov_model, OV_DEPTH_ANYTHING_PATH)
    
    # Initialize OpenVINO runtime
    print("Initializing OpenVINO runtime...")
    core = ov.Core()
    
    # Try different devices in order of preference
    device_priority = ["CPU"]
    compiled_model = None
    used_device = None
    
    for device in device_priority:
        if device in available_devices:
            try:
                print(f"Trying to compile model for {device}...")
                compiled_model = core.compile_model(OV_DEPTH_ANYTHING_PATH, device)
                used_device = device
                print(f"Successfully compiled model for {device}!")
                break
            except Exception as e:
                print(f"Failed to compile for {device}: {str(e)}")
    
    if compiled_model is None:
        print("Failed to compile model for any device. Exiting.")
        return
    
    # Get model input details
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape
    print(f"Model input shape: {input_shape}")
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set lower resolution for webcam for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Processing webcam feed using {used_device}. Press 'q' to exit.")
    print("View options:")
    print("  - Press 'f' to toggle frame skipping (process every other frame)")
    print("  - Press '1' for standard view (RGB + Depth)")
    print("  - Press '2' for 2D SLAM grid map view")
    print("  - Press '3' for accumulated SLAM map view")
    print("  - Press 'i' to show/hide info")
    print("  - Press '+'/'-' to adjust grid size")
    print("  - Press 's' to save current map")
    print("  - Press 'r' to reset accumulated map")
    
    # Parameters for performance optimization
    frame_skip = False
    frame_count = 0
    show_info = True
    view_mode = 1  # 1: RGB+Depth, 2: Grid Map, 3: Accumulated Map
    
    # Grid map parameters
    grid_size = 100  # Default grid size
    
    # Global map parameters
    global_map_size = 400
    camera_global_x = global_map_size // 2
    camera_global_y = global_map_size // 2
    camera_angle = 0
    
    # FPS calculation variables
    prev_frame_time = 0
    new_frame_time = 0
    fps_values = []
    
    # Previous frame storage
    prev_frame = None
    
    # Main processing loop
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Get current time for FPS calculation
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_values.append(fps)
        if len(fps_values) > 30:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)
        
        # Frame skipping if enabled
        frame_count += 1
        if frame_skip and frame_count % 2 != 0:
            # Just show original frame with FPS
            if show_info:
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Depth Anything V2", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('f'):
                frame_skip = not frame_skip
                print(f"Frame skipping: {'ON' if frame_skip else 'OFF'}")
            elif key == ord('i'):
                show_info = not show_info
                print(f"Info display: {'ON' if show_info else 'OFF'}")
            elif key == ord('1'):
                view_mode = 1
                print("View mode: RGB + Depth")
            elif key == ord('2'):
                view_mode = 2
                print("View mode: 2D SLAM Grid Map")
            elif key == ord('3'):
                view_mode = 3
                print("View mode: Accumulated SLAM Map")
            elif key == ord('r'):
                # Reset accumulated map
                create_accumulated_map.map = np.full((global_map_size, global_map_size), 128, dtype=np.uint8)
                create_accumulated_map.confidence = np.zeros((global_map_size, global_map_size), dtype=np.float32)
                create_accumulated_map.camera_x = global_map_size // 2
                create_accumulated_map.camera_y = global_map_size // 2
                create_accumulated_map.camera_angle = 0
                print("Accumulated map reset.")
            continue
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Start inference timing
        inference_start = time.time()
        
        # Preprocess image
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        input_image = transform({"image": input_image})["image"]
        input_image = np.expand_dims(input_image, 0)
        
        # Inference
        result = compiled_model(input_image)[0]
        
        # End inference timing
        inference_time = time.time() - inference_start
        
        # Get raw depth map (for 3D operations)
        raw_depth = get_raw_depth(result[0], w, h)
        
        # Get colored depth map (for visualization)
        depth_map_colored = get_depth_map(result[0], w, h)
        
        # Different view modes
        if view_mode == 1:
            # Standard view: RGB + Depth
            display_frame = frame.copy()
            depth_display = depth_map_colored.copy()
            
            # Add info if enabled
            if show_info:
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Device: {used_device}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(depth_display, "Standard View (Press 1,2,3 to change)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display original and depth map side by side
            stacked_frame = np.hstack((display_frame, depth_display))
            cv2.imshow("Depth Anything V2", stacked_frame)
            
        elif view_mode == 2:
            start_time = time.time()
            
            # 카메라 앞 실제 장애물만 인식하는 함수 사용
            grid_display, local_map = create_slam_grid_map_front_only(
                frame, 
                raw_depth, 
                grid_size=grid_size
            )
            
            mapping_time = time.time() - start_time
            
            # Add info as a text overlay
            if show_info:
                # Convert grid display to color for info overlay
                grid_display_color = cv2.cvtColor(grid_display, cv2.COLOR_GRAY2BGR)
                cv2.putText(grid_display_color, f"FPS: {avg_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(grid_display_color, f"Mapping: {mapping_time*1000:.1f}ms", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(grid_display_color, f"Grid size: {grid_size}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(grid_display_color, "SLAM Grid Map (Press 1,2,3)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Display the color version with info
                cv2.imshow("Depth Anything V2", grid_display_color)
            else:
                # Display the grayscale version without info
                cv2.imshow("Depth Anything V2", grid_display)
            
        elif view_mode == 3:
            if prev_frame is not None:
                # 카메라 앞 실제 장애물만 인식하는 함수 사용
                acc_map_display, mapping_time, camera_global_x, camera_global_y, camera_angle = \
                    handle_accumulated_mapping_mode_front_only(frame, raw_depth, prev_frame, grid_size)
                
                # Add info if enabled
                if show_info:
                    cv2.putText(acc_map_display, f"FPS: {avg_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(acc_map_display, f"Mapping: {mapping_time*1000:.1f}ms", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(acc_map_display, "Accumulated SLAM Map", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(acc_map_display, "Rotate camera to map 360°", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(acc_map_display, f"Camera rotation: {np.degrees(camera_angle):.1f}°", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(acc_map_display, "Press 'r' to reset map", 
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Display the map
                cv2.imshow("Depth Anything V2", acc_map_display)
            
            # Store current frame for next iteration
            prev_frame = frame.copy()
        
        # Check for user exit and other keyboard commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            frame_skip = not frame_skip
            print(f"Frame skipping: {'ON' if frame_skip else 'OFF'}")
        elif key == ord('i'):
            show_info = not show_info
            print(f"Info display: {'ON' if show_info else 'OFF'}")
        elif key == ord('1'):
            view_mode = 1
            print("View mode: RGB + Depth")
        elif key == ord('2'):
            view_mode = 2
            print("View mode: 2D SLAM Grid Map")
        elif key == ord('3'):
            view_mode = 3
            print("View mode: Accumulated SLAM Map")
        elif key == ord('+') or key == ord('='):
            grid_size = min(150, grid_size + 10)
            print(f"Grid size: {grid_size}")
        elif key == ord('-'):
            grid_size = max(50, grid_size - 10)
            print(f"Grid size: {grid_size}")
        elif key == ord('s'):
            # Save the current map
            if view_mode == 2:
                cv2.imwrite("slam_grid_map.png", grid_display)
                print("Map saved to slam_grid_map.png")
            elif view_mode == 3:
                cv2.imwrite("slam_accumulated_map.png", acc_map_display)
                print("Map saved to slam_accumulated_map.png")
        elif key == ord('r'):
            # Reset accumulated map
            create_accumulated_map.map = np.full((global_map_size, global_map_size), 128, dtype=np.uint8)
            create_accumulated_map.confidence = np.zeros((global_map_size, global_map_size), dtype=np.float32)
            create_accumulated_map.camera_x = global_map_size // 2
            create_accumulated_map.camera_y = global_map_size // 2
            create_accumulated_map.camera_angle = 0
            print("Accumulated map reset.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam processing stopped.")


if __name__ == "__main__":
    main()