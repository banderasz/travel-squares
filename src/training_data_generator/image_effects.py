import cv2
import numpy as np
import random
import math


def generate_transformation_matrices(
    image_shape,
    perspective_strength=0.15,
    rotation_angle_deg=5
):
    """
    Generate transformation matrices for perspective and rotation effects.
    These matrices can be applied to both images and bounding boxes.

    Args:
        image_shape: Tuple of (height, width) of the input image
        perspective_strength: Maximum perspective distortion (0-1)
        rotation_angle_deg: Maximum rotation angle in degrees

    Returns:
        Dict containing:
            - 'perspective_matrix': 3x3 perspective transformation matrix (or None)
            - 'rotation_matrix': 2x3 rotation transformation matrix (or None)
            - 'output_size_after_perspective': (width, height) after perspective
            - 'output_size_after_rotation': (width, height) after rotation
            - 'perspective_applied': Boolean
            - 'rotation_applied': Boolean
    """
    h, w = image_shape[:2]
    result = {
        'perspective_matrix': None,
        'rotation_matrix': None,
        'output_size_after_perspective': (w, h),
        'output_size_after_rotation': (w, h),
        'perspective_applied': False,
        'rotation_applied': False
    }

    # Generate perspective transformation matrix
    if perspective_strength > 0:
        max_offset = int(min(w, h) * perspective_strength)

        # Source points (corners of the image)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Destination points with random offsets
        dst_pts = np.float32([
            [random.randint(-max_offset//2, max_offset//2),
             random.randint(-max_offset//2, max_offset//2)],
            [w + random.randint(-max_offset//2, max_offset//2),
             random.randint(-max_offset//2, max_offset//2)],
            [w + random.randint(-max_offset//2, max_offset//2),
             h + random.randint(-max_offset//2, max_offset//2)],
            [random.randint(-max_offset//2, max_offset//2),
             h + random.randint(-max_offset//2, max_offset//2)]
        ])

        # Calculate perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Calculate output size
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        transformed_corners = perspective_matrix @ corners
        transformed_corners /= transformed_corners[2, :]

        x_coords = transformed_corners[0, :]
        y_coords = transformed_corners[1, :]

        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))

        # Adjust matrix to ensure positive coordinates
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
        final_perspective_matrix = translation_matrix @ perspective_matrix

        output_size_perspective = (max_x - min_x, max_y - min_y)

        result['perspective_matrix'] = final_perspective_matrix
        result['output_size_after_perspective'] = output_size_perspective
        result['perspective_applied'] = True

        # Update dimensions for rotation calculation
        w, h = output_size_perspective

    # Generate rotation transformation matrix
    if rotation_angle_deg > 0:
        angle = random.uniform(-rotation_angle_deg, rotation_angle_deg)

        # Calculate rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size to fit rotated image
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))

        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        result['rotation_matrix'] = rotation_matrix
        result['output_size_after_rotation'] = (new_w, new_h)
        result['rotation_applied'] = True

    return result


def apply_transformation_matrices_to_image(image, alpha_channel, transformation_matrices):
    """
    Apply pre-generated transformation matrices to an image and its alpha channel.

    Args:
        image: Input image (BGR or BGRA as float32)
        alpha_channel: Alpha channel (or None)
        transformation_matrices: Dict from generate_transformation_matrices()

    Returns:
        Tuple of (transformed_image, transformed_alpha)
    """
    result = image.copy()
    new_alpha = alpha_channel

    # Apply perspective transformation
    if transformation_matrices['perspective_applied']:
        perspective_matrix = transformation_matrices['perspective_matrix']
        output_size = transformation_matrices['output_size_after_perspective']

        result = cv2.warpPerspective(result, perspective_matrix, output_size,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        if new_alpha is not None:
            new_alpha = cv2.warpPerspective(new_alpha, perspective_matrix, output_size,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Apply rotation transformation
    if transformation_matrices['rotation_applied']:
        rotation_matrix = transformation_matrices['rotation_matrix']
        output_size = transformation_matrices['output_size_after_rotation']

        result = cv2.warpAffine(result, rotation_matrix, output_size,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        if new_alpha is not None:
            new_alpha = cv2.warpAffine(new_alpha, rotation_matrix, output_size,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return result, new_alpha


def apply_transformation_matrices_to_bounding_boxes(bounding_boxes, transformation_matrices):
    """
    Apply transformation matrices to bounding boxes.

    Args:
        bounding_boxes: List of bounding boxes as (x1, y1, x2, y2) tuples or
                       dict of {quarter_location: [(symbol_name, (x1, y1, x2, y2)), ...]}
        transformation_matrices: Dict from generate_transformation_matrices()

    Returns:
        Transformed bounding boxes in the same format as input
    """
    # Handle dict format (with quarter locations)
    if isinstance(bounding_boxes, dict):
        transformed = {}
        for quarter_loc, boxes in bounding_boxes.items():
            transformed[quarter_loc] = []
            for symbol_name, bbox in boxes:
                x1, y1, x2, y2 = bbox
                transformed_bbox = _transform_single_bbox(
                    x1, y1, x2, y2, transformation_matrices
                )
                transformed[quarter_loc].append((symbol_name, transformed_bbox))
        return transformed

    # Handle list format
    else:
        transformed = []
        for bbox in bounding_boxes:
            if len(bbox) == 4:  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
                transformed.append(_transform_single_bbox(x1, y1, x2, y2, transformation_matrices))
            else:  # (symbol_name, (x1, y1, x2, y2))
                symbol_name, (x1, y1, x2, y2) = bbox
                transformed.append((symbol_name, _transform_single_bbox(x1, y1, x2, y2, transformation_matrices)))
        return transformed


def _transform_single_bbox(x1, y1, x2, y2, transformation_matrices):
    """
    Transform a single bounding box through perspective and rotation.

    Returns:
        Tuple of (new_x1, new_y1, new_x2, new_y2)
    """
    # Get all four corners of the bounding box
    corners = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1]
    ], dtype=np.float32).T

    # Apply perspective transformation
    if transformation_matrices['perspective_applied']:
        perspective_matrix = transformation_matrices['perspective_matrix']
        transformed_corners = perspective_matrix @ corners
        transformed_corners /= transformed_corners[2, :]
        corners = transformed_corners

    # Apply rotation transformation
    if transformation_matrices['rotation_applied']:
        rotation_matrix = transformation_matrices['rotation_matrix']
        # Convert 2x3 rotation matrix to work with homogeneous coordinates
        corners_2d = corners[:2, :]  # Take only x, y (drop z)
        ones = np.ones((1, corners_2d.shape[1]))
        corners_homogeneous = np.vstack([corners_2d, ones])

        transformed_corners = rotation_matrix @ corners_homogeneous
        corners = np.vstack([transformed_corners, np.ones((1, transformed_corners.shape[1]))])

    # Get axis-aligned bounding box from transformed corners
    x_coords = corners[0, :]
    y_coords = corners[1, :]

    new_x1 = int(np.min(x_coords))
    new_y1 = int(np.min(y_coords))
    new_x2 = int(np.max(x_coords))
    new_y2 = int(np.max(y_coords))

    return (new_x1, new_y1, new_x2, new_y2)


def apply_realistic_camera_effects(
    image,
    glare_intensity=0.3,
    blur_amount=1.5,
    noise_level=0.02,
    brightness_variation=0.1,
    contrast_variation=0.1,
    shadow_strength=0.2,
    vignette_strength=0.1,
    compression_artifacts=True,
    add_background=False,
    background_type="wooden_table",
    debug=False
):
    """
    Apply realistic camera effects to simulate taking a photo.
    This function only applies visual effects that don't transform geometry
    (no perspective or rotation - those should be done separately).

    Args:
        image: Input image (BGR or BGRA)
        glare_intensity: Intensity of lens glare/reflections (0-1)
        blur_amount: Amount of motion/focus blur
        noise_level: Camera sensor noise level (0-1)
        brightness_variation: Random brightness variation (0-1)
        contrast_variation: Random contrast variation (0-1)
        shadow_strength: Strength of shadows/lighting variations (0-1)
        vignette_strength: Vignetting effect strength (0-1)
        compression_artifacts: Whether to add JPEG-like compression artifacts
        add_background: Whether to add a realistic background (only works with alpha channel)
        background_type: Type of background ("wooden_table", "fabric", "marble", "concrete")
        debug: Whether to show intermediate steps

    Returns:
        Processed image with realistic camera effects
    """
    # Handle alpha channel properly
    has_alpha = image.shape[2] == 4
    if has_alpha:
        # Separate alpha channel
        alpha_channel = image[:, :, 3]
        image_rgb = image[:, :, :3]
    else:
        image_rgb = image
        alpha_channel = None

    result = image_rgb.copy().astype(np.float32)
    h, w = result.shape[:2]

    if debug:
        print("Starting camera effects processing...")
        cv2.imshow("0. Original", cv2.resize(result.astype(np.uint8), (800, 600)))
        cv2.waitKey(1000)

    # 1. Add background first if we have alpha channel
    if add_background and has_alpha:
        background = _generate_realistic_background(h, w, background_type)
        result = _blend_with_background(result, background, alpha_channel)

        if debug:
            print("1. Added background")
            cv2.imshow("1. Background Added", cv2.resize(result.astype(np.uint8), (800, 600)))
            cv2.waitKey(1000)

    # 2. Apply lighting variations and shadows
    if shadow_strength > 0:
        result = _apply_lighting_variations(result, shadow_strength)

        if debug:
            print(f"2. Applied lighting variations (strength: {shadow_strength})")
            display_img = cv2.resize(result.astype(np.uint8), (min(800, w), min(600, h)))
            cv2.imshow("2. Lighting", display_img)
            cv2.waitKey(1000)

    # 3. Apply lens glare/reflections
    if glare_intensity > 0:
        if debug:
            result = _apply_lens_glare_debug(result, glare_intensity, debug_windows=True)
            print(f"3. Applied lens glare (intensity: {glare_intensity})")
            display_img = cv2.resize(result.astype(np.uint8), (min(800, w), min(600, h)))
            cv2.imshow("3. Glare", display_img)
            cv2.waitKey(1000)
        else:
            result = _apply_lens_glare(result, glare_intensity)

    # 4. Apply motion/focus blur
    if blur_amount > 0:
        result = _apply_realistic_blur(result, blur_amount)

        if debug:
            print(f"4. Applied blur (amount: {blur_amount})")
            display_img = cv2.resize(result.astype(np.uint8), (min(800, w), min(600, h)))
            cv2.imshow("4. Blur", display_img)
            cv2.waitKey(1000)

    # 5. Apply brightness and contrast variations
    result = _apply_exposure_variations(result, brightness_variation, contrast_variation)

    if debug:
        print(f"5. Applied exposure variations (brightness: {brightness_variation}, contrast: {contrast_variation})")
        display_img = cv2.resize(result.astype(np.uint8), (min(800, w), min(600, h)))
        cv2.imshow("5. Exposure", display_img)
        cv2.waitKey(1000)

    # 6. Apply vignetting
    if vignette_strength > 0:
        result = _apply_vignetting(result, vignette_strength)

        if debug:
            print(f"6. Applied vignetting (strength: {vignette_strength})")
            display_img = cv2.resize(result.astype(np.uint8), (min(800, w), min(600, h)))
            cv2.imshow("6. Vignetting", display_img)
            cv2.waitKey(1000)

    # 7. Add sensor noise
    if noise_level > 0:
        result = _add_camera_noise(result, noise_level)

        if debug:
            print(f"7. Added camera noise (level: {noise_level})")
            display_img = cv2.resize(result.astype(np.uint8), (min(800, w), min(600, h)))
            cv2.imshow("7. Noise", display_img)
            cv2.waitKey(1000)

    # 8. Apply compression artifacts
    if compression_artifacts:
        result = _add_compression_artifacts(result)

        if debug:
            print("8. Applied compression artifacts")
            display_img = cv2.resize(result.astype(np.uint8), (min(800, w), min(600, h)))
            cv2.imshow("8. Compression", display_img)
            cv2.waitKey(1000)

    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    if debug:
        print("9. Final result")
        display_img = cv2.resize(result, (min(800, w), min(600, h)))
        cv2.imshow("9. Final", display_img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return result


def _generate_realistic_background(height, width, background_type):
    """Generate a realistic background texture."""
    if background_type == "wooden_table":
        return _generate_wooden_table_background(height, width)
    elif background_type == "fabric":
        return _generate_fabric_background(height, width)
    elif background_type == "marble":
        return _generate_marble_background(height, width)
    elif background_type == "concrete":
        return _generate_concrete_background(height, width)
    elif background_type == "green_felt":
        return _generate_green_felt_background(height, width)
    elif background_type == "red_cloth":
        return _generate_red_cloth_background(height, width)
    elif background_type == "blue_denim":
        return _generate_blue_denim_background(height, width)
    elif background_type == "granite":
        return _generate_granite_background(height, width)
    elif background_type == "cardboard":
        return _generate_cardboard_background(height, width)
    elif background_type == "leather":
        return _generate_leather_background(height, width)
    elif background_type == "metal":
        return _generate_metal_background(height, width)
    elif background_type == "paper":
        return _generate_paper_background(height, width)
    else:
        return _generate_wooden_table_background(height, width)


def _generate_wooden_table_background(height, width):
    """Generate a wooden table texture background."""
    # Base wood color (warm brown)
    base_color = np.array([139, 117, 101], dtype=np.float32)  # BGR

    # Create base background
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    # Add wood grain pattern
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Horizontal wood grain
    grain_freq = 0.02
    grain_pattern = np.sin(y * grain_freq) * 15

    # Add some vertical variation
    vertical_variation = np.sin(x * 0.005) * 10

    # Combine patterns
    wood_pattern = grain_pattern + vertical_variation

    # Add random noise for texture
    noise = np.random.normal(0, 8, (height, width))

    # Apply patterns to each channel with slight variation
    for i in range(3):
        channel_variation = wood_pattern + noise + random.uniform(-5, 5)
        background[:, :, i] += channel_variation

    # Add some darker wood knots randomly
    num_knots = random.randint(2, 6)
    for _ in range(num_knots):
        knot_x = random.randint(0, width)
        knot_y = random.randint(0, height)
        knot_size = random.randint(20, 60)

        # Create circular knot pattern
        knot_mask = ((x - knot_x)**2 + (y - knot_y)**2) <= knot_size**2
        knot_intensity = np.exp(-((x - knot_x)**2 + (y - knot_y)**2) / (knot_size**2 / 4))

        background[:, :, 0][knot_mask] -= knot_intensity[knot_mask] * 30
        background[:, :, 1][knot_mask] -= knot_intensity[knot_mask] * 25
        background[:, :, 2][knot_mask] -= knot_intensity[knot_mask] * 20

    return np.clip(background, 0, 255)


def _generate_fabric_background(height, width):
    """Generate a fabric texture background."""
    # Base fabric color (light gray/beige)
    base_color = np.array([180, 175, 165], dtype=np.float32)
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    # Create weave pattern
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Fabric weave pattern
    weave_size = 4
    weave_x = np.sin(x * 2 * np.pi / weave_size) * 5
    weave_y = np.sin(y * 2 * np.pi / weave_size) * 5

    weave_pattern = weave_x + weave_y

    # Add fine texture noise
    texture_noise = np.random.normal(0, 3, (height, width))

    # Apply to all channels
    for i in range(3):
        background[:, :, i] += weave_pattern + texture_noise

    return np.clip(background, 0, 255)


def _generate_marble_background(height, width):
    """Generate a marble texture background."""
    # Base marble color (light gray/white)
    base_color = np.array([220, 215, 210], dtype=np.float32)
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Create marble veining
    vein_freq1 = 0.01
    vein_freq2 = 0.007

    veins = (np.sin(x * vein_freq1 + y * vein_freq2) *
             np.sin(y * vein_freq1 + x * vein_freq2 * 1.3)) * 25

    # Add fine marble texture
    marble_noise = np.random.normal(0, 5, (height, width))

    # Apply darker veining
    for i in range(3):
        background[:, :, i] += veins + marble_noise - 10

    return np.clip(background, 0, 255)


def _generate_concrete_background(height, width):
    """Generate a concrete texture background."""
    # Base concrete color (medium gray)
    base_color = np.array([140, 135, 130], dtype=np.float32)
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    # Add concrete texture with strong noise
    concrete_noise = np.random.normal(0, 12, (height, width))

    # Add some larger variations
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    large_variation = np.sin(x * 0.003) * np.sin(y * 0.004) * 15

    # Apply texture
    for i in range(3):
        background[:, :, i] += concrete_noise + large_variation

    return np.clip(background, 0, 255)


def _generate_green_felt_background(height, width):
    """Generate a green felt texture background (like a card table)."""
    # Base green felt color
    base_color = np.array([35, 85, 25], dtype=np.float32)  # Dark green BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    # Add felt fiber texture
    fiber_noise = np.random.normal(0, 4, (height, width))

    # Add directional fiber pattern
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    fiber_pattern = np.sin(x * 0.5 + y * 0.3) * 2

    # Apply texture to all channels
    for i in range(3):
        background[:, :, i] += fiber_noise + fiber_pattern + random.uniform(-3, 3)

    # Add some slight color variation patches
    num_patches = random.randint(3, 7)
    for _ in range(num_patches):
        patch_x = random.randint(0, width)
        patch_y = random.randint(0, height)
        patch_size = random.randint(40, 100)

        patch_mask = ((x - patch_x)**2 + (y - patch_y)**2) <= patch_size**2
        patch_intensity = np.exp(-((x - patch_x)**2 + (y - patch_y)**2) / (patch_size**2 / 3))

        color_shift = random.uniform(-5, 5)
        background[:, :, 1][patch_mask] += patch_intensity[patch_mask] * color_shift

    return np.clip(background, 0, 255)


def _generate_red_cloth_background(height, width):
    """Generate a red cloth texture background."""
    # Base red color
    base_color = np.array([50, 40, 140], dtype=np.float32)  # Dark red BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Cloth weave pattern - more pronounced
    weave_size = 6
    weave_x = np.sin(x * 2 * np.pi / weave_size) * 8
    weave_y = np.sin(y * 2 * np.pi / weave_size) * 8
    cross_weave = np.sin((x + y) * 2 * np.pi / weave_size) * 4

    weave_pattern = weave_x + weave_y + cross_weave

    # Add cloth folds/creases
    fold_pattern = (np.sin(x * 0.01) * np.sin(y * 0.015) * 15 +
                   np.sin(x * 0.007) * np.sin(y * 0.012) * 10)

    # Fine texture noise
    texture_noise = np.random.normal(0, 5, (height, width))

    # Apply to all channels with slight variation
    for i in range(3):
        channel_variation = weave_pattern + fold_pattern + texture_noise
        if i == 2:  # Red channel - enhance it
            channel_variation *= 1.2
        background[:, :, i] += channel_variation

    return np.clip(background, 0, 255)


def _generate_blue_denim_background(height, width):
    """Generate a blue denim texture background."""
    # Base denim blue color
    base_color = np.array([110, 70, 35], dtype=np.float32)  # Denim blue BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Denim diagonal weave pattern
    diagonal_weave = (np.sin((x + y) * 0.3) * np.cos((x - y) * 0.25) * 10 +
                     np.sin((x + y) * 0.15) * 8)

    # Add denim thread texture
    thread_noise = np.random.normal(0, 6, (height, width))

    # Occasional darker/lighter patches (wear pattern)
    num_patches = random.randint(5, 10)
    for _ in range(num_patches):
        patch_x = random.randint(0, width)
        patch_y = random.randint(0, height)
        patch_size = random.randint(50, 120)

        patch_mask = ((x - patch_x)**2 + (y - patch_y)**2) <= patch_size**2
        patch_intensity = np.exp(-((x - patch_x)**2 + (y - patch_y)**2) / (patch_size**2 / 4))

        wear_amount = random.uniform(-10, 15)
        for i in range(3):
            background[:, :, i][patch_mask] += patch_intensity[patch_mask] * wear_amount

    # Apply diagonal weave and texture
    for i in range(3):
        background[:, :, i] += diagonal_weave + thread_noise

    return np.clip(background, 0, 255)


def _generate_granite_background(height, width):
    """Generate a granite stone texture background."""
    # Base granite color (dark gray with color variation)
    base_color = np.array([80, 75, 70], dtype=np.float32)  # Gray-brown BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    # Add speckled granite pattern
    num_speckles = int(height * width * 0.003)  # Lots of small speckles

    for _ in range(num_speckles):
        speckle_x = random.randint(0, width - 1)
        speckle_y = random.randint(0, height - 1)
        speckle_size = random.randint(1, 4)

        # Random speckle color (dark or light)
        if random.random() > 0.5:
            speckle_color = random.randint(20, 50)  # Dark speckle
        else:
            speckle_color = random.randint(150, 200)  # Light speckle

        # Draw speckle
        y_start = max(0, speckle_y - speckle_size)
        y_end = min(height, speckle_y + speckle_size)
        x_start = max(0, speckle_x - speckle_size)
        x_end = min(width, speckle_x + speckle_size)

        for i in range(3):
            background[y_start:y_end, x_start:x_end, i] = speckle_color

    # Add overall texture noise
    granite_noise = np.random.normal(0, 8, (height, width))

    for i in range(3):
        background[:, :, i] += granite_noise

    return np.clip(background, 0, 255)


def _generate_cardboard_background(height, width):
    """Generate a cardboard texture background."""
    # Base cardboard color (tan/brown)
    base_color = np.array([120, 140, 160], dtype=np.float32)  # Light brown BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Cardboard fiber pattern (horizontal lines)
    fiber_lines = np.sin(y * 0.5) * 5 + np.sin(y * 0.2) * 3

    # Add corrugation pattern
    corrugation = np.sin(x * 0.3) * 4

    # Cardboard texture noise
    cardboard_noise = np.random.normal(0, 8, (height, width))

    # Add some creases/damage
    num_creases = random.randint(2, 5)
    for _ in range(num_creases):
        crease_y = random.randint(0, height)
        crease_thickness = random.randint(2, 6)
        crease_start = max(0, crease_y - crease_thickness)
        crease_end = min(height, crease_y + crease_thickness)

        for i in range(3):
            background[crease_start:crease_end, :, i] -= random.uniform(10, 20)

    # Apply all patterns
    for i in range(3):
        background[:, :, i] += fiber_lines + corrugation + cardboard_noise

    return np.clip(background, 0, 255)


def _generate_leather_background(height, width):
    """Generate a leather texture background."""
    # Base leather color (dark brown)
    base_color = np.array([40, 60, 85], dtype=np.float32)  # Brown BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Leather grain pattern (irregular)
    grain_pattern = (np.sin(x * 0.05 + np.random.random()) *
                    np.cos(y * 0.04 + np.random.random()) * 8)

    # Add pores/texture
    pore_noise = np.random.normal(0, 6, (height, width))

    # Leather creases/wrinkles
    num_wrinkles = random.randint(3, 6)
    for _ in range(num_wrinkles):
        wrinkle_x = random.randint(0, width)
        wrinkle_y = random.randint(0, height)
        wrinkle_size = random.randint(100, 200)
        wrinkle_angle = random.uniform(0, np.pi)

        # Create elongated wrinkle
        dx = x - wrinkle_x
        dy = y - wrinkle_y
        rotated_dx = dx * np.cos(wrinkle_angle) + dy * np.sin(wrinkle_angle)
        rotated_dy = -dx * np.sin(wrinkle_angle) + dy * np.cos(wrinkle_angle)

        wrinkle_dist = (rotated_dx / wrinkle_size)**2 + (rotated_dy / (wrinkle_size * 0.3))**2
        wrinkle_mask = wrinkle_dist <= 1
        wrinkle_intensity = np.exp(-wrinkle_dist * 2)

        for i in range(3):
            background[:, :, i][wrinkle_mask] -= wrinkle_intensity[wrinkle_mask] * 15

    # Apply patterns
    for i in range(3):
        background[:, :, i] += grain_pattern + pore_noise

    return np.clip(background, 0, 255)


def _generate_metal_background(height, width):
    """Generate a brushed metal texture background."""
    # Base metal color (gray)
    base_color = np.array([100, 100, 100], dtype=np.float32)  # Gray BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Brushed metal pattern (horizontal scratches)
    brush_pattern = np.sin(x * 0.8) * 3 + np.sin(x * 2.5) * 1.5

    # Add fine horizontal scratches
    num_scratches = random.randint(20, 40)
    for _ in range(num_scratches):
        scratch_y = random.randint(0, height)
        scratch_intensity = random.uniform(5, 20)
        scratch_thickness = random.randint(1, 2)

        scratch_start = max(0, scratch_y - scratch_thickness)
        scratch_end = min(height, scratch_y + scratch_thickness)

        # Vary intensity across the scratch - fix broadcasting by slicing x properly
        for i in range(3):
            variation = (np.sin(x[scratch_start:scratch_end, :] * 0.1) * scratch_intensity * 0.3)
            if random.random() > 0.5:
                background[scratch_start:scratch_end, :, i] += scratch_intensity + variation
            else:
                background[scratch_start:scratch_end, :, i] -= scratch_intensity + variation

    # Fine metal grain noise
    metal_noise = np.random.normal(0, 4, (height, width))

    # Apply patterns
    for i in range(3):
        background[:, :, i] += brush_pattern + metal_noise

    return np.clip(background, 0, 255)


def _generate_paper_background(height, width):
    """Generate a paper texture background."""
    # Base paper color (off-white)
    base_color = np.array([235, 240, 245], dtype=np.float32)  # Off-white BGR
    background = np.full((height, width, 3), base_color, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Paper fiber texture
    fiber_noise = np.random.normal(0, 5, (height, width))

    # Add subtle paper grain
    grain_pattern = (np.sin(x * 0.3) * np.cos(y * 0.25) * 2 +
                    np.sin(x * 0.15) * np.sin(y * 0.18) * 1.5)

    # Add some yellowing/age spots
    num_spots = random.randint(3, 8)
    for _ in range(num_spots):
        spot_x = random.randint(0, width)
        spot_y = random.randint(0, height)
        spot_size = random.randint(50, 150)

        spot_mask = ((x - spot_x)**2 + (y - spot_y)**2) <= spot_size**2
        spot_intensity = np.exp(-((x - spot_x)**2 + (y - spot_y)**2) / (spot_size**2 / 4))

        # Yellow tint (reduce blue channel, keep red/green)
        background[:, :, 0][spot_mask] -= spot_intensity[spot_mask] * random.uniform(10, 25)  # Blue

    # Add occasional crease/fold
    if random.random() > 0.5:
        fold_y = random.randint(height // 4, 3 * height // 4)
        fold_thickness = random.randint(3, 8)
        fold_start = max(0, fold_y - fold_thickness)
        fold_end = min(height, fold_y + fold_thickness)

        for i in range(3):
            background[fold_start:fold_end, :, i] -= random.uniform(8, 15)

    # Apply patterns
    for i in range(3):
        background[:, :, i] += fiber_noise + grain_pattern

    return np.clip(background, 0, 255)


def _blend_with_background(foreground, background, alpha_channel):
    """Blend foreground with background using alpha channel."""
    if alpha_channel is None:
        return foreground

    alpha_normalized = alpha_channel.astype(np.float32) / 255.0
    alpha_normalized = alpha_normalized[:, :, np.newaxis]

    # Alpha blending: result = foreground * alpha + background * (1 - alpha)
    result = foreground * alpha_normalized + background * (1 - alpha_normalized)

    return result


def _apply_lighting_variations(image, strength):
    """Apply realistic lighting variations and shadows - uses pure multiplicative adjustment only."""
    h, w = image.shape[:2]

    # Create gradient lighting
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    # Random lighting direction
    light_dir_x = random.uniform(-1, 1)
    light_dir_y = random.uniform(-1, 1)

    # Create lighting gradient - only darkening, never brightening
    # This prevents any color shifts
    lighting = (x * light_dir_x + y * light_dir_y) * strength * 0.05
    lighting = np.clip(lighting + 1.0, 0.97, 1.0)  # Only allow darkening (0.97 to 1.0)

    # Apply equally to all channels to maintain color balance
    for i in range(image.shape[2]):
        image[:, :, i] = image[:, :, i] * lighting

    return image


def _apply_lens_glare(image, intensity):
    """Add lens flare/glare effects - uses pure white light only."""
    h, w = image.shape[:2]

    # Random glare position (usually towards light sources)
    glare_x = random.randint(w//4, 3*w//4)
    glare_y = random.randint(h//4, 3*h//4)

    # Create distance map from glare center
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distance = np.sqrt((x - glare_x)**2 + (y - glare_y)**2)

    # Create main glare effect (exponential decay)
    max_distance = min(w, h) * 0.8
    glare_strength = intensity * np.exp(-distance / (max_distance * 0.4))

    # Apply ONLY additive white light (equal on all channels)
    # This preserves the color ratios and prevents yellow tinting
    white_light = glare_strength * intensity * 80

    # Apply the same white light to ALL channels equally
    for i in range(image.shape[2]):
        image[:, :, i] = np.minimum(image[:, :, i] + white_light, 255)

    # For very strong glare areas, blend towards pure white (255, 255, 255)
    strong_glare_mask = glare_strength > intensity * 0.3
    if np.any(strong_glare_mask):
        blend_factor = np.clip((glare_strength - intensity * 0.3) * intensity * 2, 0, 1)
        blend_factor = blend_factor[:, :, np.newaxis]

        # Target is pure white (255 on all channels)
        white_target = np.full_like(image, 255.0)
        image = image * (1 - blend_factor) + white_target * blend_factor

    # Add lens flare artifacts (using pure white only)
    if intensity > 0.2:
        _add_lens_flare_artifacts(image, glare_x, glare_y, intensity)

    return image


def _apply_exposure_variations(image, brightness_var, contrast_var):
    """Apply random exposure variations - neutral color balance."""
    # Random brightness adjustment - very conservative, affects all channels equally
    brightness_factor = 1.0 + random.uniform(-brightness_var * 0.15, brightness_var * 0.15)

    # Random contrast adjustment - very conservative, affects all channels equally
    contrast_factor = 1.0 + random.uniform(-contrast_var * 0.1, contrast_var * 0.1)

    # Apply the SAME adjustments to all channels to maintain color balance
    # This prevents color shifts
    image = image * contrast_factor * brightness_factor

    return image


def _apply_vignetting(image, strength):
    """Apply vignetting effect (darkening towards edges) - neutral color."""
    h, w = image.shape[:2]

    # Create distance map from center
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    distance = np.sqrt(x**2 + y**2)

    # Create vignette mask - very subtle, only darkening, affects all channels equally
    max_distance = math.sqrt(2)  # Corner distance
    vignette = 1.0 - (distance / max_distance) * strength * 0.15
    vignette = np.clip(vignette, 1.0 - strength * 0.15, 1.0)

    # Apply the SAME vignette to all channels
    for i in range(image.shape[2]):
        image[:, :, i] = image[:, :, i] * vignette

    return image


def _add_lens_flare_artifacts(image, center_x, center_y, intensity):
    """Add realistic lens flare artifacts - pure white spots only."""
    h, w = image.shape[:2]

    # Add multiple smaller flare spots along a line from center
    num_flares = random.randint(2, 4)

    # Line from glare center towards opposite corner
    target_x = w - center_x
    target_y = h - center_y

    for i in range(num_flares):
        # Position flares along the line
        t = (i + 1) / (num_flares + 1) * 0.7
        flare_x = int(center_x + (target_x - center_x) * t)
        flare_y = int(center_y + (target_y - center_y) * t)

        # Make sure flare is within image bounds
        if 0 <= flare_x < w and 0 <= flare_y < h:
            # Create circular flare spot
            flare_size = random.randint(10, 30) * intensity
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            flare_distance = np.sqrt((x - flare_x)**2 + (y - flare_y)**2)

            # Circular flare with soft edges
            flare_intensity = np.exp(-flare_distance**2 / (flare_size**2 / 4)) * intensity * 25

            # Intensity decreases for flares further from center
            white_intensity = 0.3 + 0.7 * (1 - i / num_flares)

            # Apply THE SAME white intensity to ALL channels
            addition = flare_intensity * white_intensity
            for c in range(image.shape[2]):
                image[:, :, c] = np.minimum(image[:, :, c] + addition, 255)


def _apply_realistic_blur(image, blur_amount):
    """Apply realistic blur (combination of motion and focus blur)."""
    if blur_amount <= 0:
        return image

    # Random choice between motion blur and Gaussian blur
    blur_type = random.choice(['motion', 'gaussian', 'mixed'])

    if blur_type == 'motion':
        # Motion blur with random direction
        kernel_size = int(blur_amount * 3) * 2 + 1
        angle = random.uniform(0, 360)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        # Draw line in the direction of motion
        angle_rad = np.deg2rad(angle)
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1

        kernel = kernel / np.sum(kernel)

        # Apply motion blur
        image = cv2.filter2D(image, -1, kernel)

    elif blur_type == 'gaussian':
        # Gaussian blur (out of focus effect)
        kernel_size = int(blur_amount * 2) * 2 + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_amount)

    else:  # mixed
        # Apply both types with reduced strength
        kernel_size = max(3, int(blur_amount) * 2 + 1)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_amount * 0.5)

        # Light motion blur
        motion_kernel_size = max(3, int(blur_amount * 1.5) * 2 + 1)
        angle = random.uniform(0, 360)
        kernel = np.zeros((motion_kernel_size, motion_kernel_size))
        center = motion_kernel_size // 2
        angle_rad = np.deg2rad(angle)

        for i in range(motion_kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < motion_kernel_size and 0 <= y < motion_kernel_size:
                kernel[y, x] = 1

        kernel = kernel / np.sum(kernel)
        image = cv2.filter2D(image, -1, kernel)

    return image


def _add_camera_noise(image, noise_level):
    """Add camera sensor noise (Gaussian noise)."""
    if noise_level <= 0:
        return image

    h, w = image.shape[:2]

    # Generate noise for each channel
    noise = np.random.normal(0, noise_level * 255, image.shape)

    # Add noise to image
    noisy_image = image + noise

    return noisy_image


def _add_compression_artifacts(image):
    """Add JPEG-like compression artifacts."""
    # Encode to JPEG with low quality and decode back
    quality = random.randint(70, 85)

    # Convert to uint8 for JPEG encoding
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    # Encode and decode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image_uint8, encode_param)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    return decoded_image.astype(np.float32)


def _apply_lens_glare_debug(image, intensity, debug_windows=False):
    """
    Debug version of lens glare that shows intermediate steps.
    """
    if debug_windows:
        print(f"\nApplying lens glare with intensity: {intensity}")

    result = _apply_lens_glare(image.copy(), intensity)

    if debug_windows:
        cv2.imshow("Lens Glare Applied", cv2.resize(result.astype(np.uint8), (800, 600)))
        cv2.waitKey(1000)

    return result


# Integration function for the picture generator
def apply_camera_effects_to_composite(
    composite_image,
    effect_preset="moderate",
    custom_params=None,
    debug=False
):
    """
    Apply camera effects to composite image with preset configurations.
    Note: This only applies visual effects. Geometric transformations (perspective, rotation)
    should be applied separately using the Board class methods.

    Args:
        composite_image: Input composite image (BGR or BGRA)
        effect_preset: "light", "moderate", "heavy", or "custom"
        custom_params: Dict of custom parameters if preset is "custom"
        debug: Whether to show intermediate transformation steps

    Returns:
        Image with applied effects
    """
    # Extended list of background types
    all_backgrounds = [
        "wooden_table", "fabric", "marble", "concrete",
        "green_felt", "red_cloth", "blue_denim", "granite",
        "cardboard", "leather", "metal", "paper"
    ]

    presets = {
        "light": {
            "glare_intensity": 0.1,
            "blur_amount": 0.5,
            "noise_level": 0.003,
            "brightness_variation": 0.01,
            "contrast_variation": 0.01,
            "shadow_strength": 0.02,
            "vignette_strength": 0.01,
            "compression_artifacts": False,
            "add_background": True,
            "background_type": random.choice(["wooden_table", "paper", "fabric"])
        },
        "moderate": {
            "glare_intensity": 0.3,
            "blur_amount": 1.0,
            "noise_level": 0.005,
            "brightness_variation": 0.02,
            "contrast_variation": 0.02,
            "shadow_strength": 0.04,
            "vignette_strength": 0.02,
            "compression_artifacts": True,
            "add_background": True,
            "background_type": random.choice(all_backgrounds[:8])  # Most common backgrounds
        },
        "heavy": {
            "glare_intensity": 0.6,
            "blur_amount": 2.0,
            "noise_level": 0.01,
            "brightness_variation": 0.03,
            "contrast_variation": 0.03,
            "shadow_strength": 0.06,
            "vignette_strength": 0.04,
            "compression_artifacts": True,
            "add_background": True,
            "background_type": random.choice(all_backgrounds)  # Any background
        }
    }

    if effect_preset == "custom" and custom_params:
        params = custom_params
    else:
        params = presets.get(effect_preset, presets["moderate"])

    # Add debug parameter to the effects
    params["debug"] = debug

    return apply_realistic_camera_effects(composite_image, **params)
