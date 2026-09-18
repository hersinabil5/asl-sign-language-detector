def normalize_landmarks(flat_landmarks):
    """Takes a flat list of 63 floats (21 landmarks x,y,z) and returns
    a normalized version: wrist-centered and scaled by hand size, so
    the same sign looks the same regardless of where it happens in
    frame or how far from the camera. Leaves an all-zero vector
    (no hand detected) unchanged."""
    if all(v == 0.0 for v in flat_landmarks):
        return flat_landmarks

    points = [
        (flat_landmarks[i], flat_landmarks[i + 1], flat_landmarks[i + 2])
        for i in range(0, len(flat_landmarks), 3)
    ]

    wrist = points[0]
    translated = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in points]

    # scale by distance from wrist to middle-finger MCP (landmark 9) —
    # a stable reference for hand size regardless of finger spread
    ref_x, ref_y, ref_z = translated[9]
    scale = (ref_x ** 2 + ref_y ** 2 + ref_z ** 2) ** 0.5
    if scale < 1e-6:
        scale = 1.0

    normalized = [(x / scale, y / scale, z / scale) for x, y, z in translated]
    return [c for point in normalized for c in point]
