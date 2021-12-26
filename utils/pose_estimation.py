import cv2 as cv
import numpy as np

#########################################################
################## POSE ESTIMATION ######################
#########################################################

PART_NAMES = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"),
    ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"),
    ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"),
    ("leftHip", "rightHip"),
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

LOCAL_MAXIMUM_RADIUS = 1

POSE_CHAIN = [
    ("nose", "leftEye"),
    ("leftEye", "leftEar"),
    ("nose", "rightEye"),
    ("rightEye", "rightEar"),
    ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
]

PARENT_CHILD_TUPLES = [
    (PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN
]

PART_CHANNELS = [
    "left_face",
    "right_face",
    "right_upper_leg_front",
    "right_lower_leg_back",
    "right_upper_leg_back",
    "left_lower_leg_front",
    "left_upper_leg_front",
    "left_upper_leg_back",
    "left_lower_leg_back",
    "right_feet",
    "right_lower_leg_front",
    "left_feet",
    "torso_front",
    "torso_back",
    "right_upper_arm_front",
    "right_upper_arm_back",
    "right_lower_arm_back",
    "left_lower_arm_front",
    "left_upper_arm_front",
    "left_upper_arm_back",
    "left_lower_arm_back",
    "right_hand",
    "right_lower_arm_front",
    "left_hand",
]


def draw_keypoints(
    img,
    instance_scores,
    keypoint_scores,
    keypoint_coords,
    min_pose_confidence=0.5,
    min_part_confidence=0.5,
):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv.KeyPoint(kc[1], kc[0], 10.0 * ks))
    out_img = cv.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if (
            keypoint_scores[left] < min_confidence
            or keypoint_scores[right] < min_confidence
        ):
            continue
        results.append(
            np.array(
                [keypoint_coords[left][::-1], keypoint_coords[right][::-1]]
            ).astype(np.int32),
        )
    return results


def draw_skeleton(
    img,
    instance_scores,
    keypoint_scores,
    keypoint_coords,
    min_pose_confidence=0.5,
    min_part_confidence=0.5,
):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence
        )
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv.polylines(
        out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0)
    )
    return out_img


def draw_skel_and_kp(
    img,
    instance_scores,
    keypoint_scores,
    keypoint_coords,
    min_pose_score=0.5,
    min_part_score=0.5,
):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score
        )
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv.KeyPoint(kc[1], kc[0], 10.0 * ks))

    out_img = cv.drawKeypoints(
        out_img,
        cv_keypoints,
        outImage=np.array([]),
        color=(255, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    out_img = cv.polylines(
        out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0)
    )
    return out_img


def unprocess_keypoint_coords(coords, image_shape, input_image_shape):

    ratio_y = input_image_shape[0] * (image_shape[0] / input_image_shape[0])
    ratio_x = input_image_shape[1] * (image_shape[1] / input_image_shape[1])

    coords[:, :, 0] *= ratio_y
    coords[:, :, 1] *= ratio_x

    return coords
