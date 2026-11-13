"""Label-free detector geometry, scene graph and table support."""
import numpy as np
import pandas as pd
BOX = ['x1', 'y1', 'x2', 'y2']
STATIC = ['confidence', 'class_name', 'cx', 'cy', 'width', 'height']
DUPLICATES = ['same_class_iou', 'higher_confidence_iou', 'overlap_count']
PAIR_COLUMNS = ['near_person', 'near_hand', 'near_item']
CHAIN_COLUMNS = ['full_chain', 'local_hand_count', 'local_item_count', 'excess_hands', 'frame_hands_per_person', 'frame_items_per_hand']
SURFACE_COLUMNS = ['on_table', 'tray_on_table', 'hands_or_surface']
BODY_COLUMNS = ['torso_hand', 'torso_item_chain']

def intersection_over_union(box, others):
    overlap = np.maximum(np.minimum(box[2:], others[:, 2:]) - np.maximum(box[:2], others[:, :2]), 0).prod(axis=1)
    union = np.prod(box[2:] - box[:2]) + (others[:, 2:] - others[:, :2]).prod(axis=1) - overlap
    return overlap / np.maximum(union, 1e-09)

def build_features(split, root):
    rows = pd.read_csv(root / f'{split}.csv', dtype={'id': str})
    rows = rows[rows.class_name != 'table'].copy()
    frames = pd.read_csv(root / f'{split}_frames.csv', usecols=['frame_id', 'timestamp', 'image_width', 'image_height'])
    rows = rows.merge(frames, on='frame_id', how='left', validate='many_to_one')
    rows['day'] = pd.to_datetime(rows.timestamp, utc=True, format='mixed').dt.strftime('%Y%m%d')
    rows['cx'] = (rows.x1 + rows.x2) / (2 * rows.image_width)
    rows['cy'] = (rows.y1 + rows.y2) / (2 * rows.image_height)
    rows['width'] = (rows.x2 - rows.x1) / rows.image_width
    rows['height'] = (rows.y2 - rows.y1) / rows.image_height
    stream = pd.read_csv(root / f'{split}_stream_predictions.csv', dtype={'detection_id': str})
    assert rows.id.is_unique and stream.detection_id.is_unique and rows.id.isin(stream.detection_id).all()
    groups = {key: group for key, group in stream.groupby(['frame_id', 'class_name'])}
    for key, group in rows.groupby(['frame_id', 'class_name']):
        candidates = groups[key]
        boxes = candidates[BOX].to_numpy()
        confidences = candidates.confidence.to_numpy()
        ids = candidates.detection_id.to_numpy()
        for index, row in group.iterrows():
            other = ids != row.id
            overlaps = intersection_over_union(row[BOX].to_numpy(dtype=float), boxes[other])
            higher = overlaps[confidences[other] > row.confidence]
            rows.loc[index, DUPLICATES] = [overlaps.max(initial=0), higher.max(initial=0), np.count_nonzero(overlaps >= 0.3)]
    assert rows[STATIC + DUPLICATES].notna().all().all()
    return rows

def box_gaps(first, second):
    """Расстояние между прямоугольниками, ноль при пересечении."""
    horizontal_vertical = np.maximum(first[:, None, :2] - second[None, :, 2:], second[None, :, :2] - first[:, None, 2:])
    return np.linalg.norm(np.maximum(horizontal_vertical, 0), axis=2)

def frame_graph(frame):
    people = frame.loc[frame.class_name == 'person']
    hands = frame.loc[frame.class_name == 'nude_hands']
    items = frame.loc[frame.class_name.isin(['cup', 'tray'])]
    person_boxes = people[BOX].to_numpy()
    hand_boxes = hands[BOX].to_numpy()
    item_boxes = items[BOX].to_numpy()
    person_scale = np.maximum((person_boxes[:, 2] - person_boxes[:, 0]) * 0.25, 20)
    hand_scale = np.maximum(np.linalg.norm(hand_boxes[:, 2:] - hand_boxes[:, :2], axis=1), 16)
    person_hand = np.exp(-box_gaps(person_boxes, hand_boxes) / person_scale[:, None])
    hand_item = np.exp(-box_gaps(hand_boxes, item_boxes) / hand_scale[:, None])
    person_conf = people.confidence.to_numpy()
    hand_conf = hands.confidence.to_numpy()
    item_conf = items.confidence.to_numpy()
    confidence_people = (person_hand * person_conf[:, None]).max(axis=0, initial=0)
    confidence_items = (hand_item * item_conf[None, :]).max(axis=1, initial=0)
    near = np.exp(-1)
    close_people = person_hand >= near
    close_items = hand_item >= near
    assigned = np.zeros_like(close_people)
    if len(people) and len(hands):
        owner = (person_hand * person_conf[:, None]).argmax(axis=0)
        hand_indices = np.arange(len(hands))
        assigned[owner, hand_indices] = close_people[owner, hand_indices]
    hands_per_person = assigned.sum(axis=1)
    person_features = pd.DataFrame({'near_hand': (person_hand * hand_conf[None, :]).max(axis=1, initial=0), 'full_chain': (person_hand * (hand_conf * confidence_items)[None, :]).max(axis=1, initial=0), 'local_hand_count': hands_per_person, 'excess_hands': np.maximum(hands_per_person - 2, 0)}, index=people.detection_id)
    hand_features = pd.DataFrame({'near_person': confidence_people, 'near_item': confidence_items, 'full_chain': confidence_people * confidence_items, 'local_hand_count': assigned.T @ hands_per_person, 'local_item_count': close_items.sum(axis=1), 'excess_hands': assigned.T @ np.maximum(hands_per_person - 2, 0)}, index=hands.detection_id)
    item_features = pd.DataFrame({'near_hand': (hand_item * hand_conf[:, None]).max(axis=0, initial=0), 'full_chain': (hand_item * (hand_conf * confidence_people)[:, None]).max(axis=0, initial=0), 'local_hand_count': close_items.sum(axis=0)}, index=items.detection_id)
    features = pd.concat([person_features, hand_features, item_features])
    features['frame_hands_per_person'] = len(hands) / max(len(people), 1)
    features['frame_items_per_hand'] = len(items) / max(len(hands), 1)
    return features

def point_support(points, supports, scales):
    rectangles = supports[BOX].to_numpy()
    delta = np.maximum(rectangles[None, :, :2] - points[:, None, :], points[:, None, :] - rectangles[None, :, 2:])
    gap = np.linalg.norm(np.maximum(delta, 0), axis=2)
    return np.exp(-gap / scales[None, :]) * supports.confidence.to_numpy()[None, :]

def table_support(frame):
    tables = frame[frame.class_name == 'table']
    cups = frame[frame.class_name == 'cup']
    trays = frame[frame.class_name == 'tray']
    items = pd.concat([cups, trays])
    centers_x = (items.x1 + items.x2) / 2
    anchors_y = np.where(items.class_name == 'cup', items.y2, (items.y1 + items.y2) / 2)
    anchors = np.column_stack([centers_x, anchors_y])
    tabletop = point_support(anchors, tables, np.maximum((tables.y2 - tables.y1).to_numpy() * 0.2, 12))
    on_table = tabletop.max(axis=1, initial=0)
    tray_scale = np.maximum((trays.y2 - trays.y1).to_numpy() * 0.2, 10)
    cup_to_tray = point_support(anchors[:len(cups)], trays, tray_scale)
    cup_via_tray = (cup_to_tray * on_table[len(cups):][None, :]).max(axis=1, initial=0)
    return pd.DataFrame({'on_table': on_table, 'tray_on_table': np.r_[cup_via_tray, np.full(len(trays), np.nan)]}, index=items.detection_id)

def torso_support(frame):
    people = frame[frame.class_name == 'person']
    hands = frame[frame.class_name == 'nude_hands']
    items = frame[frame.class_name.isin(['cup', 'tray'])]
    bodies = people.copy()
    height = people.y2 - people.y1
    width = people.x2 - people.x1
    bodies['y1'] = people.y1 + 0.2 * height
    bodies['y2'] = people.y1 + 0.82 * height
    bodies['x1'] = people.x1 - 0.1 * width
    bodies['x2'] = people.x2 + 0.1 * width
    hand_points = (hands[['x1', 'y1']].to_numpy() + hands[['x2', 'y2']].to_numpy()) / 2
    torso = point_support(hand_points, bodies, np.maximum(height.to_numpy() * 0.18, 15))
    hand_conf = hands.confidence.to_numpy()
    hand_to_body = torso.max(axis=1, initial=0)
    body_to_hand = (torso * hand_conf[:, None]).max(axis=0, initial=0)
    hand_boxes, item_boxes = (hands[BOX].to_numpy(), items[BOX].to_numpy())
    hand_scale = np.maximum(np.linalg.norm(hand_boxes[:, 2:] - hand_boxes[:, :2], axis=1), 16)
    hand_item = np.exp(-box_gaps(hand_boxes, item_boxes) / hand_scale[:, None])
    item_path = (hand_item * (hand_conf * hand_to_body)[:, None]).max(axis=0, initial=0)
    return pd.concat([pd.DataFrame({'torso_hand': body_to_hand}, index=people.detection_id), pd.DataFrame({'torso_hand': hand_to_body}, index=hands.detection_id), pd.DataFrame({'torso_item_chain': item_path}, index=items.detection_id)])

def add_graph(rows, split, root):
    stream = pd.read_csv(root / f'{split}_stream_predictions.csv', dtype={'detection_id': str})
    stream = stream[stream.frame_id.isin(rows.frame_id) & stream.class_name.isin(['person', 'nude_hands', 'tray', 'cup', 'table'])]
    graphs = []
    for _, frame in stream.groupby('frame_id'):
        graph = frame_graph(frame).join(table_support(frame)).join(torso_support(frame))
        graph['hands_or_surface'] = graph.on_table.combine(graph.full_chain, max)
        graph.loc[graph.on_table.isna(), 'hands_or_surface'] = np.nan
        graphs.append(graph)
    graph = pd.concat(graphs)
    assert graph.index.is_unique and rows.id.isin(graph.index).all()
    return rows.join(graph, on='id', validate='one_to_one')
