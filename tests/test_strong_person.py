import numpy as np
from detector_error_discovery.strong_person import fuse


def test_strong_fusion_preserves_hidden_bottom_and_legacy_low_confidence_boxes():
    old=[dict(frame_id='f',boxes=[[10,10,30,50],[100,100,140,160],[200,20,230,60]],
              confidence=[.9,.05,.8],classes=[0,0,0])]
    independent=[dict(frame_id='f',boxes=[[12,14,32,42],[102,102,142,162],[204,24,234,64],[400,400,420,450]],
                      confidence=[.9,.9,.2,.99],classes=[0,0,0,0])]
    actual=fuse(old,independent)[0]
    assert len(actual['boxes'])==3
    np.testing.assert_array_equal(actual['confidence'],old[0]['confidence'])
    np.testing.assert_array_equal(np.asarray(actual['boxes'])[:,3],[50,160,60])
    np.testing.assert_array_equal(actual['boxes'][0],[11,12,31,50])
    np.testing.assert_array_equal(actual['boxes'][1],old[0]['boxes'][1])
    np.testing.assert_array_equal(actual['boxes'][2],old[0]['boxes'][2])
