import Augmentor

p = Augmentor.Pipeline("./data/test/aug")
p.ground_truth("./data/test/aug_gt")

p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
p.random_distortion(probability=0.7, grid_width=4, grid_height=4, magnitude=8)
p.skew_tilt(probability=0.5, magnitude=0.3)
p.skew(probability=0.5, magnitude=0.3)

p.sample(395*10, multi_threaded=True)
