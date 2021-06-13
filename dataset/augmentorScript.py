import Augmentor

for i in range(0, 10):
    p = Augmentor.Pipeline(str(i))
    p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
    p.process()