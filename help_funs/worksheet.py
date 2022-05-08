import mu
import file_io
import json

data_file = "D:\\TUK\\improved_normal_inference\\paper\\pic\\00440.data0.json"
depth_file = "D:\\TUK\\improved_normal_inference\\paper\\pic\\00440.depth0_noise.png"

f = open(data_file)
data = json.load(f)
f.close()

depth = file_io.load_scaled16bitImage(depth_file,
                                      data['minDepth'],
                                      data['maxDepth'])

output_name = "00440.vertex"
mu.visual_input(depth, data, output_name)
