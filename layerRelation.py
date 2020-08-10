# import sys
# sys.path.insert(0,'/public/chenchanggui/wuqi/tf_train/caffe/python')

# import caffe
# from caffe.proto.caffe_pb2 import NetParameter
# from google.protobuf import text_format
from collections import OrderedDict

class Relation():
    def __init__(self, layer_idx_1, layer_idx_2):
        self.layer_first = layer_idx_1
        self.layer_second = layer_idx_2
        self.S_first = None
        self.S_second = None

    def __repr__(self):
        return '({}, {})'.format(self.layer_first, self.layer_second)

    def get_idxs(self):
        return self.layer_first, self.layer_second

    def set_scale_vec(self, S_first, S_second):
        if self.S_first is None and self.S_second is None:
            self.S_first = S_first
            self.S_second = S_second
        else:
            self.S_first *= S_first
            self.S_second *= S_second

    def get_scale_vec(self):
        return self.S_first, self.S_second

def count_layer_bottomsize(net_txt):
    layer_bottom_count_dict = OrderedDict()
    layer = net_txt.layer

    for layer_index in range(len(layer)):
        if layer[layer_index].type in ['Input']:
            continue
        if layer[layer_index].bottom == layer[layer_index].top:
            continue
        for bottom in layer[layer_index].bottom:
            if not bottom in layer_bottom_count_dict:
                layer_bottom_count_dict[bottom] = 1
            else:
                layer_bottom_count_dict[bottom] += 1

    return layer_bottom_count_dict


def create_relation(net_txt,layer_bottom_count_dict):
    relation_dict = OrderedDict()
    layer = net_txt.layer

    for layer_index in range(len(layer)):

        if layer[layer_index].top[0] not in layer_bottom_count_dict or layer_bottom_count_dict[layer[layer_index].top[0]]!=1:
            continue

        if layer[layer_index].type == 'Scale' and layer[layer_index+1].type == 'Convolution' and layer[layer_index+2].type == 'ReLU' :
            relation_dict[layer[layer_index].name] = Relation(layer[layer_index].name, layer[layer_index+1].name)
        elif layer[layer_index].type == 'Scale' and layer[layer_index+1].type == 'ReLU' and layer[layer_index+2].type == 'Convolution' :
            relation_dict[layer[layer_index].name] = Relation(layer[layer_index].name, layer[layer_index+2].name)
        elif layer[layer_index].type == 'Convolution' and layer[layer_index+1].type == 'ReLU' and layer[layer_index+2].type == 'Scale':
            relation_dict[layer[layer_index].name] = Relation(layer[layer_index].name, layer[layer_index+2].name)

        elif layer[layer_index].type == 'Convolution' and layer[layer_index+1].type == 'ReLU' and layer[layer_index+2].type == 'Convolution':
            relation_dict[layer[layer_index].name] = Relation(layer[layer_index].name, layer[layer_index+2].name)

        elif layer[layer_index].type == 'Convolution' and layer[layer_index+1].type == 'InnerProduct':
            relation_dict[layer[layer_index].name] = Relation(layer[layer_index].name, layer[layer_index+1].name)
        elif layer[layer_index].type == 'Convolution' and layer[layer_index+1].type == 'ReLU' and layer[layer_index+2].type == 'InnerProduct':
            relation_dict[layer[layer_index].name] = Relation(layer[layer_index].name, layer[layer_index+2].name)

    return list(relation_dict.values())


if __name__ == '__main__':
    prototxt = "../v8211_dfq/model_v8_2_11.prototxt"

    with open(prototxt, 'r') as fp:
        net_txt = NetParameter()
        text_format.Parse(fp.read(), net_txt)
        layer_bottom_count_dict = count_layer_bottomsize(net_txt)
        res = create_relation(net_txt,layer_bottom_count_dict)

        for pair in res:
            print(pair)

