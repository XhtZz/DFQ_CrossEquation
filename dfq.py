import caffe
from caffe.proto.caffe_pb2 import NetParameter
from google.protobuf import text_format

import copy
import math
import layerRelation
import numpy as np

def _layer_equalization(weight_first, weight_second, bias_first, s_range=(1e-8, 1e8), eps=0):
    num_group = 1

    pre_weight_first = copy.deepcopy(weight_first)
    pre_weight_second = copy.deepcopy(weight_second)

    if weight_first.shape[0] != weight_second.shape[1]:
        # group convolution
        num_group = weight_first.shape[0] // weight_second.shape[1]
	
    # print('%d %d' % (weight_first.shape[0],weight_second.shape[1]))
    group_channels_i = weight_first.shape[0] // num_group
    group_channels_o = weight_second.shape[0] // num_group

    S_first = np.zeros(weight_first.shape[0])
    S_second = np.zeros(weight_second.shape[1] * num_group)

    # print(weight_first.shape,weight_second.shape)

    for g in range(num_group):
        c_start_i = g * group_channels_i
        c_end_i = (g + 1) * group_channels_i
        weight_first_group = weight_first[c_start_i:c_end_i] # shape [k, c, h, w]

        c_start_o = g * group_channels_o
        c_end_o = (g + 1) * group_channels_o
        weight_second_group = weight_second[c_start_o:c_end_o]

        for ii in range(weight_second_group.shape[1]):
            range_1 = abs(weight_first_group[ii]).max() 
            range_2 = abs(weight_second_group[:, ii]).max()

            # 1 / s = (1 / r1) * sqrt(r1 * r2)
            s = (1 / (range_1 + eps)) * math.sqrt(range_1 * range_2 + eps)
            s = max(s_range[0], min(s_range[1], s))
            S_first[c_start_i + ii] = s

            weight_first[c_start_i + ii] = weight_first[c_start_i + ii] * s

            if bias_first is not None:
                bias_first[c_start_i + ii] = bias_first[c_start_i + ii] * s

            S_second[c_start_i + ii] = 1/s
            weight_second[c_start_o:c_end_o, ii] = weight_second[c_start_o:c_end_o, ii] / s

    diff_first = float(np.mean(np.square(weight_first - pre_weight_first)))
    diff_second = float(np.mean(np.square(weight_second - pre_weight_second)))

    return weight_first, weight_second, bias_first, S_first, S_second, diff_first+diff_second

def cross_layer_equalization(net, relations, s_range=[1e-8, 1e8], range_thres=0, converge_thres=2e-7, converge_count=50, eps=1e-5):
    print("Start cross layer equalization")
    diff = 10
    count = 0
    net_param = net.params

    # for i in range(net_param['conv_last'][0].data.shape[0]):
    #     print(np.max(net_param['conv_last'][0].data[i]), np.min(net_param['conv_last'][0].data[i]), net_param['conv_last'][1].data[i])

    while diff > converge_thres and count < converge_count:
        diff_tmp = 0
        for rr in relations:
            layer_first, layer_second = rr.get_idxs()
            # print('%s %s' % (layer_first,layer_second))

            # layer eualization
            net_param[layer_first][0].data[...], net_param[layer_second][0].data[...], net_param[layer_first][1].data[...], S_first, S_second, diff_cur =   \
            _layer_equalization(net_param[layer_first][0].data[...],                                                                                        \
                                net_param[layer_second][0].data[...],                                                                                       \
                                net_param[layer_first][1].data[...],                                                                                        \
                                s_range=s_range, eps=eps)
            rr.set_scale_vec(S_first, S_second)
            diff_tmp = diff_tmp + diff_cur

        diff = diff_tmp
        count += 1

        print('count', count)
        print('diff', diff)

    return net

if __name__ == '__main__':
    prototxt = "****.prototxt"
    caffemodel = "****.caffemodel"
    dfq_caffemodel = "****.caffemodel"
    with open(prototxt, 'r') as fp:
        net_text = NetParameter()
        text_format.Parse(fp.read(), net_text)
        layer_bottom_count_dict = layerRelation.count_layer_bottomsize(net_text)
        relations = layerRelation.create_relation(net_text,layer_bottom_count_dict)
    
    net = caffe.Net(prototxt,caffemodel,caffe.TEST)
    new_net = cross_layer_equalization(net, relations, converge_count = 1)
    new_net.save(dfq_caffemodel)

    # for rr in relations:
    #     print(rr)
    #     S_first,S_second = rr.get_scale_vec()
    #     print(S_first,S_second)
        
