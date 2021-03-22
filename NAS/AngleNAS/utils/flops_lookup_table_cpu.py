
from flops_test_blocks_cpu import *
import time
import sys, os
import argparse
sys.setrecursionlimit(10000)
import functools
print=functools.partial(print,flush=True)
import pickle

sys.path.append("../")
from thop import profile

class Flops_Tester():
    def __init__(self, savepath):
        self.op_flops_dict = {}
        self.savepath = savepath
        if os.path.exists(self.savepath):
            print('load {}'.format(self.savepath))
            f = open(self.savepath, 'rb')
            self.op_flops_dict = pickle.load(f)
            f.close()

        self.inp_oup_sizes = [ # inp, oup, img_h, img_w, stride
            (3,     32,     224,    224,    2),     #conv1
            (16,    32,     112,    112,    2),     #stride = 2
            (32,    32,     56,     56,     1),
            (32,    40,     56,     56,     2),     #stride = 2
            (40,    40,     28,     28,     1),
            (40,    40,     28,     28,     1),   
            (40,    40,     28,     28,     1),
            (40,    80,     28,     28,     2),     #stride = 2
            (80,    80,     14,     14,     1),
            (80,    80,     14,     14,     1),  
            (80,    80,     14,     14,     1),
            (80,    96,     14,     14,     1),
            (96,    96,     14,     14,     1),
            (96,    96,     14,     14,     1),
            (96,    96,     14,     14,     1),
            (96,    192,    14,     14,     2),     #stride = 2
            (192,   192,    7,      7,      1),
            (192,   192,    7,      7,      1),
            (192,   192,    7,      7,      1),  
            (192,   320,    7,      7,      1),
            (320,   1280,   7,      7,      1),     # post_processing
        ]
        self.layers = len(self.inp_oup_sizes)
        self.op_keys = op_keys

    def test_flops_of_block(self, op_key, inp, oup, img_h, img_w, stride):
        input_size = (1, inp, img_h, img_w)
        op = blocks_dict[op_key](inp, oup, stride)
        flops, params = profile(op, input_size)
        return flops, params

    def run(self):
        for i in range(self.layers):
            inp, oup, img_h, img_w, stride = self.inp_oup_sizes[i]
            print('layer={}'.format(i))
            if i == 0:
                flops, _ = self.test_flops_of_block(self.op_keys[i], inp, oup, img_h, img_w, stride)
                key_name = self.op_keys[i]
                if key_name not in self.op_flops_dict:
                    self.op_flops_dict[key_name] = {}
                self.op_flops_dict[key_name][(inp, oup, img_h, img_w, stride)] = flops
                print(i, key_name, inp, oup, img_h, img_w, stride, flops)
            elif i == self.layers - 1:
                flops, _ = self.test_flops_of_block(self.op_keys[-1], inp, oup, img_h, img_w, stride)
                key_name = self.op_keys[-1]
                if key_name not in self.op_flops_dict:
                    self.op_flops_dict[key_name] = {}
                self.op_flops_dict[key_name][(inp, oup, img_h, img_w, stride)] = flops
                print(i, key_name, inp, oup, img_h, img_w, stride, flops)
            else:
                for op_key in self.op_keys[1:-1]:
                    flops, _ = self.test_flops_of_block(op_key, inp, oup, img_h, img_w, stride)
                    if op_key not in self.op_flops_dict:
                        self.op_flops_dict[op_key] = {}
                    self.op_flops_dict[op_key][(inp, oup, img_h, img_w, stride)] = flops
                    print(i, op_key, inp, oup, img_h, img_w, stride, flops)
            
        print('dump {}'.format(self.savepath))
        f = open(self.savepath, 'wb')
        pickle.dump(self.op_flops_dict, f)
        f.close()
        keys = sorted(self.op_flops_dict.keys())
        for key in keys:
            info = self.op_flops_dict[key]
            tuple_key = sorted(info.keys())
            for tk in tuple_key:
                print(key, tk, info[tk])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--savepath', type=str, default='../op_flops_dict_cpu.pkl')

    args=parser.parse_args()
    tester = Flops_Tester(savepath=args.savepath)
    tester.run()

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
