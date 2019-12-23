import os

import argparse
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_engine(onnx_file_path, mode):
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1
        builder.fp16_mode = True if mode == 'fp16' else False
        builder.int8_mode = True if mode == 'int8' else False
        builder.max_workspace_size = 1 << 32  # 1GB:30

        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())
            for error in range(parser.num_errors):
                print(parser.get_error(error))

        print(len(network))
        for i in range(len(network)):
            print(network[i].type)

        engine = builder.build_cuda_engine(network)

    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_folder', type=str, required=True)
    args = parser.parse_args()

    for i in range(6):
        print('='*80)
        print(i)
        onnx_file_path = os.path.join(args.onnx_folder, '{}.onnx'.format(i))
        get_engine(onnx_file_path, mode='fp32')
