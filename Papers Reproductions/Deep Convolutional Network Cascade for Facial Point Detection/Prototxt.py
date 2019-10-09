"""
This file generate prototxt file for LEVEL-2 and LEVEL-3
"""

import sys

name_dict = {
    's0': ['F'],
    's1': ['EN', 'NM'],
    's3': ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2']}


def generate(network, level, names, mode='GPU'):
    """
    Generate template
    :param network: CNN type
    :param level: LEVEL
    :param names: CNN names
    :param mode: CPU or GPU
    """
    assert(mode == 'GPU' or mode == 'CPU')

    types = ['train', 'solver', 'deploy']
    for name in names:
        for type in types:
            templateFile = '{0}_{1}.prototxt.template'.format(network, type)
            with open(templateFile, 'r') as fd:
                template = fd.read()
                outputFile = '{0}_{1}_{2}.prototxt'.format(level, name, type)
                with open(outputFile, 'w') as fd:
                    fd.write(template.format(level=level, name=name, mode=mode))


def generate_train(network, level, names):
    for name in names:
        templateFile = 'train.template'
        with open(templateFile, 'r') as fd:
            template = fd.read()
            outputFile = '{0}_{1}_train.sh'.format(level, name)
            with open(outputFile, 'w') as fd:
                fd.write(template.format(level=level, name=name))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        mode = 'GPU'
    else:
        mode = 'CPU'
    generate('s0', 1, name_dict['s0'], mode)
    generate('s1', 1, name_dict['s1'], mode)
    generate('s3', 2, name_dict['s3'], mode)
    generate('s3', 3, name_dict['s3'], mode)

    generate_train('s0', 1, name_dict['s0'])
    generate_train('s1', 1, name_dict['s1'])
    generate_train('s3', 2, name_dict['s3'])
    generate_train('s3', 3, name_dict['s3'])

