import os

data_path = '/local/DEEPLEARNING/oid/'

classes_path = os.path.join(data_path, 'annotations', 'challenge-2018-class-descriptions-500.csv')
output_path = os.path.join(data_path, 'annotations', 'challenge-2018-classes.csv')

classes = dict()
with open(classes_path, 'r') as f_cls, open(output_path, 'w+') as f_out:
    for i, line in enumerate(f_cls):
        parts = line.strip().split(',')
        classes[parts[0]] = {'id': i, 'name': parts[1], 'original_id': parts[0]}

        f_out.write(','.join([str(i), parts[1], parts[0]]) + '\n')  # output csv: id,name,original_id
