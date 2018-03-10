'''
format of .lst file:
0       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer95.jpeg
1       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer70.jpeg
2       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer73.jpeg
3       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer17.jpeg
4       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer13.jpeg
5       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer38.jpeg
6       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer49.jpeg
7       1       /home/ubuntu/projects/images/deep_fryer/img/1_deepfryer28.jpeg
8       0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer18.jpg
9       0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer27.jpg
10      0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer50.jpg
11      0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer13.jpg
12      0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer11.jpg
13      0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer58.jpg
14      0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer6.jpg
15      0       /home/ubuntu/projects/images/deep_fryer/img/0_deepfryer36.jpg
'''

# Dependencies
import os
import sys
import random


# Global constants ----


# Functions ----
def write_lst(image_arr, base_dir, file_path):
    with open(file_path, 'w') as f:
        count = 0
        for img in image_arr:
            label = img['label']
            img_path = os.path.join(base_dir, img['filename'])
            new_line = '\t'.join([str(count), str(label), str(img_path)])
            new_line += '\n'
            f.write(new_line)
            count += 1


def split_dataset(from_idx, to_idx):
    return map['1'][from_idx: to_idx] + map['0'][from_idx: to_idx]


# Main ----
if __name__ == '__main__':
    data_dir = sys.argv[1]
    fnames = os.listdir(data_dir)

    map = {
        '0': [],
        '1': []
    }
    counter = 0

    for fn in fnames:
        arr = fn.split('_')
        category = arr[0]
        if category in ['0','1']:
            map[category].append({
                'idx': counter,
                'label': int(category),
                'filename': fn,
            })
            counter += 1

    for i in map:
        print(i, len(map[i]))

    random.shuffle(map['1'])
    random.shuffle(map['0'])

    # split data range
    min_data_len = min(len(map['1']), len(map['0']))
    train = (0, int(min_data_len * 0.8))
    validation = (int(min_data_len * 0.8), int(min_data_len * 100))
    # test = (int(min_data_len * 0.85), int(min_data_len * 1))

    train_set = split_dataset(train[0], train[1])
    train_lst_path = os.path.join(data_dir, 'train/img.lst')
    write_lst(train_set, data_dir, train_lst_path)

    validation_set = split_dataset(validation[0], validation[1])
    validation_lst_path = os.path.join(data_dir, 'validation/img.lst')
    write_lst(validation_set, data_dir, validation_lst_path)

    # test_set = split_dataset(test[0], test[1])
    # test_lst_path = os.path.join(data_dir, 'test/img.lst')
    # write_lst(test_set, data_dir, test_lst_path)
