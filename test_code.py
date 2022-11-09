'''
def cache_labels(self, path=Path('./labels.cache'), prefix=''):  # 重写
    # Cache dataset labels, check images and read shapes
    x = {}  # dict
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing(所有图片没有标注的数目和), found(找到的标注和), empty(虽然有标注文件，但是文件内啥都没写), duplicate(读取时候出现问题的样本数目)
    pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(
        self.img_files))  # 产生这么个进度条，Scanning images:   0%|                          | 0/118287 [00:00<?, ?it/s]
    for i, (im_file, lb_file) in enumerate(pbar):  # 循环每个样本，图像jpg-标注txt对
        try:
            # verify images
            im = Image.open(im_file)  # 验证图像是否可以打开
            im.verify()  # PIL verify  # 检查文件完整性
            shape = exif_size(im)  # 获得 image size
            segments = []  # instance segments
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in img_formats, f'invalid image format {im.format}'

            # verify labels
            if os.path.isfile(lb_file):
                nf += 1  # label found
                with open(lb_file, 'r') as f:
                    l = [x.split() for x in f.read().strip().splitlines()]  # 把标注txt 文件的每行(一个标注)都读取出来组成list
                    if any([len(x) > 8 for x in l]):  # is segment 如果长度大于8那么该标注是分割
                        classes = np.array([x[0] for x in l],
                                           dtype=np.float32)  # 标注的第一列代表类别，是一个字符串类型的数字， 如 '45', 这里组成当前文件的类别list：如 [45.0, 45.0, 50.0, 45.0, 49.0, 49.0, 49.0, 49.0]
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in
                                    l]  # 除了第一列，后面每两个数是一个标注的坐标，把每个实例分割框的每个点坐标 reshape 下
                        l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)),
                                           1)  # (cls, xywh) 如(8,5) 这里 xy 是目标中心点坐标，并且xywh 都是经过归一化的
                    l = np.array(l, dtype=np.float32)
                if len(l):
                    assert l.shape[1] == 5, 'labels require 5 columns each'  # 即 cls,xywh
                    assert (l >= 0).all(), 'negative labels'  # 所有值都  >= 0
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'  # bbox 坐标不能在 图像外
                    assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'  # 标注里面有重复的框
                else:
                    ne += 1  # label empty
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm += 1  # label missing
                l = np.zeros((0, 5), dtype=np.float32)
            x[im_file] = [l, shape, segments]  # x是一个dict，key 为 图像path，value：该图像的标注(如 8，5)， 图像的宽高，分割的坐标
        except Exception as e:
            nc += 1
            print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

        pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                    f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"  # 更新进度条
    pbar.close()

    if nf == 0:
        print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

    x['hash'] = get_hash(self.label_files + self.img_files)
    x['results'] = nf, nm, ne, nc, i + 1  # 统计的数目
    x['version'] = 0.1  # cache version
    torch.save(x, path)  # save for next time
    logging.info(f'{prefix}New cache created: {path}')
    return x

'''
from pycocotools.coco import COCO
import numpy as np
from skimage import io  # scikit-learn 包
import matplotlib.pyplot as plt
import pylab
from utils.general import *

def show_sample():
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # COCO 是一个类, 因此, 使用构造函数创建一个 COCO 对象, 构造函数首先会加载 json 文件,
    # 然后解析图片和标注信息的 id, 根据 id 来创建其关联关系.

    dataDir = '/home/wangyu_kyland/dataset/coco/images/val2017'
    dataType = 'val2017'
    annFile = '/home/wangyu_kyland/dataset/coco/annotations/instances_val2017.json'
    # 初始化标注数据的 COCO api
    coco = COCO(annFile)
    print("数据加载成功！")

    # 显示 COCO 数据集中的具体类和超类
    categories = coco.loadCats(coco.getCatIds())
    label_names = [cat['name'] for cat in categories]
    # for i, name in enumerate(label_names):
    #     print(name, ' == ', i)
    # print('COCO categories: \n{}\n'.format(' '.join(label_names)))
    # print("类别总数为： %d" % len(label_names))
    super_names = set([cat['supercategory'] for cat in categories])
    print('COCO supercategories: \n{}'.format(' '.join(super_names)))
    print("超类总数为：%d " % len(super_names))

    # 加载并显示指定 图片 id
    catIds = coco.getCatIds(catNms=['oven', 'refrigerator'])
    imgIds = coco.getImgIds(catIds=catIds)
    # imgIds = coco.getImgIds(imgIds=[724])
    img = coco.loadImgs(imgIds[0])[0]
    I = io.imread(img['coco_url'])
    plt.axis('off')
    plt.imshow(I)
    plt.show()

    # 加载并将 “segmentation” 标注信息显示在图片上
    # 加载并显示标注信息
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns, draw_bbox=True)


def read_lables():
    lb_file = '/home/wangyu_kyland/dataset/coco/labels/val2017/000000000139.txt'
    with open(lb_file, 'r') as f:
        l = [x.split() for x in f.read().strip().splitlines()]
        if any([len(x) > 8 for x in l]):  # is segment
            classes = np.array([x[0] for x in l], dtype=np.float32)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
        l = np.array(l, dtype=np.float32)
    print(l)

def test_segments2bbox():
    segments = [0.379609, 0.497793, 0.376938, 0.468404, 0.397375, 0.457723, 0.399156, 0.47108, 0.395594] #, 0.499108]
    segments = np.array(segments)
    segments = [segments.reshape(-1, 3)]
    x, y, z = segments[0].T

    print(segments)
    results = segments2boxes(segments)
    print(results)

if __name__ == '__main__':
    # test_segments2bbox()
    read_lables()