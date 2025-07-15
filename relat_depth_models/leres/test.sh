export PYTHONPATH="/home/cyf/03_tansparent/AdelaiDepth/LeReS/Minist_Test"

python ./tools/test_depth.py --load_ckpt res50.pth --backbone resnet50

# python ./tools/test_depth.py --load_ckpt res101.pth --backbone resnext101

# run the ResNet-50
# python ./tools/test_shape.py --load_ckpt res50.pth --backbone resnet50

# run the ResNeXt-101
# python ./tools/test_shape.py --load_ckpt res101.pth --backbone resnext101