paths=`ls /public/home/wangzhijun/Ners`
for path in $paths; do
    echo $path
    rm -r /public/home/wangzhijun/Ners/$path/seg_masks
    rm /public/home/wangzhijun/Ners/$path/ori_imgs/000000.jpg
    # rm -r /public/home/wangzhijun/Ners/$path/parsing


done