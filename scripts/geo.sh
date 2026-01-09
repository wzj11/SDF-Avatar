# echo $#
echo $1



if [[ ! -f "outputs_track/$1/objects/single.obj" ]]; then
    python src/geo_finetune.py --config configs/config_track.yaml
fi


