paths=`ls outputs_track`
# echo $paths

for path in $paths;do
    if [[ -d "outputs/$path/slats" ]];then
        echo "processing $path"
        cp -r outputs/$path/slats/* "outputs_track/$path/slats/"
    fi
done