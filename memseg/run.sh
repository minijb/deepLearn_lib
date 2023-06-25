#!/bin/zsh
source  /home/zhouhao/anaconda3/bin/activate memseg

echo 'available options'
echo $(ls -l configs | sed '1,3d' | sed -r 's/.*[0-9]{2}:[0-9]{2} //' | sed -r 's/.yaml//')

echo 'option: '
read option
python main.py --yaml_config ./configs/$option.yaml
