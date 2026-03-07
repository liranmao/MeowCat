#BSUB -J 04_multi_xenium
#BSUB -q mingyaogpu
#BSUB -n 2
#BSUB -M 120000
#BSUB -o /home/liranmao/06_he_anno/code/MeowCat/MeowCat/examples/04_multi_xenium/03062026_ex04_run.out
#BSUB -e /home/liranmao/06_he_anno/code/MeowCat/MeowCat/examples/04_multi_xenium/03062026_ex04_run.err
#BSUB -gpu "num=1"

source activate he_anno
cd /home/liranmao/06_he_anno/code/MeowCat/MeowCat/examples/04_multi_xenium
bash run.sh