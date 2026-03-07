#BSUB -J 05_multi_visium_xenium
#BSUB -q mingyaogpu
#BSUB -n 2
#BSUB -M 120000
#BSUB -o /home/liranmao/06_he_anno/code/MeowCat/MeowCat/examples/05_multi_visium_xenium/03052026_ex05_run.out
#BSUB -e /home/liranmao/06_he_anno/code/MeowCat/MeowCat/examples/05_multi_visium_xenium/03052026_ex05_run.err
#BSUB -gpu "num=1"

source activate he_anno
cd /home/liranmao/06_he_anno/code/MeowCat/MeowCat/examples/05_multi_visium_xenium
bash run.sh