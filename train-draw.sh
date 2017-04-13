export THEANO_FLAGS=device=cuda2,floatX=float32

# CAUTION SET GLOBAL VAR NORM
#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 100 --bs 128 --initial_rec_gamma 0.5 --initial_c_gamma 0.5 \
#                     --init 'ortho' --force_norm --lr_decay 5e-5 \
#                     --name 'norm_bs=128,lr=0.01,lr_decay=5e-5,rec_gamma=0.5,c_gamma=0.5,force_norm,new_decay'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 100 --bs 128 --initial_rec_gamma 1.0 --initial_c_gamma 1.0 \
#                     --layer_norm --init 'baseline' --lr_decay 1e-3 --name 'ln_bs=128,lr=0.01,lr_decay=1e-3,baseline'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 100 --bs 128 --lr_decay 1e-3 \
#                     --init 'baseline' --name 'baseline_bs=128,lr=0.01,lr_decay=1e-3,baseline'

#Leto14
#for CG in 0.1 0.3
#do
#    for RG in 0.1 0.3 0.5 0.7 0.9 1.0 1.25 1.5
#    do
#        for SCALE in 0.1 0.3 0.5 0.7 0.9 1.0
#        do
#            python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.001 --epochs 5 --bs 128 \
#                                 --initial_rec_gamma $RG --initial_c_gamma $CG --scale $SCALE \
#                                 --name 'norm_bs=128,lr=0.001,rec_gamma='$RG',c_gamma='$CG',ortho,scale='$SCALE
#        done
#    done
#done

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.001 --epochs 5 --bs 128 \
#                     --initial_rec_gamma 1.0 --initial_c_gamma 1.0 


# ----------------------------------------------------------------------------
# FINAL EXPERIMENTS
# ----------------------------------------------------------------------------

# Baseline
# CAUTION: SET GLOBAL VAR NORM
#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.001 --epochs 200 --bs 128 --lr_decay 1e-4 \
#                     --init 'baseline' --name 'baseline,lr=0.001,lr_decay=1e-4,init=baseline'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 200 --bs 128 --lr_decay 1e-3 \
#                     --init 'ortho' --name 'baseline,lr=0.01,lr_decay=1e-3,init=ortho'

# Layer Norm
# CAUTION: SET GLOBAL VAR NORM
#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.001 --epochs 200 --bs 128 --initial_rec_gamma 1.0 --initial_c_gamma 1.0 \
#                     --layer_norm --init 'baseline' --lr_decay 1e-5 --name 'ln,lr=0.001,lr_decay=1e-5,init=baseline'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.001 --epochs 200 --bs 128 --initial_rec_gamma 1.0 --initial_c_gamma 1.0 \
#                     --layer_norm --init 'ortho' --lr_decay 1e-5 --name 'ln,lr=0.001,lr_decay=1e-5,init=ortho'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.001 --epochs 200 --bs 128 --initial_rec_gamma 1.0 --initial_c_gamma 1.0 \
#                     --layer_norm --init 'baseline' --lr_decay 1e-3 --name 'ln,lr=0.001,lr_decay=1e-3,init=baseline'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.001 --epochs 200 --bs 128 --initial_rec_gamma 1.0 --initial_c_gamma 1.0 \
#                     --layer_norm --init 'ortho' --lr_decay 1e-3 --name 'ln,lr=0.001,lr_decay=1e-3,init=ortho'


# WEIGHT NORMALIZATION
#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 200 --bs 128 --initial_rec_gamma 1.0 --lr_decay 1e-3 --weight_norm \
#                     --init 'baseline' --force_norm --name 'wn,lr=0.01,lr_decay=1e-3,force,gamma=1.0,init=baseline'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 200 --bs 128 --initial_rec_gamma 1.0 --lr_decay 1e-3 --weight_norm \
#                     --init 'ortho' --force_norm --name 'wn,lr=0.01,lr_decay=1e-3,force,gamma=1.0,init=ortho'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 200 --bs 128 --initial_rec_gamma 0.5 --lr_decay 1e-3 --weight_norm \
#                     --init 'baseline' --force_norm --name 'wn,lr=0.01,lr_decay=1e-3,force,gamma=0.5,init=baseline'

#python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 200 --bs 128 --initial_rec_gamma 0.5 --lr_decay 1e-3 --weight_norm \
#                     --init 'ortho' --force_norm --name 'wn,lr=0.01,lr_decay=1e-3,force,gamma=0.5,init=ortho'

# NORM PROP
python train-draw.py --dataset 'bmnist' --attention 2,5 --niter 64 --lr 0.01 --epochs 200 --bs 128 --initial_rec_gamma 0.5 --initial_c_gamma 0.5 \
                     --init 'baseline' --force_norm --lr_decay 1e-3 \
                     --name 'norm,lr=0.01,lr_decay=1e-3,force,gamma=0.5,init=baseline'
