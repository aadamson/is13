# !/bin/bash
source .env/bin/activate
# THEANO_FLAGS='device=gpu,nvcc.fastmath=True,optimizer_including=local_ultra_fast_sigmoid' python examples/deep-sentence-forward.py -nh 30 -d 3 --emb GoogleNews-vectors-negative300.bin --ex ese_sents.pkl
THEANO_FLAGS='optimizer_including=local_ultra_fast_sigmoid' python examples/deep-sentence-forward.py -nh 30 -d 3 --emb GoogleNews-vectors-negative300.bin --ex dse_sents.pkl
