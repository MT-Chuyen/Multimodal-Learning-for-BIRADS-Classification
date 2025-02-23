python3 main.py --model cross_attn
# cross_attn = True

python3 main.py --model fusion
# Không dùng cross_attn

python3 main.py --model cross_attn --self_meta
# self_meta = True, cross_attn = True

python3 main.py --model cross_attn --self_image
# self_image = True, cross_attn = True

python3 main.py --model cross_attn --self_meta --self_image
# self_meta = True, self_image = True, cross_attn = True



# python main.py --model cross_attn --loss hierarchical
# python main.py --model cross_attn --loss attention_entropy --lambda_entropy 0.1
# python main.py --model cross_attn --loss weighted_multitask --alpha 0.4 --beta 0.6