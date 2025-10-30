import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def get_attn_mask(bsz, headn, reg_token_num=2, template_num=2, template_len=64, search_len=324, local_size = 3):
    H = search_len+reg_token_num
    W = template_num*template_len+H
    attn_mask = torch.zeros([H, W]).to(torch.bool).cuda()
    s_h = int(math.sqrt(H-reg_token_num))
    for i in range(H-reg_token_num):
        for j in range(template_num*template_len,W-reg_token_num):
            s_i = i
            s_j = j - template_num * template_len
            row_i = int(s_i / s_h)
            col_i = s_i % s_h
            row_j = int(s_j / s_h)
            col_j = s_j % s_h
            if abs(row_i - row_j) > local_size/2 or abs(col_i - col_j) > local_size/2:
                attn_mask[i][j] = True
    #attn_mask = attn_mask.cpu()
    #for i in range(400):
    #    img = attn_mask[i+template_num*template_len]
    #    img = img[template_num*template_len:]
    #    img = torch.reshape(img, (20,20))
    #    plt.imshow(img)
    #    plt.show()
    #    print(img)
    attn_mask = attn_mask.unsqueeze(0)
    ret_mask=attn_mask
    for i in range(headn-1):
        ret_mask = torch.cat((ret_mask, attn_mask), 0)
    ret_mask = ret_mask.unsqueeze(0)
    ret_mask2=ret_mask
    for i in range(bsz-1):
        ret_mask2 = torch.cat((ret_mask2, ret_mask), 0)
    return ret_mask2