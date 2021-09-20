import random
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from torch.nn import CosineSimilarity, MarginRankingLoss
from misc.config import Config
from GlobalAttention import func_attention

cfg = Config()


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.bool)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.bool)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


#################################### stochastic triplet loss ####################################

def cosine_distance(x1, x2, dim=1, eps=1e-8):
    """Returns (1 - cosine similarity) between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return 1 - (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def triplet_loss_with_cosine_distance(anc, pos, neg, margin=0.5):
    score = cosine_distance(anc, pos) - cosine_distance(anc, neg) + margin
    #     z = torch.zeros_like(score)
    return TF.relu(score)


def sent_triplet_loss(cnn_code, rnn_code, labels, neg_ids, batch_size):
    anchor_ids = labels
    positive_ids = labels
    negative_ids = neg_ids

    img_anchor = cnn_code[anchor_ids]
    text_positive = rnn_code[positive_ids]
    text_negative = rnn_code[negative_ids]

    text_anchor = rnn_code[anchor_ids]
    img_positive = cnn_code[positive_ids]
    img_negative = cnn_code[negative_ids]

    i2t_triplet_loss = triplet_loss_with_cosine_distance(img_anchor, text_positive, text_negative,
                                                         margin=cfg.sent_margin).mean()
    t2i_triplet_loss = triplet_loss_with_cosine_distance(text_anchor, img_positive, img_negative,
                                                         margin=cfg.sent_margin).mean()

    return i2t_triplet_loss, t2i_triplet_loss


######## word triplet loss #########
def word_similarity(img_features, words_emb, words_num):
    # -> 1 x nef x words_num
    word = words_emb[0, :, :words_num].unsqueeze(0).contiguous()
    # 1 x nef x 17*17
    context = img_features
    """
        word(query): 1 x nef x words_num
        context: 1 x nef x 16 x 16
        weiContext: 1 x nef x words_num
        attn: 1 x words_num x 16 x 16
    """
    weiContext, attn = func_attention(word, context, cfg.GAMMA1)
    att_maps = attn[0].unsqueeze(0).contiguous()
    # --> batch_size x words_num x nef
    word = word.transpose(1, 2).contiguous()
    weiContext = weiContext.transpose(1, 2).contiguous()
    # --> batch_size*words_num x nef
    word = word.view(1 * words_num, -1)
    weiContext = weiContext.view(1 * words_num, -1)
    #
    # -->batch_size*words_num
    row_sim = cosine_similarity(word, weiContext)
    # --> batch_size x words_num
    row_sim = row_sim.view(1, words_num)

    # Eq. (10)
    row_sim.mul_(cfg.GAMMA2).exp_()
    row_sim = row_sim.sum(dim=1, keepdim=True)
    row_sim = -torch.log(row_sim)
    return row_sim, att_maps


def triplet_loss_with_word_similarity(anc, pos, neg, words_num, flag='img', margin=0.5):
    if flag == 'img':
        pos_score, pos_attn_map = word_similarity(anc, pos, words_num)
        neg_score, _ = word_similarity(anc, neg, words_num)
        score = pos_score - neg_score + margin
    elif flag == 'text':
        pos_score, pos_attn_map = word_similarity(pos, anc, words_num)
        neg_score, _ = word_similarity(neg, anc, words_num)
        score = pos_score - neg_score + margin

    #     z = torch.zeros_like(score)
    return TF.relu(score), pos_attn_map


def words_triplet_loss(img_features, words_emb, labels, neg_ids, cap_lens, batch_size):
    anchor_ids = labels
    positive_ids = labels
    negative_ids = neg_ids

    i2t_triplet_loss_arr = []
    t2i_triplet_loss_arr = []
    attn_maps = []
    for i in range(batch_size):
        img_anchor = img_features[anchor_ids[i:i + 1]]  # 1 x nef x 16 x 16
        text_positive = words_emb[positive_ids[i:i + 1]]  # 1 x nef x 256
        text_negative = words_emb[negative_ids[i:i + 1]]  #
        i2t_triplet_loss, img_pos_attn_map = \
            triplet_loss_with_word_similarity(img_anchor, text_positive, text_negative, cap_lens[i], flag='img',
                                              margin=cfg.word_margin)
        i2t_triplet_loss_arr.append(i2t_triplet_loss)
        attn_maps.append(img_pos_attn_map)

        text_anchor = words_emb[anchor_ids[i:i + 1]]
        img_positive = img_features[positive_ids[i:i + 1]]
        img_negative = img_features[negative_ids[i:i + 1]]
        t2i_triplet_loss, _ = \
            triplet_loss_with_word_similarity(text_anchor, img_positive, img_negative, cap_lens[i], flag='text',
                                              margin=cfg.word_margin)
        t2i_triplet_loss_arr.append(t2i_triplet_loss)

    i2t_triplet_loss = torch.cat(i2t_triplet_loss_arr, 1).mean()
    t2i_triplet_loss = torch.cat(t2i_triplet_loss_arr, 1).mean()

    return i2t_triplet_loss, t2i_triplet_loss, attn_maps


##############################################################################################################

def ranking_loss(z_image, z_text, y, report_id,
                 similarity_function='dot'):
    """
    A custom ranking-based loss function
    Args:
        z_image: a mini-batch of image embedding features
        z_text: a mini-batch of text embedding features
        y: a 1D mini-batch of image-text labels 
    """
    return imposter_img_loss(z_image, z_text, y, report_id, similarity_function) + \
           imposter_txt_loss(z_image, z_text, y, report_id, similarity_function)


def imposter_img_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the image is an imposter image chosen from the batch 
    """
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)

    for i in range(batch_size):
        if similarity_function == 'dot':
            paired_similarity = torch.dot(z_image[i], z_text[i])
        if similarity_function == 'cosine':
            paired_similarity = \
                torch.dot(z_image[i], z_text[i]) / (torch.norm(z_image[i]) * torch.norm(z_text[i]))
        if similarity_function == 'l2':
            paired_similarity = -1 * torch.norm(z_image[i] - z_text[i])

        # Select an imposter image index and 
        # compute the maximum margin based on the image label difference
        j = i + 1 if i < batch_size - 1 else 0
        if report_id[i] == report_id[j]:
            # This means the imposter image comes from the same acquisition
            margin = 0
        elif y[i].item() == -1 or y[j].item() == -1:  # '-1' means unlabeled
            margin = 0.5
        else:
            margin = max(0.5, (y[i] - y[j]).abs().item())

        if similarity_function == 'dot':
            imposter_similarity = torch.dot(z_image[j], z_text[i])
        if similarity_function == 'cosine':
            imposter_similarity = \
                torch.dot(z_image[j], z_text[i]) / (torch.norm(z_image[j]) * torch.norm(z_text[i]))
        if similarity_function == 'l2':
            imposter_similarity = -1 * torch.norm(z_image[j] - z_text[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        if diff_similarity > 0:
            loss = loss + diff_similarity

    return loss / batch_size  # 'mean' reduction


def imposter_txt_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the text is an imposter text chosen from the batch 
    """
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)

    for i in range(batch_size):
        if similarity_function == 'dot':
            paired_similarity = torch.dot(z_image[i], z_text[i])
        if similarity_function == 'cosine':
            paired_similarity = \
                torch.dot(z_image[i], z_text[i]) / (torch.norm(z_image[i]) * torch.norm(z_text[i]))
        if similarity_function == 'l2':
            paired_similarity = -1 * torch.norm(z_image[i] - z_text[i])

        # Select an imposter text index and 
        # compute the maximum margin based on the image label difference
        j = i + 1 if i < batch_size - 1 else 0
        if report_id[i] == report_id[j]:
            # This means the imposter report comes from the same acquisition 
            margin = 0
        elif y[i].item() == -1 or y[j].item() == -1:  # '-1' means unlabeled
            margin = 0.5
        else:
            margin = max(0.5, (y[i] - y[j]).abs().item())

        if similarity_function == 'dot':
            imposter_similarity = torch.dot(z_text[j], z_image[i])
        if similarity_function == 'cosine':
            imposter_similarity = \
                torch.dot(z_text[j], z_image[i]) / (torch.norm(z_text[j]) * torch.norm(z_image[i]))
        if similarity_function == 'l2':
            imposter_similarity = -1 * torch.norm(z_text[j] - z_image[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        if diff_similarity > 0:
            loss = loss + diff_similarity

    return loss / batch_size  # 'mean' reduction


def dot_product_loss(z_image, z_text):
    batch_size = z_image.size(0)
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    for i in range(batch_size):
        loss = loss - torch.dot(z_image[i], z_text[i])
    return loss / batch_size
