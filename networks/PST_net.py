import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
from models.PMMs_new import PMMs
from utils import contrast_loss

class OneModel(nn.Module):
    def __init__(self, args):

        self.inplanes = 64
        self.num_pro = 3
        super(OneModel, self).__init__()

        self.model_res = resnet.Res50_Deeplab(pretrained=True)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer55 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5),
        )

        self.layer56 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5),
        )

        self.layer6 = ASPP.PSPnet()

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5),

        )

        self.layer9 = nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=True)  # numclass = 2

        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.PMMsU = PMMs(256, self.num_pro, stage_num=10)

        self.batch_size = args.batch_size
        self.mode = args.mode

    def forward(self, query_rgb, support_rgb, support_mask, query_mask = None):
        if self.mode =='train':
            logits = self.forward_train(query_rgb, support_rgb, support_mask, query_mask)
        else:
            logits = self.forward_test(query_rgb, support_rgb, support_mask)

        return logits

    def forward_train(self, query_rgb, support_rgb, support_mask, query_mask):
        # important: do not optimize the RESNET backbone
        # A means support set
        # B measns query set

        # extract A feature
        support_feature = self.extract_feature_res(support_rgb)

        # extract B feature
        query_feature = self.extract_feature_res(query_rgb)

        # generate semantic vector
        vec_pos_normal, mu_f_s, mu_b_s = self.PMMsU.generate_prototype(support_feature, support_mask)
        Prob_map_normal, P_normal = self.PMMsU.discriminative_model(query_feature, mu_f_s, mu_b_s)

        vec_pos_q, mu_f_q, mu_b_q = self.PMMsU.generate_prototype(query_feature, query_mask)
        # Match Prototypes
        mu_f_s, mu_f_q = contrast_loss.MCMFMatch(mu_f_s, mu_f_q)
        Prob_map_self, P_self = self.PMMsU.discriminative_model(query_feature, mu_f_q, mu_b_q)

        exit_feat_in, Prob_Q = self.trans2query(mu_f_s, mu_b_s, query_feature)


        # segmentation
        out, _ = self.IoM(exit_feat_in, Prob_Q)

        return support_feature, P_normal, P_self, out

    def trans2query(self, mu_f_s, mu_b_s, query_feature):
        Prob_map_normal, P_normal = self.PMMsU.discriminative_model(query_feature, mu_f_s, mu_b_s)
        b,k,w,h = P_normal.shape

        z = P_normal.view(b,k,-1)
        z_t = F.softmax(z, dim=1)

        mu = torch.cat([mu_f_s,mu_b_s],dim=1)
        mu = mu.permute(0,2,1)
        x = torch.bmm(mu,z_t)
        c = x.shape[1]
        x=x.view(b,c,w,h)

        # sup->query
        exit_feat_in = self.p_match(x, query_feature)

        return exit_feat_in, Prob_map_normal

    def forward_test(self, query_rgb, support_rgb, support_mask):
        # extract A feature
        support_feature = self.extract_feature_res(support_rgb)

        # extract B feature
        query_feature = self.extract_feature_res(query_rgb)
        
        # generate semantic vector
        vec_pos_normal, mu_f_s, mu_b_s = self.PMMsU.generate_prototype(support_feature, support_mask)

        # sup->query
        exit_feat_in, Prob_Q = self.trans2query(mu_f_s, mu_b_s, query_feature)
        
        # segmentation
        out, _ = self.IoM(exit_feat_in, Prob_Q)

        return support_feature, query_feature, Prob_Q, out

    def forward_5shot(self, query_rgb, support_rgb_batch, support_mask_batch):
        # extract B feature
        query_feature = self.extract_feature_res(query_rgb)
        # feature concate
        feature_size = query_feature.shape[-2:]

        mean = torch.zeros(5).cuda()
        for i in range(support_rgb_batch.shape[1]):
            mean[i] = torch.sum(torch.sum(support_mask_batch[:, i], dim=3), dim=2)
        avg = torch.mean(mean, dim=0)  # mean
        mean = avg / mean

        for i in range(support_rgb_batch.shape[1]):
            support_rgb = support_rgb_batch[:, i]
            support_mask = support_mask_batch[:, i]
            # extract A feature
            support_feature = self.extract_feature_res(support_rgb)
            # generate semantic vector
            support_mask_temp = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear',
                                              align_corners=True)
            if i == 0:
                feature_cat = support_feature
                mask_cat = support_mask_temp
            else:
                feature_cat = torch.cat([feature_cat,support_feature],dim=2)
                mask_cat = torch.cat([mask_cat,support_mask_temp],dim=2)
        vec_pos, mu_f_s, mu_b_s = self.PMMsU.generate_prototype(feature_cat, mask_cat)
        exit_feat_in, Prob_Q = self.trans2query(mu_f_s, mu_b_s, query_feature)
        out, _ = self.IoM(exit_feat_in, Prob_Q)
        return out, out, out, out

    def p_match(self, vec_pos,query_feature):
        feature_size = query_feature.shape[-2:]

        exit_feat_in_ = self.f_v_concate(query_feature, vec_pos, feature_size)
        exit_feat_in = self.layer55(exit_feat_in_)

        exit_feat_in = self.layer56(exit_feat_in)
        return exit_feat_in

    def extract_feature_res(self, rgb):
        out_resnet = self.model_res(rgb)
        stage2_out = out_resnet[1]
        stage3_out = out_resnet[2]
        out_23 = torch.cat([stage2_out, stage3_out], dim=1)
        feature = self.layer5(out_23)

        return feature

    def f_v_concate(self, feature, vec_pos, feature_size):
        fea_pos = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        exit_feat_in = torch.cat([feature, fea_pos], dim=1)

        return exit_feat_in

    def IoM(self, feature, history_mask):
        feature_size = feature.shape[-2:]

        history_mask = F.interpolate(history_mask, feature_size, mode='bilinear', align_corners=True)
        out = feature
        out_plus_history = torch.cat([out, history_mask], dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer9(out)

        out_softmax = F.softmax(out, dim=1)

        return out, out_softmax

    def get_loss(self, logits, query_label, support_mask):
        bce_logits_func = nn.CrossEntropyLoss()
        support_feature, P_normal, P_self, outB_side = logits

        #contrastive loss
        Prob_map, P_label = self.trans_loss(P_normal, P_self)
        loss_bce_seg2 = bce_logits_func(Prob_map, P_label.long())

        # segmentation loss
        b, c, w, h = query_label.size()
        outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
        query_label = query_label.view(b, -1)
        bb, cc, _, _ = outB_side.size()
        outB_side = outB_side.view(b, cc, w * h)
        loss_bce_seg1 = bce_logits_func(outB_side, query_label.long())

        # Merge
        loss = loss_bce_seg1 + 0.1*(loss_bce_seg2)

        return loss, loss_bce_seg1, loss_bce_seg2

    def trans_loss(self, P_normal, P_self):
        # construct prob map
        Prob_map_b = torch.sum(P_normal[:, self.num_pro:], dim=1).unsqueeze(dim=1) / self.num_pro  # background
        Prob_map_f = P_normal[:, :self.num_pro]
        Prob_map = torch.cat([Prob_map_f, Prob_map_b], dim=1)

        # construct label
        _, P_label = torch.max(P_self, dim=1)
        P_label[P_label > (self.num_pro-1)] = self.num_pro
        P_label = P_label.long()

        return Prob_map, P_label

    def get_pred(self, logits, query_image):
        outB, outA_pos, outB_side1, outB_side = logits
        w, h = query_image.size()[-2:]
        outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
        out_softmax = F.softmax(outB_side, dim=1)  # .squeeze()
        values, pred = torch.max(out_softmax, dim=1)
        return out_softmax, pred
