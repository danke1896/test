# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmcv.cnn import bias_init_with_prob, normal_init, xavier_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@HEADS.register_module()
class DoubleWeightsHead(AnchorFreeHead):
    """
    DoubleWeightsead
    
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
        
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 sample_channels,
                 sample_num=256,
                 pos_factor=0.5,
                 center_sampling=False,
                 center_sample_radius=1.5,
                 topk=10,
                 norm_on_bbox=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_sample=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.norm_on_bbox = norm_on_bbox
        self.sample_channels = sample_channels
        self.sample_num = sample_num
        self.pos_factor = pos_factor
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.topk = topk
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_sample = build_loss(loss_sample)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()

        self.conv_iou = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.sample_predict = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.sample_channels),
            nn.ReLU(),
            nn.Linear(in_features=self.sample_channels, out_features=5))

    def init_weights(self):
        """Initialize weights of the head.

        In particular, we have special initialization for classified conv's and
        regression conv's bias
        """

        super(DoubleWeightsHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)

        for m in self.sample_predict:
            xavier_init(m, gain=1.0, bias=0.0)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats,
                           self.strides)

    def forward_single(self, x, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions\
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        iou_score = self.conv_iou(reg_feat)
        return cls_score, bbox_pred, iou_score

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_scores'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_scores,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            iou_scores (list[Tensor]): IoU scores for each scale level.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(iou_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, sample_scores, weights, sample_preds, sample_labels \
            = self.get_targets(cls_scores, bbox_preds, iou_scores,
                               all_level_points,
                               gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_iou_scores = [
            iou_score.permute(0, 2, 3, 1).reshape(-1, 1)
            for iou_score in iou_scores
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_iou_scores = torch.cat(flatten_iou_scores)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_sample_scores = torch.cat(sample_scores)
        flatten_weights = torch.cat(weights)
        flatten_sample_preds = torch.cat(sample_preds)
        flatten_sample_labels = torch.cat(sample_labels)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero(as_tuple=False).reshape(-1)
        pos_labels = flatten_labels[pos_inds]

        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        # compute iou loss
        pos_scores = flatten_sample_scores[pos_inds]
        pos_scores = torch.exp(pos_scores ** 5)
        pos_weights_pred = flatten_weights[pos_inds]
        pos_weights = pos_scores * pos_weights_pred

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_weights,
                avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()

        # compute classification loss
        flatten_cls_scores = flatten_cls_scores.sigmoid()
        flatten_iou_scores = flatten_iou_scores.sigmoid()
        flatten_scores = flatten_cls_scores * flatten_iou_scores

        flatten_cls_labels = torch.zeros_like(flatten_scores, device=flatten_scores.device)
        flatten_cls_labels[pos_inds, pos_labels] = 1.0

        flatten_cls_weights = torch.ones_like(flatten_scores, device=flatten_scores.device)
        flatten_cls_weights[pos_inds, pos_labels] = pos_weights

        loss_cls = self.loss_cls(flatten_scores,
                                 flatten_cls_labels,
                                 weight=flatten_cls_weights,
                                 avg_factor=num_pos)

        # compute sample loss
        loss_samples = self.loss_sample(flatten_sample_preds,
                                        flatten_sample_labels,
                                        avg_factor=num_pos)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_samples=loss_samples)

    def get_targets(self,
                    pred_cls,
                    pred_bbox,
                    pred_ious,
                    points,
                    gt_bboxes_list,
                    gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            pred_cls (list[Tensor]): Prediction of classification.
            pred_bbox (list[Tensor]): Prediction of bbox.
            pred_ious (list[Tensor]): Prediction of iou scores.
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        num_levels = len(points)
        num_points = [center.size(0) for center in points]
        concat_points = torch.cat(points, dim=0)

        # flatten prediction information to get the cost with gt
        pred_bbox = [pred_bbox[i].detach() for i in range(len(pred_bbox))]
        pred_cls = [pred_cls[i].detach() for i in range(len(pred_cls))]
        pred_ious = [pred_ious[i].detach() for i in range(len(pred_ious))]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(cls_score.shape[0], -1, self.cls_out_channels)
            for cls_score in pred_cls
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.shape[0], -1, 4)
            for bbox_pred in pred_bbox
        ]

        flatten_iou_scores = [
            iou_score.permute(0, 2, 3, 1).reshape(iou_score.shape[0], -1, 1)
            for iou_score in pred_ious
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_iou_scores = torch.cat(flatten_iou_scores, dim=1)

        flatten_cls_scores = flatten_cls_scores.sigmoid() * flatten_iou_scores.sigmoid()

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, scores_list, weights_list, sample_list, sample_label_list = multi_apply(
            self._get_target_single,
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        scores_list = [scores.split(num_points, 0) for scores in scores_list]
        weights_list = [weights.split(num_points, 0) for weights in weights_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_scores = []
        concat_lvl_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)

            concat_lvl_scores.append(torch.cat([scores[i] for scores in scores_list]))
            concat_lvl_weights.append(torch.cat([weights[i] for weights in weights_list]))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_scores, concat_lvl_weights, sample_list, sample_label_list

    def _get_target_single(self,
                           pred_cls,
                           pred_bbox,
                           gt_bboxes,
                           gt_labels,
                           points,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        # compute iou between gt and pred
        pred_bbox = distance2bbox(points, pred_bbox)
        iou = bbox_overlaps(pred_bbox, gt_bboxes, mode='iou', eps=1e-6)

        # compute cls_score for gt
        pos_cls = pred_cls[:, gt_labels]
        neg_cls = pred_cls.sum(1)[..., None] - pos_cls

        # get the sample prediction
        pred_iou = iou.view(-1, num_gts, 1)
        pos_cls = pos_cls.view(-1, num_gts, 1)
        neg_cls = neg_cls.view(-1, num_gts, 1)

        sample_vector = torch.cat([pred_iou, pos_cls, neg_cls], dim=2)
        sample_pred = self.sample_predict(sample_vector)
        sample_scores = sample_pred[..., 0]
        sample_mui = sample_pred[..., 1:3]
        sample_sigma = sample_pred[..., 3:]

        sample_scores = F.sigmoid(sample_scores)
        sample_mui = F.relu(sample_mui)
        sample_sigma = F.relu(sample_sigma)

        # get the samples in gt bboxes
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        strides = []
        for i in range(len(num_points_per_lvl)):
            stride = torch.tensor([self.strides[i]])[None].repeat(num_points_per_lvl[i], 1)
            strides.append(stride)
        strides = torch.cat(strides).to(pred_bbox.device)
        center_weights_prior = self.center_prior(sample_mui, sample_sigma, points, gt_bboxes, strides)
        center_weights = center_weights_prior.clone().detach()
        center_weights[inside_gt_bbox_mask] = 1.0

        sample_vector = sample_scores.detach().clone()
        sample_vector[~inside_gt_bbox_mask] = 0.0
        sample_vector = sample_vector * center_weights

        # choose top-k positive samples for every target
        labels = torch.full([points.shape[0]], self.num_classes, dtype=torch.int64, device=pred_cls.device)
        bbox_target = torch.full([points.shape[0], 4], 0, dtype=pred_bbox.dtype, device=pred_bbox.device)
        weights = torch.full([points.shape[0]], 1.0, dtype=points.dtype, device=points.device)
        scores = torch.full([points.shape[0]], 1.0, dtype=points.dtype, device=points.device)
        matching_matrix = torch.full(sample_vector.shape, 0, dtype=sample_vector.dtype, device=sample_vector.device)
        topk_samples, _ = torch.topk(sample_vector, self.topk, dim=0)

        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_samples.sum(0).int(), min=1)
        for idx in range(len(gt_labels)):
            _, pos_idx = torch.topk(sample_vector[:, idx], k=dynamic_ks[idx])
            matching_matrix[pos_idx, idx] = 1.0

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.max(
                sample_vector[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0

        index = torch.where(matching_matrix > 0.0)
        labels[index[0]] = gt_labels[index[1]]
        bbox_target[index[0]] = bbox_targets[index]
        weights[index[0]] = center_weights_prior[index]
        scores[index[0]] = sample_scores[index]

        # 获取样本的标签（随机采样n个）
        pos_idx = torch.where(inside_gt_bbox_mask > 0)
        neg_idx = torch.where(inside_gt_bbox_mask == 0)

        num_pos = int(self.sample_num * self.pos_factor)
        num_neg = int(self.sample_num - num_pos)

        num_pos_max = dynamic_ks.sum()
        num_pos = min(pos_idx[0].shape[0], num_pos, num_pos_max)
        num_pos = max(num_pos - index[0].shape[0], 0)
        num_neg = min(neg_idx[0].shape[0], num_neg)

        idx_pos = torch.randperm(pos_idx[0].numel(), device=pos_idx[0].device)[:num_pos]
        idx_neg = torch.randperm(neg_idx[0].numel(), device=neg_idx[0].device)[:num_neg]
        idx_pos_gt = torch.cat([pos_idx[1][idx_pos], index[1]])
        idx_pos = torch.cat([pos_idx[0][idx_pos], index[0]])

        idx_neg_gt = neg_idx[1][idx_neg]
        idx_neg = neg_idx[0][idx_neg]
        pos_sample = sample_scores[idx_pos, idx_pos_gt]
        pos_sample_labels = torch.ones_like(pos_sample).to(pos_sample.device)
        neg_sample = sample_scores[idx_neg, idx_neg_gt]
        neg_sample_labels = torch.zeros_like(neg_sample).to(neg_sample.device)

        samples = torch.cat([pos_sample, neg_sample])
        sample_labels = torch.cat([pos_sample_labels, neg_sample_labels])

        return labels, bbox_target, scores, weights, samples, sample_labels

    def center_prior(self, sample_mui, sample_sigma, points, gt_bboxes, strides):
        """Get the center prior of each point on the feature map for each
        instance.

        Args:
            sample_mui
            sample_sigma
            points (list[Tensor]): list of coordinate
                of points on feature map. Each with shape
                (num_points, 2).
            gt_bboxes (Tensor): The gt_bboxes with shape of
                (num_gt, 4).
            gt_labels (Tensor): The gt_labels with shape of (num_gt).

            strides (Tensor): The stride for each level
        Returns:
            tuple(Tensor):

                - center_prior_weights(Tensor): Float tensor with shape \
                    of (num_points, num_gt). Each value represents \
                    the center weighting coefficient.

        """

        num_gts = len(gt_bboxes)
        num_points = len(points)
        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points, num_gts)

        mlvl_points = points[:, None, :].expand(num_points, num_gts, 2)
        strides = strides[:, None, :].expand(num_points, num_gts, 1)
        gt_center_x = ((gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2)
        gt_center_y = ((gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2)
        gt_center = torch.stack((gt_center_x, gt_center_y), dim=2)

        distance = (((mlvl_points - gt_center) / strides - sample_mui) ** 2)
        center_prior_weights = torch.exp(-distance /
                                         (2 * sample_sigma ** 2)).prod(dim=-1)

        return center_prior_weights

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_scores'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou_scores,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            iou_scores (list[Tensor]): Iou scores for each scale level.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(iou_scores)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        iou_score_list = [iou_scores[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list, iou_score_list,
                                       mlvl_points, img_shapes, scale_factors,
                                       cfg, rescale, with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    iou_scores,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            iou_scores (list[Tensor]): IoU scores for a single scale level.
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(iou_scores) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, iou_score, points in zip(
                cls_scores, bbox_preds, iou_scores, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels)
            iou_score = iou_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 1)
            scores = cls_score.sigmoid() * iou_score.sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)

            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = scores.max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
                    points = points.reshape(-1, 2)[transformed_inds, :].reshape(batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points
