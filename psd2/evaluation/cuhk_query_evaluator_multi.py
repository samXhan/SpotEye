import logging
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torchvision.ops.boxes import box_iou

from .multi_module_query_evaluator import MultiModuleQueryEvaluator
from psd2.structures import Boxes, BoxMode, pairwise_iou

logger = logging.getLogger(__name__)


class CuhkQueryEvaluatorMulti(MultiModuleQueryEvaluator):
    """
    和原来的 CuhkQueryEvaluator 完全一样的评测逻辑，
    只是多了:
    - 多模块统计接口 (这里只用 "main" 一个模块)
    - 记录案例到 case_records, 用于错误分析
    """

    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
        module_names=None,
        max_case_records: int = 500,
    ):
        if module_names is None:
            # 目前只有一个模块: "main"
            module_names = ["main"]
        super().__init__(
            dataset_name=dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            s_threds=s_threds,
            module_names=module_names,
            max_case_records=max_case_records,
        )

    def process(self, inputs, outputs):
        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_instances = query_dict["instances"].to(self._cpu_device)
            g_instances_list = in_dict["gallery"]  # list[Instances]
            q_imgid = q_instances.image_id
            q_pid = q_instances.gt_pids
            q_box = q_instances.org_gt_boxes

            # ★ query 图像路径: 用 file_name (如果你的 Instances 里没有这个字段，可以改为 image_id 或其它)
            q_img_path = getattr(q_instances, "file_name", q_imgid)

            # 用于记录案例（只记录一个 score 阈值下的 top-k 排序，避免太冗余）
            # 这里选择 det_score_thresh 中的第一个阈值作为 "case 用阈值"
            case_th = self.det_score_thresh[0]
            case_scores = []
            case_labels = []
            case_gallery_paths = []
            case_gallery_pids = []

            # 为每个阈值准备统计结构
            y_trues = {dst: [] for dst in self.det_score_thresh}
            y_scores = {dst: [] for dst in self.det_score_thresh}
            count_gts = {dst: 0 for dst in self.det_score_thresh}
            count_tps = {dst: 0 for dst in self.det_score_thresh}

            for dst in self.det_score_thresh:
                pred_instances = outputs[bi].to(self._cpu_device)

                # -------- 以下基本保持你原始逻辑 --------
                if pred_instances.has("reid_feats"):
                    feat_q = pred_instances.reid_feats
                else:
                    query_img_boxes_t, query_img_feats = self._get_gallery_dets(
                        q_imgid, dst
                    )[:, :4], self._get_gallery_feats(q_imgid, dst)
                    if query_img_boxes_t.shape[0] == 0:
                        logger.warning(
                            "Undetected query person in {} !".format(q_imgid)
                        )
                        continue
                    else:
                        ious = pairwise_iou(
                            q_box, Boxes(query_img_boxes_t, BoxMode.XYXY_ABS)
                        )
                        max_iou, nmax = torch.max(ious, dim=1)
                        if max_iou < 0.4:
                            logger.warning(
                                "Low-quality {} query person detected in {} !".format(
                                    max_iou.item(), q_imgid
                                )
                            )
                        feat_q = query_img_feats[nmax.item()]

                name2sim = {}

                # 1. 遍历 gallery
                for item in g_instances_list:
                    gallery_imname = item.image_id
                    gt_boxes = item.org_gt_boxes
                    count_gts[dst] += len(gt_boxes) > 0

                    if gallery_imname not in self.infs:
                        continue

                    det, feat_g = self._get_gallery_dets(
                        gallery_imname, dst
                    ), self._get_gallery_feats(gallery_imname, dst)
                    if det.shape[0] == 0:
                        continue

                    sim = torch.mm(feat_g, feat_q.view(-1)[:, None]).squeeze(1)
                    if gallery_imname in name2sim:
                        continue
                    name2sim[gallery_imname] = sim

                    label = torch.zeros(sim.shape[0], dtype=torch.int)
                    if len(gt_boxes) > 0:
                        hw = gt_boxes.get_sizes().squeeze(0)
                        iou_thresh = min(
                            0.5, (hw[1] * hw[0] * 1.0) / ((hw[1] + 10) * (hw[0] + 10))
                        )
                        inds = torch.argsort(sim)
                        inds = inds.tolist()[::-1]
                        inds = torch.tensor(inds, dtype=torch.long)
                        sim = name2sim[gallery_imname][inds]
                        det = det[inds]

                        for j, roi in enumerate(det[:, :4]):
                            if (
                                box_iou(roi[None, :], gt_boxes.tensor)
                                .squeeze()
                                .item()
                                >= iou_thresh
                            ):
                                label[j] = 1
                                count_tps[dst] += 1
                                break

                    # 加入总体统计
                    y_trues[dst].extend(label.tolist())
                    y_scores[dst].extend(sim.tolist())

                    # ★ 如果是我们挑选记录案例的阈值，把每个 detection 也存起来，方便后续排序分析
                    if dst == case_th:
                        # gallery 图像路径，优先用 file_name 没有就用 image_id
                        g_img_path = getattr(item, "file_name", gallery_imname)
                        pid_val = int(item.gt_pids.item()) if hasattr(item, "gt_pids") else -1
                        for s, lab in zip(sim.tolist(), label.tolist()):
                            case_scores.append(s)
                            case_labels.append(lab)
                            case_gallery_paths.append(g_img_path)
                            case_gallery_pids.append(pid_val)

                # 2. 计算 AP / top-k
                y_score = np.asarray(y_scores[dst])
                y_true = np.asarray(y_trues[dst])
                if len(y_true) == 0:
                    continue
                assert count_tps[dst] <= count_gts[dst]
                recall_rate = (
                    0.0
                    if count_gts[dst] == 0
                    else count_tps[dst] * 1.0 / count_gts[dst]
                )
                ap = (
                    0
                    if count_tps[dst] == 0
                    else average_precision_score(y_true, y_score) * recall_rate
                )
                self.aps[dst].append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true_o = y_true[inds]
                acc_main = [min(1, sum(y_true_o[:k])) for k in self.topks]
                self.accs[dst].append(acc_main)

                # ★ 多模块统计: 目前只有 "main" 一个模块
                module_ap_dict = {"main": ap}
                module_acc_dict = {"main": acc_main}
                self.update_metrics_for_modules(dst, module_ap_dict, module_acc_dict)

            # -------- 在 case_th 下，基于 case_scores 记录一个 query 的 top-k 排名 --------
            if len(case_scores) > 0:
                case_scores = np.asarray(case_scores)
                case_labels = np.asarray(case_labels)
                inds = np.argsort(case_scores)[::-1]

                case_scores_sorted = case_scores[inds]
                case_labels_sorted = case_labels[inds]
                gallery_paths_sorted = [case_gallery_paths[i] for i in inds]
                gallery_pids_sorted = [case_gallery_pids[i] for i in inds]

                # topK 范围内的 score / label
                topK = max(self.topks)
                top_scores = case_scores_sorted[:topK].tolist()
                top_labels = case_labels_sorted[:topK].tolist()
                top_gallery_paths = gallery_paths_sorted[:topK]
                top_gallery_pids = gallery_pids_sorted[:topK]

                module_scores = {"main": top_scores}
                is_match_flags = [bool(x) for x in top_labels]
                gt_pid_int = int(q_pid.item()) if torch.is_tensor(q_pid) else int(q_pid)

                self.record_case(
                    query_img_path=q_img_path,
                    gallery_img_paths=top_gallery_paths,
                    gt_pid=gt_pid_int,
                    gallery_pids=top_gallery_pids,
                    module_scores=module_scores,
                    is_match_flags=is_match_flags,
                    extra_info={
                        "det_score_th": float(case_th),
                        "dataset_name": self.dataset_name,
                    },
                )
