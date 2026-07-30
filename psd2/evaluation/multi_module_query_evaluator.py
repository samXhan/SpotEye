# psd2/evaluation/multi_module_query_evaluator.py

from collections import OrderedDict
from copy import copy as shallow_copy
import itertools
import json
import logging
import os
from typing import Dict, List

import numpy as np
import torch

import psd2.utils.comm as comm
from .query_evaluator import QueryEvaluator

logger = logging.getLogger(__name__)


class MultiModuleQueryEvaluator(QueryEvaluator):
    """
    在原有 QueryEvaluator 基础上增加:
    - 多模块指标统计: module_aps / module_accs
    - 案例记录: case_records, 用于后续错误分析
    """

    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
        module_names=None,
        max_case_records: int = 500,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            s_threds=s_threds,
        )

        if module_names is None:
            # 至少有一个模块: "main" = 原始评测逻辑
            module_names = ["main"]

        self.module_names: List[str] = module_names

        # 多模块: 每个模块 & 每个 score 阈值都有一套 AP / topk 统计
        self.module_aps: Dict[str, Dict[float, List[float]]] = {
            m: {st: [] for st in self.det_score_thresh} for m in self.module_names
        }
        self.module_accs: Dict[str, Dict[float, List[List[float]]]] = {
            m: {st: [] for st in self.det_score_thresh} for m in self.module_names
        }

        # query 案例列表 (用于错误分析): 每条是一个 dict
        self.case_records: List[dict] = []
        self.max_case_records = max_case_records

    def reset(self):
        super().reset()
        self.module_aps = {
            m: {st: [] for st in self.det_score_thresh} for m in self.module_names
        }
        self.module_accs = {
            m: {st: [] for st in self.det_score_thresh} for m in self.module_names
        }
        self.case_records = []

    # ============ 子类在 process() 中调用的两个辅助接口 ============

    def update_metrics_for_modules(
        self,
        det_score_th: float,
        module_ap_dict: Dict[str, float],
        module_acc_dict: Dict[str, List[float]],
    ):
        """
        在子类 process() 里用:
        - det_score_th: 当前使用的 score 阈值
        - module_ap_dict:   { "main": ap_main, "aux": ap_aux, ... }
        - module_acc_dict:  { "main": [top1, top5, top10], "aux": [...] }
        """
        assert det_score_th in self.det_score_thresh, f"Unknown th {det_score_th}"

        for m in self.module_names:
            if m not in module_ap_dict or m not in module_acc_dict:
                continue
            self.module_aps[m][det_score_th].append(float(module_ap_dict[m]))
            self.module_accs[m][det_score_th].append(
                [float(x) for x in module_acc_dict[m]]
            )

    def record_case(
        self,
        query_img_path: str,
        gallery_img_paths: List[str],
        gt_pid: int,
        gallery_pids: List[int],
        module_scores: Dict[str, List[float]],
        is_match_flags: List[bool],
        extra_info: dict = None,
    ):
        """
        记录一条 query 的检索结果, 用于后续错误分析.
        """
        if len(self.case_records) >= self.max_case_records:
            return

        rec = {
            "query_img": query_img_path,
            "gallery_imgs": gallery_img_paths,
            "gt_pid": int(gt_pid),
            "gallery_pids": [int(x) for x in gallery_pids],
            "modules": {
                m: [float(s) for s in module_scores.get(m, [])]
                for m in self.module_names
            },
            "is_match": [bool(x) for x in is_match_flags],
        }
        if extra_info is not None:
            rec["extra"] = extra_info

        self.case_records.append(rec)

    # ================= evaluate: 多模块 + 写出 query_cases.json ==============

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps, dst=0)
            accs_all = comm.gather(self.accs, dst=0)
            module_aps_all = comm.gather(self.module_aps, dst=0)
            module_accs_all = comm.gather(self.module_accs, dst=0)
            case_records_all = comm.gather(self.case_records, dst=0)

            if not comm.is_main_process():
                return {}

            aps = {}
            accs = {}
            module_aps = {
                m: {st: [] for st in self.det_score_thresh} for m in self.module_names
            }
            module_accs = {
                m: {st: [] for st in self.det_score_thresh} for m in self.module_names
            }
            case_records = list(itertools.chain(*case_records_all))

            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))
                for m in self.module_names:
                    module_aps[m][dst] = list(
                        itertools.chain(*[ma[m][dst] for ma in module_aps_all])
                    )
                    module_accs[m][dst] = list(
                        itertools.chain(*[mac[m][dst] for mac in module_accs_all])
                    )
        else:
            aps = self.aps
            accs = self.accs
            module_aps = self.module_aps
            module_accs = self.module_accs
            case_records = self.case_records

        # 原 search 结果
        search_result = OrderedDict()
        search_result["search"] = {}
        for dst in self.det_score_thresh:
            logger.info("Search eval_{:.2f} on {} queries. ".format(dst, len(aps[dst])))
            mAP = float(np.mean(aps[dst])) if len(aps[dst]) > 0 else 0.0
            search_result["search"][f"mAP_{dst:.2f}"] = mAP

            if len(accs[dst]) > 0:
                acc = np.mean(np.array(accs[dst]), axis=0)
            else:
                acc = np.zeros(len(self.topks), dtype=np.float32)
            for i, v in enumerate(acc.tolist()):
                k = self.topks[i]
                search_result["search"][f"top{k:02d}_{dst:.2f}"] = float(v)

        # 多模块统计
        search_result["modules"] = {}
        for m in self.module_names:
            mod_res = {}
            for dst in self.det_score_thresh:
                m_ap_list = module_aps[m][dst]
                if len(m_ap_list) > 0:
                    m_mAP = float(np.mean(m_ap_list))
                else:
                    m_mAP = 0.0
                mod_res[f"mAP_{dst:.2f}"] = m_mAP

                m_acc_list = module_accs[m][dst]
                if len(m_acc_list) > 0:
                    m_acc = np.mean(np.array(m_acc_list), axis=0)
                else:
                    m_acc = np.zeros(len(self.topks), dtype=np.float32)
                for i, v in enumerate(m_acc.tolist()):
                    k = self.topks[i]
                    mod_res[f"top{k:02d}_{dst:.2f}"] = float(v)

            search_result["modules"][m] = mod_res

        # 写案例 JSON
        if comm.is_main_process():
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                cases_path = os.path.join(self._output_dir, "query_cases.json")
                with open(cases_path, "w", encoding="utf-8") as f:
                    json.dump(case_records, f, indent=2, ensure_ascii=False)
                logger.info(
                    f"[MultiModuleQueryEvaluator] Saved {len(case_records)} cases to {cases_path}"
                )
            except Exception as e:
                logger.error(f"Failed to save query_cases.json: {e}")

        return shallow_copy(search_result)
