import copy
from collections import Counter, OrderedDict
import csv
import itertools
import json
import logging
import os

import numpy as np
import psd2.utils.comm as comm
import torch

from .evaluator import DatasetEvaluator


logger = logging.getLogger(__name__)  # setup_logger()


class QueryEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
    ) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        # {image name: torch concatenated [boxes, scores, reid features]
        inf_rts = torch.load(
            os.path.join(
                self._output_dir,
                "_gallery_gt_inf.pt"
                if "GT" not in self.dataset_name
                else "_gallery_gt_infgt.pt",
            ),
            map_location=self._cpu_device,
            weights_only=False
        )
        self.gts = inf_rts["gts"]
        self.infs = inf_rts["infs"]
        self.gtfs = inf_rts["gt_fnames"]
        self.topks = [1, 5, 10]
        # statistics
        self.det_score_thresh = s_threds
        self.ignore_cam_id = False
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}
        self.query_failures = {st: [] for st in self.det_score_thresh}
        self.query_records = {st: [] for st in self.det_score_thresh}

        # Opt-in eval hooks: both OFF unless the corresponding parameter is
        # explicitly declared via environment variable.
        #   QUERY_AUDIT_METHOD   -> enables the query-audit CSV hook (and names
        #                           the `method` column). Matches the documented
        #                           `QUERY_AUDIT_METHOD=<name> python ...` usage.
        #   QUERY_FAILURE_DEBUG  -> enables the query-failure debug JSON hook.
        # When disabled, the per-query record builders no-op and nothing is
        # dumped, so normal eval carries no audit/failure overhead.
        self._query_audit_enabled = bool(os.environ.get("QUERY_AUDIT_METHOD"))
        self._query_failure_enabled = bool(os.environ.get("QUERY_FAILURE_DEBUG"))

    def reset(self):
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}
        self.query_failures = {st: [] for st in self.det_score_thresh}
        self.query_records = {st: [] for st in self.det_score_thresh}

    def _get_gallery_dets(self, img_id, score_thred):
        img_dets_all = self.infs[img_id][:, :5]
        return img_dets_all[img_dets_all[:, 4] >= score_thred]

    def _get_gallery_feats(self, img_id, score_thred):
        img_save_all = self.infs[img_id]
        return img_save_all[img_save_all[:, 4] >= score_thred][:, 5:]

    def _get_gt_boxs(self, img_id):
        return self.gts[img_id][:, :4]

    def _get_gt_ids(self, img_id):
        return self.gts[img_id][:, 4].long()

    def process(self, inputs, outputs):
        raise NotImplementedError

    def _to_python_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            value = value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _flatten_text_values(self, value):
        texts = []
        if value is None:
            return texts
        if isinstance(value, str):
            value = value.strip()
            if value:
                texts.append(value)
            return texts
        if isinstance(value, (list, tuple)):
            for item in value:
                texts.extend(self._flatten_text_values(item))
        return texts

    def _extract_query_texts(self, query_dict):
        texts = []
        seen = set()
        for text in self._flatten_text_values(query_dict.get("raw_descriptions")):
            if text not in seen:
                texts.append(text)
                seen.add(text)
        return texts

    def _infer_method_name(self):
        method_name = os.environ.get("QUERY_AUDIT_METHOD")
        if method_name:
            return method_name
        out_dir = os.path.abspath(self._output_dir)
        base = os.path.basename(out_dir)
        if base in {"inference", "custom_eval", "inference_TTA"}:
            parent = os.path.basename(os.path.dirname(out_dir))
            return parent or base
        return base

    def _make_query_summary_record(
        self,
        det_score_thresh,
        query_img_id,
        query_img_path=None,
        query_pid=None,
        ap=None,
        sorted_scores=None,
        sorted_labels=None,
        sorted_gallery_ids=None,
        status=None,
    ):
        # Query-audit hook: skip all record building unless explicitly enabled.
        if not self._query_audit_enabled:
            return None
        scores = (
            np.asarray(sorted_scores, dtype=np.float32).reshape(-1)
            if sorted_scores is not None
            else np.asarray([], dtype=np.float32)
        )
        labels = (
            np.asarray(sorted_labels, dtype=np.int32).reshape(-1)
            if sorted_labels is not None
            else np.asarray([], dtype=np.int32)
        )
        gallery_ids = (
            list(sorted_gallery_ids)
            if sorted_gallery_ids is not None
            else [None] * len(labels)
        )

        topk_hit = {}
        for k in self.topks:
            topk_hit[f"top{k}_correct"] = bool(np.sum(labels[:k]) > 0)

        top1_gallery_img_id = None
        top1_score = None
        top1_correct = False
        if len(labels) > 0:
            top1_gallery_img_id = (
                str(gallery_ids[0]) if gallery_ids[0] is not None else None
            )
            top1_score = float(scores[0])
            top1_correct = bool(labels[0])

        match_inds = np.flatnonzero(labels > 0)
        best_gt_match_rank_global = None
        best_gt_match_similarity = None
        if len(match_inds) > 0:
            best_idx = int(match_inds[0])
            best_gt_match_rank_global = best_idx + 1
            best_gt_match_similarity = float(scores[best_idx])

        if status is None:
            if len(labels) == 0:
                status = "no_ranked_gallery_det"
            elif top1_correct:
                status = "top1_hit"
            elif best_gt_match_rank_global is not None:
                status = "top1_miss"
            else:
                status = "gt_not_retrieved"

        query_img_id = str(query_img_id)
        query_img_path = (
            str(query_img_path) if query_img_path is not None else query_img_id
        )
        pair_key = os.path.basename(query_img_id) or os.path.basename(query_img_path)

        return {
            "dataset_name": self.dataset_name,
            "setting": self.dataset_name,
            "method": self._infer_method_name(),
            "query_img_id": query_img_id,
            "query_img_path": query_img_path,
            "pair_key": pair_key,
            "query_pid": self._to_python_scalar(query_pid),
            "det_score_thresh": float(det_score_thresh),
            "ap": float(ap) if ap is not None else None,
            "status": status,
            "num_ranked_candidates": int(len(labels)),
            "top1_correct": bool(top1_correct),
            "top5_correct": bool(topk_hit.get("top5_correct", False)),
            "top10_correct": bool(topk_hit.get("top10_correct", False)),
            "top1_gallery_img_id": top1_gallery_img_id,
            "top1_score": top1_score,
            "best_gt_match_rank_global": best_gt_match_rank_global,
            "best_gt_match_similarity": best_gt_match_similarity,
        }

    def _record_query_summary(self, det_score_thresh, **kwargs):
        record = self._make_query_summary_record(det_score_thresh, **kwargs)
        if record is not None:
            self.query_records[det_score_thresh].append(record)
        return record

    def _make_gt_gallery_record(self, gallery_img_id, gallery_img_path, gt_boxes_xyxy):
        return {
            "gallery_img_id": str(gallery_img_id) if gallery_img_id is not None else None,
            "gallery_img_path": str(gallery_img_path)
            if gallery_img_path is not None
            else str(gallery_img_id),
            "gt_boxes_xyxy": gt_boxes_xyxy if gt_boxes_xyxy is not None else [],
            "iou_match_thresh": None,
            "num_detections_after_thresh": 0,
            "top_det_rank_in_gallery": None,
            "top_det_rank_global": None,
            "top_det_similarity": None,
            "top_det_score": None,
            "top_det_box_xyxy": None,
            "matched": False,
            "matched_rank_in_gallery": None,
            "matched_rank_global": None,
            "matched_similarity": None,
            "matched_det_score": None,
            "matched_det_box_xyxy": None,
            "status": "missing_inference_result",
        }

    def _collect_gt_gallery_records_without_query(
        self, gallery_instances, det_score_thresh
    ):
        records = []
        for item in gallery_instances:
            gt_boxes = item.org_gt_boxes
            if len(gt_boxes) == 0:
                continue
            gallery_img_id = item.image_id
            gallery_img_path = getattr(item, "file_name", gallery_img_id)
            record = self._make_gt_gallery_record(
                gallery_img_id,
                gallery_img_path,
                gt_boxes.tensor.tolist(),
            )
            if gallery_img_id not in self.infs:
                records.append(record)
                continue

            det = self._get_gallery_dets(gallery_img_id, det_score_thresh)
            record["num_detections_after_thresh"] = int(det.shape[0])
            if det.shape[0] == 0:
                record["status"] = "no_detection_after_thresh"
            else:
                record["top_det_rank_in_gallery"] = 1
                record["top_det_score"] = float(det[0, 4])
                record["top_det_box_xyxy"] = det[0, :4].tolist()
                record["status"] = "query_feature_missing_no_rank"
            records.append(record)
        return records

    def _finalize_gt_gallery_records(
        self, gt_gallery_records, sorted_candidate_uids=None
    ):
        if not gt_gallery_records:
            return [], [], None

        candidate_rank_map = {}
        if sorted_candidate_uids is not None:
            candidate_rank_map = {
                int(uid): rank + 1
                for rank, uid in enumerate(sorted_candidate_uids)
            }

        finalized_records = []
        gt_gallery_img_ids = []
        best_gt_match_rank_global = None
        for raw_record in gt_gallery_records:
            record = copy.deepcopy(raw_record)
            top_det_uid = record.pop("_top_det_uid", None)
            matched_det_uid = record.pop("_matched_det_uid", None)
            if top_det_uid is not None:
                record["top_det_rank_global"] = candidate_rank_map.get(
                    int(top_det_uid)
                )
            if matched_det_uid is not None:
                record["matched_rank_global"] = candidate_rank_map.get(
                    int(matched_det_uid)
                )
            gt_gallery_img_id = record.get("gallery_img_id")
            if gt_gallery_img_id is not None:
                gt_gallery_img_ids.append(gt_gallery_img_id)
            matched_rank_global = record.get("matched_rank_global")
            if matched_rank_global is not None and (
                best_gt_match_rank_global is None
                or matched_rank_global < best_gt_match_rank_global
            ):
                best_gt_match_rank_global = matched_rank_global
            finalized_records.append(record)

        return finalized_records, gt_gallery_img_ids, best_gt_match_rank_global

    def _record_query_failure(
        self,
        det_score_thresh,
        query_img_id,
        query_img_path=None,
        query_pid=None,
        query_texts=None,
        sorted_scores=None,
        sorted_labels=None,
        sorted_gallery_ids=None,
        sorted_gallery_paths=None,
        sorted_det_boxes=None,
        sorted_det_scores=None,
        sorted_candidate_uids=None,
        gt_gallery_records=None,
        failure_reason="top1_miss",
    ):
        # Query-failure hook: skip all record building unless explicitly enabled.
        if not self._query_failure_enabled:
            return
        top_predictions = []
        top1_gallery_img_id = None
        top1_gallery_img_path = None
        top1_score = None
        top1_is_match = False
        topk_hit = {f"top{k}": False for k in self.topks}
        num_ranked_candidates = 0
        (
            gt_gallery_records_out,
            gt_gallery_img_ids,
            best_gt_match_rank_global,
        ) = self._finalize_gt_gallery_records(
            gt_gallery_records,
            sorted_candidate_uids=sorted_candidate_uids,
        )

        if sorted_scores is not None and sorted_labels is not None:
            scores = np.asarray(sorted_scores)
            labels = np.asarray(sorted_labels).astype(np.int32)
            gallery_ids = (
                list(sorted_gallery_ids)
                if sorted_gallery_ids is not None
                else [None] * len(labels)
            )
            gallery_paths = (
                list(sorted_gallery_paths)
                if sorted_gallery_paths is not None
                else gallery_ids
            )
            det_boxes = (
                list(sorted_det_boxes)
                if sorted_det_boxes is not None
                else [None] * len(labels)
            )
            det_scores = (
                list(sorted_det_scores)
                if sorted_det_scores is not None
                else [None] * len(labels)
            )
            num_ranked_candidates = len(labels)
            topn = min(max(self.topks), len(labels))
            for rank in range(topn):
                pred = {
                    "rank": rank + 1,
                    "gallery_img_id": str(gallery_ids[rank])
                    if gallery_ids[rank] is not None
                    else None,
                    "gallery_img_path": str(gallery_paths[rank])
                    if gallery_paths[rank] is not None
                    else None,
                    "score": float(scores[rank]),
                    "is_match": bool(labels[rank]),
                    "pred_box_xyxy": det_boxes[rank],
                    "det_score": float(det_scores[rank])
                    if det_scores[rank] is not None
                    else None,
                }
                top_predictions.append(pred)

            if top_predictions:
                top1_gallery_img_id = top_predictions[0]["gallery_img_id"]
                top1_gallery_img_path = top_predictions[0]["gallery_img_path"]
                top1_score = top_predictions[0]["score"]
                top1_is_match = top_predictions[0]["is_match"]

            for k in self.topks:
                topk_hit[f"top{k}"] = bool(np.sum(labels[:k]) > 0)

        record = {
            "query_img_id": str(query_img_id),
            "query_img_path": str(query_img_path)
            if query_img_path is not None
            else str(query_img_id),
            "query_pid": self._to_python_scalar(query_pid),
            "query_text": query_texts[0] if query_texts else None,
            "query_texts": list(query_texts) if query_texts else [],
            "det_score_thresh": float(det_score_thresh),
            "failure_reason": failure_reason,
            "num_ranked_candidates": int(num_ranked_candidates),
            "top1_gallery_img_id": top1_gallery_img_id,
            "top1_gallery_img_path": top1_gallery_img_path,
            "top1_score": top1_score,
            "top1_is_match": top1_is_match,
            "topk_hit": topk_hit,
            "top_predictions": top_predictions,
            "gt_gallery_img_ids": gt_gallery_img_ids,
            "num_gt_gallery_images": len(gt_gallery_img_ids),
            "best_gt_match_rank_global": best_gt_match_rank_global,
            "gt_gallery_records": gt_gallery_records_out,
        }
        self.query_failures[det_score_thresh].append(record)

    def _dump_query_failure_debug(self, query_failures):
        if self._distributed and not comm.is_main_process():
            return

        payload = {
            "dataset_name": self.dataset_name,
            "thresholds": {},
            "hard_queries": [],
            "common_wrong_top1_galleries": [],
        }
        query_counter = Counter()
        query_paths = {}
        query_thresholds = {}
        wrong_gallery_counter = Counter()
        wrong_gallery_paths = {}

        for dst in self.det_score_thresh:
            records = query_failures.get(dst, [])
            payload["thresholds"][f"{dst:.2f}"] = {
                "num_failed_queries_top1": len(records),
                "records": records,
            }
            logger.info(
                "[QueryEvalDebug] eval_%0.2f top1 failures/no candidates: %d",
                dst,
                len(records),
            )
            if records:
                preview = ", ".join(
                    record["query_img_id"] for record in records[:20]
                )
                logger.info(
                    "[QueryEvalDebug] eval_%0.2f failed query ids (first %d): %s",
                    dst,
                    min(20, len(records)),
                    preview,
                )
            for record in records:
                query_img_id = record["query_img_id"]
                query_counter[query_img_id] += 1
                query_paths[query_img_id] = record.get("query_img_path", query_img_id)
                query_thresholds.setdefault(query_img_id, []).append(float(dst))

                wrong_gallery_id = record.get("top1_gallery_img_id")
                if wrong_gallery_id:
                    wrong_gallery_counter[wrong_gallery_id] += 1
                    wrong_gallery_paths[wrong_gallery_id] = record.get(
                        "top1_gallery_img_path", wrong_gallery_id
                    )

        payload["hard_queries"] = [
            {
                "query_img_id": query_img_id,
                "query_img_path": query_paths.get(query_img_id, query_img_id),
                "fail_count": int(fail_count),
                "failed_thresholds": sorted(query_thresholds.get(query_img_id, [])),
            }
            for query_img_id, fail_count in query_counter.most_common(50)
        ]
        payload["common_wrong_top1_galleries"] = [
            {
                "gallery_img_id": gallery_img_id,
                "gallery_img_path": wrong_gallery_paths.get(
                    gallery_img_id, gallery_img_id
                ),
                "appear_count": int(appear_count),
            }
            for gallery_img_id, appear_count in wrong_gallery_counter.most_common(50)
        ]

        try:
            os.makedirs(self._output_dir, exist_ok=True)
            save_path = os.path.join(self._output_dir, "query_failure_debug.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            if payload["hard_queries"]:
                preview = ", ".join(
                    f"{item['query_img_id']}({item['fail_count']})"
                    for item in payload["hard_queries"][:10]
                )
                logger.info("[QueryEvalDebug] hard queries: %s", preview)
            if payload["common_wrong_top1_galleries"]:
                preview = ", ".join(
                    f"{item['gallery_img_id']}({item['appear_count']})"
                    for item in payload["common_wrong_top1_galleries"][:10]
                )
                logger.info("[QueryEvalDebug] frequent wrong top1 galleries: %s", preview)
            logger.info("[QueryEvalDebug] saved details to %s", save_path)
        except Exception as exc:
            logger.error("[QueryEvalDebug] failed to save debug file: %s", exc)

    def _dump_query_audit_records(self, query_records):
        if self._distributed and not comm.is_main_process():
            return

        rows = []
        for dst in self.det_score_thresh:
            rows.extend(query_records.get(dst, []))
        rows = [r for r in rows if r is not None]
        if not rows:
            logger.warning("[QueryAudit] no query records to save")
            return

        fieldnames = [
            "dataset_name",
            "setting",
            "method",
            "query_img_id",
            "query_img_path",
            "pair_key",
            "query_pid",
            "det_score_thresh",
            "ap",
            "status",
            "num_ranked_candidates",
            "top1_correct",
            "top5_correct",
            "top10_correct",
            "top1_gallery_img_id",
            "top1_score",
            "best_gt_match_rank_global",
            "best_gt_match_similarity",
        ]
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            csv_path = os.path.join(self._output_dir, "query_audit_records.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key) for key in fieldnames})
            logger.info(
                "[QueryAudit] saved %d query-threshold records to %s",
                len(rows),
                csv_path,
            )
        except Exception as exc:
            logger.error("[QueryAudit] failed to save query audit records: %s", exc)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps, dst=0)
            accs_all = comm.gather(self.accs, dst=0)
            query_failures_all = comm.gather(self.query_failures, dst=0)
            query_records_all = comm.gather(self.query_records, dst=0)

            if not comm.is_main_process():
                return {}
            aps = {}
            accs = {}
            query_failures = {}
            query_records = {}
            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))
                query_failures[dst] = list(
                    itertools.chain(*[item[dst] for item in query_failures_all])
                )
                query_records[dst] = list(
                    itertools.chain(*[item[dst] for item in query_records_all])
                )

        else:
            aps = self.aps
            accs = self.accs
            query_failures = self.query_failures
            query_records = self.query_records
        search_result = OrderedDict()
        search_result["search"] = {}

        for dst in self.det_score_thresh:
            logger.info("Search eval_{:.2f} on {} queries. ".format(dst, len(aps[dst])))
            mAP = float(np.mean(aps[dst])) if len(aps[dst]) > 0 else 0.0
            search_result["search"].update({"mAP_{:.2f}".format(dst): mAP})
            acc = (
                np.mean(np.array(accs[dst]), axis=0)
                if len(accs[dst]) > 0
                else np.zeros(len(self.topks), dtype=np.float32)
            )
            # logger.info(str(acc))
            for i, v in enumerate(acc.tolist()):
                # logger.info("{:.2f} on {} acc. ".format(v, i))
                k = self.topks[i]
                search_result["search"].update({"top{:2d}_{:.2f}".format(k, dst): v})

        if self._query_failure_enabled:
            self._dump_query_failure_debug(query_failures)
        if self._query_audit_enabled:
            self._dump_query_audit_records(query_records)

        return copy.deepcopy(search_result)

    def _get_query_gallery_gts(self, query_dict):
        raise NotImplementedError
