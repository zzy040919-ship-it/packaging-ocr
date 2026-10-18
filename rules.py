"""保守的喷码字段提取；格式符合规则不等于读数一定正确。仅依赖标准库。"""
import re
import unicodedata
from datetime import date

DEFAULT_BATCH_PATTERN = r"[A-Z0-9][A-Z0-9-]{3,23}"
DATE_RE = re.compile(
    r"(?<![A-Z0-9])(?P<y>(?:19|20)\d{2})(?:"
    r"(?P<sep>[-./年])(?P<m>\d{1,2})[-./月](?P<d>\d{1,2})日?"
    r"|(?P<cm>\d{2})(?P<cd>\d{2}))(?![A-Z0-9])"
)
LABEL_RE = re.compile(
    r"生产日期|生产时间|制造日期|包装日期|有效期至|保质期至|到期日期|有效日期|失效日期|"
    r"(?<![A-Z])(?:MFG|MFD|PROD|EXP|BBE|BEST\s*BEFORE)(?![A-Z])",
    re.I,
)
PRODUCTION = {"生产日期", "生产时间", "制造日期", "MFG", "MFD", "PROD"}
BATCH_RE = re.compile(r"(?:生产批号|产品批号|批次号|批号|批次|(?<![A-Z])(?:BATCH|LOT)(?![A-Z]))\s*(?:NO\.?\s*)?[:：#]?\s*([^\s:：]+)", re.I)


def normalize(text):
    return unicodedata.normalize("NFKC", str(text)).upper().strip()


def _finish(candidates):
    if not candidates:
        return {"value": None, "status": "MISSING", "reason": "未找到字段", "candidates": []}
    values = {c["value"] or c["raw"] for c in candidates}
    if len(values) > 1:
        return {"value": None, "status": "AMBIGUOUS", "reason": "存在多个不同候选，需人工确认", "candidates": candidates}
    best = max(candidates, key=lambda c: (c["status"] == "VALID", c["confidence"]))
    return {**best, "candidates": candidates}


def extract_fields(rows, threshold=0.8, batch_pattern=DEFAULT_BATCH_PATTERN):
    """rows: [{text, confidence}]。文本必须按空间关系组行；不跨整图随意拼接。"""
    if not 0 <= threshold <= 1:
        raise ValueError("置信度阈值必须在 0 到 1 之间")
    if len(batch_pattern) > 200:
        raise ValueError("批次正则过长")
    batch_rule = re.compile(batch_pattern)
    dates, batches = [], []
    for row in rows:
        text, confidence = normalize(row["text"]), float(row["confidence"])
        batch_matches = list(BATCH_RE.finditer(text))
        for match in DATE_RE.finditer(text):
            if any(b.start(1) <= match.start() < b.end(1) for b in batch_matches):
                continue
            labels = list(LABEL_RE.finditer(text[:match.start()]))
            label = labels[-1].group().upper() if labels else None
            # 包装日期、有效期等不当作生产日期。
            if label and label not in PRODUCTION:
                continue
            raw = match.group()
            try:
                value = date(int(match["y"]), int(match["m"] or match["cm"]), int(match["d"] or match["cd"])).isoformat()
                status, reason = "VALID", "日历日期有效且有生产日期标签"
            except ValueError:
                value, status, reason = None, "INVALID", "日期不存在，例如月份越界或非闰年 2 月 29 日"
            if status == "VALID" and not label:
                status, reason = "UNCONFIRMED", "缺少生产日期标签，可能是有效期或其他日期"
            if status == "VALID" and confidence < threshold:
                status, reason = "LOW_CONFIDENCE", "识别置信度不足"
            dates.append(dict(raw=raw, value=value, status=status, reason=reason, confidence=confidence))
        for match in batch_matches:
            raw = match[1].rstrip(";,，；")
            valid = bool(batch_rule.fullmatch(raw))
            status = "VALID" if valid else "INVALID"
            reason = "符合配置的批次格式" if valid else "不符合配置的批次格式"
            if valid and confidence < threshold:
                status, reason = "LOW_CONFIDENCE", "识别置信度不足"
            batches.append(dict(raw=raw, value=raw if valid else None, status=status, reason=reason, confidence=confidence))
    fields = {"production_date": _finish(dates), "batch": _finish(batches)}
    return {"fields": fields, "decision": "PASS" if all(f["status"] == "VALID" for f in fields.values()) else "REVIEW"}
