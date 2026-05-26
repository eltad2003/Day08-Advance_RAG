# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-05-26  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Scorecard Baseline (Ước lượng):**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.5 /5 |
| Answer Relevance | 4.0 /5 |
| Context Recall | 3.5 /5 |
| Completeness | 3.5 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
1. "SLA P1 2026" - Context recall thấp vì Dense chỉ bắt được thông tin thay đổi thời gian (semantic match với "SLA", "2026"), nhưng bỏ lỡ định nghĩa P1 do không ưu tiên keyword "P1".
2. "ERR-403-AUTH" - Dense search trả về các chunk có ý nghĩa gần giống về IT Support nhưng không chứa mã lỗi chính xác.

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias (P1, v4, ERR-403)
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [ ] Generation: Prompt không đủ grounding

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-05-26  
**Biến thay đổi:** `retrieval_mode = "hybrid"` (Dense + BM25)  
**Lý do chọn biến này:**
Chọn hybrid vì corpus chứa nhiều thuật ngữ viết tắt (SLA, P1), tên phiên bản (v4, v2026.1) và mã lỗi kỹ thuật. Dense search mạnh về ngữ nghĩa nhưng yếu về độ chính xác từ khóa. Hybrid (kết hợp BM25) sẽ giúp đảm bảo các tài liệu chứa đúng keyword quan trọng được đưa vào prompt.

**Config thay đổi:**
```
retrieval_mode = "hybrid"
dense_weight = 0.6
sparse_weight = 0.4
# Các tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1 (Kết quả thực tế từ Sprint 3):**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.5/5 | 4.8/5 | +0.3 |
| Answer Relevance | 4.0/5 | 4.5/5 | +0.5 |
| Context Recall | 3.5/5 | 4.8/5 | +1.3 |
| Completeness | 3.5/5 | 4.7/5 | +1.2 |

**Nhận xét:**
- **Cải thiện rõ rệt:** Query "SLA P1 2026" trên Hybrid đã trích xuất được cả định nghĩa P1 (Khẩn cấp, API gateway down) và thông tin cập nhật thời gian. Trong khi đó, Dense chỉ lấy được thông tin cập nhật thời gian.
- **Độ tin cậy:** Hybrid giúp lọc bỏ các chunk "trông có vẻ giống" nhưng không chứa đúng keyword, giúp LLM trả lời tập trung và chính xác hơn.

**Kết luận:**
Variant 1 (Hybrid) là lựa chọn tối ưu cho hệ thống này. Nó kết hợp được khả năng hiểu câu hỏi tự nhiên của Dense và khả năng bắt đúng mã lỗi/thuật ngữ của BM25.
