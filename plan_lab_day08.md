# Kế hoạch hoàn thiện Lab Day 08 — Full RAG Pipeline

Tài liệu này phác thảo các bước chi tiết để hoàn thành bài lab RAG Pipeline, đảm bảo đáp ứng đầy đủ tiêu chí trong `SCORING.md`.

---

## 📋 Tổng quan nhiệm vụ

1.  **Sprint 1: Indexing** — Xây dựng pipeline xử lý dữ liệu và lưu vào vector store.
2.  **Sprint 2: Baseline RAG** — Xây dựng hệ thống trả lời cơ bản (Dense Retrieval).
3.  **Sprint 3: Tuning** — Cải tiến hệ thống (Hybrid/Rerank) và so sánh.
4.  **Sprint 4: Evaluation & Reporting** — Chấm điểm, chạy grading questions và viết báo cáo.

---

## 🚀 Lộ trình thực hiện (Step-by-Step)

### Phase 0: Setup (15 phút)
- [ ] Cài đặt dependencies: `pip install -r requirements.txt chromadb rank-bm25 sentence-transformers`.
- [ ] Cấu hình file `.env`: Điền `OPENAI_API_KEY` hoặc `GOOGLE_API_KEY`.
- [ ] Kiểm tra cấu trúc thư mục và dữ liệu mẫu trong `data/docs/`.

### Phase 1: Sprint 1 — Build Index (45 phút)
**File: `index.py`**
- [ ] **Implement `get_embedding()`**: Sử dụng `SentenceTransformer` (local) hoặc `OpenAI` (API).
- [ ] **Hoàn thiện `preprocess_document()`**: 
    - Dùng Regex parse metadata (Source, Department, Effective Date, Access) từ header file text.
    - Chuẩn hóa text (xóa dòng trống thừa).
- [ ] **Nâng cấp `chunk_document()` & `_split_by_size()`**: 
    - Cắt theo paragraph (`\n\n`) thay vì cắt cứng theo số ký tự.
    - Đảm bảo mỗi chunk mang đầy đủ metadata của document.
- [ ] **Implement `build_index()`**: Khởi tạo ChromaDB client, loop qua các file, embed và upsert.
- [ ] **Kiểm chứng**: Chạy `python index.py` và gọi `list_chunks()` để xem kết quả.

### Phase 2: Sprint 2 — Baseline RAG (60 phút)
**File: `rag_answer.py`**
- [ ] **Implement `retrieve_dense()`**: 
    - Embed query.
    - Tìm kiếm similarity trong ChromaDB.
    - Tính toán score (1 - distance).
- [ ] **Implement `call_llm()`**: Kết nối với OpenAI (gpt-4o-mini) hoặc Gemini. Đặt `temperature=0`.
- [ ] **Hoàn thiện `rag_answer()` pipeline**: Kết nối Retrieval -> Generation.
- [ ] **Kiểm chứng**: Chạy `python rag_answer.py` với các câu hỏi mẫu. Đảm bảo có citation `[1]` và biết nói "Không đủ dữ liệu" khi gặp câu hỏi ngoài phạm vi.

### Phase 3: Sprint 3 — Tuning (60 phút)
**File: `rag_answer.py`**
- [ ] **Chọn Variant**: Đề xuất chọn **Hybrid Retrieval** (Dense + BM25) vì dễ implement và hiệu quả cao với dữ liệu chứa mã lỗi/keyword.
- [ ] **Implement `retrieve_sparse()`**: Sử dụng `rank-bm25` trên toàn bộ corpus.
- [ ] **Implement `retrieve_hybrid()`**: Kết hợp kết quả bằng Reciprocal Rank Fusion (RRF).
- [ ] **So sánh**: Chạy `compare_retrieval_strategies()` để lấy evidence cho sự cải tiến.
- [ ] **Ghi log**: Cập nhật `docs/tuning-log.md`.

### Phase 4: Sprint 4 — Eval & Grading (60 phút)
**File: `eval.py`**
- [ ] **Implement Scoring**: 
    - Hoàn thiện `score_context_recall()` (tự động).
    - Cài đặt cơ chế chấm Faithfulness/Relevance/Completeness (Thủ công hoặc LLM-as-Judge).
- [ ] **Chạy Scorecard**:
    - Chạy `run_scorecard(BASELINE_CONFIG)`.
    - Chạy `run_scorecard(VARIANT_CONFIG)`.
    - Chạy `compare_ab()` để so sánh delta.
- [ ] **Grading Run (QUAN TRỌNG - 17:00)**:
    - Khi có `grading_questions.json`, chạy pipeline với cấu hình tốt nhất.
    - Lưu kết quả vào `logs/grading_run.json` theo đúng format yêu cầu.
- [ ] **Documentation**:
    - Hoàn thiện `docs/architecture.md` (có sơ đồ ASCII/Mermaid).
    - Viết `reports/group_report.md`.
    - Mỗi thành viên viết `reports/individual/[ten].md` (500-800 từ).

---

## 🛠 Phân vai (Gợi ý)

| Thành viên | Vai trò | Nhiệm vụ chính |
| :--- | :--- | :--- |
| **Thành viên A** | Tech Lead | Quản lý logic `rag_answer.py`, nối các module, chạy grading run. |
| **Thành viên B** | Retrieval Owner | Phụ trách `index.py`, chunking, metadata, implement Hybrid/BM25. |
| **Thành viên C** | Eval Owner | Phụ trách `eval.py`, chấm điểm, phân tích sai sót của pipeline. |
| **Thành viên D** | Doc Owner | Viết Architecture docs, Tuning log, Group report. |

---

## ⚠️ Lưu ý quan trọng
1. **Hallucination = Penalty**: Thà trả lời "Không biết" còn hơn bịa thông tin.
2. **Deadline**: Code và Logs phải commit trước **18:00**. Report có thể nộp sau.
3. **A/B Rule**: Chỉ thay đổi một biến duy nhất khi so sánh để kết quả có ý nghĩa.
4. **Metadata**: Đảm bảo `source`, `section`, `effective_date` luôn chính xác vì nó ảnh hưởng trực tiếp đến điểm Context Recall.
