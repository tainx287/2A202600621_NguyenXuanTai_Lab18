# Individual Reflection — Lab 18

**Tên:** Nguyễn Văn Tài  
**Module phụ trách:** M1, M2, M3, M4, M5

---

## 1. Đóng góp kỹ thuật

- Module đã implement: Tất cả các module từ M1 đến M5.
- Các hàm/class chính đã viết:
  - `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware` trong `src/m1_chunking.py`.
  - `segment_vietnamese`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion` trong `src/m2_search.py`.
  - `CrossEncoderReranker` trong `src/m3_rerank.py`.
  - `evaluate_ragas`, `failure_analysis` trong `src/m4_eval.py`.
  - `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `_enrich_single_call` trong `src/m5_enrichment.py`.
- Số tests pass: 37/37

## 2. Kiến thức học được

- Khái niệm mới nhất: RRF (Reciprocal Rank Fusion) để kết hợp BM25 và Dense Search, cùng với Contextual Prepend để giảm retrieval failure.
- Điều bất ngờ nhất: Sự cải thiện rõ rệt của Hybrid Search so với chỉ sử dụng Dense Search hay BM25 đơn lẻ khi xử lý tiếng Việt.
- Kết nối với bài giảng (slide nào): Slide về Advanced Chunking và Hybrid Search / Reranking.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Tách từ tiếng Việt bị lỗi khi BM25 khớp do underthesea dùng dấu gạch dưới `_`.
- Cách giải quyết: Thay thế dấu gạch dưới bằng khoảng trắng trong hàm `segment_vietnamese`.
- Thời gian debug: 1 giờ.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Sử dụng thêm các mô hình Reranker nhẹ hơn để tối ưu hóa thời gian tải và suy luận trên CPU.
- Module nào muốn thử tiếp: Module 5 Enrichment kết hợp với Graph RAG.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
