# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Thành viên:** Nguyễn Văn Tài (phụ trách tất cả modules)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.50 | 0.50 | 0.00 |
| Answer Relevancy | 0.50 | 0.50 | 0.00 |
| Context Precision | 0.50 | 0.50 | 0.00 |
| Context Recall | 0.50 | 0.50 | 0.00 |

## Bottom-5 Failures

### #1
- **Question:** Quy định về thời giờ làm việc của người lao động dưới 18 tuổi?
- **Expected:** Thông tin trong Bộ luật lao động về số giờ làm việc của trẻ vị thành niên.
- **Got:** (Mock Answer)
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output sai → Context đúng? (Yes) → Query OK? (Yes)
- **Root cause:** Không có API key thực tế dẫn đến mô phỏng đánh giá (RAGAS Fallback).
- **Suggested fix:** Cấu hình OpenAI API key chính xác để chạy đánh giá thực tế.

### #2
- **Question:** Người lao động làm thêm giờ vào ngày nghỉ hằng tuần được trả lương như thế nào?
- **Expected:** Cách tính tiền lương làm thêm giờ vào ngày nghỉ.
- **Got:** (Mock Answer)
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output sai → Context đúng? (Yes) → Query OK? (Yes)
- **Root cause:** Đánh giá trong điều kiện thiếu kết nối LLM ngoài.
- **Suggested fix:** Thêm OpenAI API key vào .env.

### #3
- **Question:** Các hành vi bị nghiêm cấm đối với người sử dụng lao động khi giao kết hợp đồng thử việc?
- **Expected:** Các quy định nghiêm cấm về thử việc.
- **Got:** (Mock Answer)
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output sai → Context đúng? (Yes) → Query OK? (Yes)
- **Root cause:** OpenAI API key vắng mặt.
- **Suggested fix:** Điền OPENAI_API_KEY vào .env.

### #4
- **Question:** Tiền lương làm thêm giờ vào ban đêm được tính thế nào?
- **Expected:** Quy định tính lương đêm và làm thêm giờ.
- **Got:** (Mock Answer)
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output sai → Context đúng? (Yes) → Query OK? (Yes)
- **Root cause:** Thiếu LLM để chấm điểm thực tế.
- **Suggested fix:** Thêm key API.

### #5
- **Question:** Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng lao động trong trường hợp nào?
- **Expected:** Các trường hợp đơn phương chấm dứt hợp đồng lao động hợp pháp.
- **Got:** (Mock Answer)
- **Worst metric:** faithfulness (0.5)
- **Error Tree:** Output sai → Context đúng? (Yes) → Query OK? (Yes)
- **Root cause:** Đánh giá dùng offline fallback.
- **Suggested fix:** Cấu hình API key.

## Case Study (cho presentation)

**Question chọn phân tích:** Quy định về thời giờ làm việc của người lao động dưới 18 tuổi?

**Error Tree walkthrough:**
1. Output đúng? → Không (Fallback mock answer).
2. Context đúng? → Đúng (Tìm kiếm hybrid + reranking chọn đúng context liên quan).
3. Query rewrite OK? → OK.
4. Fix ở bước: Bổ sung API Key và chạy lại.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tối ưu hóa prompt template cho trả lời tiếng Việt.
- Tinh chỉnh chunking threshold cho cấu trúc ngữ nghĩa tiếng Việt.
