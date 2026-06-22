# Hướng Dẫn Chi Tiết Thực Hiện Lab 18: Production RAG Pipeline

Tài liệu này cung cấp hướng dẫn từng bước chi tiết để thiết lập, lập trình, chạy thử nghiệm và đánh giá hệ thống **Production RAG Pipeline** cho Lab 18.

---

## 📌 Tổng Quan Hệ Thống

Hệ thống RAG nâng cao trong bài lab này bao gồm 5 Module chính được kết nối với nhau:
```
[Dữ liệu MD/PDF] ──> M1 Chunking ──> M5 Enrichment ──> M2 Hybrid Search ──> M3 Rerank ──> LLM ──> M4 RAGAS Eval
```

Mục tiêu là cải tiến giải pháp Baseline (chỉ sử dụng Paragraph Chunking + Dense Search đơn giản) bằng các kỹ thuật nâng cao giúp tăng độ chính xác tìm kiếm (Context Precision, Context Recall), độ chính xác câu trả lời (Faithfulness, Answer Relevancy) và tối ưu hóa chi phí API LLM.

---

## 🛠️ Bước 1: Setup Môi Trường & Chạy Baseline (10 phút)

1. **Khởi động Qdrant Vector Database**:
   Qdrant được chạy cục bộ qua Docker:
   ```bash
   docker compose up -d
   ```
   *Kiểm tra:* Truy cập giao diện Qdrant Web UI tại `http://localhost:6333/dashboard`.

2. **Cài đặt các thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình biến môi trường**:
   Sao chép file mẫu `.env.example` thành `.env` và điền khóa API OpenAI của bạn:
   ```bash
   cp .env.example .env
   # Sửa file .env: OPENAI_API_KEY=your_key_here
   ```

4. **Tải trước các Mô hình (Tránh Timeout khi chạy chính thức)**:
   Chạy các lệnh Python sau để tải trước các mô hình Sentence Transformer, Embedding và Reranker về máy cục bộ:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
   python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
   ```

5. **Chạy thử nghiệm Baseline**:
   Chạy script baseline trước khi bắt đầu thực hiện các module nâng cao để có điểm so sánh:
   ```bash
   python naive_baseline.py
   ```
   *Kết quả:* Điểm số baseline sẽ được lưu vào file `naive_baseline_report.json`. Ghi lại các điểm số này để so sánh hiệu năng.

---

## 💻 Bước 2: Hướng Dẫn Lập Trình Từng Module

### 🔹 Module 1: Advanced Chunking (`src/m1_chunking.py`)
Mục tiêu là chia tài liệu thành các phân mảnh nhỏ hơn nhưng bảo toàn ý nghĩa ngữ nghĩa và cấu trúc.

1. **Semantic Chunking (`chunk_semantic`)**:
   *   **Ý tưởng**: Tách văn bản thành các câu. Dùng mô hình embedding (`all-MiniLM-L6-v2`) để lấy vector cho từng câu. Tính độ tương đồng Cosine giữa 2 câu liên tiếp. Nếu độ tương đồng nhỏ hơn ngưỡng cấu hình (`SEMANTIC_THRESHOLD` trong `config.py`), tạo một chunk mới.
   *   **Gợi ý thuật toán**:
       ```python
       from sentence_transformers import SentenceTransformer
       import re
       import numpy as np

       # Tách câu bằng regex phù hợp
       sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)
       sentences = [s.strip() for s in sentences if s.strip()]
       if not sentences:
           return []

       model = SentenceTransformer("all-MiniLM-L6-v2")
       embeddings = model.encode(sentences)

       # Tính độ tương đồng cosine
       def cosine_similarity(a, b):
           return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

       chunks = []
       current_group = [sentences[0]]
       for i in range(1, len(sentences)):
           sim = cosine_similarity(embeddings[i-1], embeddings[i])
           if sim < threshold:
               chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "strategy": "semantic"}))
               current_group = [sentences[i]]
           else:
               current_group.append(sentences[i])
       if current_group:
           chunks.append(Chunk(text=" ".join(current_group), metadata={**metadata, "strategy": "semantic"}))
       ```

2. **Hierarchical Chunking (`chunk_hierarchical`)**:
   *   **Ý tưởng**: Tạo ra 2 mức độ chunk. Chunk cha (lớn hơn, vd: 2048 ký tự) chứa toàn bộ ngữ cảnh xung quanh và các chunk con (nhỏ hơn, vd: 256 ký tự) phục vụ tìm kiếm chính xác. Khi tìm thấy chunk con phù hợp, ta trả về nội dung của chunk cha để cung cấp đầy đủ ngữ cảnh nhất cho LLM.
   *   **Gợi ý thuật toán**:
       1. Tách văn bản theo các đoạn (`\n\n`) thành các chunk cha với độ dài tối đa là `parent_size`. Mỗi chunk cha có mã định danh duy nhất (VD: `parent_{index}`).
       2. Với mỗi chunk cha, chia nhỏ tiếp thành các chunk con có độ dài tối đa là `child_size`. Gán metadata `parent_id` trong mỗi chunk con trỏ về ID của chunk cha tương ứng.

3. **Structure-Aware Chunking (`chunk_structure_aware`)**:
   *   **Ý tưởng**: Phân tích cú pháp các tiêu đề Markdown (`#`, `##`, `###`) để tách tài liệu theo cấu trúc phân mục. Tránh cắt nhỏ bảng biểu (`table`), khối mã lệnh (`code blocks`), hoặc các danh sách (`lists`) để giữ tính trọn vẹn của thông tin.
   *   **Gợi ý thuật toán**:
       Dùng regex `re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)` để tách cấu trúc. Duyệt qua các phân đoạn: nếu dòng hiện tại là tiêu đề, lưu lại làm tiêu đề của phần (section) hiện hành và gộp nội dung phía dưới vào section đó. Trả về các Chunk với metadata chứa khóa `"section"`.

*Kiểm tra Module 1:* Chạy lệnh: `pytest tests/test_m1.py`

---

### 🔹 Module 2: Hybrid Search (`src/m2_search.py`)
Mục tiêu là kết hợp thế mạnh của tìm kiếm từ khóa (BM25 - bắt tốt các thuật ngữ chính xác) và tìm kiếm ngữ nghĩa (Dense - bắt tốt ngữ cảnh và từ đồng nghĩa).

1. **Vietnamese Word Segmentation (`segment_vietnamese`)**:
   Sử dụng thư viện `underthesea` để tách từ tiếng Việt. Tuy nhiên, `underthesea` dùng dấu gạch dưới `_` để nối các từ ghép (Ví dụ: `nghỉ_phép`). Vì vậy ta cần thay thế dấu `_` thành dấu cách `" "` để BM25 không xem từ ghép đó là một từ đơn độc lập khó khớp.
   ```python
   from underthesea import word_tokenize
   segmented = word_tokenize(text, format="text")
   return segmented.replace("_", " ")
   ```

2. **BM25 Search (`BM25Search`)**:
   *   **Index**: Với mỗi chunk, thực hiện tách từ bằng hàm `segment_vietnamese` sau đó split theo khoảng trắng để tạo tập hợp token cho tài liệu đó. Khởi tạo `BM25Okapi(self.corpus_tokens)`.
   *   **Search**: Tách từ truy vấn (query), lấy điểm số BM25 cho tất cả tài liệu, sắp xếp giảm dần và lấy các tài liệu có điểm số tốt nhất (lọc các kết quả có điểm > 0).

3. **Dense Search (`DenseSearch`)**:
   *   **Index**: Khởi tạo Qdrant collection với kích thước vector `EMBEDDING_DIM = 1024` và độ đo tương đồng `Distance.COSINE`. Mã hóa toàn bộ text của chunks bằng mô hình `BAAI/bge-m3`. Upload các điểm dữ liệu `PointStruct` lên Qdrant thông qua phương thức `upsert`.
   *   **Search**: Mã hóa query thành vector, dùng phương thức `client.query_points()` (phù hợp với qdrant-client phiên bản mới) để truy vấn.

4. **Reciprocal Rank Fusion (`reciprocal_rank_fusion`)**:
   *   **Công thức**: Kết hợp 2 danh sách kết quả được xếp hạng từ BM25 và Dense Search bằng phương pháp RRF:
       $$\text{RRF Score}(d) = \sum_{m \in \text{methods}} \frac{1}{k + \text{rank}_m(d) + 1}$$
       *(trong đó $k$ mặc định là 60, và $\text{rank}$ bắt đầu từ 0).*
   *   **Gợi ý thuật toán**: Duyệt qua từng danh sách, tính điểm RRF cộng dồn cho mỗi văn bản dựa trên thứ tự xếp hạng (rank) của nó, sắp xếp danh sách kết quả theo điểm RRF giảm dần và lấy `top_k` phần tử đầu tiên với thuộc tính `method="hybrid"`.

*Kiểm tra Module 2:* Chạy lệnh: `pytest tests/test_m2.py`

---

### 🔹 Module 3: Reranking (`src/m3_rerank.py`)
Mục tiêu là sắp xếp lại danh sách kết quả tìm kiếm thô (thường lấy khoảng 20 kết quả) để chọn ra 3 kết quả tốt nhất và phù hợp nhất đưa vào ngữ cảnh gửi cho LLM.

*   Sử dụng mô hình Cross-Encoder `BAAI/bge-reranker-v2-m3` thông qua lớp `sentence_transformers.CrossEncoder`.
*   Tạo danh sách cặp `(query, doc_text)` cho tất cả các văn bản đầu vào.
*   Dùng hàm `model.predict(pairs)` để dự đoán điểm mức độ liên quan.
*   Sắp xếp lại các tài liệu theo điểm dự đoán giảm dần, trả về danh sách kết quả thuộc kiểu `RerankResult` giới hạn ở `top_k` phần tử tốt nhất.

*Kiểm tra Module 3:* Chạy lệnh: `pytest tests/test_m3.py`

---

### 🔹 Module 4: RAGAS Evaluation (`src/m4_eval.py`)
Mục tiêu là tự động hóa quá trình đánh giá chất lượng hệ thống RAG bằng RAGAS framework.

1. **RAGAS Evaluation (`evaluate_ragas`)**:
   *   Chuyển đổi các danh sách dữ liệu (questions, answers, contexts, ground_truths) thành định dạng Hugging Face `Dataset`.
   *   Gọi hàm `evaluate` từ thư viện `ragas` với 4 độ đo chính: `faithfulness`, `answer_relevancy`, `context_precision`, và `context_recall`.
   *   Trả về một từ điển chứa điểm trung bình của 4 độ đo cùng danh sách chi tiết điểm số từng câu hỏi (`per_question`).

2. **Failure Analysis (`failure_analysis`)**:
   *   **Ý tưởng**: Định vị lỗi dựa trên Diagnostic Tree:
       *   `faithfulness` thấp: LLM đang hallucinate (bốc thuốc/bịa thông tin) -> Cần thắt chặt Prompt, giảm nhiệt độ (`temperature`).
       *   `context_recall` thấp: Tìm kiếm bị bỏ sót các chunk chứa thông tin đúng -> Cần cải thiện giải thuật Chunking hoặc bổ sung tìm kiếm BM25.
       *   `context_precision` thấp: Context chứa quá nhiều chunk rác/không liên quan -> Cần tối ưu Reranking hoặc bổ sung bộ lọc metadata.
       *   `answer_relevancy` thấp: Câu trả lời lạc đề, không đúng trọng tâm câu hỏi -> Cần tối ưu hóa Prompt Template trả lời.
   *   **Thuật toán**: Tính điểm trung bình của 4 chỉ số trên từng câu hỏi, sắp xếp tăng dần để tìm ra $N$ câu hỏi có điểm số kém nhất. Với mỗi câu hỏi tệ nhất, tìm chỉ số có điểm thấp nhất và gán nhãn `diagnosis` cùng giải pháp `suggested_fix` tương ứng từ cây chẩn đoán lỗi.

*Kiểm tra Module 4:* Chạy lệnh: `pytest tests/test_m4.py`

---

### 🔹 Module 5: Enrichment Pipeline (`src/m5_enrichment.py`)
Mục tiêu là "làm giàu" dữ liệu chunk trước khi tạo vector embedding để tăng chất lượng khớp từ khóa và ngữ cảnh.

Ta sẽ triển khai chế độ **Combined Single-Call Mode** (gộp 4 yêu cầu làm giàu vào 1 cuộc gọi API duy nhất) để tiết kiệm chi phí và tối ưu hóa thời gian phản hồi:
*   **Prompt thiết kế**: Yêu cầu LLM phân tích chunk văn bản hiện tại và trả về định dạng JSON chứa các thuộc tính:
    1.  `summary`: Tóm tắt ngắn gọn 2-3 câu.
    2.  `questions`: Danh sách 3 câu hỏi giả định mà đoạn văn có thể trả lời (HyQA).
    3.  `context`: Câu ngắn gọn mô tả vị trí và vai trò của đoạn văn này trong tài liệu tổng thể (Contextual Prepend).
    4.  `metadata`: Trích xuất tự động từ khóa, thực thể (entities), và nhãn phân loại (category).
*   **Fallback**: Cần hiện thực các khối `try/except` cùng giải pháp fallback bằng giải thuật xử lý chuỗi cơ bản trong Python để đảm bảo chương trình không bị crash và vẫn chạy được khi không có khóa API OpenAI.

*Kiểm tra Module 5:* Chạy lệnh: `pytest tests/test_m5.py`

---

## 🚀 Bước 3: Chạy Toàn Bộ Hệ Thống & Đánh Giá

1.  **Chạy pipeline chính thức**:
    ```bash
    python src/pipeline.py
    ```
    Script này sẽ liên kết 5 module bạn đã hiện thực: tải tài liệu, thực hiện chunking, làm giàu dữ liệu, lập chỉ mục vector, truy vấn và chấm điểm RAGAS. Điểm số kết quả được lưu tại `reports/ragas_report.json`.

2.  **Chạy kiểm tra tính đầy đủ trước khi nộp bài**:
    ```bash
    python check_lab.py
    ```
    Script này tự động kiểm tra xem bạn có thiếu file nào trong danh sách nộp bài hay không, kiểm tra các tests có vượt qua hay không và đếm các dòng chứa dấu đánh dấu `# TODO`. Hãy đảm bảo không còn dòng TODO nào và toàn bộ kiểm thử đều vượt qua (100% pass).

---

## 📝 Bước 4: Viết Báo Cáo & Reflection

Sau khi chạy xong hệ thống, bạn cần hoàn thiện các tài liệu trong thư mục `analysis/`:

1.  **Failure Analysis (`analysis/failure_analysis.md`)**:
    *   Mở file `reports/ragas_report.json` vừa được tạo ra.
    *   Lấy danh sách 5 câu hỏi có điểm số kém nhất (Bottom-5 worst questions).
    *   Điền thông tin phân tích vào file `analysis/failure_analysis.md` kèm theo chẩn đoán lỗi (Diagnostic) và phương án khắc phục dựa trên các lỗi thực tế xuất hiện.

2.  **Personal Reflection (`analysis/reflections/reflection_[HọTên].md`)**:
    *   **Phần 1: Mapping bài giảng**: Nêu rõ cách các khái niệm lý thuyết trong slide (Semantic chunking, BM25 + Dense Fusion, Cross-Encoder Reranking, RAGAS, Contextual Embeddings) tương ứng với những hàm cụ thể bạn vừa viết.
    *   **Phần 2: Khó khăn & giải pháp**: Ghi lại các lỗi thực tế bạn đã gặp phải trong quá trình debug (kèm lỗi hệ thống chi tiết) và cách xử lý.
    *   **Phần 3: Action Plan**: Lên kế hoạch chi tiết áp dụng các cấu trúc RAG nâng cao này vào đồ án môn học hoặc dự án thực tế của cá nhân/nhóm bạn.

---

Chúc các bạn thực hiện bài lab thành công và đạt kết quả tốt nhất!
