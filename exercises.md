# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```

```
vinuni ❯ ./vinuni/bin/python -m pytest Day01-lab-assignment/tests -v
================================== test session starts ==================================
platform linux -- Python 3.14.5, pytest-9.0.3, pluggy-1.6.0 -- /home/vuvanhoc/Study/Applied_AI_Talent/vinuni/bin/python
cachedir: .pytest_cache
rootdir: /home/vuvanhoc/Study/Applied_AI_Talent
plugins: anyio-4.13.0
collected 19 items                                                                      

Day01-lab-assignment/tests/test_solution.py::TestCallOpenAI::test_latency_is_positive_float PASSED [  5%]
Day01-lab-assignment/tests/test_solution.py::TestCallOpenAI::test_returns_non_empty_string PASSED [ 10%]
Day01-lab-assignment/tests/test_solution.py::TestCallOpenAI::test_returns_tuple_of_two PASSED [ 15%]
Day01-lab-assignment/tests/test_solution.py::TestCallOpenAIMini::test_latency_is_positive_float PASSED [ 21%]
Day01-lab-assignment/tests/test_solution.py::TestCallOpenAIMini::test_returns_non_empty_string PASSED [ 26%]
Day01-lab-assignment/tests/test_solution.py::TestCallOpenAIMini::test_returns_tuple_of_two PASSED [ 31%]
Day01-lab-assignment/tests/test_solution.py::TestCompareModels::test_cost_estimate_is_non_negative PASSED [ 36%]
Day01-lab-assignment/tests/test_solution.py::TestCompareModels::test_latency_values_are_positive PASSED [ 42%]
Day01-lab-assignment/tests/test_solution.py::TestCompareModels::test_responses_are_non_empty_strings PASSED [ 47%]
Day01-lab-assignment/tests/test_solution.py::TestCompareModels::test_returns_dict_with_required_keys PASSED [ 52%]
Day01-lab-assignment/tests/test_solution.py::TestStreamingChatbot::test_exits_on_quit PASSED [ 57%]
Day01-lab-assignment/tests/test_solution.py::TestStreamingChatbot::test_function_exists_and_is_callable PASSED [ 63%]
Day01-lab-assignment/tests/test_solution.py::TestRetryWithBackoff::test_raises_after_max_retries PASSED [ 68%]
Day01-lab-assignment/tests/test_solution.py::TestRetryWithBackoff::test_retries_on_transient_exception PASSED [ 73%]
Day01-lab-assignment/tests/test_solution.py::TestRetryWithBackoff::test_succeeds_on_first_try PASSED [ 78%]
Day01-lab-assignment/tests/test_solution.py::TestBatchCompare::test_result_contains_prompt_key PASSED [ 84%]
Day01-lab-assignment/tests/test_solution.py::TestBatchCompare::test_returns_correct_length PASSED [ 89%]
Day01-lab-assignment/tests/test_solution.py::TestFormatComparisonTable::test_contains_column_headers PASSED [ 94%]
Day01-lab-assignment/tests/test_solution.py::TestFormatComparisonTable::test_returns_string PASSED [100%]

================================== 19 passed in 1.57s ===================================

```

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp như 0.0, phản hồi thường ổn định, trực tiếp và ít biến thể hơn. Khi tăng lên 0.5, 1.0 và 1.5, câu trả lời có xu hướng đa dạng, sáng tạo và bất ngờ hơn, nhưng cũng dễ kém nhất quán hoặc lan man hơn.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt khoảng 0.2–0.4 cho chatbot hỗ trợ khách hàng. Mức này giúp câu trả lời nhất quán, ít bịa hơn và vẫn đủ tự nhiên để giao tiếp với người dùng.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Tổng số token mỗi ngày là 10.000 × 3 × 350 = 10.500.000 token. Với giá output trong bài, GPT-4o là 0.010 USD/1K token còn GPT-4o-mini là 0.0006 USD/1K token, nên GPT-4o đắt hơn khoảng 0.010 / 0.0006 = 16,7 lần. Ước tính chi phí mỗi ngày: GPT-4o khoảng 105 USD, GPT-4o-mini khoảng 6,3 USD.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> GPT-4o xứng đáng khi tác vụ cần chất lượng lập luận cao, ví dụ phân tích tài liệu phức tạp, tư vấn chuyên sâu hoặc xử lý yêu cầu quan trọng của khách hàng doanh nghiệp. GPT-4o-mini phù hợp hơn cho các tác vụ nhiều lượt, đơn giản và nhạy chi phí như FAQ, phân loại ticket, tóm tắt ngắn hoặc chatbot hỗ trợ cơ bản.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi dài hoặc người dùng cần cảm giác hệ thống đang xử lý ngay, ví dụ chatbot tương tác, giải thích từng bước, viết nội dung dài hoặc hỗ trợ coding. Nó giúp giảm cảm giác chờ đợi vì người dùng thấy kết quả xuất hiện dần. Non-streaming phù hợp hơn khi phản hồi ngắn, cần xử lý hậu kỳ trước khi hiển thị, hoặc khi ứng dụng cần nhận toàn bộ output để parse JSON, kiểm tra format, lưu log hay đưa vào một bước xử lý tiếp theo.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định
